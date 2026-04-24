"""Tests for C26 Step 4 — brain benchmark harness."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.evaluation.brain_benchmark import (
    BenchmarkScenario,
    DEFAULT_TARGET_SCORE,
    PlanScore,
    run_brain_benchmark,
    score_plan_vs_outcome,
    write_benchmark_report,
)


def _scenario(direction="UP", pnl=2.0, regime="TRENDING"):
    return BenchmarkScenario(
        scenario_id="s1", asset="BTC",
        context={"ema": 8}, regime=regime,
        actual_outcome={
            "actual_direction": direction,
            "pnl_pct": pnl, "max_adverse_pct": 0.5,
        },
    )


# ── scorer ──────────────────────────────────────────────────────────────


def test_scorer_rewards_correct_direction():
    s = _scenario(direction="UP", pnl=2.0)
    score = score_plan_vs_outcome(
        {"direction": "LONG", "thesis": "breakout",
         "expected_pnl_pct_range": [1.0, 3.0]},
        s.actual_outcome,
    )
    assert score.direction_correct is True
    assert score.composite >= 0.5


def test_scorer_penalizes_wrong_direction():
    s = _scenario(direction="UP", pnl=2.0)
    score = score_plan_vs_outcome(
        {"direction": "SHORT", "thesis": "breakdown"},
        s.actual_outcome,
    )
    assert score.direction_correct is False
    assert score.composite < 0.5


def test_scorer_flat_when_flat():
    s = _scenario(direction="FLAT", pnl=0.0)
    score_skip = score_plan_vs_outcome({"direction": "SKIP"}, s.actual_outcome)
    score_flat = score_plan_vs_outcome({"direction": "FLAT"}, s.actual_outcome)
    assert score_skip.direction_correct is True
    assert score_flat.direction_correct is True


def test_scorer_thesis_quality_rewards_evidence():
    s = _scenario()
    bare = score_plan_vs_outcome({"direction": "LONG"}, s.actual_outcome)
    with_thesis = score_plan_vs_outcome(
        {"direction": "LONG",
         "thesis": "breakout confirmed on 4h volume with macro alignment",
         "supporting_evidence": ["ml_ensemble", "multi_strategy"]},
        s.actual_outcome,
    )
    assert with_thesis.thesis_quality > bare.thesis_quality


def test_scorer_authority_clean_when_no_violations():
    s = _scenario()
    clean = score_plan_vs_outcome({"direction": "LONG"}, s.actual_outcome)
    dirty = score_plan_vs_outcome(
        {"direction": "LONG", "authority_violations": ["HTF_DISAGREEMENT"]},
        s.actual_outcome,
    )
    assert clean.authority_clean is True
    assert dirty.authority_clean is False


# ── runner ──────────────────────────────────────────────────────────────


def test_run_benchmark_with_injected_runners():
    scenarios = [
        _scenario(direction="UP", pnl=2.0),
        _scenario(direction="DOWN", pnl=-1.5),
        _scenario(direction="FLAT", pnl=0.0),
    ]

    def local(s):
        # Perfect local — always matches the outcome
        d = {"UP": "LONG", "DOWN": "SHORT", "FLAT": "FLAT"}.get(
            s.actual_outcome["actual_direction"], "FLAT"
        )
        return {"direction": d, "thesis": "perfect",
                "supporting_evidence": ["ml_ensemble"]}

    def ref(s):
        # Reference always says FLAT — baseline
        return {"direction": "FLAT", "thesis": "ref stub"}

    result = run_brain_benchmark(
        scenarios=scenarios,
        local_runner=local,
        reference_runner=ref,
    )
    # Local won 3/3 (UP + DOWN direction; on FLAT both agree on
    # direction but local has richer thesis → higher composite).
    assert result.n_scenarios == 3
    assert result.local_wins == 3
    assert result.ref_wins == 0
    assert result.ties == 0


def test_benchmark_quality_score_math():
    scenarios = [_scenario() for _ in range(10)]

    def perfect_local(s):
        return {"direction": "LONG", "thesis": "ok",
                "supporting_evidence": ["x"]}

    def losing_ref(s):
        return {"direction": "SHORT", "thesis": "stub"}

    result = run_brain_benchmark(
        scenarios=scenarios,
        local_runner=perfect_local,
        reference_runner=losing_ref,
    )
    # All 10 scenarios have direction=UP — local is right every time
    assert result.brain_quality_score == 1.0
    assert result.beats_target(DEFAULT_TARGET_SCORE) is True


def test_benchmark_empty_scenarios_graceful():
    result = run_brain_benchmark(scenarios=[])
    assert result.n_scenarios == 0
    assert result.brain_quality_score == 0.0
    assert result.beats_target() is False


def test_report_writer_creates_file(tmp_path, monkeypatch):
    import src.evaluation.brain_benchmark as bb
    monkeypatch.setattr(bb, "PROJECT_ROOT", tmp_path)
    scenarios = [_scenario()]

    result = run_brain_benchmark(
        scenarios=scenarios,
        local_runner=lambda s: {"direction": "LONG", "thesis": "ok"},
        reference_runner=lambda s: {"direction": "FLAT"},
    )
    path = write_benchmark_report(result, stamp="test-stamp")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "Brain Quality Score" in text
    assert "test-stamp" in text


# ── skill integration ──────────────────────────────────────────────────


def test_brain_benchmark_skill_returns_ok(tmp_path, monkeypatch):
    """End-to-end: skill loads, runs with empty scenarios, emits SkillResult."""
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    action_path = root / "skills" / "brain_benchmark" / "action.py"
    spec = importlib.util.spec_from_file_location("_bb_action", str(action_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Stub the runner so this test stays offline
    import src.evaluation.brain_benchmark as bb
    monkeypatch.setattr(bb, "_load_warm_store_scenarios", lambda limit: [])
    monkeypatch.setattr(bb, "_load_curated_scenarios", lambda: [])
    monkeypatch.setattr(bb, "PROJECT_ROOT", tmp_path)

    result = mod.run({"ref_model": "claude-haiku-4-5", "max_scenarios": 50})
    assert result.ok is True
    assert "brain_quality_score" in result.data
    assert "target" in result.data
