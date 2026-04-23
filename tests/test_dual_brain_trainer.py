"""Tests for src/ai/dual_brain_trainer.py — orchestration with stub backend."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

import pytest

from src.ai.dual_brain_trainer import (
    DEFAULT_MIN_SAMPLES,
    DISABLE_AGENTIC_ENV,
    CycleReport,
    StubBackend,
    TrainerResult,
    TrainingJob,
    _hot_swap,
    _train_one_brain,
    persist_report,
    run_cycle,
    split_samples,
)
from src.ai.training_data_filter import ExperienceSample


# ── split_samples ───────────────────────────────────────────────────────


def _es(ts, pnl=1.0, matched=True, label="positive"):
    return ExperienceSample(
        decision_id=f"d-{ts}", asset="BTC", ts=ts, pnl_pct=pnl,
        direction="LONG", matched_thesis=matched, label=label,
        plan={"direction": "LONG", "size_pct": 5.0},
        outcome={"pnl_pct": pnl}, self_critique={"matched_thesis": matched},
        scanner_tag={"top_signals": ["ema_cross"]},
    )


def test_split_empty():
    train, val = split_samples([])
    assert train == [] and val == []


def test_split_preserves_time_order():
    now = time.time()
    samples = [_es(now - i * 60) for i in range(10)]
    train, val = split_samples(samples, validation_split=0.2)
    # Validation should be newest.
    assert all(v.ts >= t.ts for t in train for v in val)
    assert 1 <= len(val) <= 3


def test_split_never_empty_train():
    train, val = split_samples([_es(time.time())], validation_split=0.2)
    assert len(train) == 1
    assert len(val) == 0


# ── StubBackend ────────────────────────────────────────────────────────


def test_stub_backend_records_trained_tags():
    sb = StubBackend()
    sb.train("base", [{"prompt": "x", "completion": "y"}], "base-act-1")
    assert sb.trained_tags == ["base-act-1"]


def test_stub_backend_infer_differs_by_model():
    sb = StubBackend(
        incumbent_direction_match=False, challenger_direction_match=True,
    )
    sb.trained_tags.append("challenger")
    sample = {"direction": "LONG"}
    inc = json.loads(sb.infer("incumbent", sample))
    cha = json.loads(sb.infer("challenger", sample))
    assert inc["direction"] != cha["direction"]
    assert cha["direction"] == "LONG"


# ── _train_one_brain ───────────────────────────────────────────────────


def _make_samples(n=60, matched=True):
    now = time.time()
    return [_es(now - i * 60, matched=matched) for i in range(n)]


def test_train_one_brain_happy_path():
    sb = StubBackend(challenger_direction_match=True,
                     incumbent_direction_match=False)
    train, val = split_samples(_make_samples(60))
    # Pretend-format: since ExperienceSample has plan + scanner_tag populated
    # above, the scanner formatter returns a row; analyst always returns one.
    from src.ai.training_data_filter import format_analyst_sft_example
    result = _train_one_brain(
        sb, brain="analyst", incumbent="base:32b",
        challenger_tag="base:32b-act-1",
        train_set=train, val_set=val,
        format_fn=format_analyst_sft_example,
        min_improvement_pct=2.0, max_regression_pct=5.0,
    )
    assert result.training_ok is True
    assert result.gate is not None
    assert result.promoted is True


def test_train_one_brain_rejects_when_backend_fails():
    sb = StubBackend(train_ok=False)
    train, val = split_samples(_make_samples(60))
    from src.ai.training_data_filter import format_analyst_sft_example
    result = _train_one_brain(
        sb, brain="analyst", incumbent="base", challenger_tag="new",
        train_set=train, val_set=val,
        format_fn=format_analyst_sft_example,
        min_improvement_pct=2.0, max_regression_pct=5.0,
    )
    assert result.training_ok is False
    assert result.promoted is False


def test_train_one_brain_insufficient_rows():
    sb = StubBackend()
    train, val = split_samples(_make_samples(5))     # fewer than 10 rows
    from src.ai.training_data_filter import format_analyst_sft_example
    result = _train_one_brain(
        sb, brain="analyst", incumbent="base", challenger_tag="new",
        train_set=train, val_set=val,
        format_fn=format_analyst_sft_example,
        min_improvement_pct=2.0, max_regression_pct=5.0,
    )
    assert result.training_ok is False
    assert "insufficient" in (result.swap_error or "")


# ── run_cycle (full integration with tmp warm_store) ───────────────────


@pytest.fixture
def warm_store_with_samples(tmp_path, monkeypatch):
    """Seed enough filtered samples to pass min_samples=20."""
    db = tmp_path / "w.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute("""
        CREATE TABLE decisions (
            decision_id TEXT PRIMARY KEY,
            trace_id TEXT, symbol TEXT, ts_ns INTEGER, direction INTEGER,
            confidence REAL, consensus TEXT, veto INTEGER, raw_signal INTEGER,
            final_action TEXT, authority_violations TEXT, payload_json TEXT,
            component_signals TEXT DEFAULT '{}',
            plan_json TEXT DEFAULT '{}',
            self_critique TEXT DEFAULT '{}'
        )
    """)
    conn.execute("""
        CREATE TABLE outcomes (
            outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id TEXT, symbol TEXT, direction TEXT,
            entry_price REAL, exit_price REAL, pnl_pct REAL, pnl_usd REAL,
            duration_s REAL, exit_reason TEXT, regime TEXT,
            entry_ts REAL, exit_ts REAL, payload_json TEXT
        )
    """)
    now = time.time()
    for i in range(40):
        ts_ns = int((now - i * 3600) * 1_000_000_000)
        pnl = 1.2 if i % 2 == 0 else -1.0
        matched = i % 2 == 0   # positives matched, negatives honest
        conn.execute(
            "INSERT INTO decisions (decision_id, symbol, ts_ns, direction, "
            "final_action, plan_json, self_critique, component_signals) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (f"r-{i}", "BTC", ts_ns, 1, "LONG",
             json.dumps({"direction": "LONG", "size_pct": 5.0}),
             json.dumps({"matched_thesis": matched}),
             json.dumps({"top_signals": ["ema_cross"]})),
        )
        conn.execute(
            "INSERT INTO outcomes (decision_id, symbol, direction, pnl_pct, "
            "exit_ts, payload_json) VALUES (?,?,?,?,?,?)",
            (f"r-{i}", "BTC", "LONG", pnl, now - i * 3600,
             json.dumps({"pnl_pct": pnl})),
        )
    conn.commit()
    conn.close()
    monkeypatch.setenv("ACT_WARM_DB_PATH", str(db))
    return db


def test_run_cycle_happy(warm_store_with_samples, monkeypatch):
    monkeypatch.delenv(DISABLE_AGENTIC_ENV, raising=False)
    sb = StubBackend()
    report = run_cycle(
        sb, min_samples=5,
        analyst_incumbent="a:32b", scanner_incumbent="s:7b",
        pause_agentic=True,
    )
    assert report.error is None
    assert report.analyst is not None
    assert report.scanner is not None
    # Both brains were trained.
    assert report.analyst.training_ok is True
    assert report.scanner.training_ok is True
    # Gate outcomes depend on challenger > incumbent; stub default has
    # challenger_direction_match=True so both should promote.
    assert report.analyst.promoted is True
    assert report.scanner.promoted is True
    # Env restored after cycle.
    assert os.environ.get(DISABLE_AGENTIC_ENV) in (None, "", "0")


def test_run_cycle_pauses_and_restores_env(warm_store_with_samples, monkeypatch):
    monkeypatch.setenv(DISABLE_AGENTIC_ENV, "original")
    sb = StubBackend()
    run_cycle(sb, min_samples=5,
              analyst_incumbent="a", scanner_incumbent="s",
              pause_agentic=True)
    # Restored to the caller's original value.
    assert os.environ.get(DISABLE_AGENTIC_ENV) == "original"


def test_run_cycle_insufficient_samples(warm_store_with_samples):
    sb = StubBackend()
    report = run_cycle(
        sb, min_samples=10_000,   # unattainable
        analyst_incumbent="a", scanner_incumbent="s",
        pause_agentic=False,
    )
    assert report.error is not None
    assert "< min_samples" in report.error
    assert report.analyst is None


# ── Hot-swap + persist ─────────────────────────────────────────────────


def test_hot_swap_sets_env(monkeypatch):
    monkeypatch.delenv("ACT_ANALYST_MODEL", raising=False)
    err = _hot_swap("analyst", "my-new:32b")
    assert err is None
    assert os.environ.get("ACT_ANALYST_MODEL") == "my-new:32b"


def test_persist_report_writes_file(tmp_path):
    report = CycleReport(
        started_at=time.time() - 1, finished_at=time.time(),
        filter_stats={"total": 5, "kept_positive": 3},
    )
    path = persist_report(report, out_dir=str(tmp_path))
    assert path is not None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["filter_stats"]["total"] == 5


# ── Skill integration ──────────────────────────────────────────────────


def test_skill_registered():
    from src.skills.registry import get_registry
    reg = get_registry(refresh=True)
    assert reg.get("fine-tune-brain") is not None


def test_skill_requires_confirm():
    from src.skills.registry import get_registry
    reg = get_registry(refresh=True)
    r = reg.dispatch("fine-tune-brain", {}, invoker="operator")
    assert r.ok is False
    assert "confirm" in (r.error or "").lower()


def test_skill_dry_run(warm_store_with_samples):
    from src.skills.registry import get_registry
    reg = get_registry(refresh=True)
    r = reg.dispatch("fine-tune-brain",
                     {"confirm": True, "dry_run": True, "min_samples": 5},
                     invoker="operator")
    # Dry-run uses StubBackend — must complete ok + report artifacts.
    assert r.ok is True
    assert r.data.get("backend") == "stub"
    assert r.data.get("analyst") is not None
