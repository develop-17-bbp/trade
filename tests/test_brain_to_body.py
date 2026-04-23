"""Tests for src/learning/brain_to_body.py — C9 controller."""
from __future__ import annotations

import pytest

from src.learning.brain_to_body import (
    DEFAULT_GENETIC_CADENCE_S,
    DISABLE_ENV,
    EMERGENCY_LEVEL_CAUTION,
    EMERGENCY_LEVEL_NORMAL,
    EMERGENCY_LEVEL_STRESS,
    BodyController,
    BodyControls,
    compute_controls,
    current_emergency_level,
    current_exploration_bias,
    current_genetic_cadence_s,
    current_priority_agents,
    get_controller,
)


# ── compute_controls: neutral + emergency axis ─────────────────────────


def test_neutral_inputs_yield_neutral_controls():
    c = compute_controls(scan_scores=[], analyst_verdicts=[],
                          critique_matches=[], top_signals=[])
    assert c.exploration_bias == 1.0
    assert c.genetic_cadence_s == DEFAULT_GENETIC_CADENCE_S
    assert c.emergency_level == EMERGENCY_LEVEL_NORMAL
    assert c.priority_agents           # default fallback set
    assert c.scans_considered == 0


def test_emergency_mode_env_forces_stress():
    c = compute_controls(
        scan_scores=[60], analyst_verdicts=["plan"] * 10,
        critique_matches=[True] * 10, top_signals=["trend"],
        emergency_mode=True,
    )
    assert c.emergency_level == EMERGENCY_LEVEL_STRESS
    # Stress → explore harder.
    assert c.exploration_bias < 1.0
    # Stress → genetic cadence tightened.
    assert c.genetic_cadence_s < DEFAULT_GENETIC_CADENCE_S


def test_low_match_rate_triggers_stress():
    c = compute_controls(
        scan_scores=[50] * 10, analyst_verdicts=["plan"] * 10,
        critique_matches=[False] * 20,         # 0% match
        top_signals=["trend"],
    )
    assert c.emergency_level == EMERGENCY_LEVEL_STRESS


def test_mediocre_match_rate_triggers_caution():
    # 40% match over 20 critiques → below 50% threshold, above 30%.
    matches = [True] * 8 + [False] * 12
    c = compute_controls(
        scan_scores=[50] * 10, analyst_verdicts=["plan"] * 10,
        critique_matches=matches, top_signals=["trend"],
    )
    assert c.emergency_level == EMERGENCY_LEVEL_CAUTION


def test_high_parse_failure_rate_triggers_stress():
    c = compute_controls(
        scan_scores=[50] * 10,
        analyst_verdicts=["parse_failures"] * 6 + ["plan"] * 4,
        critique_matches=[], top_signals=[],
    )
    assert c.emergency_level == EMERGENCY_LEVEL_STRESS


def test_high_skip_rate_triggers_caution_and_tightens_genetic():
    verdicts = (["skip"] * 19) + ["plan"]  # 95% skip
    c = compute_controls(
        scan_scores=[55] * 10, analyst_verdicts=verdicts,
        critique_matches=[True] * 10, top_signals=[],
    )
    assert c.emergency_level == EMERGENCY_LEVEL_CAUTION
    # Genetic cadence should be < default (evolve sooner).
    assert c.genetic_cadence_s < DEFAULT_GENETIC_CADENCE_S


# ── compute_controls: hot path → exploit ───────────────────────────────


def test_hot_scanner_high_match_exploits_bandit():
    c = compute_controls(
        scan_scores=[72, 74, 76, 75] * 3,           # avg > 70
        analyst_verdicts=["plan"] * 15,
        critique_matches=[True] * 13 + [False] * 2,  # ~87% match, >10 samples
        top_signals=["trend", "breakout"],
    )
    assert c.emergency_level == EMERGENCY_LEVEL_NORMAL
    assert c.exploration_bias > 1.5   # exploit hard


def test_warm_scanner_moderate_match_mild_exploit():
    c = compute_controls(
        scan_scores=[63] * 10,
        analyst_verdicts=["plan"] * 10,
        critique_matches=[True] * 6 + [False] * 4,
        top_signals=["momentum"],
    )
    assert 1.0 < c.exploration_bias < 1.5


# ── Priority agents from top_signals ──────────────────────────────────


def test_priority_agents_mapped_from_signals():
    c = compute_controls(
        scan_scores=[55], analyst_verdicts=[],
        critique_matches=[], top_signals=["macro", "whale", "breakout"],
    )
    # Expect one agent per keyword match, deduped, order-preserving.
    assert "ask_regime_intelligence" in c.priority_agents   # macro
    assert "ask_sentiment_decoder" in c.priority_agents     # whale
    assert "ask_trend_momentum" in c.priority_agents        # breakout


def test_priority_agents_default_when_no_signals():
    c = compute_controls(scan_scores=[], analyst_verdicts=[],
                          critique_matches=[], top_signals=[])
    # Must fall back to a safety-first chain.
    assert c.priority_agents
    assert "ask_regime_intelligence" in c.priority_agents


def test_priority_agents_capped_at_five():
    many = ["trend", "whale", "macro", "pattern", "liquidity",
            "correlation", "reversal", "risk"]
    c = compute_controls(scan_scores=[], analyst_verdicts=[],
                          critique_matches=[], top_signals=many)
    assert len(c.priority_agents) <= 5


# ── BodyControls serialization ─────────────────────────────────────────


def test_to_dict_round_trip():
    c = compute_controls(
        scan_scores=[60, 70], analyst_verdicts=["plan", "skip"],
        critique_matches=[True, True], top_signals=["trend"],
    )
    d = c.to_dict()
    for k in ("exploration_bias", "genetic_cadence_s", "emergency_level",
              "priority_agents", "reason", "computed_at",
              "avg_opportunity_score", "analyst_match_rate",
              "analyst_skip_rate", "parse_failure_rate"):
        assert k in d


# ── Controller singleton + refresh ────────────────────────────────────


def test_get_controller_singleton():
    a = get_controller()
    b = get_controller()
    assert a is b


def test_controller_disabled_by_env(monkeypatch, tmp_path):
    monkeypatch.setenv(DISABLE_ENV, "1")
    bc = BodyController()
    controls = bc.refresh()
    assert controls.reason == "disabled by env"
    assert controls.exploration_bias == 1.0


def test_controller_refresh_tolerates_missing_subsystems(tmp_path, monkeypatch):
    """Every subsystem (brain_memory, warm_store, readiness_gate) might
    be unavailable in fresh-install / test envs. Refresh must not raise
    and must return a valid BodyControls."""
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    monkeypatch.setenv("ACT_WARM_DB_PATH", str(tmp_path / "no_such.sqlite"))
    bc = BodyController()
    controls = bc.refresh()
    assert isinstance(controls, BodyControls)
    # Contract: valid emergency level string + default-or-tighter cadence.
    assert controls.emergency_level in (EMERGENCY_LEVEL_NORMAL,
                                        EMERGENCY_LEVEL_CAUTION,
                                        EMERGENCY_LEVEL_STRESS)
    assert controls.genetic_cadence_s <= DEFAULT_GENETIC_CADENCE_S


# ── Convenience functions honor the singleton ─────────────────────────


def test_convenience_fns_read_singleton(monkeypatch):
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    bc = get_controller()
    import src.learning.brain_to_body as bb
    monkeypatch.setattr(bc, "_current", BodyControls(
        exploration_bias=1.7, genetic_cadence_s=3600,
        emergency_level=EMERGENCY_LEVEL_CAUTION,
        priority_agents=["ask_risk_guardian"],
        reason="test",
    ), raising=False)
    assert abs(current_exploration_bias() - 1.7) < 1e-6
    assert current_genetic_cadence_s() == 3600
    assert current_emergency_level() == EMERGENCY_LEVEL_CAUTION
    assert "ask_risk_guardian" in current_priority_agents()


# ── Thompson bandit integration ───────────────────────────────────────


def test_thompson_bandit_honors_controller_bias(monkeypatch):
    """When the controller sets exploration_bias != 1.0, Thompson
    sampling uses it instead of the static EMERGENCY_EXPLOIT_BIAS."""
    from src.learning.thompson_bandit import _effective_exploit_bias

    bc = get_controller()
    monkeypatch.setattr(bc, "_current", BodyControls(exploration_bias=0.5),
                         raising=False)
    # Controller says EXPLORE (0.5); should win over the static 3.0
    # even when emergency_mode is True.
    assert _effective_exploit_bias(emergency_mode=True) == pytest.approx(0.5)


def test_thompson_bandit_falls_back_when_controls_neutral(monkeypatch):
    from src.learning.thompson_bandit import EMERGENCY_EXPLOIT_BIAS, _effective_exploit_bias

    bc = get_controller()
    monkeypatch.setattr(bc, "_current", BodyControls(exploration_bias=1.0),
                         raising=False)
    # Neutral controller → fall back to static emergency path.
    assert _effective_exploit_bias(emergency_mode=True) == EMERGENCY_EXPLOIT_BIAS
    assert _effective_exploit_bias(emergency_mode=False) == 1.0


# ── LLM tool registration ──────────────────────────────────────────────


def test_get_body_controls_tool_registered():
    from src.ai.trade_tools import build_default_registry
    reg = build_default_registry()
    assert reg.get("get_body_controls") is not None


def test_get_body_controls_tool_returns_dict(monkeypatch):
    import json
    bc = get_controller()
    monkeypatch.setattr(bc, "_current", BodyControls(
        exploration_bias=1.2, emergency_level=EMERGENCY_LEVEL_NORMAL,
        priority_agents=["ask_trend_momentum"], reason="test",
    ), raising=False)

    from src.ai.trade_tools import build_default_registry
    reg = build_default_registry()
    out = json.loads(reg.dispatch("get_body_controls", {}))
    assert out["emergency_level"] == EMERGENCY_LEVEL_NORMAL
    assert abs(out["exploration_bias"] - 1.2) < 0.01
