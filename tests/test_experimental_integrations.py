"""Tests for the three queued multi-day integrations:
  - dual_path_reasoning
  - hierarchical_orchestrator
  - skeptic_persona

Each must:
  1. Be DORMANT when its env flag is unset (no compute, no logs).
  2. Run in shadow mode without affecting any TradePlan output.
  3. Have bounded outputs (no prompt bloat).
  4. Use anti-overfit safeguards (sample-size warnings, weight caps).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest


# ── Dual-Path Reasoning ─────────────────────────────────────────────────


def test_dual_path_dormant_when_env_unset(monkeypatch):
    monkeypatch.delenv("ACT_DUAL_PATH", raising=False)
    from src.ai import dual_path_reasoning as dpr
    assert dpr.is_enabled() is False
    assert dpr.is_authoritative() is False


def test_dual_path_shadow_mode_enables_but_not_authoritative(monkeypatch):
    monkeypatch.setenv("ACT_DUAL_PATH", "shadow")
    from src.ai import dual_path_reasoning as dpr
    assert dpr.is_enabled() is True
    assert dpr.is_authoritative() is False


def test_dual_path_authoritative_only_with_explicit_one(monkeypatch):
    monkeypatch.setenv("ACT_DUAL_PATH", "1")
    from src.ai import dual_path_reasoning as dpr
    assert dpr.is_enabled() is True
    assert dpr.is_authoritative() is True


def test_dual_path_weights_use_prior_when_no_outcomes(monkeypatch):
    """Below MIN_SAMPLES the weights MUST be 50/50 prior — no overfit
    on 2 outcomes."""
    from src.ai import dual_path_reasoning as dpr
    # Force-empty by monkeypatching the read function.
    monkeypatch.setattr(dpr, "_read_recent_outcomes", lambda **kw: [])
    fw, sw, n, src = dpr.compute_weights(min_samples=20)
    assert fw == 0.5 and sw == 0.5
    assert n == 0
    assert src == "prior"


def test_dual_path_synthesis_bounds_confidence(monkeypatch):
    from src.ai import dual_path_reasoning as dpr
    monkeypatch.setattr(dpr, "_read_recent_outcomes", lambda **kw: [])
    fact = dpr.PathVerdict(path="fact", direction="LONG", confidence=0.9)
    subj = dpr.PathVerdict(path="subjectivity", direction="LONG", confidence=0.9)
    s = dpr.synthesize(fact, subj)
    assert 0.0 <= s.confidence <= 1.0
    assert s.direction == "LONG"
    # Both agree → agreement bonus applied
    assert s.confidence > 0.85


def test_dual_path_synthesis_disagreement_yields_skip_or_lower_conf(monkeypatch):
    from src.ai import dual_path_reasoning as dpr
    monkeypatch.setattr(dpr, "_read_recent_outcomes", lambda **kw: [])
    fact = dpr.PathVerdict(path="fact", direction="LONG", confidence=0.6)
    subj = dpr.PathVerdict(path="subjectivity", direction="SHORT", confidence=0.6)
    s = dpr.synthesize(fact, subj)
    # 50/50 prior + opposing direction → blended ≈ 0 → SKIP
    assert s.direction == "SKIP"


def test_dual_path_verdict_rationale_is_capped():
    from src.ai.dual_path_reasoning import PathVerdict
    long_text = "x" * 2000
    v = PathVerdict(path="fact", direction="LONG", confidence=0.5,
                    rationale=long_text, inputs_used=["x"] * 50)
    d = v.to_dict()
    assert len(d["rationale"]) <= 400
    assert len(d["inputs_used"]) <= 20


# ── Hierarchical Orchestrator ───────────────────────────────────────────


def test_hierarchy_dormant_when_env_unset(monkeypatch):
    monkeypatch.delenv("ACT_AGENT_HIERARCHY", raising=False)
    from src.agents import hierarchical_orchestrator as hier
    assert hier.is_enabled() is False
    assert hier.is_authoritative() is False


def test_hierarchy_shadow_enables_but_not_authoritative(monkeypatch):
    monkeypatch.setenv("ACT_AGENT_HIERARCHY", "shadow")
    from src.agents import hierarchical_orchestrator as hier
    assert hier.is_enabled() is True
    assert hier.is_authoritative() is False


def test_hierarchy_decision_aggregates_5_analyst_votes(monkeypatch):
    from src.agents import hierarchical_orchestrator as hier
    votes = {
        "market_structure": {"direction": 1, "confidence": 0.7},
        "regime_intelligence": {"direction": 1, "confidence": 0.6},
        "trend_momentum": {"direction": 1, "confidence": 0.8},
        "mean_reversion": {"direction": 0, "confidence": 0.4},
        "pattern_matcher": {"direction": 1, "confidence": 0.65},
        "trade_timing": {"direction": 1, "confidence": 0.7},
        "risk_guardian": {"direction": 0, "confidence": 0.5},
        "loss_prevention_guardian": {"direction": 0, "confidence": 0.5},
        "authority_compliance_guardian": {"direction": 0, "confidence": 0.5},
        "portfolio_optimizer": {"direction": 1, "confidence": 0.6},
        "decision_auditor": {"direction": 1, "confidence": 0.6},
    }
    d = hier.hierarchical_decide("BTC", votes)
    assert d.analyst_team.direction == 1
    assert d.trader.direction == 1
    assert d.final_direction == 1
    assert d.final_confidence > 0


def test_hierarchy_risk_team_can_veto():
    from src.agents import hierarchical_orchestrator as hier
    votes = {
        "market_structure": {"direction": 1, "confidence": 0.8},
        "regime_intelligence": {"direction": 1, "confidence": 0.8},
        "trend_momentum": {"direction": 1, "confidence": 0.8},
        "mean_reversion": {"direction": 1, "confidence": 0.8},
        "pattern_matcher": {"direction": 1, "confidence": 0.8},
        "trade_timing": {"direction": 1, "confidence": 0.8},
        # Risk team strongly disagrees
        "risk_guardian": {"direction": -1, "confidence": 0.8},
        "loss_prevention_guardian": {"direction": -1, "confidence": 0.8},
        "authority_compliance_guardian": {"direction": -1, "confidence": 0.7},
        "portfolio_optimizer": {"direction": 1, "confidence": 0.6},
        "decision_auditor": {"direction": 1, "confidence": 0.6},
    }
    d = hier.hierarchical_decide("BTC", votes)
    assert d.risk_team.veto is True
    assert d.final_direction == 0
    assert d.final_confidence == 0.0


def test_hierarchy_low_sample_calibration_flag():
    from src.agents import hierarchical_orchestrator as hier
    # Empty votes → empty stages → low sample (warm_store will be near-empty)
    d = hier.hierarchical_decide("BTC", {})
    # Sample size from warm_store; on a fresh test box may be small.
    assert d.confidence_calibration in ("ok", "low_sample")
    # All stage rationales bounded
    assert len(d.analyst_team.rationale) <= 200
    assert len(d.trader.rationale) <= 200
    assert len(d.risk_team.rationale) <= 200
    assert len(d.orchestrator.rationale) <= 200


# ── Skeptic Persona ──────────────────────────────────────────────────────


def test_skeptic_dormant_when_env_unset(monkeypatch):
    monkeypatch.delenv("ACT_SKEPTIC", raising=False)
    monkeypatch.delenv("ACT_SKEPTIC_WEIGHT", raising=False)
    from src.agents import skeptic_persona as sk
    assert sk.is_enabled() is False
    assert sk.is_authoritative() is False
    # evaluate returns None when dormant
    v = sk.evaluate("BTC", "LONG", 0.85, tick_snap={})
    assert v is None


def test_skeptic_enabled_but_zero_weight_is_advisory_only(monkeypatch):
    monkeypatch.setenv("ACT_SKEPTIC", "1")
    monkeypatch.setenv("ACT_SKEPTIC_WEIGHT", "0.0")
    from src.agents import skeptic_persona as sk
    assert sk.is_enabled() is True
    assert sk.is_authoritative() is False
    assert sk.get_veto_weight() == 0.0


def test_skeptic_stays_quiet_on_low_consensus_confidence(monkeypatch):
    monkeypatch.setenv("ACT_SKEPTIC", "1")
    from src.agents import skeptic_persona as sk
    # 0.5 consensus < 0.7 threshold → quiet
    assert sk.evaluate("BTC", "LONG", 0.5, tick_snap={"macro_bias": -0.5}) is None


def test_skeptic_stays_quiet_on_skip_consensus(monkeypatch):
    monkeypatch.setenv("ACT_SKEPTIC", "1")
    from src.agents import skeptic_persona as sk
    assert sk.evaluate("BTC", "SKIP", 0.85, tick_snap={"macro_bias": -0.5}) is None


def test_skeptic_finds_counter_signals(monkeypatch):
    monkeypatch.setenv("ACT_SKEPTIC", "1")
    from src.agents import skeptic_persona as sk
    snap = {
        "macro_bias": -0.6,           # bearish, contradicts LONG
        "hurst_value": 0.40,           # mean-reverting
        "regime": "CRISIS",            # crisis regardless
        "ml_meta_dir": -1,             # ML disagrees
        "vpin_toxic": True,            # toxic flow
        "spread_pct": 1.69,
        "open_positions_same_asset": 5,
    }
    v = sk.evaluate("BTC", "LONG", 0.85, tick_snap=snap)
    assert v is not None
    assert len(v.counter_signals) >= 4
    assert v.veto_strength > 0
    assert "macro_bias_bearish" in " ".join(v.counter_signals)
    # Output bounded
    assert len(v.skeptic_argument) <= 400
    assert len(v.counter_signals) <= 10


def test_skeptic_veto_only_with_weight(monkeypatch):
    monkeypatch.setenv("ACT_SKEPTIC", "1")
    monkeypatch.setenv("ACT_SKEPTIC_WEIGHT", "0.0")  # advisory only
    from src.agents import skeptic_persona as sk
    snap = {"macro_bias": -0.6, "hurst_value": 0.40, "regime": "CRISIS",
            "ml_meta_dir": -1, "vpin_toxic": True}
    v = sk.evaluate("BTC", "LONG", 0.85, tick_snap=snap)
    assert v is not None
    assert v.veto_applied is False  # weight 0 → no veto

    monkeypatch.setenv("ACT_SKEPTIC_WEIGHT", "1.0")  # max weight
    # Re-evaluate with same snap; veto_applied becomes True when
    # weight × strength >= 0.6
    v2 = sk.evaluate("BTC", "LONG", 0.85, tick_snap=snap)
    assert v2 is not None
    assert v2.veto_applied is True


def test_skeptic_format_for_brain_empty_when_quiet(monkeypatch):
    monkeypatch.delenv("ACT_SKEPTIC", raising=False)
    from src.agents import skeptic_persona as sk
    assert sk.format_for_brain(None) == ""


def test_skeptic_format_for_brain_renders_advisory(monkeypatch):
    monkeypatch.setenv("ACT_SKEPTIC", "1")
    from src.agents import skeptic_persona as sk
    v = sk.evaluate("BTC", "LONG", 0.85,
                    tick_snap={"macro_bias": -0.6, "hurst_value": 0.40})
    line = sk.format_for_brain(v)
    assert "SKEPTIC" in line
    assert len(line) <= 500
