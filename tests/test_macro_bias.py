"""Tests for src/trading/macro_bias.py — aggregator that turns the 12 economic
intelligence layers into a signed bias + size multiplier."""
from __future__ import annotations

import pytest


def _summary(
    bullish=0, bearish=0, active=12, composite="NEUTRAL",
    confidence=0.5, macro_risk=30, pre_event=False,
    usd_regime="neutral", crisis=False,
):
    return {
        "composite_signal": composite,
        "composite_confidence": confidence,
        "usd_regime": usd_regime,
        "macro_risk": macro_risk,
        "pre_event_flag": pre_event,
        "top_risks": [], "top_tailwinds": [],
        "active_layers": active,
        "total_layers": 12,
        "bullish_count": bullish,
        "bearish_count": bearish,
        "crisis": crisis,
    }


def test_neutral_input_produces_neutral_bias():
    from src.trading.macro_bias import compute_macro_bias

    b = compute_macro_bias(_summary(bullish=3, bearish=3))
    assert b.signed_bias == 0.0
    assert b.crisis is False
    assert b.size_multiplier == 1.0   # neutral macro = 1.0x, no amplification
    assert "neutral" in b.reasons[0]


def test_strong_bullish_produces_positive_bias():
    from src.trading.macro_bias import compute_macro_bias

    b = compute_macro_bias(_summary(bullish=7, bearish=1, composite="BULLISH", confidence=0.75))
    assert b.signed_bias > 0
    assert b.crisis is False
    # Size multiplier should amplify above 1.0 but below cap
    assert 1.0 < b.size_multiplier <= 1.5


def test_strong_bearish_produces_negative_bias():
    from src.trading.macro_bias import compute_macro_bias

    b = compute_macro_bias(_summary(bullish=1, bearish=7, composite="BEARISH", confidence=0.75))
    assert b.signed_bias < 0
    # magnitude produces amplification, but direction is negative
    assert 1.0 < b.size_multiplier <= 1.5   # still amplifies ABS bias


def test_crisis_flag_forces_zero_size():
    from src.trading.macro_bias import compute_macro_bias

    b = compute_macro_bias(_summary(bullish=2, bearish=4, crisis=True))
    assert b.crisis is True
    assert b.size_multiplier == 0.0
    assert b.signed_bias == -1.0


def test_high_macro_risk_triggers_crisis():
    """macro_risk >= 80 AND confidence >= 0.70 should auto-crisis even without explicit flag."""
    from src.trading.macro_bias import compute_macro_bias

    b = compute_macro_bias(_summary(bullish=0, bearish=10, macro_risk=85, confidence=0.8))
    assert b.crisis is True
    assert b.size_multiplier == 0.0


def test_high_macro_risk_but_low_confidence_is_not_crisis():
    """macro_risk >= 80 but confidence < 0.70 should NOT trigger crisis."""
    from src.trading.macro_bias import compute_macro_bias

    b = compute_macro_bias(_summary(bullish=0, bearish=10, macro_risk=85, confidence=0.3))
    assert b.crisis is False


def test_pre_event_halves_bias():
    from src.trading.macro_bias import compute_macro_bias

    no_event = compute_macro_bias(_summary(bullish=7, bearish=1, confidence=0.75, pre_event=False))
    pre_event = compute_macro_bias(_summary(bullish=7, bearish=1, confidence=0.75, pre_event=True))
    assert pre_event.signed_bias < no_event.signed_bias
    assert abs(pre_event.signed_bias) == pytest.approx(abs(no_event.signed_bias) * 0.5, rel=0.01)


def test_aligned_long_with_bullish_bias():
    from src.trading.macro_bias import compute_macro_bias

    b = compute_macro_bias(_summary(bullish=6, bearish=1, confidence=0.6))
    assert b.aligned("LONG") is True
    assert b.aligned("SHORT") is False


def test_aligned_neutral_accepts_both():
    from src.trading.macro_bias import compute_macro_bias

    b = compute_macro_bias(_summary(bullish=3, bearish=3))
    assert b.aligned("LONG") is True
    assert b.aligned("SHORT") is True


def test_apply_direction_alignment_fades_unaligned():
    from src.trading.macro_bias import compute_macro_bias, apply_direction_alignment

    bull_bias = compute_macro_bias(_summary(bullish=7, bearish=1, confidence=0.75))
    aligned = apply_direction_alignment(bull_bias, "LONG", 1.0)
    unaligned = apply_direction_alignment(bull_bias, "SHORT", 1.0)
    assert aligned == 1.0
    assert unaligned < 1.0 and unaligned > 0.0


def test_missing_summary_returns_neutral():
    from src.trading.macro_bias import compute_macro_bias

    b = compute_macro_bias(None)
    assert b.signed_bias == 0.0
    assert b.crisis is False
    assert b.size_multiplier == 1.0

    b2 = compute_macro_bias({})
    assert b2.signed_bias == 0.0


def test_malformed_summary_returns_neutral():
    from src.trading.macro_bias import compute_macro_bias

    b = compute_macro_bias({"bullish_count": "not a number"})
    assert b.signed_bias == 0.0
    assert "malformed" in b.reasons[0]


def test_is_enabled_respects_flag(monkeypatch):
    from src.trading.macro_bias import is_enabled

    monkeypatch.delenv("ACT_ROBINHOOD_HARDEN", raising=False)
    assert is_enabled() is False
    monkeypatch.setenv("ACT_ROBINHOOD_HARDEN", "1")
    assert is_enabled() is True
    monkeypatch.setenv("ACT_ROBINHOOD_HARDEN", "0")
    assert is_enabled() is False
