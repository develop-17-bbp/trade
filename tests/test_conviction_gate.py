"""Tests for src/trading/conviction_gate.py — tiered entry filter."""
from __future__ import annotations

import pytest

from src.trading.conviction_gate import evaluate
from src.trading.macro_bias import MacroBias


def _bias(signed=0.0, crisis=False):
    return MacroBias(signed_bias=signed, crisis=crisis, size_multiplier=1.0, confidence=0.7)


def test_sniper_when_all_aligned_bullish():
    r = evaluate(
        direction="LONG",
        tf_1h_direction="RISING", tf_4h_direction="RISING",
        hurst_regime="trending",
        multi_strategy_counts={"long": 7, "short": 1, "flat": 28},
        macro_bias=_bias(signed=+0.4),
    )
    assert r.tier == "sniper"
    assert r.passed is True
    assert r.size_multiplier == 3.0


def test_normal_when_tf_aligned_but_hurst_not_trending():
    r = evaluate(
        direction="LONG",
        tf_1h_direction="RISING", tf_4h_direction="RISING",
        hurst_regime="random",
        multi_strategy_counts={"long": 4, "short": 1, "flat": 31},
        macro_bias=_bias(signed=+0.1),
    )
    assert r.tier == "normal"
    assert r.passed is True
    assert r.size_multiplier == 1.0


def test_reject_when_tf_not_aligned():
    r = evaluate(
        direction="LONG",
        tf_1h_direction="FALLING", tf_4h_direction="RISING",
        hurst_regime="trending",
        multi_strategy_counts={"long": 5, "short": 0, "flat": 31},
        macro_bias=_bias(signed=+0.4),
    )
    assert r.tier == "reject"
    assert r.passed is False
    assert r.size_multiplier == 0.0
    assert any("tf_not_aligned" in x for x in r.reasons)


def test_reject_when_multi_strategy_weak():
    r = evaluate(
        direction="LONG",
        tf_1h_direction="RISING", tf_4h_direction="RISING",
        hurst_regime="trending",
        multi_strategy_counts={"long": 2, "short": 3, "flat": 31},  # only 2 long
        macro_bias=_bias(signed=+0.3),
    )
    assert r.tier == "reject"
    assert any("multistrat_weak" in x for x in r.reasons)


def test_reject_when_macro_crisis():
    r = evaluate(
        direction="LONG",
        tf_1h_direction="RISING", tf_4h_direction="RISING",
        hurst_regime="trending",
        multi_strategy_counts={"long": 10, "short": 0, "flat": 26},
        macro_bias=MacroBias(crisis=True, size_multiplier=0.0),
    )
    assert r.tier == "reject"
    assert "macro_crisis" in r.reasons
    assert r.checks["macro_crisis_free"] is False


def test_reject_when_macro_misaligned_with_direction():
    """LONG trade but macro bias strongly BEARISH → reject."""
    r = evaluate(
        direction="LONG",
        tf_1h_direction="RISING", tf_4h_direction="RISING",
        hurst_regime="trending",
        multi_strategy_counts={"long": 6, "short": 1, "flat": 29},
        macro_bias=_bias(signed=-0.5),
    )
    assert r.tier == "reject"
    assert any("macro_misaligned" in x for x in r.reasons)


def test_normal_with_neutral_macro_still_passes():
    """Neutral macro (bias=0) aligns with both LONG and SHORT — should pass to normal."""
    r = evaluate(
        direction="LONG",
        tf_1h_direction="RISING", tf_4h_direction="RISING",
        hurst_regime="random",
        multi_strategy_counts={"long": 3, "short": 1, "flat": 32},
        macro_bias=_bias(signed=0.0),
    )
    assert r.tier == "normal"


def test_sniper_requires_stronger_macro_than_normal():
    """Strong TF + strategies + macro aligned but WEAK magnitude -> normal, not sniper."""
    r = evaluate(
        direction="LONG",
        tf_1h_direction="RISING", tf_4h_direction="RISING",
        hurst_regime="trending",
        multi_strategy_counts={"long": 8, "short": 0, "flat": 28},
        macro_bias=_bias(signed=+0.05),  # barely bullish — below 0.20 threshold
    )
    assert r.tier == "normal"
    assert r.checks["macro_strong_lean"] is False


def test_short_direction_handling():
    """SHORT trade with bearish everything → sniper."""
    r = evaluate(
        direction="SHORT",
        tf_1h_direction="FALLING", tf_4h_direction="FALLING",
        hurst_regime="trending",
        multi_strategy_counts={"long": 0, "short": 6, "flat": 30},
        macro_bias=_bias(signed=-0.4),
    )
    assert r.tier == "sniper"


def test_unknown_direction_rejects():
    r = evaluate(
        direction="FLAT",
        tf_1h_direction="RISING", tf_4h_direction="RISING",
        hurst_regime="trending",
        multi_strategy_counts={"long": 6, "short": 0, "flat": 30},
        macro_bias=_bias(signed=+0.3),
    )
    assert r.tier == "reject"
    assert any("unknown_direction" in x for x in r.reasons)


def test_case_insensitive_direction():
    r = evaluate(
        direction="buy",
        tf_1h_direction="rising", tf_4h_direction="RISING",
        hurst_regime="TRENDING",
        multi_strategy_counts={"long": 4, "short": 1, "flat": 31},
        macro_bias=_bias(signed=+0.1),
    )
    assert r.tier == "normal"


def test_result_to_dict_schema():
    """UI + logging consumers need this shape to be stable."""
    r = evaluate(
        direction="LONG",
        tf_1h_direction="RISING", tf_4h_direction="RISING",
        hurst_regime="trending",
        multi_strategy_counts={"long": 6, "short": 0, "flat": 30},
        macro_bias=_bias(signed=+0.3),
    )
    d = r.to_dict()
    required = {"tier", "passed", "direction", "size_multiplier", "checks", "reasons"}
    assert required.issubset(d.keys())
