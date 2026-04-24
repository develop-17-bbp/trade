"""Tests for conviction_gate regime hysteresis (C19).

Fresh entry uses adopt-thresholds (strict); in-position uses
hold-thresholds (looser) so a signal dip mid-trade doesn't force an exit.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.trading.conviction_gate import evaluate


@dataclass
class _StubMacro:
    signed_bias: float = 0.25
    crisis: bool = False

    def aligned(self, direction: str) -> bool:
        # Aligned with LONGs when bias > 0.
        if direction.upper() == "LONG":
            return self.signed_bias > 0
        return self.signed_bias < 0


def test_fresh_entry_requires_5_strategies_for_sniper():
    r = evaluate(
        direction="LONG",
        tf_1h_direction="RISING", tf_4h_direction="RISING",
        hurst_regime="trending",
        multi_strategy_counts={"long": 4, "short": 0, "flat": 0},
        macro_bias=_StubMacro(signed_bias=0.30),
        in_position=False,
    )
    # Only 4 strategies agreeing — not enough for sniper, but enough for normal (>=3)
    assert r.tier == "normal"


def test_holding_accepts_4_strategies_for_sniper():
    """Hysteresis: while in position, sniper floor drops to 4."""
    r = evaluate(
        direction="LONG",
        tf_1h_direction="RISING", tf_4h_direction="RISING",
        hurst_regime="trending",
        multi_strategy_counts={"long": 4, "short": 0, "flat": 0},
        macro_bias=_StubMacro(signed_bias=0.20),
        in_position=True,
    )
    assert r.tier == "sniper"
    # Size multiplier for sniper is 3x
    assert r.size_multiplier == 3.0


def test_holding_accepts_2_strategies_for_normal():
    """Hysteresis: normal floor drops from 3 to 2 while holding."""
    r = evaluate(
        direction="LONG",
        tf_1h_direction="RISING", tf_4h_direction="RISING",
        hurst_regime="random",   # not trending, so not sniper
        multi_strategy_counts={"long": 2, "short": 0, "flat": 0},
        macro_bias=_StubMacro(signed_bias=0.15),
        in_position=True,
    )
    assert r.tier == "normal"


def test_fresh_entry_rejects_2_strategies():
    r = evaluate(
        direction="LONG",
        tf_1h_direction="RISING", tf_4h_direction="RISING",
        hurst_regime="trending",
        multi_strategy_counts={"long": 2, "short": 0, "flat": 0},
        macro_bias=_StubMacro(signed_bias=0.30),
        in_position=False,
    )
    assert r.tier == "reject"


def test_macro_magnitude_relaxed_when_holding():
    """Hysteresis: sniper macro magnitude drops 30% when holding."""
    # bias=0.14 is under fresh-entry 0.20 but above hold-threshold 0.14
    r_fresh = evaluate(
        direction="LONG",
        tf_1h_direction="RISING", tf_4h_direction="RISING",
        hurst_regime="trending",
        multi_strategy_counts={"long": 6, "short": 0, "flat": 0},
        macro_bias=_StubMacro(signed_bias=0.14),
        in_position=False,
    )
    assert r_fresh.tier == "normal"   # macro_strong_lean fails -> normal not sniper

    r_hold = evaluate(
        direction="LONG",
        tf_1h_direction="RISING", tf_4h_direction="RISING",
        hurst_regime="trending",
        multi_strategy_counts={"long": 6, "short": 0, "flat": 0},
        macro_bias=_StubMacro(signed_bias=0.14),
        in_position=True,
    )
    assert r_hold.tier == "sniper"    # hold threshold is 0.14 → passes


def test_crisis_still_rejects_regardless_of_hysteresis():
    r = evaluate(
        direction="LONG",
        tf_1h_direction="RISING", tf_4h_direction="RISING",
        hurst_regime="trending",
        multi_strategy_counts={"long": 6, "short": 0, "flat": 0},
        macro_bias=_StubMacro(signed_bias=0.30, crisis=True),
        in_position=True,
    )
    assert r.tier == "reject"
