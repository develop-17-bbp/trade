"""Authority-rule enforcement tests.

These tests prove each of the non-negotiable authority rules actually
vetoes a trade when its precondition fires. If any of these regress,
the guardian is silently passing rule-violating trades.
"""

from __future__ import annotations

import pytest

from src.ai.authority_rules import (
    ASSET_TRADE_PERMISSIONS,
    AUTHORITY_MAX_HOLD_HOURS,
    DEFAULT_MAX_HOLD_HOURS,
    UNIVERSAL_RULES,
    get_max_hold_hours,
    validate_authority_entry,
)


# ── Per-asset max hold ──────────────────────────────────────────────────


def test_btc_allows_swing_hold():
    # BTC can swing → 10 days
    assert get_max_hold_hours("BTC") == 240.0
    assert get_max_hold_hours("BTCUSDT") == 240.0


def test_eth_capped_at_intraday():
    # ETH intraday only → 48h
    assert get_max_hold_hours("ETH") == 48.0
    assert get_max_hold_hours("ETHUSDT") == 48.0


def test_unknown_alt_defaults_to_intraday_cap():
    # Unknown altcoin should get the safest (shortest) cap.
    assert get_max_hold_hours("SOL") == DEFAULT_MAX_HOLD_HOURS
    assert get_max_hold_hours("") == DEFAULT_MAX_HOLD_HOURS


# ── Asset permission: ETH cannot swing ──────────────────────────────────


def test_eth_swing_rejected():
    ok, violations = validate_authority_entry(
        quant_state={},
        context={"raw_signal": 1, "asset": "ETH", "trade_type": "swing"},
    )
    assert ok is False
    assert any("ASSET_PERMISSION" in v for v in violations)


def test_btc_swing_accepted():
    ok, violations = validate_authority_entry(
        quant_state={},
        context={
            "raw_signal": 1,
            "asset": "BTC",
            "trade_type": "swing",
            "htf_trend_direction": 1,
        },
    )
    # May still fail other checks if data missing, but NOT for permission.
    assert not any("ASSET_PERMISSION" in v for v in violations)


def test_eth_intraday_accepted():
    ok, violations = validate_authority_entry(
        quant_state={},
        context={
            "raw_signal": 1,
            "asset": "ETH",
            "trade_type": "intraday",
            "htf_trend_direction": 1,
        },
    )
    assert not any("ASSET_PERMISSION" in v for v in violations)


# ── Higher-TF agreement ─────────────────────────────────────────────────


def test_htf_disagreement_vetoes():
    ok, violations = validate_authority_entry(
        quant_state={},
        context={
            "raw_signal": 1,            # want to go LONG
            "asset": "BTC",
            "trade_type": "intraday",
            "htf_trend_direction": -1,  # higher TF is DOWN
        },
    )
    assert ok is False
    assert any("HTF_DISAGREEMENT" in v for v in violations)


def test_htf_neutral_does_not_veto():
    # Weak/no HTF trend → don't block the trade
    ok, violations = validate_authority_entry(
        quant_state={},
        context={
            "raw_signal": 1,
            "asset": "BTC",
            "trade_type": "intraday",
            "htf_trend_direction": 0,
        },
    )
    assert not any("HTF_DISAGREEMENT" in v for v in violations)


# ── Wick entry ──────────────────────────────────────────────────────────


def test_wick_entry_vetoes():
    ok, violations = validate_authority_entry(
        quant_state={},
        context={
            "raw_signal": 1,
            "asset": "BTC",
            "trade_type": "intraday",
            "entry_on_wick": True,
        },
    )
    assert ok is False
    assert any("WICK_ENTRY" in v for v in violations)


# ── Small body ──────────────────────────────────────────────────────────


def test_small_body_vetoes():
    ok, violations = validate_authority_entry(
        quant_state={},
        context={
            "raw_signal": 1,
            "asset": "BTC",
            "trade_type": "intraday",
            "candle_body_pct": 0.001,
            "avg_body_pct_10_50": 0.005,   # current body 5× smaller than avg
        },
    )
    assert ok is False
    assert any("SMALL_BODY" in v for v in violations)


def test_strong_body_passes():
    ok, violations = validate_authority_entry(
        quant_state={},
        context={
            "raw_signal": 1,
            "asset": "BTC",
            "trade_type": "intraday",
            "candle_body_pct": 0.006,
            "avg_body_pct_10_50": 0.005,
        },
    )
    assert not any("SMALL_BODY" in v for v in violations)


# ── Universal rules completeness ────────────────────────────────────────


def test_seven_universal_rules_defined():
    # The authority PDF enumerates seven universal rules (memory:trading_rules).
    # If this count changes, the audit system needs to know.
    assert len(UNIVERSAL_RULES) == 7


def test_btc_has_all_three_permissions():
    assert set(ASSET_TRADE_PERMISSIONS["BTC"]) == {"scalp", "intraday", "swing"}


def test_eth_missing_swing_permission():
    assert "swing" not in ASSET_TRADE_PERMISSIONS["ETH"]
    assert "intraday" in ASSET_TRADE_PERMISSIONS["ETH"]
