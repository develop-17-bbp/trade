"""Tests for src/trading/trade_plan.py — the compiled plan-mode artifact."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from src.trading.trade_plan import (
    DEFAULT_VALIDITY_SECONDS,
    ExitCondition,
    TPLevel,
    TradePlan,
    ToolCallEvidence,
)


# ── Happy-path construction ─────────────────────────────────────────────


def _long_plan(**overrides):
    base = dict(
        asset="BTC",
        direction="LONG",
        entry_tier="normal",
        entry_price=60000.0,
        size_pct=5.0,
        sl_price=58800.0,
        tp_levels=[TPLevel(price=61800.0, fraction=0.5), TPLevel(price=63000.0, fraction=0.5)],
        expected_hold_bars=24,
        thesis="EMA-align + macro bias",
        confidence=0.7,
    )
    base.update(overrides)
    return TradePlan(**base)


def test_long_plan_constructs_and_validates():
    plan = _long_plan()
    assert plan.asset == "BTC"
    assert plan.direction == "LONG"
    assert plan.entry_tier == "normal"
    assert plan.size_multiplier() == 1.0
    assert plan.is_stale() is False


def test_short_plan_sl_above_entry_is_required():
    plan = TradePlan(
        asset="ETH", direction="SHORT", entry_tier="normal",
        entry_price=3000.0, size_pct=5.0, sl_price=3060.0,
    )
    assert plan.direction == "SHORT"


def test_sniper_tier_multiplier():
    plan = _long_plan(entry_tier="sniper", size_pct=15.0)
    assert plan.size_multiplier() == 3.0


def test_asset_is_upcased():
    plan = _long_plan(asset="btc")
    assert plan.asset == "BTC"


# ── Validation failures ─────────────────────────────────────────────────


def test_long_with_sl_above_entry_rejected():
    with pytest.raises(ValidationError):
        _long_plan(sl_price=60500.0)


def test_short_with_sl_below_entry_rejected():
    with pytest.raises(ValidationError):
        TradePlan(
            asset="BTC", direction="SHORT", entry_tier="normal",
            entry_price=60000.0, size_pct=5.0, sl_price=59000.0,
        )


def test_long_with_tp_below_entry_rejected():
    with pytest.raises(ValidationError):
        _long_plan(tp_levels=[TPLevel(price=59000.0, fraction=0.5)])


def test_tp_fractions_over_1_rejected():
    with pytest.raises(ValidationError):
        _long_plan(tp_levels=[
            TPLevel(price=61000.0, fraction=0.6),
            TPLevel(price=62000.0, fraction=0.6),
        ])


def test_sniper_oversize_rejected():
    with pytest.raises(ValidationError):
        _long_plan(entry_tier="sniper", size_pct=25.0)


def test_normal_oversize_rejected():
    with pytest.raises(ValidationError):
        _long_plan(entry_tier="normal", size_pct=10.0)


def test_size_pct_above_hard_cap_rejected():
    with pytest.raises(ValidationError):
        _long_plan(size_pct=35.0)


# ── Skip plans are permissive ───────────────────────────────────────────


def test_skip_factory_bypasses_risk_checks():
    plan = TradePlan.skip("BTC", thesis="no setup")
    assert plan.direction == "SKIP"
    assert plan.entry_tier == "skip"
    assert plan.size_pct == 0.0
    assert plan.size_multiplier() == 0.0


def test_flat_plan_zeroes_size():
    plan = TradePlan(
        asset="BTC", direction="FLAT", entry_tier="normal",
        entry_price=1.0, size_pct=5.0, sl_price=1.0,
    )
    # model_validator should coerce tier/size to skip semantics
    assert plan.entry_tier == "skip"
    assert plan.size_pct == 0.0


# ── Staleness ───────────────────────────────────────────────────────────


def test_valid_until_default_is_compiled_at_plus_window():
    plan = _long_plan()
    delta = plan.valid_until - plan.compiled_at
    assert abs(delta.total_seconds() - DEFAULT_VALIDITY_SECONDS) < 0.1


def test_is_stale_after_valid_until():
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    plan = _long_plan(valid_until=past)
    assert plan.is_stale() is True


# ── Serialization round-trip ────────────────────────────────────────────


def test_to_dict_is_json_serialisable():
    plan = _long_plan(
        supporting_evidence=[ToolCallEvidence(tool="get_macro_bias", summary="+0.3 LONG-aligned")],
        exit_conditions=[ExitCondition(kind="regime_change", params={"to": "RANGING"})],
    )
    d = plan.to_dict()
    # Must survive a round-trip through json — critical for warm_store plan_json storage
    s = json.dumps(d)
    back = json.loads(s)
    assert back["asset"] == "BTC"
    assert back["direction"] == "LONG"
    assert back["supporting_evidence"][0]["tool"] == "get_macro_bias"
    assert back["exit_conditions"][0]["kind"] == "regime_change"
    assert isinstance(back["compiled_at"], str)  # ISO string, not datetime
