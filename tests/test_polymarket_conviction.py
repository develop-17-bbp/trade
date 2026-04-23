"""Tests for Polymarket-specific conviction gate."""
from __future__ import annotations

import time

import pytest

from src.trading.polymarket_conviction import (
    NORMAL_MIN_EDGE,
    NORMAL_MIN_EV_PCT,
    PolymarketConvictionResult,
    SNIPER_MIN_EDGE,
    SNIPER_MIN_EV_PCT,
    evaluate,
)


def _market(yes=0.40, vol=20_000, hours=12, end_override=None):
    now = time.time()
    end = end_override if end_override is not None else now + hours * 3600
    return {
        "market_id": "test-mkt",
        "question": "Will BTC close above $70k by Friday?",
        "yes_price": yes,
        "no_price": 1.0 - yes,
        "end_ts": end,
        "volume_24h": vol,
    }


# ── Input validation ───────────────────────────────────────────────────


def test_invalid_side_rejects():
    r = evaluate(
        market=_market(), proposed_side="MAYBE",
        estimated_probability=0.6, equity_usd=10_000,
    )
    assert r.tier == "reject"
    assert "invalid side" in r.reasons[0].lower()


def test_invalid_probability_rejects():
    r = evaluate(
        market=_market(), proposed_side="YES",
        estimated_probability=1.5, equity_usd=10_000,
    )
    assert r.tier == "reject"
    assert "probability" in r.reasons[0].lower()


def test_expired_market_rejects():
    r = evaluate(
        market=_market(end_override=time.time() - 3600), proposed_side="YES",
        estimated_probability=0.6, equity_usd=10_000,
    )
    assert r.tier == "reject"
    assert "expired" in r.reasons[0].lower()


def test_malformed_market_rejects():
    r = evaluate(
        market={"yes_price": "nonsense"}, proposed_side="YES",
        estimated_probability=0.6, equity_usd=10_000,
    )
    # yes_price gets float("nonsense") -> ValueError → caught.
    assert r.tier == "reject"


# ── Sniper tier ────────────────────────────────────────────────────────


def test_sniper_promotes_with_big_edge_and_liquid_short_market():
    # yes_price = 0.40 (market thinks 40%)
    # estimated = 0.60 (we think 60%) → +0.20 edge on YES
    # High vol, 12h to expiry.
    r = evaluate(
        market=_market(yes=0.40, vol=20_000, hours=12),
        proposed_side="YES", estimated_probability=0.60,
        equity_usd=10_000,
    )
    assert r.tier == "sniper"
    assert r.passed is True
    assert r.shares > 0
    assert r.expected_value_usd > 0


def test_sniper_rejects_when_expiry_too_far():
    r = evaluate(
        market=_market(yes=0.40, vol=20_000, hours=200),   # >24h
        proposed_side="YES", estimated_probability=0.60,
        equity_usd=10_000,
    )
    # Falls through to normal (still liquid, still edge)
    assert r.tier in ("normal", "reject")


def test_sniper_rejects_when_illiquid():
    r = evaluate(
        market=_market(yes=0.40, vol=500, hours=12),    # below sniper liquidity
        proposed_side="YES", estimated_probability=0.60,
        equity_usd=10_000,
    )
    # Should fall through to normal if volume is still above normal floor.
    assert r.tier in ("normal", "reject")


# ── Normal tier ────────────────────────────────────────────────────────


def test_normal_tier_on_moderate_edge():
    # Moderate edge 0.08 — above normal (0.05), below sniper (0.12).
    r = evaluate(
        market=_market(yes=0.40, vol=5_000, hours=48),
        proposed_side="YES", estimated_probability=0.48,
        equity_usd=10_000,
    )
    assert r.tier == "normal"
    assert r.passed is True
    assert r.shares > 0


def test_normal_rejects_when_edge_too_small():
    r = evaluate(
        market=_market(yes=0.40, vol=5_000, hours=48),
        proposed_side="YES", estimated_probability=0.42,   # edge 0.02 < 0.05
        equity_usd=10_000,
    )
    assert r.tier == "reject"


def test_normal_rejects_when_too_illiquid():
    r = evaluate(
        market=_market(yes=0.40, vol=100, hours=48),   # far below normal floor
        proposed_side="YES", estimated_probability=0.52,
        equity_usd=10_000,
    )
    assert r.tier == "reject"


# ── NO side semantics ──────────────────────────────────────────────────


def test_no_side_with_positive_edge_promotes():
    # yes=0.70 → no=0.30. Estimated true yes = 0.50, so NO has positive edge.
    r = evaluate(
        market=_market(yes=0.70, vol=20_000, hours=12),
        proposed_side="NO", estimated_probability=0.50,
        equity_usd=10_000,
    )
    assert r.tier in ("sniper", "normal")
    assert r.passed is True


def test_no_side_with_negative_edge_rejects():
    # yes=0.70 implied → no=0.30. Estimated true yes = 0.80, so NO is
    # LESS likely than market thinks → negative edge on NO.
    r = evaluate(
        market=_market(yes=0.70, vol=20_000, hours=12),
        proposed_side="NO", estimated_probability=0.80,
        equity_usd=10_000,
    )
    assert r.tier == "reject"


# ── Portfolio sizing caps ──────────────────────────────────────────────


def test_portfolio_exposure_cap_reduces_shares():
    # Fresh equity $10k — sniper cap 3% = $300 room.
    r1 = evaluate(
        market=_market(yes=0.40, vol=20_000, hours=12),
        proposed_side="YES", estimated_probability=0.60,
        equity_usd=10_000, existing_polymarket_exposure_usd=0.0,
    )
    assert r1.shares > 0
    # If existing exposure already ate the 15% portfolio cap, shares=0.
    r2 = evaluate(
        market=_market(yes=0.40, vol=20_000, hours=12),
        proposed_side="YES", estimated_probability=0.60,
        equity_usd=10_000, existing_polymarket_exposure_usd=1_500.0,
    )
    assert r2.shares == 0 or r2.cost_usd == 0.0


def test_zero_equity_rejects():
    r = evaluate(
        market=_market(yes=0.40, vol=20_000, hours=12),
        proposed_side="YES", estimated_probability=0.60,
        equity_usd=0.0,
    )
    assert r.shares == 0


# ── Result serialization ───────────────────────────────────────────────


def test_to_dict_round_trip():
    r = evaluate(
        market=_market(), proposed_side="YES",
        estimated_probability=0.60, equity_usd=10_000,
    )
    d = r.to_dict()
    assert d["tier"] in ("sniper", "normal", "reject")
    assert "reasons" in d and isinstance(d["reasons"], list)
    assert "edge" in d


def test_tunable_thresholds_sane():
    assert SNIPER_MIN_EDGE > NORMAL_MIN_EDGE
    assert SNIPER_MIN_EV_PCT > NORMAL_MIN_EV_PCT
