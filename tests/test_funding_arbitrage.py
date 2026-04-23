"""Tests for src/trading/funding_arbitrage.py — dormant-on-Robinhood carry."""
from __future__ import annotations

import pytest

from src.trading.funding_arbitrage import (
    describe_dormancy_reason,
    evaluate_funding_opportunity,
    is_enabled,
)


# ── evaluate_funding_opportunity ────────────────────────────────────────


def test_zero_funding_is_skip():
    r = evaluate_funding_opportunity(0.0)
    assert r.tier == "skip"
    assert r.direction == ""
    assert r.reason == "funding_neutral"


def test_positive_funding_shorts_perp():
    # Longs pay shorts → we want to be short the perp.
    r = evaluate_funding_opportunity(0.03, expected_hold_days=7.0, min_annualised_return=5.0)
    assert r.direction == "SHORT_PERP_LONG_SPOT"
    assert r.tier == "enter"


def test_negative_funding_longs_perp():
    r = evaluate_funding_opportunity(-0.03, expected_hold_days=7.0, min_annualised_return=5.0)
    assert r.direction == "LONG_PERP_SHORT_SPOT"
    assert r.tier == "enter"


def test_low_funding_skips_below_threshold():
    # 0.001% per 8h → ~1.1% annualised gross, definitely below 10% min.
    r = evaluate_funding_opportunity(0.001, min_annualised_return=10.0)
    assert r.tier == "skip"
    assert "below_min" in r.reason


def test_annualised_math():
    # 0.01% per 8h = 0.03% per day = 10.95% annualised (365d).
    r = evaluate_funding_opportunity(0.01, expected_hold_days=7.0)
    assert abs(r.annualised_pct - (0.01 * 3 * 365)) < 0.01


def test_to_dict_has_expected_keys():
    r = evaluate_funding_opportunity(0.05, expected_hold_days=7.0, min_annualised_return=5.0)
    d = r.to_dict()
    for key in (
        "tier", "direction", "funding_pct_per_8h", "annualised_pct",
        "expected_net_annual_pct", "notional_usd", "reason", "legs",
    ):
        assert key in d
    assert d["legs"]["perp"]["side"] in ("LONG", "SHORT")
    assert d["legs"]["spot"]["side"] in ("LONG", "SHORT")


def test_legs_are_opposite_sides():
    r = evaluate_funding_opportunity(0.05, expected_hold_days=7.0, min_annualised_return=5.0)
    assert r.legs["perp"]["side"] != r.legs["spot"]["side"]


# ── is_enabled ──────────────────────────────────────────────────────────


def test_robinhood_only_is_always_disabled(monkeypatch):
    monkeypatch.setenv("ACT_FUNDING_ARB", "1")
    cfg = {"funding_arb": {"enabled": True}, "exchanges": [{"name": "robinhood"}]}
    assert is_enabled(cfg) is False


def test_env_off_wins(monkeypatch):
    monkeypatch.setenv("ACT_FUNDING_ARB", "0")
    cfg = {"funding_arb": {"enabled": True}, "exchanges": [{"name": "bybit"}]}
    assert is_enabled(cfg) is False


def test_env_on_with_perp_venue(monkeypatch):
    monkeypatch.setenv("ACT_FUNDING_ARB", "1")
    cfg = {"exchanges": [{"name": "bybit"}]}
    assert is_enabled(cfg) is True


def test_config_enable_with_perp(monkeypatch):
    monkeypatch.delenv("ACT_FUNDING_ARB", raising=False)
    cfg = {"funding_arb": {"enabled": True}, "exchanges": [{"name": "delta"}]}
    assert is_enabled(cfg) is True


def test_default_off(monkeypatch):
    monkeypatch.delenv("ACT_FUNDING_ARB", raising=False)
    cfg = {"exchanges": [{"name": "bybit"}]}
    assert is_enabled(cfg) is False


# ── describe_dormancy_reason ────────────────────────────────────────────


def test_dormancy_no_perp_venue(monkeypatch):
    monkeypatch.setenv("ACT_FUNDING_ARB", "1")
    cfg = {"funding_arb": {"enabled": True}, "exchanges": [{"name": "robinhood"}]}
    reason = describe_dormancy_reason(cfg)
    assert "no_perp_venue" in reason


def test_dormancy_env_disabled(monkeypatch):
    monkeypatch.setenv("ACT_FUNDING_ARB", "0")
    cfg = {"exchanges": [{"name": "bybit"}]}
    assert describe_dormancy_reason(cfg).startswith("disabled_by_env")


def test_dormancy_default_off(monkeypatch):
    monkeypatch.delenv("ACT_FUNDING_ARB", raising=False)
    cfg = {"exchanges": [{"name": "bybit"}]}
    assert describe_dormancy_reason(cfg).startswith("disabled_by_default")


def test_dormancy_active(monkeypatch):
    monkeypatch.setenv("ACT_FUNDING_ARB", "1")
    cfg = {"funding_arb": {"enabled": True}, "exchanges": [{"name": "bybit"}]}
    assert describe_dormancy_reason(cfg) == "active"
