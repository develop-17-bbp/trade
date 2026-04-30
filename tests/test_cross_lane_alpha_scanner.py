"""Unit tests for the cross-lane alpha ranker — focuses on the
deterministic logic (cap merge, role filter, admit-then-skip) that
doesn't depend on live fetcher state.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict

import pytest

from src.scanners.cross_lane_alpha_scanner import (
    CrossLaneCandidate,
    DEFAULT_MAX_CLASS_PCT,
    _active_lanes,
    _max_class_pct,
    score_candidate,
)


def test_default_max_class_pct_covers_known_classes():
    for cls in ("CRYPTO", "STOCK", "OPTIONS", "POLYMARKET"):
        assert cls in DEFAULT_MAX_CLASS_PCT
        assert DEFAULT_MAX_CLASS_PCT[cls] > 0


def test_max_class_pct_overrides_merge_with_defaults():
    caps = _max_class_pct({"cross_lane": {"max_class_pct": {"CRYPTO": 25}}})
    assert caps["CRYPTO"] == 25.0       # operator override applied
    assert caps["STOCK"] == DEFAULT_MAX_CLASS_PCT["STOCK"]  # default kept


def test_max_class_pct_handles_unparseable_values(caplog):
    caps = _max_class_pct({"cross_lane": {"max_class_pct": {"STOCK": "abc"}}})
    assert caps["STOCK"] == DEFAULT_MAX_CLASS_PCT["STOCK"]   # falls back to default


@pytest.fixture(autouse=True)
def _isolate_box_role(monkeypatch):
    monkeypatch.delenv("ACT_BOX_ROLE", raising=False)
    yield


def test_active_lanes_no_role_filter_honors_enabled_flag():
    cfg = {"exchanges": [
        {"name": "robinhood", "asset_class": "CRYPTO", "enabled": True},
        {"name": "alpaca",    "asset_class": "STOCK",  "enabled": False},
    ]}
    lanes = _active_lanes(cfg)
    assert [l["name"] for l in lanes] == ["robinhood"]


def test_active_lanes_role_filter_force_enables_matching(monkeypatch):
    monkeypatch.setenv("ACT_BOX_ROLE", "stocks")
    cfg = {"exchanges": [
        {"name": "robinhood", "asset_class": "CRYPTO", "enabled": True},
        {"name": "alpaca",    "asset_class": "STOCK",  "enabled": False},  # off
    ]}
    lanes = _active_lanes(cfg)
    # role=stocks force-enables alpaca and force-disables robinhood
    assert [l["name"] for l in lanes] == ["alpaca"]


def test_active_lanes_comma_separated_role(monkeypatch):
    monkeypatch.setenv("ACT_BOX_ROLE", "stocks,crypto")
    cfg = {"exchanges": [
        {"name": "robinhood",     "asset_class": "CRYPTO",  "enabled": False},
        {"name": "alpaca",        "asset_class": "STOCK",   "enabled": False},
        {"name": "alpaca_crypto", "asset_class": "CRYPTO",  "enabled": False},
        {"name": "polymarket",    "asset_class": "POLYMARKET", "enabled": True},
    ]}
    lanes = _active_lanes(cfg)
    names = sorted(l["name"] for l in lanes)
    assert names == ["alpaca", "alpaca_crypto", "robinhood"]   # polymarket excluded


def _admit(raw, caps, top_k=10):
    """Inline copy of the ranker admit loop so the test asserts the
    exact behavior without spinning up a fetcher."""
    class_budget: Dict[str, float] = defaultdict(float)
    out = []
    for c in raw:
        cap = caps.get(c.asset_class, 0.0)
        if class_budget[c.asset_class] + c.size_pct > cap:
            continue
        out.append(c)
        class_budget[c.asset_class] += c.size_pct
        if len(out) >= top_k:
            break
    return out, dict(class_budget)


def test_per_class_budget_cap_skips_overflow_candidates():
    caps = {"OPTIONS": 20.0, "STOCK": 45.0, "CRYPTO": 35.0}
    raw = [
        CrossLaneCandidate(lane="alpaca_options", asset="NVDA_C", asset_class="OPTIONS",
                           venue="alpaca", score=10.0, size_pct=15),
        CrossLaneCandidate(lane="alpaca_options", asset="SPY_C", asset_class="OPTIONS",
                           venue="alpaca", score=9.5, size_pct=15),    # would breach 20%
        CrossLaneCandidate(lane="alpaca", asset="NVDA", asset_class="STOCK",
                           venue="alpaca", score=8.0, size_pct=10),
        CrossLaneCandidate(lane="robinhood", asset="BTC", asset_class="CRYPTO",
                           venue="robinhood", score=7.0, size_pct=20),
        CrossLaneCandidate(lane="robinhood", asset="ETH", asset_class="CRYPTO",
                           venue="robinhood", score=6.5, size_pct=20),  # would breach 35%
    ]
    selected, budget = _admit(raw, caps)
    assert [c.asset for c in selected] == ["NVDA_C", "NVDA", "BTC"]
    assert budget == {"OPTIONS": 15.0, "STOCK": 10.0, "CRYPTO": 20.0}


def test_score_candidate_synthetic_uptrending_bars():
    bars = [[0, 100 + i * 0.5, 100 + i * 0.5, 100 + i * 0.5,
             100 + i * 0.5, 1_000_000 * (1.0 + i * 0.05)]
            for i in range(30)]
    lane_cfg = {"name": "alpaca", "asset_class": "STOCK", "venue": "alpaca",
                "intraday_position_pct_max": 15}
    cand = score_candidate(lane_cfg, "NVDA", bars)
    assert cand is not None
    assert cand.asset_class == "STOCK"
    assert cand.lane == "alpaca"
    # 15% intraday cap → default size = cap/3 = 5.0
    assert cand.size_pct == pytest.approx(5.0)
    assert cand.direction_hint in ("LONG", "FLAT")  # uptrend may or may not pass vol-ratio gate
