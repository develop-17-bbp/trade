"""Tests for src/trading/timeframe_profile.py — profile resolution + shape."""
from __future__ import annotations

import os

import pytest

from src.trading.timeframe_profile import (
    POSITION_PROFILE,
    SCALP_PROFILE,
    SWING_PROFILE,
    TimeframeProfile,
    get_profile,
    is_swing_or_higher,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("ACT_TIMEFRAME_PROFILE", raising=False)
    yield


def test_default_is_scalp():
    assert get_profile().name == "scalp"
    assert get_profile({}).name == "scalp"


def test_config_selects_swing():
    p = get_profile({"timeframe_profile": "swing"})
    assert p.name == "swing"
    assert p.primary_tf == "1h"


def test_env_overrides_config(monkeypatch):
    monkeypatch.setenv("ACT_TIMEFRAME_PROFILE", "position")
    p = get_profile({"timeframe_profile": "swing"})
    assert p.name == "position"
    assert p.primary_tf == "4h"


def test_unknown_falls_back_to_scalp():
    p = get_profile({"timeframe_profile": "does_not_exist"})
    assert p is SCALP_PROFILE


def test_env_whitespace_and_case_insensitive(monkeypatch):
    monkeypatch.setenv("ACT_TIMEFRAME_PROFILE", "  SWING  ")
    assert get_profile().name == "swing"


def test_profiles_are_frozen():
    with pytest.raises(Exception):
        SCALP_PROFILE.primary_tf = "10m"  # type: ignore[misc]


def test_scalp_shape_matches_existing_defaults():
    # These values are what the pre-profile code path used. Regressions here
    # would mean the swing/position additions accidentally shifted scalp.
    assert SCALP_PROFILE.primary_tf == "5m"
    assert SCALP_PROFILE.poll_interval_s == 60
    assert SCALP_PROFILE.sniper_min_move_pct == 5.0
    assert SCALP_PROFILE.normal_min_move_pct == 2.5
    assert SCALP_PROFILE.min_hold_minutes == 1440


def test_swing_thresholds_survive_robinhood_spread():
    # Robinhood round-trip spread is 1.69%. Every accepted tier must clear it.
    assert SWING_PROFILE.sniper_min_move_pct > 1.69
    assert SWING_PROFILE.normal_min_move_pct > 1.69


def test_position_is_slowest():
    assert POSITION_PROFILE.poll_interval_s >= SWING_PROFILE.poll_interval_s
    assert POSITION_PROFILE.max_hold_days >= SWING_PROFILE.max_hold_days
    assert POSITION_PROFILE.target_trades_per_week <= SWING_PROFILE.target_trades_per_week


def test_is_swing_or_higher():
    assert is_swing_or_higher(SWING_PROFILE) is True
    assert is_swing_or_higher(POSITION_PROFILE) is True
    assert is_swing_or_higher(SCALP_PROFILE) is False


def test_to_dict_roundtrip():
    d = SWING_PROFILE.to_dict()
    assert d["name"] == "swing"
    assert d["primary_tf"] == "1h"
    assert isinstance(d["poll_interval_s"], int)


def test_profile_dataclass_type():
    assert isinstance(SCALP_PROFILE, TimeframeProfile)
    assert isinstance(SWING_PROFILE, TimeframeProfile)
    assert isinstance(POSITION_PROFILE, TimeframeProfile)
