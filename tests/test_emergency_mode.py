"""Tests for emergency-mode signal — readiness_gate + scheduler integration."""
from __future__ import annotations

import os

import pytest

from src.orchestration.readiness_gate import (
    EMERGENCY_ENV,
    GateState,
    is_emergency_mode,
    publish_emergency_mode,
)
from src.orchestration.scheduler import emergency_aware_rate


# ── is_emergency_mode ───────────────────────────────────────────────────


def _state(rolling_sharpe: float, samples: int, target_annual: float = 25.0) -> GateState:
    return GateState(
        open_=False, reasons=[],
        details={
            "rolling_sharpe": rolling_sharpe,
            "sharpe_samples": samples,
            "target_annual_return_pct": target_annual,
        },
    )


def test_emergency_false_when_sharpe_above_ratio():
    # target_annual=25 → target_sharpe=25/30≈0.833; emergency if < 0.7×target ≈ 0.583
    assert is_emergency_mode(_state(rolling_sharpe=1.0, samples=50)) is False


def test_emergency_true_when_sharpe_below_ratio():
    assert is_emergency_mode(_state(rolling_sharpe=0.3, samples=50)) is True


def test_emergency_false_when_samples_too_sparse():
    # Even if Sharpe is bad, < 10 samples is not enough evidence.
    assert is_emergency_mode(_state(rolling_sharpe=0.1, samples=5)) is False


def test_emergency_false_when_target_zero():
    assert is_emergency_mode(_state(rolling_sharpe=0.1, samples=50, target_annual=0.0)) is False


# ── publish_emergency_mode ──────────────────────────────────────────────


def test_publish_sets_env_true(monkeypatch):
    monkeypatch.delenv(EMERGENCY_ENV, raising=False)
    publish_emergency_mode(True)
    assert os.environ[EMERGENCY_ENV] == "1"


def test_publish_clears_env(monkeypatch):
    monkeypatch.setenv(EMERGENCY_ENV, "1")
    publish_emergency_mode(False)
    assert os.environ[EMERGENCY_ENV] == "0"


# ── emergency_aware_rate ────────────────────────────────────────────────


def test_rate_unchanged_when_flag_off(monkeypatch):
    monkeypatch.delenv(EMERGENCY_ENV, raising=False)
    fn = emergency_aware_rate(base_rate_s=3600.0)
    assert fn() == 3600.0


def test_rate_halved_when_flag_on(monkeypatch):
    monkeypatch.setenv(EMERGENCY_ENV, "1")
    fn = emergency_aware_rate(base_rate_s=3600.0, factor=0.5)
    assert fn() == 1800.0


def test_rate_custom_factor(monkeypatch):
    monkeypatch.setenv(EMERGENCY_ENV, "1")
    fn = emergency_aware_rate(base_rate_s=1000.0, factor=0.25)
    assert fn() == 250.0


def test_rate_factor_clamped_to_safe_range(monkeypatch):
    monkeypatch.setenv(EMERGENCY_ENV, "1")
    # factor=0.01 should clamp up to 0.1 minimum (don't allow absurd cadence)
    fn = emergency_aware_rate(base_rate_s=1000.0, factor=0.01)
    assert fn() == 100.0
    # factor=2.0 should clamp down to 1.0 (never accelerate unwanted)
    fn2 = emergency_aware_rate(base_rate_s=1000.0, factor=2.0)
    assert fn2() == 1000.0


def test_rate_responds_to_flap(monkeypatch):
    monkeypatch.delenv(EMERGENCY_ENV, raising=False)
    fn = emergency_aware_rate(base_rate_s=600.0, factor=0.5)
    assert fn() == 600.0
    monkeypatch.setenv(EMERGENCY_ENV, "1")
    assert fn() == 300.0
    monkeypatch.setenv(EMERGENCY_ENV, "0")
    assert fn() == 600.0
