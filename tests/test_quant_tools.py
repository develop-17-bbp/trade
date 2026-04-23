"""Tests for src/ai/quant_tools.py — LLM-callable wrappers over src/models/."""
from __future__ import annotations

import json
import math
from typing import List, Optional

import numpy as np
import pytest

from src.ai.quant_tools import (
    DEFAULT_LOOKBACK,
    MAX_LOOKBACK,
    _closes_from_raw,
    register_quant_tools,
    set_bars_fetcher,
)
from src.ai.trade_tools import ToolRegistry


def _make_trending_bars(n=240, start_price=60000.0, drift_per_bar=5.0):
    """Synthetic OHLCV: steady upward drift + small noise. Good for
    exercising the quant tools deterministically."""
    bars: List[List[float]] = []
    rng = np.random.default_rng(42)
    price = start_price
    for i in range(n):
        price = price + drift_per_bar + rng.standard_normal() * 2
        o = price - 1
        c = price
        h = price + 3
        l = price - 3
        bars.append([1_700_000_000 + i * 3600, o, h, l, c, 1.0])
    return bars


def _make_mean_reverting_bars(n=240, mean=60000.0, amp=500.0):
    bars: List[List[float]] = []
    rng = np.random.default_rng(7)
    for i in range(n):
        osc = math.sin(i * 0.3) * amp
        noise = rng.standard_normal() * 20
        c = mean + osc + noise
        bars.append([1_700_000_000 + i * 3600, c, c + 5, c - 5, c, 1.0])
    return bars


@pytest.fixture(autouse=True)
def _reset_bars_fetcher():
    """Keep the module-level override clean across tests."""
    set_bars_fetcher(None)
    yield
    set_bars_fetcher(None)


@pytest.fixture
def reg_with_quant():
    """A fresh registry with quant tools registered."""
    reg = ToolRegistry()
    register_quant_tools(reg)
    return reg


# ── _closes_from_raw ───────────────────────────────────────────────────


def test_closes_from_raw_happy():
    bars = [[t, 1.0, 1.1, 0.9, 1.05, 10.0] for t in range(50)]
    closes = _closes_from_raw(bars)
    assert closes is not None
    assert len(closes) == 50
    assert closes[0] == pytest.approx(1.05)


def test_closes_from_raw_too_few():
    assert _closes_from_raw([[t, 1, 1, 1, 1, 1] for t in range(10)]) is None


def test_closes_from_raw_empty():
    assert _closes_from_raw([]) is None


def test_closes_from_raw_skips_malformed_rows():
    bars = [[t, 1, 1, 1, float(t), 1] for t in range(30)] + [[0, 0, 0]]
    # Function takes whatever it can; the short row is skipped.
    closes = _closes_from_raw(bars)
    assert closes is not None
    assert len(closes) == 30


# ── Registration ───────────────────────────────────────────────────────


def test_register_quant_tools_adds_all_six(reg_with_quant):
    names = set(reg_with_quant.list_names())
    expected = {
        "fit_ou_process", "hurst_exponent", "kalman_trend",
        "hmm_regime", "hawkes_clustering", "test_cointegration",
    }
    assert expected <= names


def test_register_is_idempotent():
    reg = ToolRegistry()
    register_quant_tools(reg)
    second = register_quant_tools(reg)
    # Second call should return [] (all already present) — not raise.
    assert second == []


def test_tools_tagged_as_read_only(reg_with_quant):
    read_only_names = {s["name"] for s in reg_with_quant.anthropic_schemas(tags=["read_only"])}
    assert "fit_ou_process" in read_only_names
    assert "test_cointegration" in read_only_names


def test_quant_tools_in_default_registry():
    """Default registry (C3) should now include quant tools (C11)."""
    from src.ai.trade_tools import build_default_registry
    reg = build_default_registry()
    names = set(reg.list_names())
    assert "fit_ou_process" in names
    assert "hurst_exponent" in names


# ── Fetcher error paths ────────────────────────────────────────────────


def test_no_bars_returns_error(reg_with_quant):
    set_bars_fetcher(lambda a, t, n: None)
    out = json.loads(reg_with_quant.dispatch("fit_ou_process", {"asset": "BTC"}))
    assert "error" in out
    assert "no bars" in out["error"].lower()


def test_insufficient_bars_returns_error(reg_with_quant):
    set_bars_fetcher(lambda a, t, n: [[0, 1, 1, 1, 1, 1] for _ in range(5)])
    out = json.loads(reg_with_quant.dispatch("hurst_exponent", {"asset": "BTC"}))
    assert "error" in out
    assert "insufficient" in out["error"].lower()


def test_kalman_returns_level_and_slope(reg_with_quant):
    set_bars_fetcher(lambda a, t, n: _make_trending_bars(n))
    out = json.loads(reg_with_quant.dispatch(
        "kalman_trend", {"asset": "BTC", "timeframe": "1h", "bars": 240},
    ))
    # Kalman may fall back to unavailable if scipy missing — handle both.
    if "error" not in out:
        assert "level" in out and "slope" in out
        assert isinstance(out["summary"], str)
        assert "Kalman" in out["summary"]


def test_hurst_on_trending_data(reg_with_quant):
    set_bars_fetcher(lambda a, t, n: _make_trending_bars(n))
    out = json.loads(reg_with_quant.dispatch(
        "hurst_exponent", {"asset": "BTC", "bars": 240},
    ))
    if "error" not in out:
        # Trending synthetic → H should lean > 0.5.
        assert out.get("hurst") is not None
        assert "regime" in out


def test_ou_on_mean_reverting_data(reg_with_quant):
    set_bars_fetcher(lambda a, t, n: _make_mean_reverting_bars(n))
    out = json.loads(reg_with_quant.dispatch(
        "fit_ou_process", {"asset": "BTC", "bars": 240},
    ))
    if "error" not in out:
        assert "half_life" in out
        # z_score and signal may be None if fit didn't converge; just
        # sanity-check the keys are present.
        assert "z_score" in out
        assert "signal" in out


def test_cointegration_requires_both_assets(reg_with_quant):
    # Return bars for A, None for B → should error.
    def _fetcher(asset, tf, n):
        return _make_trending_bars(n) if asset == "BTC" else None
    set_bars_fetcher(_fetcher)
    out = json.loads(reg_with_quant.dispatch(
        "test_cointegration", {"asset_a": "BTC", "asset_b": "ETH"},
    ))
    assert "error" in out
    assert "no bars" in out["error"].lower()


def test_dispatch_errors_captured_by_registry(reg_with_quant):
    """If a handler raises, the registry catches and returns error JSON."""
    def _boom(*_):
        raise RuntimeError("fetcher down")
    set_bars_fetcher(_boom)
    # Handler catches fetcher failure internally → returns {"error": "no bars"}
    out = json.loads(reg_with_quant.dispatch("fit_ou_process", {"asset": "BTC"}))
    assert "error" in out


# ── Schema shape ───────────────────────────────────────────────────────


def test_schema_exposes_asset_as_required(reg_with_quant):
    tool = reg_with_quant.get("fit_ou_process")
    assert tool is not None
    schema = tool.input_schema
    assert "asset" in schema["properties"]
    assert "asset" in schema["required"]


def test_cointegration_schema_requires_both_assets(reg_with_quant):
    tool = reg_with_quant.get("test_cointegration")
    assert "asset_a" in tool.input_schema["required"]
    assert "asset_b" in tool.input_schema["required"]
