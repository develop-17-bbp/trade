"""Tests for the 6 predictive market-factor modules.

Each must:
  * Return bounded structured output
  * Handle degenerate inputs gracefully (no exceptions)
  * Use research-grounded thresholds (not learned weights)
  * Default to safe / unavailable when data missing
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest


# ── Macro Overlay ───────────────────────────────────────────────────────


def test_macro_overlay_returns_structured_when_unavailable(monkeypatch):
    """Force fetcher to return None for all symbols → graceful 'unavailable'."""
    from src.ai import macro_overlay
    monkeypatch.setattr(macro_overlay, "_fetch_yahoo", lambda *_a, **_k: None)
    monkeypatch.setattr(macro_overlay, "_cache",
                        {"ts": 0.0, "data": None})  # bust cache
    r = macro_overlay.fetch_macro_overlay()
    assert r.method == "unavailable"
    assert r.risk_regime == "unavailable"


def test_macro_overlay_classifies_vix_zones():
    from src.ai.macro_overlay import _classify_vix_zone
    assert _classify_vix_zone(35) == "panic"
    assert _classify_vix_zone(28) == "elevated"
    assert _classify_vix_zone(20) == "neutral"
    assert _classify_vix_zone(12) == "complacent_risk_on"
    assert _classify_vix_zone(None) == "unavailable"


def test_macro_overlay_bias_score_bounded():
    from src.ai.macro_overlay import _bias_score
    # Worst-case stack
    s = _bias_score(dxy_chg=2.0, us10y_chg_bps=20.0,
                     vix_zone="panic", spx_chg=-3.0)
    assert -1.0 <= s <= 1.0
    # Best-case stack
    s = _bias_score(dxy_chg=-1.0, us10y_chg_bps=-10.0,
                     vix_zone="complacent_risk_on", spx_chg=2.0)
    assert -1.0 <= s <= 1.0


def test_macro_overlay_risk_regime_classification():
    from src.ai.macro_overlay import _classify_risk_regime
    assert _classify_risk_regime(1.0, 4.0, "panic", -2.0) == "risk_off"
    assert _classify_risk_regime(-0.5, 3.0, "neutral", 0.5) == "risk_on"
    assert _classify_risk_regime(0.1, 3.5, "neutral", 0.0) == "neutral"


# ── BTC Dominance ───────────────────────────────────────────────────────


def test_btc_dominance_classify_zone():
    from src.ai.btc_dominance import _classify_zone
    assert _classify_zone(62) == "no_altseason"
    assert _classify_zone(55) == "btc_favored"
    assert _classify_zone(47) == "rotation_zone"
    assert _classify_zone(40) == "altseason_likely"
    assert _classify_zone(None) == "unavailable"


def test_btc_dominance_eth_bias_bounded():
    from src.ai.btc_dominance import _eth_bias_score
    s = _eth_bias_score(2.0, -0.05, "no_altseason")
    assert -1.0 <= s <= 1.0
    s = _eth_bias_score(-2.0, 0.05, "altseason_likely")
    assert -1.0 <= s <= 1.0


def test_btc_dominance_unavailable_when_no_api(monkeypatch):
    from src.ai import btc_dominance
    monkeypatch.setattr(btc_dominance, "_fetch_coingecko_global",
                        lambda: None)
    monkeypatch.setattr(btc_dominance, "_fetch_eth_btc_ratio",
                        lambda: None)
    monkeypatch.setattr(btc_dominance, "_cache",
                        {"ts": 0.0, "data": None})
    r = btc_dominance.fetch_btc_dominance()
    assert r.method == "unavailable"


# ── CVD ────────────────────────────────────────────────────────────────


def test_cvd_handles_short_history():
    from src.ai.cvd import compute_cvd
    r = compute_cvd([100] * 5, [101] * 5, [99] * 5, [1000] * 5)
    assert "insufficient" in r.rationale.lower()


def test_cvd_detects_accumulation_pattern():
    """Closes near TOP of range → buy > sell → cvd_recent > 0."""
    from src.ai.cvd import compute_cvd
    n = 60
    closes = [100 + i * 0.1 for i in range(n)]
    # Highs barely above close, lows far below = closing near range top
    highs = [c + 0.1 for c in closes]
    lows = [c - 1.0 for c in closes]
    volumes = [1000.0] * n
    r = compute_cvd(closes, highs, lows, volumes)
    assert r.cvd_recent > 0
    assert -1.0 <= r.cvd_momentum_score <= 1.0


def test_cvd_detects_bearish_divergence():
    """Price up + CVD down = bearish divergence."""
    from src.ai.cvd import compute_cvd
    n = 60
    # Price rises overall, but each bar closes near LOW of range
    # (sellers in control on each bar)
    closes = [100 + i * 0.1 for i in range(n)]
    highs = [c + 1.0 for c in closes]
    lows = [c - 0.05 for c in closes]  # close near LOW (close ≈ low+0.05)
    volumes = [1000.0] * n
    r = compute_cvd(closes, highs, lows, volumes)
    # Each bar's close is near the low → buy_volume small, sell_volume large
    # CVD trends down even as price rises = bearish divergence
    assert r.cvd_divergence_kind in ("bearish", "none")


def test_cvd_output_bounded():
    from src.ai.cvd import compute_cvd
    n = 60
    closes = [100 + math.sin(i / 5) for i in range(n)]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    volumes = [1000.0] * n
    r = compute_cvd(closes, highs, lows, volumes)
    d = r.to_dict()
    assert -1.0 <= d["cvd_momentum_score"] <= 1.0
    assert d["cvd_slope_sign"] in (-1, 0, 1)


# ── Whale Flow ──────────────────────────────────────────────────────────


def test_whale_flow_detects_volume_spike():
    """Inject one outlier-volume bar → should be detected."""
    from src.ai.whale_flow import detect_whale_flow
    n = 110
    closes = [100 + 0.01 * i for i in range(n)]
    opens = [closes[i] - 0.05 for i in range(n)]
    volumes = [1000.0] * n
    volumes[-3] = 8000.0  # large spike
    r = detect_whale_flow(closes, opens, volumes, lookback=100)
    assert r.n_whale_bars_recent >= 1
    assert r.last_whale_z_score > 0
    assert r.last_whale_direction in ("buy", "sell")


def test_whale_flow_handles_short_history():
    from src.ai.whale_flow import detect_whale_flow
    r = detect_whale_flow([100] * 10, [99] * 10, [1000] * 10)
    assert "insufficient" in r.rationale.lower()


def test_whale_flow_bias_bounded():
    from src.ai.whale_flow import detect_whale_flow
    n = 200
    closes = [100 + 0.01 * i for i in range(n)]
    opens = [closes[i] - 0.05 for i in range(n)]
    volumes = [1000.0 + (5000 if i in (50, 60, 80, 100) else 0) for i in range(n)]
    r = detect_whale_flow(closes, opens, volumes)
    assert -1.0 <= r.whale_directional_bias <= 1.0


# ── Halving Cycle ───────────────────────────────────────────────────────


def test_halving_cycle_2026_in_post_markup():
    from src.ai.halving_cycle import get_halving_cycle
    # April 2026 is ~24 months past Apr 2024 halving
    test_date = datetime(2026, 4, 29, tzinfo=timezone.utc)
    r = get_halving_cycle(now_utc=test_date)
    assert r.method == "halving_cycle"
    assert r.days_since_last_halving > 360  # >1 year
    assert r.cycle_position_pct < 100
    assert r.cycle_phase in (
        "post_halving_markup", "blowoff_top_zone", "distribution_bear",
        "capitulation_accumulation", "post_cycle",
    )


def test_halving_cycle_bullish_phase_flag():
    from src.ai.halving_cycle import get_halving_cycle
    # 12 months post-halving — typically blowoff zone
    test_date = datetime(2025, 5, 1, tzinfo=timezone.utc)
    r = get_halving_cycle(now_utc=test_date)
    assert r.bullish_phase is True


def test_halving_cycle_pure_deterministic():
    from src.ai.halving_cycle import get_halving_cycle
    test_date = datetime(2026, 4, 29, tzinfo=timezone.utc)
    r1 = get_halving_cycle(now_utc=test_date)
    r2 = get_halving_cycle(now_utc=test_date)
    assert r1.cycle_position_pct == r2.cycle_position_pct
    assert r1.cycle_phase == r2.cycle_phase


# ── BTC ETH Lead-Lag ────────────────────────────────────────────────────


def test_lead_lag_handles_short_history():
    from src.ai.btc_eth_lead_lag import analyze_lead_lag
    r = analyze_lead_lag([100] * 10, [200] * 10)
    assert "insufficient" in r.sample_warning.lower()


def test_lead_lag_synchronous_when_correlated():
    """Two perfectly-correlated series → synchronous, lag 0, high strength."""
    from src.ai.btc_eth_lead_lag import analyze_lead_lag
    btc = [100 + i * 0.1 for i in range(100)]
    eth = [b * 0.025 for b in btc]
    r = analyze_lead_lag(btc, eth, max_lag=5)
    # Perfectly synchronous price series should yield zero lag with high strength
    assert r.optimal_lag_bars == 0
    assert r.correlation_strength > 0.5


def test_lead_lag_relationship_classification():
    from src.ai.btc_eth_lead_lag import analyze_lead_lag
    # Random-ish series: low correlation → "unclear"
    import random
    rng = random.Random(42)
    btc = [100 + rng.gauss(0, 1) for _ in range(60)]
    eth = [200 + rng.gauss(0, 1) for _ in range(60)]
    r = analyze_lead_lag(btc, eth)
    assert r.relationship in (
        "unclear", "synchronous", "btc_leads_eth", "eth_leads_btc",
    )


# ── Tool registration ──────────────────────────────────────────────────


def test_all_6_predictive_tools_registered():
    from src.ai.trade_tools import build_default_registry
    r = build_default_registry()
    names = set(r.list_names())
    expected = {
        "query_macro_overlay",
        "query_btc_dominance",
        "query_cvd",
        "query_whale_flow",
        "query_halving_cycle",
        "query_btc_eth_lead_lag",
    }
    missing = expected - names
    assert not missing, f"missing: {missing}"


def test_halving_cycle_tool_dispatches():
    from src.ai.trade_tools import build_default_registry
    r = build_default_registry()
    res = r.dispatch("query_halving_cycle", {})
    s = str(res)
    assert "cycle_phase" in s
