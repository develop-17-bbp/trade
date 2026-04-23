"""
Quant-tool registry — expose ACT's existing quant/ML models as LLM tools.

ACT has ~20 quantitative models under `src/models/` already (OU process,
cointegration, HMM regime, Hurst exponent, Kalman filter, Hawkes point
process, fractional differentiation, volatility regime, ...). Historically
these are called inline from the executor, never from the agentic loop.

This module binds them as LLM-callable tools in the existing ToolRegistry
(C3) so the Analyst brain can request a quant-grounded answer before
compiling a TradePlan.

Design rules:
  * Lean output — same ≤500-char digest discipline as `web_context.py`.
    Each tool returns a short summary + 2-3 key numbers, never raw
    arrays. No 200-bar arrays leak into the parent LLM context.
  * Graceful — every tool catches its own errors, returns a structured
    `{"error": "..."}` dict rather than raising. Missing scipy /
    statsmodels / bar source → "unavailable" in the digest.
  * No hot-path cost — bar fetches route through the existing
    `PriceFetcher` in `src/data/fetcher.py`. If no price source is
    reachable, we return "no bars" rather than stalling the agentic
    loop.
  * Zero hard deps on the executor — tools are standalone. Tests can
    inject a bars-fetcher; production uses the real one.

Tools registered (all read_only):
  * fit_ou_process(asset, timeframe, bars)
  * hurst_exponent(asset, timeframe, bars)
  * kalman_trend(asset, timeframe, bars)
  * hmm_regime(asset, timeframe, bars)
  * hawkes_clustering(asset, timeframe, bars)
  * test_cointegration(asset_a, asset_b, timeframe, bars)

Usage:
    from src.ai.trade_tools import build_default_registry
    from src.ai.quant_tools import register_quant_tools
    reg = build_default_registry()
    register_quant_tools(reg)
    # LLM now sees six new tools alongside in-process + web.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# Default timeframe + lookback; operator can override via env or config.
DEFAULT_TF = "1h"
DEFAULT_LOOKBACK = 200
MAX_LOOKBACK = 1000


# ── Bars fetcher (injectable for tests) ─────────────────────────────────


BarsFetcher = Callable[[str, str, int], Optional[List[List[float]]]]


def _default_fetch_bars(asset: str, timeframe: str, limit: int) -> Optional[List[List[float]]]:
    """Pull OHLCV bars via the project's existing PriceFetcher.

    Returns raw OHLCV rows (list of [ts, open, high, low, close, volume])
    or None on any failure. Never raises.
    """
    try:
        from src.data.fetcher import PriceFetcher
        symbol = asset.upper()
        # Try a zero-arg constructor; fall back to the singleton if it
        # expects args (it typically takes a config dict).
        try:
            pf = PriceFetcher()
        except Exception:
            try:
                pf = PriceFetcher({})
            except Exception as e:
                logger.debug("quant_tools: PriceFetcher construct failed: %s", e)
                return None
        raw = pf.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return list(raw or [])
    except Exception as e:
        logger.debug("quant_tools: fetch_bars(%s, %s) failed: %s", asset, timeframe, e)
        return None


# Module-level override so tests can swap the fetcher without touching
# the global data layer. Production code leaves this as None.
_fetch_bars_override: Optional[BarsFetcher] = None


def set_bars_fetcher(fn: Optional[BarsFetcher]) -> None:
    """Test helper — inject a custom bars fetcher. Pass None to restore."""
    global _fetch_bars_override
    _fetch_bars_override = fn


def _fetch_bars(asset: str, timeframe: str, limit: int) -> Optional[List[List[float]]]:
    fn = _fetch_bars_override or _default_fetch_bars
    return fn(asset, timeframe, limit)


def _closes_from_raw(raw: List[List[float]]) -> Optional[np.ndarray]:
    """Extract close column from an OHLCV rows list. Robust to varying
    row widths — some sources return [ts, o, h, l, c, v] and some
    return [ts, o, h, l, c]."""
    try:
        closes: List[float] = []
        for row in raw or []:
            if not row or len(row) < 5:
                continue
            closes.append(float(row[4]))
        if len(closes) < 20:
            return None
        return np.asarray(closes, dtype=np.float64)
    except Exception:
        return None


# ── Individual tool handlers ────────────────────────────────────────────


def _ou_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    asset = str(args.get("asset") or "BTC").upper()
    tf = str(args.get("timeframe") or DEFAULT_TF)
    bars = min(MAX_LOOKBACK, int(args.get("bars") or DEFAULT_LOOKBACK))
    raw = _fetch_bars(asset, tf, bars)
    if not raw:
        return {"error": f"no bars for {asset}/{tf}"}
    closes = _closes_from_raw(raw)
    if closes is None:
        return {"error": f"insufficient bars ({len(raw)})"}
    try:
        from src.models.ou_process import OUProcess
        ou = OUProcess()
        # fit_and_signal wants log-prices; OUProcess handles that internally.
        result = ou.fit_and_signal(closes, window=min(bars - 1, 252))
    except Exception as e:
        return {"error": f"ou_fit_failed: {type(e).__name__}: {e}"}
    # Compact digest — drop any array fields, keep the headline numbers.
    half_life = result.get("half_life")
    theta = result.get("theta")
    z = result.get("z_score") or result.get("z")
    sig = result.get("signal") or 0
    summary = (
        f"OU({asset}/{tf}): half_life={half_life} theta={theta} "
        f"z_score={z} signal={sig}"
    )
    return {
        "summary": summary,
        "half_life": half_life, "theta": theta,
        "z_score": z, "signal": int(sig or 0),
    }


def _hurst_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    asset = str(args.get("asset") or "BTC").upper()
    tf = str(args.get("timeframe") or DEFAULT_TF)
    bars = min(MAX_LOOKBACK, int(args.get("bars") or DEFAULT_LOOKBACK))
    raw = _fetch_bars(asset, tf, bars)
    if not raw:
        return {"error": f"no bars for {asset}/{tf}"}
    closes = _closes_from_raw(raw)
    if closes is None:
        return {"error": f"insufficient bars ({len(raw)})"}
    try:
        from src.models.hurst import HurstExponent
        result = HurstExponent().compute(closes, window=min(bars - 1, 200))
    except Exception as e:
        return {"error": f"hurst_failed: {type(e).__name__}: {e}"}
    h = result.get("hurst")
    regime = result.get("regime") or (
        "trending" if (h or 0) > 0.55 else
        "mean_reverting" if (h or 0) < 0.45 else "random_walk"
    )
    return {
        "summary": f"Hurst({asset}/{tf}): H={h:.3f} regime={regime}" if h is not None else "Hurst: unavailable",
        "hurst": h,
        "regime": regime,
    }


def _kalman_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    asset = str(args.get("asset") or "BTC").upper()
    tf = str(args.get("timeframe") or DEFAULT_TF)
    bars = min(MAX_LOOKBACK, int(args.get("bars") or DEFAULT_LOOKBACK))
    raw = _fetch_bars(asset, tf, bars)
    if not raw:
        return {"error": f"no bars for {asset}/{tf}"}
    closes = _closes_from_raw(raw)
    if closes is None:
        return {"error": f"insufficient bars ({len(raw)})"}
    try:
        from src.models.kalman_filter import KalmanTrendFilter
        result = KalmanTrendFilter().latest(closes)
    except Exception as e:
        return {"error": f"kalman_failed: {type(e).__name__}: {e}"}
    level = result.get("level")
    slope = result.get("slope")
    return {
        "summary": f"Kalman({asset}/{tf}): level={level:.4f} slope={slope:+.5f}" if level is not None else "Kalman: unavailable",
        "level": level,
        "slope": slope,
    }


def _hmm_regime_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    asset = str(args.get("asset") or "BTC").upper()
    tf = str(args.get("timeframe") or DEFAULT_TF)
    bars = min(MAX_LOOKBACK, int(args.get("bars") or DEFAULT_LOOKBACK))
    raw = _fetch_bars(asset, tf, bars)
    if not raw:
        return {"error": f"no bars for {asset}/{tf}"}
    closes = _closes_from_raw(raw)
    if closes is None or len(closes) < 60:
        return {"error": "insufficient bars for HMM (need 60+)"}
    try:
        returns = np.diff(np.log(closes))
        # Rolling 20-bar realized vol — compact estimate.
        vol = np.zeros_like(returns)
        for i in range(len(returns)):
            lo = max(0, i - 20)
            vol[i] = float(np.std(returns[lo:i + 1]) or 0.0)
        from src.models.hmm_regime import HMMRegimeDetector
        det = HMMRegimeDetector(n_states=4)
        det.fit(returns, vol)
        result = det.predict(returns, vol)
    except Exception as e:
        return {"error": f"hmm_failed: {type(e).__name__}: {e}"}
    regime = result.get("regime") or result.get("state_label")
    conf = result.get("confidence") or 0.0
    return {
        "summary": f"HMM({asset}/{tf}): regime={regime} confidence={conf:.2f}",
        "regime": regime,
        "confidence": conf,
    }


def _hawkes_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    asset = str(args.get("asset") or "BTC").upper()
    tf = str(args.get("timeframe") or DEFAULT_TF)
    bars = min(MAX_LOOKBACK, int(args.get("bars") or DEFAULT_LOOKBACK))
    threshold_pct = float(args.get("threshold_pct") or 0.02)
    raw = _fetch_bars(asset, tf, bars)
    if not raw:
        return {"error": f"no bars for {asset}/{tf}"}
    closes = _closes_from_raw(raw)
    if closes is None:
        return {"error": f"insufficient bars ({len(raw)})"}
    try:
        from src.models.hawkes_process import HawkesProcess
        hp = HawkesProcess()
        events = hp.detect_events(closes, threshold_pct=threshold_pct)
        # events may be (times_array, meta) or a dict; handle both.
        if isinstance(events, tuple) and len(events) >= 1:
            times = events[0]
        elif isinstance(events, dict):
            times = events.get("event_times") or events.get("times") or []
        else:
            times = events or []
        if len(times) < 2:
            return {"summary": f"Hawkes({asset}/{tf}): too few events ({len(times)})",
                    "events": len(times)}
        params = hp.fit(np.asarray(times, dtype=np.float64), max_iter=50)
        intensity = hp.current_intensity(np.asarray(times, dtype=np.float64), len(closes))
    except Exception as e:
        return {"error": f"hawkes_failed: {type(e).__name__}: {e}"}
    return {
        "summary": (
            f"Hawkes({asset}/{tf}): events={len(times)} "
            f"alpha={params.get('alpha')} beta={params.get('beta')} "
            f"intensity={intensity:.3f}"
        ),
        "events": len(times),
        "alpha": params.get("alpha"),
        "beta": params.get("beta"),
        "intensity": intensity,
    }


def _cointegration_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    a = str(args.get("asset_a") or "BTC").upper()
    b = str(args.get("asset_b") or "ETH").upper()
    tf = str(args.get("timeframe") or DEFAULT_TF)
    bars = min(MAX_LOOKBACK, int(args.get("bars") or DEFAULT_LOOKBACK))
    raw_a = _fetch_bars(a, tf, bars)
    raw_b = _fetch_bars(b, tf, bars)
    if not raw_a or not raw_b:
        return {"error": f"no bars for {a}/{b}/{tf}"}
    ca = _closes_from_raw(raw_a)
    cb = _closes_from_raw(raw_b)
    if ca is None or cb is None:
        return {"error": "insufficient bars"}
    # Align lengths — both ends should be the same "now".
    n = min(len(ca), len(cb))
    if n < 60:
        return {"error": f"insufficient aligned bars ({n})"}
    ca = ca[-n:]
    cb = cb[-n:]
    try:
        from src.models.cointegration import CointegrationEngine
        ce = CointegrationEngine()
        result = ce.test_pair(ca, cb)
    except Exception as e:
        return {"error": f"cointegration_failed: {type(e).__name__}: {e}"}
    pvalue = result.get("p_value") or result.get("pvalue")
    beta = result.get("hedge_ratio") or result.get("beta")
    half_life = result.get("half_life")
    z = result.get("z_score") or result.get("z")
    cointegrated = bool(result.get("cointegrated", False))
    return {
        "summary": (
            f"Coint({a},{b}/{tf}): cointegrated={cointegrated} "
            f"p={pvalue} β={beta} half_life={half_life} z={z}"
        ),
        "cointegrated": cointegrated,
        "p_value": pvalue,
        "beta": beta,
        "half_life": half_life,
        "z_score": z,
    }


# ── Registration ────────────────────────────────────────────────────────


def register_quant_tools(registry) -> List[str]:
    """Register all quant tools into the given ToolRegistry.

    Returns the list of tool names added. Imports Tool lazily so the
    module still imports cleanly in environments without trade_tools.
    """
    from src.ai.trade_tools import Tool

    added: List[str] = []

    specs = [
        ("fit_ou_process",
         "Fit an Ornstein-Uhlenbeck mean-reversion model to recent "
         "closes. Returns half_life, theta (equilibrium), z_score, signal.",
         {
             "type": "object",
             "properties": {
                 "asset": {"type": "string"},
                 "timeframe": {"type": "string", "enum": ["5m", "15m", "1h", "4h", "1d"]},
                 "bars": {"type": "integer", "minimum": 50, "maximum": MAX_LOOKBACK},
             },
             "required": ["asset"],
         },
         _ou_handler),
        ("hurst_exponent",
         "Compute Hurst exponent H on recent closes. H>0.55 trending, "
         "H<0.45 mean-reverting, else random-walk.",
         {
             "type": "object",
             "properties": {
                 "asset": {"type": "string"},
                 "timeframe": {"type": "string"},
                 "bars": {"type": "integer", "minimum": 50, "maximum": MAX_LOOKBACK},
             },
             "required": ["asset"],
         },
         _hurst_handler),
        ("kalman_trend",
         "Run a Kalman trend filter on recent closes. Returns latest "
         "level + slope (smoothed trend direction).",
         {
             "type": "object",
             "properties": {
                 "asset": {"type": "string"},
                 "timeframe": {"type": "string"},
                 "bars": {"type": "integer", "minimum": 50, "maximum": MAX_LOOKBACK},
             },
             "required": ["asset"],
         },
         _kalman_handler),
        ("hmm_regime",
         "Fit an HMM regime classifier on recent returns + vol. Returns "
         "current regime label + posterior confidence.",
         {
             "type": "object",
             "properties": {
                 "asset": {"type": "string"},
                 "timeframe": {"type": "string"},
                 "bars": {"type": "integer", "minimum": 80, "maximum": MAX_LOOKBACK},
             },
             "required": ["asset"],
         },
         _hmm_regime_handler),
        ("hawkes_clustering",
         "Detect clustered events (volatility bursts) using a Hawkes "
         "self-exciting point process. Returns event count + alpha/beta "
         "+ current intensity.",
         {
             "type": "object",
             "properties": {
                 "asset": {"type": "string"},
                 "timeframe": {"type": "string"},
                 "bars": {"type": "integer", "minimum": 100, "maximum": MAX_LOOKBACK},
                 "threshold_pct": {"type": "number", "minimum": 0.005, "maximum": 0.1},
             },
             "required": ["asset"],
         },
         _hawkes_handler),
        ("test_cointegration",
         "Engle-Granger cointegration test between two assets. Returns "
         "cointegrated bool, p-value, hedge ratio β, half-life, z-score.",
         {
             "type": "object",
             "properties": {
                 "asset_a": {"type": "string"},
                 "asset_b": {"type": "string"},
                 "timeframe": {"type": "string"},
                 "bars": {"type": "integer", "minimum": 60, "maximum": MAX_LOOKBACK},
             },
             "required": ["asset_a", "asset_b"],
         },
         _cointegration_handler),
    ]

    for name, desc, schema, handler in specs:
        try:
            registry.register(Tool(
                name=name, description=f"[QUANT] {desc}",
                input_schema=schema, handler=handler, tag="read_only",
            ))
            added.append(name)
        except ValueError:
            # Already registered (refresh scenario) — skip silently.
            logger.debug("quant tool %s already registered", name)
            continue

    return added
