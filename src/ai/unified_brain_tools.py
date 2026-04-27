"""Unified-brain tool pack — C26 Step 2.

Exposes the ACT subsystems that weren't yet callable as LLM tools:

  * `query_ml_ensemble`      → LightGBM + LSTM + PatchTST + RL aggregate
  * `query_multi_strategy`   → 36-strategy engine consensus
  * `find_similar_trades`    → MemoryVault with age-decayed retrieval (C19)
  * `monte_carlo_var`        → monte_carlo_bt.py probability-of-ruin
  * `evt_tail_risk`          → evt_risk.py fat-tail VaR
  * `get_macro_bias`         → macro_bias.py signed tilt
  * `get_economic_layer`     → economic_intelligence.py layer snapshot
  * `request_genetic_candidate` → genetic_strategy_engine hall-of-fame
                                  DNA (with strategy_repository fallback)
  * `run_full_backtest`      → full_engine.py event-driven (slow, decisive)
  * `query_12_agents`        → 13 fixed + transient persona debate verdict
  * `query_strategy_universe`→ top-k positive-EV strategies for current regime

Plus the Unit-8 Robinhood read-only query tools so the brain can ASK
the venue directly (operator directive: "LLM should do autonomously by
using read only robinhood api"):

  * `query_robinhood_balance`   → buying power + cash + status
  * `query_robinhood_positions` → BTC/ETH holdings summary
  * `query_robinhood_quote`     → live bid/ask/mid/spread
  * `query_recent_robinhood_fills` → recent live + paper fills

These four are STRICTLY READ-ONLY. They never call any place_order /
submit_trade endpoint. The brain proposes via TradePlan; the executor
handles submission.

Each handler is lazy-imported so a missing dependency does not block
the registry build. Every return is a compact dict; serialisation and
capping happens in `ToolRegistry.dispatch`. Every handler catches its
own exceptions — returns `{"error": ...}` rather than raising.

Registered via `register_unified_brain_tools(registry)` which
`build_default_registry` calls in a try/except block.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Handlers ───────────────────────────────────────────────────────────


def _handle_ml_ensemble(args: Dict[str, Any]) -> Dict[str, Any]:
    """Roll up ML-predictor outputs so the Analyst can see the ensemble's
    joint opinion rather than calling four separate tools."""
    asset = str(args.get("asset") or "BTC").upper()
    out: Dict[str, Any] = {"asset": asset, "models": {}}

    # LightGBM
    try:
        from src.ai.ml_models import get_lgbm_state
        out["models"]["lightgbm"] = get_lgbm_state(asset)
    except Exception as e:
        out["models"]["lightgbm"] = {"error": str(e)[:80]}

    # LSTM ensemble
    try:
        from src.models.lstm_ensemble import get_latest_signal
        out["models"]["lstm"] = get_latest_signal(asset)
    except Exception as e:
        out["models"]["lstm"] = {"error": str(e)[:80]}

    # PatchTST
    try:
        from src.ai.patchtst_model import get_latest_forecast
        out["models"]["patchtst"] = get_latest_forecast(asset)
    except Exception as e:
        out["models"]["patchtst"] = {"error": str(e)[:80]}

    # RL agent
    try:
        from src.ai.reinforcement_learning import get_agent_state
        out["models"]["rl"] = get_agent_state(asset)
    except Exception as e:
        out["models"]["rl"] = {"error": str(e)[:80]}

    # Consensus summary
    bull, bear, neutral = 0, 0, 0
    for m in out["models"].values():
        if isinstance(m, dict):
            sig = str(m.get("signal") or m.get("direction") or "").upper()
            if sig in ("LONG", "BULLISH", "BUY"):
                bull += 1
            elif sig in ("SHORT", "BEARISH", "SELL"):
                bear += 1
            else:
                neutral += 1
    out["consensus"] = {"bullish": bull, "bearish": bear, "neutral": neutral}
    return out


def _handle_multi_strategy(args: Dict[str, Any]) -> Dict[str, Any]:
    """36-strategy engine votes; returns counts by direction and the
    top-5 bullish + top-5 bearish strategies for the Analyst."""
    asset = str(args.get("asset") or "BTC").upper()
    try:
        from src.trading.multi_strategy_engine import get_latest_consensus
        consensus = get_latest_consensus(asset)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:80]}"}
    # Expect {votes: {"ema_trend": "LONG", ...}, score: float}
    votes = consensus.get("votes") or {}
    longs = [k for k, v in votes.items() if str(v).upper() == "LONG"]
    shorts = [k for k, v in votes.items() if str(v).upper() == "SHORT"]
    return {
        "asset": asset,
        "total_strategies": len(votes),
        "long_count": len(longs),
        "short_count": len(shorts),
        "flat_count": len(votes) - len(longs) - len(shorts),
        "consensus_score": round(float(consensus.get("score") or 0.0), 3),
        "sample_longs": longs[:5],
        "sample_shorts": shorts[:5],
    }


def _handle_find_similar_trades(args: Dict[str, Any]) -> Dict[str, Any]:
    """MemoryVault RAG with age-decayed weighting (C19). Returns top-k
    semantically-similar past trade outcomes."""
    try:
        from src.ai.memory_vault import MemoryVault
    except Exception as e:
        return {"error": f"memory_vault import failed: {e}"}
    asset = str(args.get("asset") or "BTC").upper()
    regime = str(args.get("regime") or "UNKNOWN")
    k = int(args.get("k") or 3)
    k = max(1, min(10, k))
    try:
        vault = MemoryVault()
        results = vault.find_similar_trades(
            asset=asset,
            current_regime=regime,
            current_funding=float(args.get("funding") or 0.0),
            current_sentiment=args.get("sentiment") or {},
            proposed_signal=int(args.get("proposed_signal") or 0),
            top_k=k,
        )
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:80]}"}

    out = []
    for r in results or []:
        out.append({
            "id": r.get("id", ""),
            "similarity": round(float(r.get("similarity", 0.0)), 3),
            "age_factor": round(float(r.get("age_factor", 1.0)), 3),
            "weighted_score": round(float(r.get("weighted_score", 0.0)), 3),
            "regime_match": bool(r.get("regime_match", False)),
            "pnl_pct": (r.get("metadata") or {}).get("pnl_pct", 0.0),
            "direction": (r.get("metadata") or {}).get("direction", ""),
        })
    return {"asset": asset, "regime": regime, "count": len(out), "results": out}


def _handle_monte_carlo_var(args: Dict[str, Any]) -> Dict[str, Any]:
    """Probability-of-ruin + VaR via Monte-Carlo simulation on empirical
    trade-outcome distribution."""
    try:
        from src.backtesting.monte_carlo_bt import run_monte_carlo_var
    except Exception as e:
        return {"error": f"monte_carlo import failed: {e}"}
    asset = str(args.get("asset") or "BTC").upper()
    size_pct = float(args.get("size_pct") or 1.0)
    horizon_bars = int(args.get("horizon_bars") or 24)
    try:
        result = run_monte_carlo_var(asset=asset, size_pct=size_pct,
                                     horizon_bars=horizon_bars)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:80]}"}
    return {
        "asset": asset,
        "size_pct": size_pct,
        "horizon_bars": horizon_bars,
        "var_95": round(float(result.get("var_95", 0.0)), 4),
        "cvar_95": round(float(result.get("cvar_95", 0.0)), 4),
        "prob_ruin": round(float(result.get("prob_ruin", 0.0)), 4),
        "n_sims": int(result.get("n_sims", 0)),
    }


def _handle_evt_tail_risk(args: Dict[str, Any]) -> Dict[str, Any]:
    """EVT-based fat-tail VaR at 99% — tells the Analyst what an outlier
    adverse move looks like for this asset."""
    try:
        from src.risk.evt_risk import get_latest_evt_fit
    except Exception as e:
        return {"error": f"evt_risk import failed: {e}"}
    asset = str(args.get("asset") or "BTC").upper()
    try:
        fit = get_latest_evt_fit(asset)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:80]}"}
    return {
        "asset": asset,
        "xi": round(float(fit.get("xi", 0.0)), 4),
        "sigma": round(float(fit.get("sigma", 0.0)), 6),
        "threshold": round(float(fit.get("threshold", 0.0)), 6),
        "var_99": round(float(fit.get("var_99", 0.0)), 4),
        "exceedances": int(fit.get("exceedances", 0)),
    }


def _handle_macro_bias(_args: Dict[str, Any]) -> Dict[str, Any]:
    """Signed macro tilt from macro_bias.compute_macro_bias."""
    try:
        from src.trading.macro_bias import compute_macro_bias
        from src.data.economic_intelligence import get_intelligence
    except Exception as e:
        return {"error": f"macro_bias import failed: {e}"}
    try:
        ei = get_intelligence()
        summary = ei.get_macro_summary() if ei else {}
        bias = compute_macro_bias(summary)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:80]}"}
    return {
        "signed_bias": round(float(bias.signed_bias), 3),
        "crisis": bool(bias.crisis),
        "size_multiplier": round(float(bias.size_multiplier), 2),
        "confidence": round(float(bias.confidence), 2),
        "composite_signal": str(bias.composite_signal or "NEUTRAL"),
        "reasons": list(bias.reasons)[:5],
    }


def _handle_economic_layer(args: Dict[str, Any]) -> Dict[str, Any]:
    """Snapshot of one economic_intelligence layer (usd_strength,
    central_bank, geopolitical, etc.)."""
    try:
        from src.data.economic_intelligence import get_intelligence
    except Exception as e:
        return {"error": f"econ import failed: {e}"}
    layer_name = str(args.get("layer") or "").strip()
    if not layer_name:
        return {"error": "missing required arg 'layer'"}
    try:
        ei = get_intelligence()
        snap = ei.get_layer_snapshot(layer_name) if ei else None
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:80]}"}
    if not snap:
        return {"layer": layer_name, "available": False}
    return {
        "layer": layer_name,
        "available": True,
        "signal": str(snap.get("signal", "NEUTRAL")),
        "confidence": round(float(snap.get("confidence", snap.get("conf", 0.0))), 2),
        "value": snap.get("value"),
        "notes": str(snap.get("notes", ""))[:200],
    }


def _handle_genetic_candidate(args: Dict[str, Any]) -> Dict[str, Any]:
    """Request a challenger strategy from the genetic hall of fame."""
    try:
        from src.trading.strategy_repository import get_repo
    except Exception as e:
        return {"error": f"repo import failed: {e}"}
    regime = args.get("regime")
    try:
        repo = get_repo()
        recs = repo.search(status="challenger", regime=regime, limit=1) or []
        if not recs:
            recs = repo.search(status="candidate", regime=regime, limit=1) or []
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:80]}"}
    if not recs:
        return {"available": False, "regime": regime}
    rec = recs[0]
    return {
        "available": True,
        "strategy_id": getattr(rec, "strategy_id", ""),
        "regime": getattr(rec, "regime_tag", "") or regime,
        "live_wins": int(getattr(rec, "live_wins", 0) or 0),
        "live_losses": int(getattr(rec, "live_losses", 0) or 0),
        "live_sharpe": round(float(getattr(rec, "live_sharpe", 0.0) or 0.0), 2),
        "dna_preview": str(getattr(rec, "dna", {}))[:200],
    }


def _handle_full_backtest(args: Dict[str, Any]) -> Dict[str, Any]:
    """Event-driven full backtest — slow (30-120s) but decisive.
    Different from vectorized `backtest_hypothesis` which runs in <2s."""
    try:
        from src.backtesting.full_engine import run_event_driven_backtest
    except Exception as e:
        return {"error": f"full_engine import failed: {e}"}
    asset = str(args.get("asset") or "BTC").upper()
    dna = args.get("plan_dna") or {}
    bars = int(args.get("bars") or 720)       # 30 days of 1h bars
    try:
        result = run_event_driven_backtest(asset=asset, dna=dna, bars=bars)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:80]}"}
    return {
        "asset": asset,
        "bars_tested": bars,
        "total_trades": int(result.get("total_trades", 0)),
        "win_rate": round(float(result.get("win_rate", 0.0)), 3),
        "sharpe": round(float(result.get("sharpe", 0.0)), 2),
        "max_drawdown_pct": round(float(result.get("max_drawdown_pct", 0.0)), 2),
        "total_return_pct": round(float(result.get("total_return_pct", 0.0)), 2),
        "avg_trade_pct": round(float(result.get("avg_trade_pct", 0.0)), 3),
    }


# ── Registration ───────────────────────────────────────────────────────


def register_unified_brain_tools(registry) -> int:
    """Register the 9 unified-brain tools. Safe to call multiple times
    against a fresh registry; raises on duplicate tool names in an
    existing registry (caller should use try/except)."""
    from src.ai.trade_tools import Tool

    tools = [
        Tool(
            name="query_ml_ensemble",
            description=(
                "[ML-ENSEMBLE] Joint signal from LightGBM + LSTM + "
                "PatchTST + RL agent. Returns per-model signal + "
                "consensus counts (bullish/bearish/neutral). Use this "
                "to see what the ML stack jointly predicts."
            ),
            input_schema={
                "type": "object",
                "properties": {"asset": {"type": "string", "enum": ["BTC", "ETH"]}},
                "required": ["asset"],
            },
            handler=_handle_ml_ensemble, tag="read_only",
        ),
        Tool(
            name="query_multi_strategy",
            description=(
                "[MULTI-STRATEGY] Consensus vote across ACT's 36-strategy "
                "engine (EMA trend, mean reversion, volatility breakout, "
                "Pine Script strategies, etc.). Returns long/short/flat "
                "counts + score + top-5 strategies agreeing on each side."
            ),
            input_schema={
                "type": "object",
                "properties": {"asset": {"type": "string", "enum": ["BTC", "ETH"]}},
                "required": ["asset"],
            },
            handler=_handle_multi_strategy, tag="read_only",
        ),
        Tool(
            name="find_similar_trades",
            description=(
                "[MEMORY] Age-decayed semantic search over past trades "
                "in MemoryVault. Returns top-k similar setups + their "
                "realized PnL. Use to answer 'have I seen this setup "
                "before — what happened?'"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "asset": {"type": "string"},
                    "regime": {"type": "string"},
                    "k": {"type": "integer", "minimum": 1, "maximum": 10},
                    "funding": {"type": "number"},
                    "sentiment": {"type": "object"},
                    "proposed_signal": {"type": "integer"},
                },
                "required": ["asset"],
            },
            handler=_handle_find_similar_trades, tag="read_only",
        ),
        Tool(
            name="monte_carlo_var",
            description=(
                "[RISK] Monte-Carlo VaR, CVaR, and probability-of-ruin "
                "for the proposed trade size. Uses empirical outcome "
                "distribution from warm_store. Returns bounded decimal "
                "fractions (0.0-1.0)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "asset": {"type": "string"},
                    "size_pct": {"type": "number", "minimum": 0.1, "maximum": 20.0},
                    "horizon_bars": {"type": "integer", "minimum": 4, "maximum": 240},
                },
                "required": ["asset", "size_pct"],
            },
            handler=_handle_monte_carlo_var, tag="read_only",
        ),
        Tool(
            name="evt_tail_risk",
            description=(
                "[RISK] Extreme-Value-Theory fat-tail VaR at 99%. "
                "Returns GPD fit parameters (xi, sigma, threshold) and "
                "var_99 as decimal fraction. Use this to reason about "
                "outlier adverse moves beyond normal-distribution VaR."
            ),
            input_schema={
                "type": "object",
                "properties": {"asset": {"type": "string"}},
                "required": ["asset"],
            },
            handler=_handle_evt_tail_risk, tag="read_only",
        ),
        Tool(
            name="get_macro_bias",
            description=(
                "[MACRO] Signed macro tilt across 12 economic layers. "
                "Returns signed_bias in [-1, +1] (+=bullish), crisis "
                "flag, size_multiplier, composite signal, and reasons."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=_handle_macro_bias, tag="read_only",
        ),
        Tool(
            name="get_economic_layer",
            description=(
                "[MACRO] Snapshot of a single economic_intelligence "
                "layer. Valid names: usd_strength, central_bank, "
                "geopolitical, macro_indicators, onchain, "
                "social_sentiment, equity_correlation, institutional, "
                "regulatory, mining_economics, derivatives, defi_liquidity."
            ),
            input_schema={
                "type": "object",
                "properties": {"layer": {"type": "string"}},
                "required": ["layer"],
            },
            handler=_handle_economic_layer, tag="read_only",
        ),
        Tool(
            name="request_genetic_candidate",
            description=(
                "[STRATEGY] Ask the genetic hall-of-fame for a challenger "
                "strategy to try. Returns best available challenger DNA "
                "for the requested regime, or available=False if none."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "regime": {"type": "string"},
                },
            },
            handler=_handle_genetic_candidate, tag="read_only",
        ),
        Tool(
            name="run_full_backtest",
            description=(
                "[BACKTEST] Decisive event-driven backtest on last N "
                "bars (slow, 30-120s). Distinct from the fast "
                "backtest_hypothesis (vectorized, <2s). Use this only "
                "for high-conviction setups where a quick check passed."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "asset": {"type": "string"},
                    "plan_dna": {"type": "object"},
                    "bars": {"type": "integer", "minimum": 120, "maximum": 2880},
                },
                "required": ["asset", "plan_dna"],
            },
            handler=_handle_full_backtest, tag="read_only",
        ),
    ]

    added = 0
    for t in tools:
        try:
            registry.register(t)
            added += 1
        except ValueError:
            # duplicate — tolerated on repeat registration (e.g. tests)
            pass
    # Also classify each new tool in tool_metadata so audit sees them.
    try:
        from src.ai.tool_metadata import TOOL_CLASSIFICATIONS, ToolClassification
        _classifications = {
            "query_ml_ensemble": ("realtime", "analysis", "experimental"),
            "query_multi_strategy": ("minute", "analysis", "crypto_native"),
            "find_similar_trades": ("daily", "query", "internal"),
            "monte_carlo_var": ("hour", "analysis", "experimental"),
            "evt_tail_risk": ("hour", "analysis", "equity_borrowed"),
            "get_macro_bias": ("hour", "analysis", "equity_borrowed"),
            "get_economic_layer": ("hour", "data_fetch", "equity_borrowed"),
            "request_genetic_candidate": ("daily", "query", "internal"),
            "run_full_backtest": ("hour", "analysis", "experimental"),
        }
        for name, (tm, it, rg) in _classifications.items():
            if name not in TOOL_CLASSIFICATIONS:
                TOOL_CLASSIFICATIONS[name] = ToolClassification(
                    timeliness=tm, intent_type=it, regulatory=rg,
                )
    except Exception:
        pass
    return added
