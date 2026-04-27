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


# ── Unit-8 Robinhood read-only handlers ─────────────────────────────────
# All four call only read endpoints on RobinhoodCryptoClient. They never
# trigger an order. The brain calls these to *see* venue state; placing
# trades stays the executor's job (TradePlan → submit_trade_plan).


def _rh_client():
    """Lazy-build a RobinhoodCryptoClient using env credentials. Returns
    None on missing creds so the handler can degrade gracefully."""
    import os
    api_key = os.environ.get("ROBINHOOD_API_KEY")
    private_key = os.environ.get("ROBINHOOD_PRIVATE_KEY_BASE64")
    if not api_key or not private_key:
        return None
    try:
        from src.integrations.robinhood_crypto import RobinhoodCryptoClient
        return RobinhoodCryptoClient(api_key=api_key, private_key_b64=private_key)
    except Exception as e:
        logger.debug(f"robinhood client init failed: {e}")
        return None


def _handle_robinhood_balance(args: Dict[str, Any]) -> Dict[str, Any]:
    cli = _rh_client()
    if cli is None:
        return {"error": "robinhood_credentials_missing_or_client_unavailable"}
    try:
        acc = cli.get_account() or {}
        return {
            "account_status": acc.get("status", "unknown"),
            "buying_power_usd": float(acc.get("buying_power", 0) or 0),
            "buying_power_currency": acc.get("buying_power_currency", "USD"),
            "account_number": acc.get("account_number", ""),
        }
    except Exception as e:
        return {"error": f"balance_query_failed: {e}"[:200]}


def _handle_robinhood_positions(args: Dict[str, Any]) -> Dict[str, Any]:
    assets = args.get("assets") or ["BTC", "ETH"]
    if isinstance(assets, str):
        assets = [a.strip().upper() for a in assets.split(",") if a.strip()]
    cli = _rh_client()
    if cli is None:
        return {"error": "robinhood_credentials_missing_or_client_unavailable"}
    try:
        h = cli.get_holdings(assets=assets) or {}
        results = h.get("results", []) if isinstance(h, dict) else []
        out = []
        for pos in results:
            out.append({
                "asset": pos.get("asset_code", ""),
                "quantity": float(pos.get("total_quantity", 0) or 0),
                "available": float(pos.get("quantity_available_for_trading", 0) or 0),
            })
        return {"positions": out, "count": len(out)}
    except Exception as e:
        return {"error": f"positions_query_failed: {e}"[:200]}


def _handle_robinhood_quote(args: Dict[str, Any]) -> Dict[str, Any]:
    asset = str(args.get("asset") or "BTC").upper()
    symbol = f"{asset}-USD"
    cli = _rh_client()
    if cli is None:
        return {"error": "robinhood_credentials_missing_or_client_unavailable"}
    try:
        q = cli.get_best_price(symbol=symbol) or {}
        results = q.get("results", []) if isinstance(q, dict) else []
        if not results:
            return {"asset": asset, "error": "no_quote_returned"}
        r = results[0]
        bid = float(r.get("bid_inclusive_of_buy_spread", r.get("bid", 0)) or 0)
        ask = float(r.get("ask_inclusive_of_sell_spread", r.get("ask", 0)) or 0)
        mid = (bid + ask) / 2.0 if (bid and ask) else 0.0
        spread_pct = ((ask - bid) / mid * 100.0) if mid else 0.0
        return {
            "asset": asset,
            "bid": bid, "ask": ask, "mid": mid,
            "spread_pct": round(spread_pct, 4),
        }
    except Exception as e:
        return {"error": f"quote_query_failed: {e}"[:200]}


def _handle_robinhood_fills(args: Dict[str, Any]) -> Dict[str, Any]:
    asset = args.get("asset")
    limit = int(args.get("limit") or 10)
    limit = max(1, min(50, limit))
    cli = _rh_client()
    if cli is None:
        return {"error": "robinhood_credentials_missing_or_client_unavailable"}
    try:
        symbol = f"{str(asset).upper()}-USD" if asset else None
        o = cli.get_orders(symbol=symbol, state="filled", limit=limit) or {}
        results = o.get("results", []) if isinstance(o, dict) else []
        out = []
        for r in results[:limit]:
            out.append({
                "id": r.get("id", "")[:32],
                "symbol": r.get("symbol", ""),
                "side": r.get("side", ""),
                "quantity": float(r.get("filled_asset_quantity", 0) or 0),
                "price": float(r.get("average_price", 0) or 0),
                "ts": r.get("updated_at", ""),
            })
        return {"fills": out, "count": len(out)}
    except Exception as e:
        return {"error": f"fills_query_failed: {e}"[:200]}


# ── Brain authority tools (gap closure: full body→brain visibility) ─────


def _handle_accuracy_engine(args: Dict[str, Any]) -> Dict[str, Any]:
    """Per-component accuracy weights + position-size multiplier the
    brain can see (was previously gate-only, blind to LLM)."""
    regime = str(args.get("regime") or "unknown")
    try:
        from src.learning.accuracy_engine import AccuracyEngine
        eng = AccuracyEngine()
        weights = eng.get_ensemble_weights(regime) or {}
        return {
            "regime": regime,
            "ensemble_weights": {k: round(float(v), 3) for k, v in weights.items()},
            "position_size_multiplier": round(float(eng.get_position_size_multiplier()), 3),
            "avg_slippage_bps": round(float(eng.get_avg_slippage()), 2),
            "effective_spread_pct": round(float(eng.get_effective_spread()), 3),
            "should_skip": bool(eng.should_skip_trade()[0]) if hasattr(eng, "should_skip_trade") else False,
        }
    except Exception as e:
        return {"error": f"accuracy_query_failed: {e}"[:200]}


def _handle_position_limits(args: Dict[str, Any]) -> Dict[str, Any]:
    """Dynamic position limits + current usage so the brain sizes
    plans within the cap before submitting."""
    asset = str(args.get("asset") or "BTC").upper()
    regime = str(args.get("regime") or "unknown")
    try:
        from src.risk.dynamic_position_limits import DynamicPositionLimits
        lim = DynamicPositionLimits()
        return {
            "asset": asset,
            "regime": regime,
            "max_position_pct": round(float(lim.get_max_position_pct(asset, regime)), 2),
            "max_position_usd": round(float(lim.get_max_position_usd(asset, regime)), 2),
            "stats": lim.get_stats() if hasattr(lim, "get_stats") else {},
        }
    except Exception as e:
        return {"error": f"position_limits_query_failed: {e}"[:200]}


def _handle_recent_plans(args: Dict[str, Any]) -> Dict[str, Any]:
    """Brain reads its OWN prior TradePlans (last N for this asset)
    so cross-tick continuity is observable. Closes the cross-tick
    gap where the brain only saw critiques but not direction."""
    asset = str(args.get("asset") or "").upper()
    limit = int(args.get("limit") or 5)
    limit = max(1, min(20, limit))
    try:
        import sqlite3, json as _json
        from src.orchestration.warm_store import get_store
        store = get_store()
        conn = sqlite3.connect(store.db_path, timeout=2.0)
        try:
            sql = (
                "SELECT ts_ns, symbol, plan_json, final_action "
                "FROM decisions "
                + ("WHERE symbol = ? " if asset else "")
                + "ORDER BY ts_ns DESC LIMIT ?"
            )
            params = ((asset, limit) if asset else (limit,))
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        out: List[Dict[str, Any]] = []
        for ts_ns, sym, raw, action in rows:
            try:
                p = _json.loads(raw or "{}")
            except Exception:
                p = {}
            out.append({
                "ts_ns": int(ts_ns),
                "asset": sym,
                "direction": p.get("direction", "?"),
                "thesis": str(p.get("thesis", ""))[:200],
                "confidence": p.get("confidence"),
                "final_action": action,
            })
        return {"plans": out, "count": len(out)}
    except Exception as e:
        return {"error": f"recent_plans_query_failed: {e}"[:200]}


def _handle_profit_protector(args: Dict[str, Any]) -> Dict[str, Any]:
    """Trailing-stop + trade-quality state per asset (was gate-only)."""
    asset = str(args.get("asset") or "BTC").upper()
    try:
        from src.risk.profit_protector import ProfitProtector
        pp = ProfitProtector()
        out = {"asset": asset}
        try:
            out["profit_status"] = pp.get_profit_status()
        except Exception:
            out["profit_status"] = {}
        try:
            trail = pp.trailing_stops.get(asset) if hasattr(pp, "trailing_stops") else None
            if trail:
                out["trailing_stop"] = {
                    "stop_price": float(trail.get("stop_price", 0)),
                    "high_water": float(trail.get("high_water", 0)),
                    "active": True,
                }
            else:
                out["trailing_stop"] = {"active": False}
        except Exception:
            out["trailing_stop"] = {"active": False}
        return out
    except Exception as e:
        return {"error": f"profit_protector_query_failed: {e}"[:200]}


def _handle_champion_gate(args: Dict[str, Any]) -> Dict[str, Any]:
    """Current champion adapter state — which fine-tuned LoRA is
    serving + last gate decision. Read-only window into model lifecycle."""
    try:
        from src.ai.champion_gate import get_champion_state
        state = get_champion_state() or {}
        return {
            "champion_id": str(state.get("champion_id", "default"))[:64],
            "challenger_id": str(state.get("challenger_id", ""))[:64],
            "last_gate_decision": str(state.get("last_decision", "n/a"))[:64],
            "champion_metrics": state.get("champion_metrics") or {},
        }
    except ImportError:
        return {"error": "champion_gate_no_state_api"}
    except Exception as e:
        return {"error": f"champion_gate_query_failed: {e}"[:200]}


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
        Tool(
            name="query_robinhood_balance",
            description=(
                "[VENUE] Live Robinhood account balance (read-only). "
                "Returns buying_power_usd + account_status. Use to check "
                "available capital before sizing. NEVER places orders."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=_handle_robinhood_balance, tag="read_only",
        ),
        Tool(
            name="query_robinhood_positions",
            description=(
                "[VENUE] Live Robinhood crypto holdings (read-only). "
                "Returns per-asset quantity + available-for-trading. "
                "Use to check what's already held before opening more."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "assets": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
            handler=_handle_robinhood_positions, tag="read_only",
        ),
        Tool(
            name="query_robinhood_quote",
            description=(
                "[VENUE] Live Robinhood bid/ask/mid for an asset (read-"
                "only). Returns spread_pct so the brain can verify the "
                "round-trip cost before proposing a TradePlan."
            ),
            input_schema={
                "type": "object",
                "properties": {"asset": {"type": "string", "enum": ["BTC", "ETH"]}},
                "required": ["asset"],
            },
            handler=_handle_robinhood_quote, tag="read_only",
        ),
        Tool(
            name="query_recent_robinhood_fills",
            description=(
                "[VENUE] Recent filled Robinhood orders (read-only). "
                "Optional filter by asset. Use to verify what really "
                "executed vs. what TradePlans proposed."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "asset": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
            },
            handler=_handle_robinhood_fills, tag="read_only",
        ),
        Tool(
            name="query_accuracy_engine",
            description=(
                "[LEARNING] Per-component ensemble weights (LGBM, LSTM, "
                "PatchTST, RL, multi-strategy, LLM, agents) for the "
                "current regime + position-size multiplier. Brain uses "
                "this to weight its own confidence and size honestly."
            ),
            input_schema={
                "type": "object",
                "properties": {"regime": {"type": "string"}},
            },
            handler=_handle_accuracy_engine, tag="read_only",
        ),
        Tool(
            name="query_position_limits",
            description=(
                "[RISK] Dynamic position limits per asset + regime. "
                "Returns max_position_pct + max_position_usd so the brain "
                "sizes within the cap before submit_trade_plan."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "asset": {"type": "string", "enum": ["BTC", "ETH"]},
                    "regime": {"type": "string"},
                },
                "required": ["asset"],
            },
            handler=_handle_position_limits, tag="read_only",
        ),
        Tool(
            name="query_recent_plans",
            description=(
                "[MEMORY] Brain reads its OWN prior TradePlans for an "
                "asset (direction/thesis/confidence/terminated_reason). "
                "Cross-tick continuity: 'what did I propose 3 ticks ago "
                "and how did it terminate?'"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "asset": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                },
            },
            handler=_handle_recent_plans, tag="read_only",
        ),
        Tool(
            name="query_profit_protector",
            description=(
                "[RISK] Trailing-stop + trade-quality state per asset. "
                "Brain sees high-water mark, active stop price, and "
                "whether the protector is trailing — useful when "
                "deciding HOLD vs EXIT."
            ),
            input_schema={
                "type": "object",
                "properties": {"asset": {"type": "string", "enum": ["BTC", "ETH"]}},
                "required": ["asset"],
            },
            handler=_handle_profit_protector, tag="read_only",
        ),
        Tool(
            name="query_champion_gate",
            description=(
                "[MODEL] Current champion adapter id + challenger + last "
                "gate decision (model lifecycle window). Returns 'no_state_api' "
                "if champion_gate.get_champion_state isn't implemented."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=_handle_champion_gate, tag="read_only",
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
            "query_robinhood_balance": ("realtime", "data_fetch", "internal"),
            "query_robinhood_positions": ("realtime", "data_fetch", "internal"),
            "query_robinhood_quote": ("realtime", "data_fetch", "internal"),
            "query_recent_robinhood_fills": ("realtime", "data_fetch", "internal"),
            "query_accuracy_engine": ("hour", "query", "internal"),
            "query_position_limits": ("minute", "query", "internal"),
            "query_recent_plans": ("realtime", "query", "internal"),
            "query_profit_protector": ("realtime", "query", "internal"),
            "query_champion_gate": ("hour", "query", "internal"),
        }
        for name, (tm, it, rg) in _classifications.items():
            if name not in TOOL_CLASSIFICATIONS:
                TOOL_CLASSIFICATIONS[name] = ToolClassification(
                    timeliness=tm, intent_type=it, regulatory=rg,
                )
    except Exception:
        pass
    return added
