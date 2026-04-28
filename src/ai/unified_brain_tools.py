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

    # Temporal transformer (research model, online-update capable).
    # Only include if a saved checkpoint exists so we don't add noise
    # from an untrained dummy.
    try:
        import os as _os
        ckpt = _os.path.join("models", f"temporal_transformer_{asset.lower()}.pkl")
        if _os.path.exists(ckpt):
            from src.ai.temporal_transformer import TemporalTransformer
            tt = TemporalTransformer()
            tt.load_model(ckpt)
            from src.data.fetcher import PriceFetcher
            try:
                pf = PriceFetcher()
                bars = pf.get_recent_bars(asset, timeframe="1h", n=64) or []
                import numpy as _np
                if bars:
                    closes = _np.array([float(b.get("close", 0)) for b in bars])
                    out["models"]["temporal_transformer"] = tt.forecast_return(closes)
            except Exception:
                out["models"]["temporal_transformer"] = {"status": "no_recent_bars"}
    except Exception:
        # Silent — temporal transformer is optional research model
        pass

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


# ── Strategy & pattern tools (Items 1-5 from 2026 strategy literature) ──


def _fetch_recent_bars(asset: str, timeframe: str = "1h", n: int = 100):
    """Best-effort recent OHLCV — falls back gracefully when unavailable."""
    try:
        from src.data.fetcher import PriceFetcher
        pf = PriceFetcher()
        bars = pf.get_recent_bars(asset, timeframe=timeframe, n=n) or []
        if not bars:
            return None
        highs = [float(b.get("high", 0)) for b in bars]
        lows = [float(b.get("low", 0)) for b in bars]
        closes = [float(b.get("close", 0)) for b in bars]
        volumes = [float(b.get("volume", 0)) for b in bars]
        return highs, lows, closes, volumes
    except Exception:
        return None


# ── Backtest-rigor tools (overfitting defense) ──────────────────────────


def _handle_deflated_sharpe(args: Dict[str, Any]) -> Dict[str, Any]:
    """Compute Deflated Sharpe Ratio (Bailey-López de Prado 2014)
    on a returns series. Penalizes Sharpe for selection bias from
    multiple-trial optimization.
    """
    returns = args.get("returns") or []
    n_trials = int(args.get("n_trials") or 1)
    if not isinstance(returns, list) or len(returns) < 5:
        return {"error": "need_at_least_5_returns"}
    try:
        from src.backtesting.overfitting_metrics import deflated_sharpe
        result = deflated_sharpe([float(r) for r in returns],
                                  n_trials=n_trials)
        return result.to_dict()
    except Exception as e:
        return {"error": f"dsr_failed: {e}"[:200]}


def _handle_pbo(args: Dict[str, Any]) -> Dict[str, Any]:
    """Probability of Backtest Overfitting (Bailey et al. 2017)
    over an M-strategies × T-periods returns matrix."""
    matrix = args.get("returns_matrix") or []
    if (not isinstance(matrix, list) or len(matrix) < 2
            or not all(isinstance(r, list) for r in matrix)):
        return {"error": "returns_matrix_must_be_M_strategies_x_T_periods"}
    try:
        from src.backtesting.overfitting_metrics import probability_of_backtest_overfitting
        result = probability_of_backtest_overfitting(
            [[float(v) for v in row] for row in matrix],
        )
        return result.to_dict()
    except Exception as e:
        return {"error": f"pbo_failed: {e}"[:200]}


def _handle_purged_walk_forward(args: Dict[str, Any]) -> Dict[str, Any]:
    """Purged walk-forward validation with embargo (López de Prado 2017).

    Strategy is a 'momentum' baseline that buys/sells based on the
    sign of the prior return — pure-function so it's reproducible
    and parameter-free. The brain calls this to validate a hypothesis
    against leak-free folds.
    """
    returns = args.get("returns") or []
    n_folds = int(args.get("n_folds") or 5)
    embargo_pct = float(args.get("embargo_pct") or 0.05)
    feature_window = int(args.get("feature_window") or 20)
    if not isinstance(returns, list) or len(returns) < 100:
        return {"error": "need_at_least_100_returns_for_purged_wf"}
    try:
        from src.backtesting.purged_walk_forward import purged_walk_forward
        # Default strategy: lag-1 momentum — return at t mirrors sign
        # of return at t-1. Replace with operator-supplied strategy
        # for richer evaluation.
        def _momentum(rets):
            out = [0.0]
            for i in range(1, len(rets)):
                out.append(rets[i] if rets[i - 1] > 0 else -rets[i])
            return out
        result = purged_walk_forward(
            [float(r) for r in returns],
            strategy_fn=_momentum,
            n_folds=n_folds, embargo_pct=embargo_pct,
            feature_window=feature_window,
        )
        return result.to_dict()
    except Exception as e:
        return {"error": f"purged_wf_failed: {e}"[:200]}


def _handle_realistic_slippage(args: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate realistic slippage scaled by size + volatility + latency
    + session — anti-optimistic vs the flat-spread default."""
    venue = str(args.get("venue") or "robinhood")
    size_pct = float(args.get("size_pct_of_equity") or 1.0)
    vol_pct = float(args.get("volatility_pct") or 1.0)
    latency_ms = float(args.get("latency_ms") or 200.0)
    session = str(args.get("session") or "US")
    try:
        from src.backtesting.realistic_slippage import estimate_slippage
        result = estimate_slippage(
            venue=venue, size_pct_of_equity=size_pct,
            volatility_pct=vol_pct, latency_ms=latency_ms,
            session=session,
        )
        return result.to_dict()
    except Exception as e:
        return {"error": f"slippage_estimate_failed: {e}"[:200]}


def _handle_strategy_universe(args: Dict[str, Any]) -> Dict[str, Any]:
    """Top-K strategies from the 242-strategy universe filtered by
    regime. Brain reads to see WHICH strategies are firing now (not
    just the aggregate vote). Anti-noise: returns at most 10
    strategies, sorted by absolute signal strength."""
    asset = str(args.get("asset") or "BTC").upper()
    regime = str(args.get("regime") or "")
    k = max(1, min(20, int(args.get("top_k") or 10)))
    try:
        bars = _fetch_recent_bars(asset, timeframe="1h", n=200)
        if bars is None:
            return {"asset": asset, "error": "no_recent_bars"}
        highs, lows, closes, volumes = bars
        from src.trading.strategy_universe import StrategyUniverse
        u = StrategyUniverse()
        signals = u.evaluate_all(closes, highs, lows, volumes) or {}
        consensus, confidence = u.get_consensus(signals, regime=regime) if regime else u.get_consensus(signals)
        # Top-K by absolute signal strength
        ranked = sorted(
            ((name, sig) for name, sig in signals.items() if sig != 0),
            key=lambda kv: abs(kv[1]), reverse=True,
        )[:k]
        return {
            "asset": asset,
            "regime": regime or "any",
            "consensus": consensus,
            "confidence": round(float(confidence), 3),
            "total_strategies": len(signals),
            "active_strategies": len(ranked),
            "top_k": [
                {"name": str(n)[:40], "signal": int(s)}
                for n, s in ranked
            ],
        }
    except Exception as e:
        return {"error": f"strategy_universe_failed: {e}"[:200]}


def _handle_liquidity_sweep(args: Dict[str, Any]) -> Dict[str, Any]:
    """ICT liquidity-sweep / stop-hunt reversal detector."""
    asset = str(args.get("asset") or "BTC").upper()
    timeframe = str(args.get("timeframe") or "1h")
    lookback = int(args.get("lookback") or 20)
    try:
        from src.trading.strategies.liquidity_sweep import detect_liquidity_sweep
        bars = _fetch_recent_bars(asset, timeframe=timeframe, n=max(50, lookback + 10))
        if bars is None:
            return {"asset": asset, "error": "no_recent_bars"}
        highs, lows, closes, volumes = bars
        det = detect_liquidity_sweep(highs, lows, closes, volumes, lookback=lookback)
        return {"asset": asset, "timeframe": timeframe, **det.to_dict()}
    except Exception as e:
        return {"error": f"liquidity_sweep_failed: {e}"[:200]}


def _handle_pair_trading_signal(args: Dict[str, Any]) -> Dict[str, Any]:
    """Active BTC-ETH pair trading signal (statistical arbitrage).

    Reads the cointegration z-score already computed by the executor's
    pairs check (surfaced via tick_state.pair_z_score). Returns an
    actionable suggestion when |z| > entry threshold.
    """
    z_entry = float(args.get("z_entry") or 2.0)
    z_exit = float(args.get("z_exit") or 0.5)
    try:
        from src.ai import tick_state as _ts
        # Read from BOTH assets — same cointegration signal but each
        # asset's snapshot has it.
        snap_btc = _ts.get("BTC")
        snap_eth = _ts.get("ETH")
        snap = snap_btc or snap_eth or {}
        z = float(snap.get("pair_z_score", 0.0))
        cointegrated = bool(snap.get("pair_cointegrated", False))
        hedge_ratio = float(snap.get("pair_hedge_ratio", 0.0))
        signal = str(snap.get("pair_signal", "NONE"))
        action = "HOLD"
        rationale = f"z={z:+.2f}; abs(z) < {z_entry} (entry threshold)"
        if cointegrated and abs(z) >= z_entry:
            if z > 0:
                action = "SHORT_BTC_LONG_ETH"
                rationale = (
                    f"z={z:+.2f} > {z_entry} → BTC rich vs ETH; "
                    "short BTC + long ETH (hedged). "
                    "On longs-only Robinhood, can only LONG ETH (half-leg)."
                )
            else:
                action = "LONG_BTC_SHORT_ETH"
                rationale = (
                    f"z={z:+.2f} < -{z_entry} → BTC cheap vs ETH; "
                    "long BTC + short ETH (hedged). "
                    "On longs-only Robinhood, can only LONG BTC (half-leg)."
                )
        elif cointegrated and abs(z) < z_exit:
            action = "EXIT_PAIR"
            rationale = f"z={z:+.2f} < {z_exit} → mean reverted, close pair"
        return {
            "z_score": round(z, 3),
            "cointegrated": cointegrated,
            "hedge_ratio": round(hedge_ratio, 3),
            "underlying_signal": signal,
            "action": action,
            "z_entry_threshold": z_entry,
            "z_exit_threshold": z_exit,
            "rationale": rationale[:300],
            "venue_note": (
                "Robinhood spot is longs-only. On crossing entry "
                "threshold, the brain can fire only the LONG leg of the "
                "pair (half-leg pair). Full pair requires a venue "
                "supporting shorts."
            ),
        }
    except Exception as e:
        return {"error": f"pair_signal_failed: {e}"[:200]}


def _handle_session_bias(args: Dict[str, Any]) -> Dict[str, Any]:
    """Session-aware volume bias + conviction multiplier."""
    try:
        from src.trading.strategies.session_bias import current_session
        return current_session()
    except Exception as e:
        return {"error": f"session_bias_failed: {e}"[:200]}


def _handle_grid_chop(args: Dict[str, Any]) -> Dict[str, Any]:
    """Grid trading suggestions for ranging / CHOP regimes."""
    asset = str(args.get("asset") or "BTC").upper()
    try:
        from src.ai import tick_state as _ts
        snap = _ts.get(asset) or {}
        from src.trading.strategies.grid_chop import grid_advisory
        return grid_advisory(
            asset=asset,
            current_price=float(snap.get("price", 0.0)),
            atr=float(snap.get("atr", 0.0)),
            regime=str(snap.get("regime", "unknown")),
            hurst_value=float(snap.get("hurst_value", 0.5)),
            spread_pct=float(snap.get("spread_pct", 1.69)),
        )
    except Exception as e:
        return {"error": f"grid_chop_failed: {e}"[:200]}


def _handle_wyckoff_phase(args: Dict[str, Any]) -> Dict[str, Any]:
    """Wyckoff phase detector (accumulation/markup/distribution/markdown)."""
    asset = str(args.get("asset") or "BTC").upper()
    timeframe = str(args.get("timeframe") or "4h")
    lookback = int(args.get("lookback") or 50)
    try:
        from src.trading.strategies.wyckoff_phase import detect_phase
        bars = _fetch_recent_bars(asset, timeframe=timeframe, n=max(60, lookback + 10))
        if bars is None:
            return {"asset": asset, "error": "no_recent_bars"}
        highs, lows, closes, volumes = bars
        v = detect_phase(closes, highs, lows, volumes, lookback=lookback)
        return {"asset": asset, "timeframe": timeframe, **v.to_dict()}
    except Exception as e:
        return {"error": f"wyckoff_failed: {e}"[:200]}


def _handle_system_health(args: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate system health: error counts by severity + which
    components have recent errors. Brain knows when to trust which
    signal — e.g. if news API is down, sentiment may be stale.

    Anti-noise: returns ONLY aggregate counts and per-component last-
    error timestamps. No raw stack traces.
    """
    try:
        from src.monitoring.auto_healer import AutoHealer
        ah = AutoHealer({"max_log_tail": 200})
        try:
            tail = ah._tail_log(max_lines=200) if hasattr(ah, "_tail_log") else []
            errors = ah._detect_errors(tail) if hasattr(ah, "_detect_errors") else []
        except Exception:
            errors, tail = [], []
        sev_counts = {"ERROR": 0, "WARNING": 0, "CRITICAL": 0}
        components: Dict[str, int] = {}
        for e in errors[:50]:
            lvl = str(e.get("level", "ERROR")).upper()
            sev_counts[lvl] = sev_counts.get(lvl, 0) + 1
            comp = str(e.get("component", "unknown"))[:30]
            components[comp] = components.get(comp, 0) + 1
        return {
            "log_lines_scanned": len(tail),
            "error_severity_counts": sev_counts,
            "components_with_errors": dict(sorted(components.items(),
                                                   key=lambda kv: kv[1],
                                                   reverse=True)[:10]),
            "advisory": (
                "Trust feeds whose component has 0 errors. Down-weight "
                "or skip tools whose component appears in the error list."
            ),
        }
    except Exception as e:
        return {"error": f"system_health_failed: {e}"[:200]}


def _handle_decision_audit_summary(args: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate audit of past decisions — win-rate by conviction tier,
    by regime, by pattern_score bucket. Pure stats, anti-overfitting:
    no individual data points exposed; the brain reads patterns not
    instances."""
    asset = str(args.get("asset") or "").upper()
    lookback = max(20, min(500, int(args.get("lookback") or 100)))
    try:
        import sqlite3, json as _json
        from src.orchestration.warm_store import get_store
        store = get_store()
        conn = sqlite3.connect(store.db_path, timeout=2.0)
        try:
            sql = (
                "SELECT plan_json, self_critique, final_action FROM decisions "
                "WHERE plan_json != '{}' "
                + ("AND symbol = ? " if asset else "")
                + "ORDER BY ts_ns DESC LIMIT ?"
            )
            params = ((asset, lookback) if asset else (lookback,))
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        # Bucket by tier, regime, pattern_score; report WR per bucket.
        from collections import defaultdict
        buckets = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})

        def _bucket(key, win, pnl):
            b = buckets[key]
            b["n"] += 1
            if win:
                b["wins"] += 1
            b["pnl"] += float(pnl or 0)

        for plan_raw, crit_raw, _action in rows:
            try:
                plan = _json.loads(plan_raw or "{}")
                crit = _json.loads(crit_raw or "{}")
            except Exception:
                continue
            tier = str(plan.get("entry_tier", "?"))
            direction = str(plan.get("direction", "?"))
            pnl = crit.get("realized_pnl_pct")
            if pnl is None:
                continue
            win = float(pnl) > 0
            _bucket(f"tier:{tier}", win, pnl)
            _bucket(f"direction:{direction}", win, pnl)
            score = plan.get("pattern_score") or 0
            try:
                score_bucket = "score>=8" if int(score) >= 8 else (
                    "score:5-7" if int(score) >= 5 else "score<5")
                _bucket(score_bucket, win, pnl)
            except Exception:
                pass

        out = {}
        for k, v in buckets.items():
            n = max(1, v["n"])
            out[k] = {
                "samples": v["n"],
                "win_rate": round(v["wins"] / n, 3),
                "avg_pnl_pct": round(v["pnl"] / n, 3),
            }
        return {
            "lookback_decisions": len(rows),
            "asset_filter": asset or "ALL",
            "patterns": out,
            "advisory": (
                "Use win_rate by bucket to calibrate conviction. If a "
                "bucket has <10 samples treat it as low-confidence stat."
            ),
        }
    except Exception as e:
        return {"error": f"decision_audit_failed: {e}"[:200]}


def _handle_chart_vision(args: Dict[str, Any]) -> Dict[str, Any]:
    """Visual chart pattern summary. Renders OHLCV to PNG and asks the
    vision model for pattern recognition. Disabled by default; returns
    'not_enabled' when the feature flag is off (anti-noise)."""
    asset = str(args.get("asset") or "BTC").upper()
    timeframe = str(args.get("timeframe") or "1h")
    try:
        from src.ai.chart_vision import is_enabled, chart_summary_section
        if not is_enabled():
            return {"asset": asset, "enabled": False,
                    "advisory": "chart_vision disabled — set ACT_CHART_VISION=1 to enable"}
        summary = chart_summary_section(asset, timeframe=timeframe) or {}
        return {"asset": asset, "timeframe": timeframe, "summary": summary}
    except Exception as e:
        return {"error": f"chart_vision_failed: {e}"[:200]}


def _handle_trade_verifier_state(args: Dict[str, Any]) -> Dict[str, Any]:
    """Recent post-trade SelfCritique entries — predicted vs realized PnL,
    catalyst hits/misses, slippage. Brain reads to avoid repeating
    setups that lost in similar conditions."""
    asset = str(args.get("asset") or "").upper()
    limit = max(1, min(20, int(args.get("limit") or 5)))
    try:
        import sqlite3, json as _json
        from src.orchestration.warm_store import get_store
        store = get_store()
        conn = sqlite3.connect(store.db_path, timeout=2.0)
        try:
            sql = (
                "SELECT ts_ns, symbol, plan_json, self_critique, final_action "
                "FROM decisions WHERE self_critique != '{}' AND self_critique IS NOT NULL "
                + ("AND symbol = ? " if asset else "")
                + "ORDER BY ts_ns DESC LIMIT ?"
            )
            params = ((asset, limit) if asset else (limit,))
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        out = []
        for ts_ns, sym, plan_raw, crit_raw, action in rows:
            try:
                plan = _json.loads(plan_raw or "{}")
                crit = _json.loads(crit_raw or "{}")
            except Exception:
                plan, crit = {}, {}
            out.append({
                "ts_ns": int(ts_ns), "asset": sym,
                "predicted_direction": plan.get("direction", "?"),
                "predicted_pnl_range": plan.get("expected_pnl_pct_range", []),
                "realized_pnl_pct": crit.get("realized_pnl_pct"),
                "verdict": crit.get("verdict", ""),
                "miss_reasons": str(crit.get("miss_reasons", ""))[:200],
                "lessons": str(crit.get("lessons", ""))[:200],
                "final_action": action,
            })
        return {"verifications": out, "count": len(out)}
    except Exception as e:
        return {"error": f"verifier_query_failed: {e}"[:200]}


def _handle_feature_drift(args: Dict[str, Any]) -> Dict[str, Any]:
    """Feature distribution drift (PSI, z-score) so brain knows when ML
    model decay is detected before circuit breaker fires."""
    try:
        from src.monitoring.drift_detector import DriftDetector
        det = DriftDetector()
        # DriftDetector tracks features the executor feeds it. Without
        # a singleton accessor we can only describe known-monitored features.
        features = ["rsi", "ema_slope", "atr_pct", "vol_z", "spread_pct"]
        out = {}
        for f in features:
            try:
                psi, status = det.check_feature(f)
                out[f] = {"psi": round(float(psi), 4), "status": str(status)}
            except Exception:
                out[f] = {"psi": 0.0, "status": "no_baseline"}
        return {"features": out, "advisory": "psi>0.25 = significant drift; rebaseline required"}
    except Exception as e:
        return {"error": f"drift_query_failed: {e}"[:200]}


def _handle_circuit_breaker_state(args: Dict[str, Any]) -> Dict[str, Any]:
    """Circuit-breaker state across registered components (open/closed/
    half-open) so brain reasons about whether to trade conservatively."""
    try:
        # Each subsystem owns its breaker; we don't have a registry.
        # Surface a graceful status that says "no central registry yet"
        # so brain doesn't loop on this.
        return {
            "state": "n/a",
            "advisory": (
                "Circuit breakers are per-subsystem (no central registry). "
                "Watch tick_state.last_refusal and emergency_level instead."
            ),
        }
    except Exception as e:
        return {"error": f"breaker_query_failed: {e}"[:200]}


def _handle_on_chain_signals(args: Dict[str, Any]) -> Dict[str, Any]:
    """On-chain metrics — blockchain stats, mempool, stablecoins, hashrate.
    Independent of news_digest so brain can ask 'what's the on-chain
    picture specifically?'"""
    try:
        from src.data.on_chain_fetcher import OnChainFetcher
        f = OnChainFetcher()
        out: Dict[str, Any] = {}
        try:
            out["blockchain_stats"] = f._fetch_blockchain_com_stats() or {}
        except Exception:
            out["blockchain_stats"] = {}
        try:
            out["mempool"] = f._fetch_mempool_stats() or {}
        except Exception:
            out["mempool"] = {}
        try:
            out["stablecoins"] = f.fetch_defillama_stablecoins() or {}
        except Exception:
            out["stablecoins"] = {}
        try:
            out["hashrate"] = f._fetch_mempool_hashrate() or {}
        except Exception:
            out["hashrate"] = {}
        return out
    except Exception as e:
        return {"error": f"on_chain_query_failed: {e}"[:200]}


def _handle_institutional_flows(args: Dict[str, Any]) -> Dict[str, Any]:
    """Institutional flows — macro correlations, stablecoin flows, options
    sentiment, long/short ratio, cross-exchange spreads."""
    asset = str(args.get("asset") or "BTC").upper()
    try:
        from src.data.institutional_fetcher import InstitutionalFetcher
        f = InstitutionalFetcher()
        out = f.get_all_institutional(asset) or {}
        return {"asset": asset, "metrics": out}
    except Exception as e:
        return {"error": f"institutional_query_failed: {e}"[:200]}


def _handle_adaptation_state(args: Dict[str, Any]) -> Dict[str, Any]:
    """Bandit + credit-assigner state. Brain sees per-strategy posterior
    means + per-component credit weights so it understands its own
    adaptation."""
    asset = str(args.get("asset") or "BTC").upper()
    out: Dict[str, Any] = {"asset": asset}
    try:
        from src.learning.thompson_bandit import top_k_by_posterior_mean
        from src.trading.strategy_repository import get_repo
        try:
            repo = get_repo()
            recs = (repo.search(status="champion", limit=50) +
                    repo.search(status="challenger", limit=50) +
                    repo.search(status="candidate", limit=50))
            top = top_k_by_posterior_mean(recs, k=5) or []
            out["top_strategies_by_posterior"] = [
                {"id": str(getattr(t, "strategy_id", "") or "")[:40],
                 "posterior_mean": round(float(getattr(t, "posterior_mean", 0)), 4),
                 "alpha": float(getattr(t, "alpha", 0)),
                 "beta": float(getattr(t, "beta", 0))}
                for t in top
            ]
        except Exception as e:
            out["bandit_error"] = str(e)[:120]
    except ImportError:
        out["bandit_error"] = "thompson_bandit_unavailable"
    try:
        from src.learning.credit_assigner import CreditAssigner
        ca = CreditAssigner()
        try:
            weights = ca.weights() if hasattr(ca, "weights") else {}
            out["credit_weights"] = {k: round(float(v), 4) for k, v in (weights or {}).items()}
        except Exception as e:
            out["credit_error"] = str(e)[:120]
    except ImportError:
        out["credit_error"] = "credit_assigner_unavailable"
    return out


def _handle_genetic_evolution_state(args: Dict[str, Any]) -> Dict[str, Any]:
    """Full evolution state: generations run, best fitness, mutation
    rate, stagnation, pareto front size. Brain reads this to know
    whether the genetic loop is healthy (still discovering new
    strategies) or stagnant (mutation rate maxed, no improvement).
    """
    try:
        from src.trading.genetic_strategy_engine import GeneticStrategyEngine
        # The executor instantiates a singleton; reuse its hall-of-fame
        # via the file the engine writes (logs/genetic_evolution_history.jsonl
        # has per-generation entries; hall-of-fame DNA is in
        # data/strategy_repo.sqlite). Build a fresh engine to read summary.
        eng = GeneticStrategyEngine()
        try:
            eng.load_hall_of_fame_from_repo() if hasattr(eng, "load_hall_of_fame_from_repo") else None
        except Exception:
            pass
        summary = eng.get_evolution_summary() if hasattr(eng, "get_evolution_summary") else {}
        return summary or {"error": "no_evolution_summary"}
    except Exception as e:
        return {"error": f"genetic_state_query_failed: {e}"[:200]}


def _handle_genetic_hall_of_fame(args: Dict[str, Any]) -> Dict[str, Any]:
    """Top-N hall-of-fame strategy DNAs (entry/exit rules + fitness +
    win rate + total PnL). Brain inspects which evolved patterns are
    winning so it can incorporate that intuition into its own
    reasoning (e.g. "the top hall-of-fame strategy uses RSI<30 +
    BB-low for entry — does the current setup match?").
    """
    limit = int(args.get("limit") or 5)
    limit = max(1, min(20, limit))
    try:
        import sqlite3
        import json as _json
        # strategy_repo.sqlite stores promoted strategies with status,
        # regime, sharpe — query top by fitness/sharpe.
        repo_path = "data/strategy_repo.sqlite"
        out: List[Dict[str, Any]] = []
        try:
            conn = sqlite3.connect(repo_path, timeout=2.0)
            try:
                rows = conn.execute(
                    "SELECT name, regime, dna_json, sharpe, win_rate, total_pnl, status "
                    "FROM strategies WHERE status IN ('champion','challenger') "
                    "ORDER BY sharpe DESC NULLS LAST LIMIT ?",
                    (int(limit),),
                ).fetchall()
            finally:
                conn.close()
            for name, regime, dna_json, sharpe, wr, pnl, status in rows:
                try:
                    dna = _json.loads(dna_json or "{}")
                except Exception:
                    dna = {}
                out.append({
                    "name": str(name)[:40],
                    "regime": str(regime),
                    "status": str(status),
                    "sharpe": round(float(sharpe or 0), 3),
                    "win_rate": round(float(wr or 0), 3),
                    "total_pnl": round(float(pnl or 0), 2),
                    "entry_rule": str(dna.get("entry_rule", ""))[:120],
                    "exit_rule": str(dna.get("exit_rule", ""))[:120],
                })
        except sqlite3.OperationalError:
            return {"error": "strategy_repo_not_initialized"}
        return {"hall_of_fame": out, "count": len(out)}
    except Exception as e:
        return {"error": f"hall_of_fame_query_failed: {e}"[:200]}


def _handle_recovery_plan(args: Dict[str, Any]) -> Dict[str, Any]:
    """When the portfolio is stuck (many opens, large net negative),
    rank positions by current_pnl_pct_net and identify:
      - profitable_closes: positions ALREADY net positive — close
        these immediately to realize gains
      - near_breakeven: positions with net > -0.3% — likely to flip
        positive on small upward move; hold and watch
      - deep_losers: positions worse than -2% net — accept the loss
        OR hold for trend recovery, depending on thesis
      - best_partial_candidates: top N positions by gross gain that
        could be partial-closed at +2% gross targets

    Brain calls this in stuck-portfolio recovery mode to plan the
    sequence of closes rather than blanket-close everything.
    """
    asset_filter = str(args.get("asset") or "").upper().strip()
    try:
        from src.data.robinhood_fetcher import get_active_paper_fetcher
        pf = get_active_paper_fetcher()
        if pf is None:
            return {"error": "no_active_paper_fetcher"}
        # Pull live tick_state for each asset to evaluate trend-
        # favorability per position (the brain wants to know if the
        # trend favors this specific direction RIGHT NOW).
        from src.ai import tick_state as _ts
        _trend_by_asset: Dict[str, Dict[str, Any]] = {}
        for _a in {str(p.asset).upper() for p in pf.positions.values()}:
            _snap = _ts.get(_a)
            _trend_by_asset[_a] = {
                "ema_dir": str(_snap.get("ema_direction", "?")),
                "regime": str(_snap.get("regime", "?")),
                "hurst_value": float(_snap.get("hurst_value", 0.5)),
                "hurst_regime": str(_snap.get("hurst_regime", "?")),
                "kalman_trend": str(_snap.get("kalman_trend", "?")),
            }
        rows: List[Dict[str, Any]] = []
        for trade_id, p in pf.positions.items():
            if asset_filter and str(p.asset).upper() != asset_filter:
                continue
            _t = _trend_by_asset.get(str(p.asset).upper(), {})
            # trend favours LONG if price > entry, EMA rising, hurst > 0.5,
            # regime trending; mirror for SHORT.
            ema_aligned = (
                (p.direction == "LONG" and _t.get("ema_dir") == "RISING") or
                (p.direction == "SHORT" and _t.get("ema_dir") == "FALLING")
            )
            hurst_trending = float(_t.get("hurst_value", 0.5)) > 0.5
            kalman_aligned = (
                (p.direction == "LONG" and _t.get("kalman_trend") == "UP") or
                (p.direction == "SHORT" and _t.get("kalman_trend") == "DOWN")
            )
            regime_ok = _t.get("regime", "").upper() not in ("CRISIS", "")
            trend_favors = sum([ema_aligned, hurst_trending, kalman_aligned, regime_ok])
            rows.append({
                "trade_id": trade_id,
                "asset": p.asset,
                "direction": p.direction,
                "entry_price": round(float(p.entry_price), 2),
                "current_price": round(float(getattr(p, "current_price", 0)), 2),
                "pnl_pct_gross": round(float(getattr(p, "current_pnl_pct", 0) or 0), 3),
                "pnl_pct_net": round(float(getattr(p, "current_pnl_pct_net", 0) or 0), 3),
                "pnl_usd_net": round(float(getattr(p, "current_pnl_usd_net", 0) or 0), 2),
                "trend_favors_score": int(trend_favors),  # 0-4; >=3 = favorable
                "trend_signals": {
                    "ema_aligned": ema_aligned,
                    "hurst_trending": hurst_trending,
                    "kalman_aligned": kalman_aligned,
                    "regime_ok": regime_ok,
                },
                "recommendation": (
                    "PARTIAL_CLOSE" if (float(getattr(p, "current_pnl_pct", 0) or 0) >= 2.5
                                         and trend_favors >= 3)
                    else "HOLD" if (trend_favors >= 2)
                    else "EVALUATE_THESIS"  # trend against — check news/macro before close
                ),
            })
        if not rows:
            return {"asset": asset_filter or "ALL", "open_count": 0,
                    "advisory": "no open positions"}
        rows.sort(key=lambda r: r["pnl_pct_net"], reverse=True)

        profitable = [r for r in rows if r["pnl_pct_net"] > 0]
        near_be = [r for r in rows if -0.3 <= r["pnl_pct_net"] <= 0]
        deep_losers = [r for r in rows if r["pnl_pct_net"] < -2.0]
        best_partial = [r for r in rows if r["pnl_pct_gross"] >= 2.0][:10]

        total_close_now_usd = sum(r["pnl_usd_net"] for r in rows)
        return {
            "asset": asset_filter or "ALL",
            "open_count": len(rows),
            "close_all_now_realized_usd": round(total_close_now_usd, 2),
            "profitable_closes": profitable[:10],
            "near_breakeven_count": len(near_be),
            "near_breakeven_sample": near_be[:5],
            "deep_losers_count": len(deep_losers),
            "deep_losers_sample": deep_losers[:5],
            "best_partial_candidates": best_partial,
            "advisory": (
                f"profitable_closes: {len(profitable)} positions are NET "
                f"positive — close them to realize gains. "
                f"near_breakeven: {len(near_be)} small upward move flips "
                "them positive; hold short-term. "
                f"deep_losers: {len(deep_losers)} accept loss OR hold for "
                "trend if thesis intact. "
                f"close_all_now_realized = ${total_close_now_usd:.2f}."
            ),
        }
    except Exception as e:
        return {"error": f"recovery_plan_failed: {e}"[:200]}


def _handle_open_positions_detail(args: Dict[str, Any]) -> Dict[str, Any]:
    """Per-position breakdown — entry price, current PnL, age, original
    thesis, trade_id. Brain reads this to decide HOLD/EXIT/PARTIAL on
    each open trade individually (not just aggregate stats)."""
    asset_filter = str(args.get("asset") or "").upper().strip()
    try:
        from src.data.robinhood_fetcher import get_active_paper_fetcher
        pf = get_active_paper_fetcher()
        if pf is None:
            return {"error": "no_active_paper_fetcher"}
        import time as _time
        from datetime import datetime as _dt, timezone as _tz
        out: List[Dict[str, Any]] = []
        for trade_id, p in pf.positions.items():
            if asset_filter and str(p.asset).upper() != asset_filter:
                continue
            _et_raw = getattr(p, "entry_time", None)
            age_min = 0.0
            if isinstance(_et_raw, str):
                try:
                    _et = _dt.fromisoformat(_et_raw.replace("Z", "+00:00"))
                    age_min = (_dt.now(tz=_tz.utc) - _et).total_seconds() / 60.0
                except Exception:
                    age_min = 0.0
            _entry = float(p.entry_price)
            _peak = float(getattr(p, "peak_price", _entry) or _entry)
            _cur = float(getattr(p, "current_price", _entry) or _entry)
            _pnl_gross = float(getattr(p, "current_pnl_pct", 0) or 0)
            _pnl_net = float(getattr(p, "current_pnl_pct_net", _pnl_gross) or _pnl_gross)
            # Distance from peak (negative = below peak; for LONG that
            # means giving back gains; for SHORT means giving back gains).
            if p.direction == "LONG":
                _dist_from_peak_pct = ((_cur - _peak) / _peak * 100.0) if _peak else 0.0
            else:
                _dist_from_peak_pct = ((_peak - _cur) / _peak * 100.0) if _peak else 0.0
            out.append({
                "trade_id": trade_id,
                "asset": p.asset,
                "direction": p.direction,
                "entry_price": round(_entry, 2),
                "current_price": round(_cur, 2),
                "peak_price": round(_peak, 2),
                "distance_from_peak_pct": round(_dist_from_peak_pct, 3),
                "current_pnl_pct_gross": round(_pnl_gross, 3),
                "current_pnl_pct_net": round(_pnl_net, 3),
                "current_pnl_usd_gross": round(float(getattr(p, "current_pnl_usd", 0) or 0), 2),
                "current_pnl_usd_net": round(float(getattr(p, "current_pnl_usd_net", 0) or 0), 2),
                "quantity": float(p.quantity),
                "age_min": round(age_min, 1),
                "bars_held": int(getattr(p, "bars_held", 0) or 0),
                "score": int(getattr(p, "score", 0)),
                "thesis": str(getattr(p, "reasoning", ""))[:200],
                "sl_price": round(float(getattr(p, "sl_price", 0)), 2),
                "tp_price": round(float(getattr(p, "tp_price", 0)), 2),
            })
        return {"positions": out, "count": len(out)}
    except Exception as e:
        return {"error": f"open_positions_query_failed: {e}"[:200]}


def _handle_close_paper_position(args: Dict[str, Any]) -> Dict[str, Any]:
    """Brain-initiated paper position close. Closes the first matching
    open position by asset (or specific trade_id if provided). Supports
    partial close via `fraction` (0.0-1.0; 1.0 = full close, default).
    Reason is logged.

    NEVER closes real-capital positions — paper-only by design.
    """
    import os as _os
    if _os.environ.get("ACT_REAL_CAPITAL_ENABLED", "").strip() == "1":
        return {"error": "close_paper_position_disabled_in_real_capital_mode"}

    asset = str(args.get("asset") or "").upper().strip()
    trade_id = str(args.get("trade_id") or "").strip()
    reason = str(args.get("reason") or "brain_initiated_close")[:200]
    try:
        fraction = float(args.get("fraction") or 1.0)
    except Exception:
        fraction = 1.0
    fraction = max(0.05, min(1.0, fraction))
    if not asset and not trade_id:
        return {"error": "asset_or_trade_id_required"}
    try:
        from src.data.robinhood_fetcher import get_active_paper_fetcher
        pf = get_active_paper_fetcher()
        if pf is None:
            return {"error": "no_active_paper_fetcher"}
        target_asset = asset
        if trade_id and trade_id in pf.positions:
            target_asset = pf.positions[trade_id].asset

        # Partial close: split the position before record_exit fires.
        # Find the target position, reduce its quantity, and create a
        # closed-portion record by calling record_exit with the carved-
        # off slice.
        if fraction < 1.0:
            target_pos = None
            target_key = None
            if trade_id and trade_id in pf.positions:
                target_key = trade_id
                target_pos = pf.positions[trade_id]
            else:
                for k, p in pf.positions.items():
                    if str(p.asset).upper() == target_asset:
                        target_key, target_pos = k, p
                        break
            if target_pos is None:
                return {"closed": False, "reason": "no_open_position", "asset": target_asset}
            close_qty = float(target_pos.quantity) * fraction
            target_pos.quantity = float(target_pos.quantity) - close_qty
            # Synthetic record_exit for the closed slice — reuse helper
            # by temporarily swapping qty.
            full_qty = target_pos.quantity + close_qty
            target_pos.quantity = close_qty
            closed = pf.record_exit(asset=target_asset, reason=f"brain partial {fraction:.0%}: {reason}")
            # restore remaining qty into a fresh position copy
            if closed is not None:
                from copy import copy as _copy
                remaining = _copy(closed)
                remaining.exit_price = None
                remaining.exit_time = None
                remaining.exit_reason = ""
                remaining.final_pnl_pct = 0.0
                remaining.final_pnl_usd = 0.0
                remaining.status = "open"
                remaining.quantity = full_qty - close_qty
                pf.positions[target_key] = remaining
            pf.save_state()
            if closed is None:
                return {"closed": False, "reason": "partial_close_failed"}
            return {
                "closed": True, "partial": True, "fraction": fraction,
                "asset": closed.asset, "direction": closed.direction,
                "exit_price": float(closed.exit_price or closed.current_price),
                "pnl_pct": round(float(closed.final_pnl_pct or 0), 3),
                "pnl_usd": round(float(closed.final_pnl_usd or 0), 2),
                "remaining_qty": round(full_qty - close_qty, 8),
            }

        # Full close.
        closed = pf.record_exit(asset=target_asset, reason=f"brain: {reason}")
        if closed is None:
            return {"closed": False, "reason": "no_open_position_for_asset", "asset": target_asset}
        pf.save_state()
        return {
            "closed": True, "partial": False,
            "asset": closed.asset, "direction": closed.direction,
            "entry_price": float(closed.entry_price),
            "exit_price": float(closed.exit_price or closed.current_price),
            "pnl_pct": round(float(closed.final_pnl_pct or 0), 3),
            "pnl_usd": round(float(closed.final_pnl_usd or 0), 2),
            "reason_logged": f"brain: {reason}",
        }
    except Exception as e:
        return {"error": f"close_failed: {e}"[:200]}


def _handle_modify_paper_position(args: Dict[str, Any]) -> Dict[str, Any]:
    """Adjust SL/TP on an open paper position. Brain calls this when
    the trade is going its way and it wants to lock in profit (raise
    SL toward entry) or extend the runway (raise TP)."""
    import os as _os
    if _os.environ.get("ACT_REAL_CAPITAL_ENABLED", "").strip() == "1":
        return {"error": "modify_paper_position_disabled_in_real_capital_mode"}

    asset = str(args.get("asset") or "").upper().strip()
    trade_id = str(args.get("trade_id") or "").strip()
    new_sl = args.get("new_sl_price")
    new_tp = args.get("new_tp_price")
    if new_sl is None and new_tp is None:
        return {"error": "must_provide_new_sl_price_or_new_tp_price"}
    if not asset and not trade_id:
        return {"error": "asset_or_trade_id_required"}
    try:
        from src.data.robinhood_fetcher import get_active_paper_fetcher
        pf = get_active_paper_fetcher()
        if pf is None:
            return {"error": "no_active_paper_fetcher"}
        target_pos = None
        target_key = None
        if trade_id and trade_id in pf.positions:
            target_key = trade_id
            target_pos = pf.positions[trade_id]
        else:
            for k, p in pf.positions.items():
                if str(p.asset).upper() == asset:
                    target_key, target_pos = k, p
                    break
        if target_pos is None:
            return {"error": "no_open_position", "asset": asset}
        old_sl = float(getattr(target_pos, "sl_price", 0))
        old_tp = float(getattr(target_pos, "tp_price", 0))
        if new_sl is not None:
            target_pos.sl_price = float(new_sl)
        if new_tp is not None:
            target_pos.tp_price = float(new_tp)
        pf.save_state()
        return {
            "modified": True,
            "trade_id": target_key,
            "old_sl": old_sl, "new_sl": float(target_pos.sl_price),
            "old_tp": old_tp, "new_tp": float(target_pos.tp_price),
        }
    except Exception as e:
        return {"error": f"modify_failed: {e}"[:200]}


def _handle_sizing_preview(args: Dict[str, Any]) -> Dict[str, Any]:
    """Preview the final size_pct after the body's modulation pipeline.

    The brain proposes size_pct; the body multiplies through:
      AdaptiveFeedback (recent WR) × SelfEvolvingOverlay (evolved risk)
      × AccuracyEngine (per-component weight) × DynamicPositionLimits
      (regime cap) → final size_pct.

    Brain calls this BEFORE submit_trade_plan so the proposal already
    accounts for the modulation rather than being surprised by a
    smaller final fill.
    """
    asset = str(args.get("asset") or "BTC").upper()
    proposed = float(args.get("proposed_size_pct") or 2.0)
    regime = str(args.get("regime") or "unknown")
    out: Dict[str, Any] = {
        "asset": asset, "regime": regime,
        "proposed_size_pct": proposed,
    }
    factor = 1.0
    factors: Dict[str, float] = {}
    try:
        from src.learning.adaptive_feedback import AdaptiveFeedbackLoop
        af = AdaptiveFeedbackLoop()
        ctx = af.get_adaptive_context(asset, regime) if hasattr(af, "get_adaptive_context") else {}
        size_mult = float(ctx.get("size_multiplier", 1.0) or 1.0)
        factors["adaptive_feedback"] = size_mult
        factor *= size_mult
    except Exception:
        factors["adaptive_feedback"] = 1.0
    try:
        from src.trading.self_evolving_overlay import SelfEvolvingOverlay
        ov = SelfEvolvingOverlay().get_overrides() or {}
        sm = float((ov.get("risk_params") or {}).get("size_mult", 1.0) or 1.0)
        factors["evolved_overlay"] = sm
        factor *= sm
    except Exception:
        factors["evolved_overlay"] = 1.0
    try:
        from src.learning.accuracy_engine import AccuracyEngine
        ae = AccuracyEngine()
        sm = float(ae.get_position_size_multiplier()) if hasattr(ae, "get_position_size_multiplier") else 1.0
        factors["accuracy_engine"] = sm
        factor *= sm
    except Exception:
        factors["accuracy_engine"] = 1.0
    cap_pct = 100.0
    try:
        from src.risk.dynamic_position_limits import DynamicPositionLimits
        cap_pct = float(DynamicPositionLimits().get_max_position_pct(asset, regime))
        factors["position_limits_cap_pct"] = cap_pct
    except Exception:
        factors["position_limits_cap_pct"] = cap_pct
    final = min(proposed * factor, cap_pct)
    out.update({
        "factors": {k: round(v, 3) for k, v in factors.items()},
        "modulation_factor": round(factor, 3),
        "final_size_pct": round(final, 3),
        "would_be_capped_by_limits": bool(proposed * factor > cap_pct),
        "advisory": (
            "If final_size_pct < 1.0%, the modulation is suppressing your "
            "size — consider raising proposed_size_pct or finding a "
            "higher-conviction setup. If would_be_capped_by_limits is "
            "true, the regime cap is binding."
        ),
    })
    return out


def _handle_venue_capabilities(args: Dict[str, Any]) -> Dict[str, Any]:
    """What operations the active venue supports — brain reads this so
    it doesn't propose impossible actions (e.g. SHORT on Robinhood)."""
    import os as _os
    venue = str(args.get("venue") or "robinhood").lower()
    caps = {
        "venue": venue,
        "supports_long": True,
        "supports_short": False,
        "supports_leverage": False,
        "max_leverage": 1.0,
        "supports_partial_close": True,
        "supports_modify_sl_tp": True,
        "supports_limit_orders": True,
        "round_trip_spread_pct": 1.69,
    }
    if venue == "bybit":
        caps.update(supports_short=True, supports_leverage=True,
                    max_leverage=10.0, round_trip_spread_pct=0.055)
    elif venue == "polymarket":
        caps.update(supports_long=True, supports_short=True,
                    supports_leverage=False, round_trip_spread_pct=2.0)
    # Override from env (cost_gate is single source of truth at runtime)
    try:
        from src.trading.cost_gate import get_spread_pct
        caps["round_trip_spread_pct"] = float(get_spread_pct(venue))
    except Exception:
        pass
    return caps


def _handle_champion_gate(args: Dict[str, Any]) -> Dict[str, Any]:
    """Current champion adapter state — LLM/LoRA champion (ai/champion_gate)
    + ML model champion (ml/champion_gate, lightgbm). Brain reads to
    know which model versions are serving."""
    out: Dict[str, Any] = {}
    try:
        from src.ai.champion_gate import get_champion_state
        state = get_champion_state() or {}
        out["llm_champion"] = {
            "champion_id": str(state.get("champion_id", "default"))[:64],
            "challenger_id": str(state.get("challenger_id", ""))[:64],
            "last_gate_decision": str(state.get("last_decision", "n/a"))[:64],
            "champion_metrics": state.get("champion_metrics") or {},
        }
    except ImportError:
        out["llm_champion"] = {"error": "ai_champion_gate_no_state_api"}
    except Exception as e:
        out["llm_champion"] = {"error": f"{type(e).__name__}: {e}"[:120]}
    # ML model champion (lightgbm) — separate file, ml/champion_gate.py.
    # The module exposes evaluate_and_gate; champion is the file path
    # of the active production model. Surface what we can.
    try:
        import os as _os
        prod_path = _os.path.join("models", "lgbm_btc.txt")
        challenger_path = _os.path.join("models", "lgbm_btc_optimized.txt")
        out["ml_champion"] = {
            "champion_path": prod_path,
            "champion_exists": _os.path.exists(prod_path),
            "challenger_path": challenger_path,
            "challenger_exists": _os.path.exists(challenger_path),
        }
    except Exception as e:
        out["ml_champion"] = {"error": f"{type(e).__name__}: {e}"[:120]}
    return out


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
        Tool(
            name="query_deflated_sharpe",
            description=(
                "[BACKTEST] Deflated Sharpe Ratio (Bailey-López de Prado "
                "2014). Adjusts observed Sharpe for selection bias "
                "(n_trials), non-normality (skew + kurtosis), and "
                "small sample. Returns probability the TRUE Sharpe > 0. "
                "Standard authority-submission metric."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "returns": {"type": "array", "items": {"type": "number"}},
                    "n_trials": {"type": "integer", "minimum": 1},
                },
                "required": ["returns"],
            },
            handler=_handle_deflated_sharpe, tag="read_only",
        ),
        Tool(
            name="query_probability_of_backtest_overfitting",
            description=(
                "[BACKTEST] Probability of Backtest Overfitting (Bailey "
                "et al. 2017). Combinatorial symmetric splits over "
                "M-strategies × T-periods returns matrix. PBO=0.5 means "
                "the in-sample winner is a coin-flip out-of-sample (= "
                "pure overfit); PBO < 0.2 = robust."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "returns_matrix": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                    },
                },
                "required": ["returns_matrix"],
            },
            handler=_handle_pbo, tag="read_only",
        ),
        Tool(
            name="query_purged_walk_forward",
            description=(
                "[BACKTEST] Purged walk-forward validation with embargo "
                "(López de Prado 2017). Per-fold metrics + overfit "
                "indicator (train_sharpe - test_sharpe). Default "
                "strategy is lag-1 momentum (parameter-free)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "returns": {"type": "array", "items": {"type": "number"}},
                    "n_folds": {"type": "integer", "minimum": 3, "maximum": 10},
                    "embargo_pct": {"type": "number", "minimum": 0.01, "maximum": 0.3},
                    "feature_window": {"type": "integer", "minimum": 5, "maximum": 200},
                },
                "required": ["returns"],
            },
            handler=_handle_purged_walk_forward, tag="read_only",
        ),
        Tool(
            name="query_realistic_slippage",
            description=(
                "[BACKTEST] Realistic slippage estimate scaled by size + "
                "volatility + latency + session. Returns expected + "
                "90th-percentile upper bound. Brain reads BEFORE "
                "submit_trade_plan to budget honest cost (anti-optimistic "
                "vs flat-spread default)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "venue": {"type": "string"},
                    "size_pct_of_equity": {"type": "number", "minimum": 0.1, "maximum": 30.0},
                    "volatility_pct": {"type": "number", "minimum": 0.1, "maximum": 50.0},
                    "latency_ms": {"type": "number", "minimum": 10.0, "maximum": 5000.0},
                    "session": {"type": "string"},
                },
            },
            handler=_handle_realistic_slippage, tag="read_only",
        ),
        Tool(
            name="query_strategy_universe",
            description=(
                "[STRATEGY] Top-K active strategies from the 242-strategy "
                "universe ranked by absolute signal strength. Returns "
                "consensus + confidence + active_strategies + top_k list "
                "(name, signal). Brain sees WHICH strategies are firing, "
                "not just the aggregate vote."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "asset": {"type": "string", "enum": ["BTC", "ETH"]},
                    "regime": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["asset"],
            },
            handler=_handle_strategy_universe, tag="read_only",
        ),
        Tool(
            name="query_liquidity_sweep",
            description=(
                "[STRATEGY] ICT liquidity-sweep / stop-hunt reversal "
                "detector. Returns detected=True when a recent wick "
                "swept a swing high/low and reversed back inside the "
                "range, with confluence count (1-5: sweep, reversal "
                "strength, FVG presence, volume spike, body strength). "
                "Highest-EV reversal pattern in 2026 SMC literature."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "asset": {"type": "string", "enum": ["BTC", "ETH"]},
                    "timeframe": {"type": "string", "enum": ["5m", "15m", "1h", "4h"]},
                    "lookback": {"type": "integer", "minimum": 10, "maximum": 50},
                },
                "required": ["asset"],
            },
            handler=_handle_liquidity_sweep, tag="read_only",
        ),
        Tool(
            name="query_pair_trading_signal",
            description=(
                "[STRATEGY] BTC-ETH statistical arbitrage. Reads the "
                "cointegration z-score and returns an actionable "
                "suggestion (LONG_BTC_SHORT_ETH / SHORT_BTC_LONG_ETH / "
                "EXIT_PAIR / HOLD). Note: Robinhood spot is longs-only "
                "so only the LONG leg of a pair is firable as a half-leg."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "z_entry": {"type": "number", "minimum": 1.0, "maximum": 4.0},
                    "z_exit": {"type": "number", "minimum": 0.0, "maximum": 1.5},
                },
            },
            handler=_handle_pair_trading_signal, tag="read_only",
        ),
        Tool(
            name="query_session_bias",
            description=(
                "[STRATEGY] Current UTC session + volume share + "
                "conviction multiplier (0.5-1.0). US session ~55% of "
                "BTC volume, EU ~30%, Asia ~10%. Use multiplier to scale "
                "confidence DOWN during low-volume sessions, never up."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=_handle_session_bias, tag="read_only",
        ),
        Tool(
            name="query_grid_chop",
            description=(
                "[STRATEGY] Grid trading suggestions for ranging / CHOP "
                "regimes. Returns up to 5 rungs with buy_price + target + "
                "expected_pnl_pct that clears spread × 1.5. Empty list "
                "when regime not ranging. Each rung is a SUGGESTION — "
                "concentration cap (3/asset) limits how many fire."
            ),
            input_schema={
                "type": "object",
                "properties": {"asset": {"type": "string", "enum": ["BTC", "ETH"]}},
                "required": ["asset"],
            },
            handler=_handle_grid_chop, tag="read_only",
        ),
        Tool(
            name="query_wyckoff_phase",
            description=(
                "[STRATEGY] Wyckoff phase detector — accumulation / "
                "markup / distribution / markdown / unclear. Pure "
                "structural heuristic (no ML, no parameter learning). "
                "Confidence flagged low_sample when bars < 50."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "asset": {"type": "string", "enum": ["BTC", "ETH"]},
                    "timeframe": {"type": "string", "enum": ["1h", "4h", "1d"]},
                    "lookback": {"type": "integer", "minimum": 30, "maximum": 200},
                },
                "required": ["asset"],
            },
            handler=_handle_wyckoff_phase, tag="read_only",
        ),
        Tool(
            name="query_system_health",
            description=(
                "[HEALTH] Aggregate error counts by severity + components "
                "with recent errors. Brain knows which feeds to trust. "
                "Anti-noise: only counts and component names, no stack "
                "traces. Use when a tool returns surprising data — was "
                "its component erroring?"
            ),
            input_schema={"type": "object", "properties": {}},
            handler=_handle_system_health, tag="read_only",
        ),
        Tool(
            name="query_decision_audit_summary",
            description=(
                "[AUDIT] Aggregate win-rate by conviction tier / direction "
                "/ pattern_score bucket from recent decisions. Pure stats, "
                "no individual data points. Calibrate conviction from your "
                "own historical performance — buckets with <10 samples are "
                "low-confidence."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "asset": {"type": "string"},
                    "lookback": {"type": "integer", "minimum": 20, "maximum": 500},
                },
            },
            handler=_handle_decision_audit_summary, tag="read_only",
        ),
        Tool(
            name="query_chart_vision",
            description=(
                "[VISION] Visual chart pattern summary via vision model. "
                "Disabled by default; returns 'not_enabled' unless "
                "ACT_CHART_VISION=1. Use sparingly — slow and noisy if "
                "abused."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "asset": {"type": "string", "enum": ["BTC", "ETH"]},
                    "timeframe": {"type": "string", "enum": ["5m", "15m", "1h", "4h", "1d"]},
                },
                "required": ["asset"],
            },
            handler=_handle_chart_vision, tag="read_only",
        ),
        Tool(
            name="query_trade_verifier_state",
            description=(
                "[VERIFY] Recent post-trade SelfCritique entries from "
                "trade_verifier — predicted vs realized PnL, miss reasons, "
                "lessons. Brain reads to avoid repeating losing setups."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "asset": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                },
            },
            handler=_handle_trade_verifier_state, tag="read_only",
        ),
        Tool(
            name="query_feature_drift",
            description=(
                "[DRIFT] Feature distribution drift (PSI per feature) so "
                "brain knows when ML inputs have shifted significantly. "
                "psi > 0.25 = drift; rebaseline needed."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=_handle_feature_drift, tag="read_only",
        ),
        Tool(
            name="query_circuit_breaker_state",
            description=(
                "[BREAKER] Circuit-breaker overall status. Returns 'n/a' "
                "currently — no central registry. Brain watches "
                "tick_state.last_refusal and emergency_level instead."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=_handle_circuit_breaker_state, tag="read_only",
        ),
        Tool(
            name="query_on_chain_signals",
            description=(
                "[ON-CHAIN] Independent on-chain metrics: blockchain_stats, "
                "mempool, stablecoin flows, hashrate. Use when news doesn't "
                "explain a move — on-chain often does."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=_handle_on_chain_signals, tag="read_only",
        ),
        Tool(
            name="query_institutional_flows",
            description=(
                "[INSTITUTIONAL] Macro correlations, stablecoin flows, "
                "options sentiment, long/short ratio, cross-exchange "
                "spreads. Use to detect smart-money positioning."
            ),
            input_schema={
                "type": "object",
                "properties": {"asset": {"type": "string", "enum": ["BTC", "ETH"]}},
            },
            handler=_handle_institutional_flows, tag="read_only",
        ),
        Tool(
            name="query_adaptation_state",
            description=(
                "[ADAPTATION] Thompson-bandit posterior means per strategy "
                "+ credit-assigner weights per component. Brain sees its "
                "own learning state — which strategies are favored, which "
                "components are credited for recent wins."
            ),
            input_schema={
                "type": "object",
                "properties": {"asset": {"type": "string"}},
            },
            handler=_handle_adaptation_state, tag="read_only",
        ),
        Tool(
            name="query_genetic_evolution_state",
            description=(
                "[GENETIC] Full evolution state: total_generations_run, "
                "best_evolved_fitness, best_evolved_strategy (entry+exit "
                "rules), population_size, current_mutation_rate, "
                "stagnation_generations, pareto_front_size. Brain reads "
                "to know if the loop is discovering or stagnant."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=_handle_genetic_evolution_state, tag="read_only",
        ),
        Tool(
            name="query_genetic_hall_of_fame",
            description=(
                "[GENETIC] Top-N hall-of-fame strategies from "
                "strategy_repo.sqlite — name, regime, status (champion/"
                "challenger), sharpe, win_rate, total_pnl, entry_rule, "
                "exit_rule. Brain inspects winning evolved patterns to "
                "match against the current setup."
            ),
            input_schema={
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 20}},
            },
            handler=_handle_genetic_hall_of_fame, tag="read_only",
        ),
        Tool(
            name="query_recovery_plan",
            description=(
                "[RECOVERY] Stuck-portfolio analysis: ranks opens by "
                "current_pnl_pct_net, identifies profitable_closes (already "
                "net positive — close to realize), near_breakeven (small "
                "upward move flips positive), deep_losers (accept or hold "
                "for trend), and best_partial_candidates (gross >2% — "
                "partial-close to lock realized gains). Use when "
                "open_positions_same_asset >> cap and avg_unrealized_net "
                "is significantly negative (stuck-portfolio recovery)."
            ),
            input_schema={
                "type": "object",
                "properties": {"asset": {"type": "string"}},
            },
            handler=_handle_recovery_plan, tag="read_only",
        ),
        Tool(
            name="query_open_positions_detail",
            description=(
                "[POSITIONS] Per-position breakdown for paper trades: "
                "trade_id, asset, direction, entry/current/PnL, age_min, "
                "thesis, SL/TP. Brain calls this to decide HOLD/EXIT/"
                "PARTIAL on each open trade individually."
            ),
            input_schema={
                "type": "object",
                "properties": {"asset": {"type": "string"}},
            },
            handler=_handle_open_positions_detail, tag="read_only",
        ),
        Tool(
            name="close_paper_position",
            description=(
                "[WRITE] Close a paper position (full or partial). Pass "
                "asset (closes oldest open) or trade_id (specific). "
                "fraction=1.0 = full close (default); 0.5 = close half. "
                "Use when thesis broken, target reached, macro shift, "
                "or trailing-stop logic says exit. STRICTLY paper-only."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "asset": {"type": "string"},
                    "trade_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "fraction": {"type": "number", "minimum": 0.05, "maximum": 1.0},
                },
            },
            handler=_handle_close_paper_position, tag="write",
        ),
        Tool(
            name="modify_paper_position",
            description=(
                "[WRITE] Adjust SL or TP on an open paper position. Use "
                "to lock in profit (raise SL toward entry on a winner) "
                "or extend runway (raise TP). Pass asset or trade_id "
                "plus new_sl_price and/or new_tp_price."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "asset": {"type": "string"},
                    "trade_id": {"type": "string"},
                    "new_sl_price": {"type": "number"},
                    "new_tp_price": {"type": "number"},
                },
            },
            handler=_handle_modify_paper_position, tag="write",
        ),
        Tool(
            name="query_sizing_preview",
            description=(
                "[SIZING] Preview the final size_pct after body's "
                "modulation pipeline (AdaptiveFeedback × SelfEvolvingOverlay "
                "× AccuracyEngine × DynamicPositionLimits cap). Call BEFORE "
                "submit_trade_plan so your proposal accounts for the "
                "modulation rather than being surprised by a smaller fill."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "asset": {"type": "string", "enum": ["BTC", "ETH"]},
                    "regime": {"type": "string"},
                    "proposed_size_pct": {"type": "number", "minimum": 0.1, "maximum": 30.0},
                },
                "required": ["asset", "proposed_size_pct"],
            },
            handler=_handle_sizing_preview, tag="read_only",
        ),
        Tool(
            name="query_venue_capabilities",
            description=(
                "[VENUE] What operations the active venue supports — "
                "supports_long/short, max_leverage, partial close, "
                "modify SL/TP, limit orders, round-trip spread. Brain "
                "reads this BEFORE proposing actions so it doesn't ask "
                "for SHORT on Robinhood (longs-only) or 10x leverage "
                "on a spot venue."
            ),
            input_schema={
                "type": "object",
                "properties": {"venue": {"type": "string"}},
            },
            handler=_handle_venue_capabilities, tag="read_only",
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
            "query_open_positions_detail": ("realtime", "query", "internal"),
            "close_paper_position": ("realtime", "control", "internal"),
            "modify_paper_position": ("realtime", "control", "internal"),
            "query_venue_capabilities": ("daily", "query", "internal"),
            "query_sizing_preview": ("realtime", "query", "internal"),
            "query_recovery_plan": ("realtime", "query", "internal"),
            "query_genetic_evolution_state": ("hour", "query", "internal"),
            "query_genetic_hall_of_fame": ("hour", "query", "internal"),
            "query_trade_verifier_state": ("realtime", "query", "internal"),
            "query_system_health": ("realtime", "query", "internal"),
            "query_strategy_universe": ("realtime", "analysis", "internal"),
            "query_deflated_sharpe": ("daily", "analysis", "internal"),
            "query_probability_of_backtest_overfitting": ("daily", "analysis", "internal"),
            "query_purged_walk_forward": ("daily", "analysis", "internal"),
            "query_realistic_slippage": ("realtime", "analysis", "internal"),
            "query_liquidity_sweep": ("realtime", "analysis", "internal"),
            "query_pair_trading_signal": ("realtime", "analysis", "internal"),
            "query_session_bias": ("hour", "query", "internal"),
            "query_grid_chop": ("realtime", "analysis", "internal"),
            "query_wyckoff_phase": ("hour", "analysis", "internal"),
            "query_decision_audit_summary": ("hour", "query", "internal"),
            "query_chart_vision": ("minute", "analysis", "internal"),
            "query_feature_drift": ("hour", "query", "internal"),
            "query_circuit_breaker_state": ("realtime", "query", "internal"),
            "query_on_chain_signals": ("hour", "data_fetch", "internal"),
            "query_institutional_flows": ("hour", "data_fetch", "internal"),
            "query_adaptation_state": ("hour", "query", "internal"),
        }
        for name, (tm, it, rg) in _classifications.items():
            if name not in TOOL_CLASSIFICATIONS:
                TOOL_CLASSIFICATIONS[name] = ToolClassification(
                    timeliness=tm, intent_type=it, regulatory=rg,
                )
    except Exception:
        pass
    return added
