"""
Broker snapshot + journal aggregates for the local dashboard.
Balance/equity always comes from the exchange client (same path as TradingExecutor).
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.monitoring.journal import TradeJournal


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def session_baseline_path() -> str:
    return os.path.join(project_root(), "logs", "broker_session.json")


def read_session_baseline() -> Dict[str, Any]:
    path = session_baseline_path()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_yaml_config() -> Dict[str, Any]:
    cfg_path = os.path.join(project_root(), "config.yaml")
    if not os.path.isfile(cfg_path):
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def load_journal_trades_jsonl() -> List[dict]:
    jlog = logging.getLogger("src.monitoring.journal")
    prev = jlog.level
    try:
        jlog.setLevel(logging.WARNING)
        j = TradeJournal()
        return j.load_trades()
    finally:
        jlog.setLevel(prev)


def aggregate_journal(trades: List[dict]) -> Dict[str, Any]:
    """Total trades, realized PnL (journal), per-day PnL and cumulative series."""
    total = len(trades)
    net_pnl = sum(float(t.get("pnl_usd", 0) or 0) for t in trades)
    wins = sum(1 for t in trades if float(t.get("pnl_usd", 0) or 0) > 0)
    losses = sum(1 for t in trades if float(t.get("pnl_usd", 0) or 0) < 0)
    by_day: Dict[str, float] = defaultdict(float)
    for t in trades:
        ts = str(t.get("timestamp", "") or "")
        day = ts[:10] if len(ts) >= 10 else "unknown"
        by_day[day] += float(t.get("pnl_usd", 0) or 0)
    days_sorted = sorted(by_day.keys())
    daily = [{"date": d, "pnl": round(by_day[d], 2)} for d in days_sorted if d != "unknown"]
    cum = 0.0
    cumulative: List[Dict[str, Any]] = []
    for row in daily:
        cum += row["pnl"]
        cumulative.append({"date": row["date"], "cumulative_pnl": round(cum, 2)})
    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / total, 4) if total else 0.0,
        "journal_net_pnl_usd": round(net_pnl, 2),
        "daily_pnl": daily,
        "cumulative_pnl": cumulative,
    }


def _connect_broker() -> Tuple[Any, Any, str, Optional[str], Dict[str, Any]]:
    """
    Returns (price_fetcher, client, label, error, config).
    client is BybitClient, DeltaClient, or None.
    """
    from dotenv import load_dotenv
    from src.data.fetcher import PriceFetcher

    root = project_root()
    load_dotenv(os.path.join(root, ".env"), override=True)
    config = load_yaml_config()
    exchanges = config.get("exchanges") or []
    if exchanges:
        ex_name = exchanges[0].get("name", "bybit")
    else:
        ex_name = (config.get("exchange") or {}).get("name", "bybit")
    mode = config.get("mode", "paper")
    testnet = mode in ("testnet", "paper")

    _quiet = (
        logging.getLogger("src.data.fetcher"),
        logging.getLogger("src.monitoring.journal"),
    )
    _prev_levels = [(lg, lg.level) for lg in _quiet]
    _buf = io.StringIO()
    try:
        for lg in _quiet:
            lg.setLevel(logging.WARNING)
        with contextlib.redirect_stdout(_buf):
            pf = PriceFetcher(exchange_name=ex_name, testnet=testnet)
    finally:
        for lg, lvl in _prev_levels:
            lg.setLevel(lvl)

    if getattr(pf, "bybit", None) and pf.bybit.available:
        return pf, pf.bybit, "bybit", None, config
    if getattr(pf, "delta", None) and pf.delta.available:
        return pf, pf.delta, "delta", None, config
    if getattr(pf, "alpaca", None) and pf.alpaca.available:
        return pf, None, "alpaca", None, config

    return pf, None, "none", "Broker not connected (check API keys and config)", config


def fetch_broker_account() -> Tuple[Dict[str, Any], Optional[str]]:
    _pf, client, label, err, _cfg = _connect_broker()
    if err:
        return {}, err
    if label == "alpaca":
        acct = _pf.alpaca.get_account()
        if acct.get("error"):
            return {}, str(acct.get("error"))
        eq = float(acct.get("equity", 0) or 0)
        cash = float(acct.get("cash", 0) or 0)
        return {
            "exchange": "alpaca",
            "equity": eq,
            "cash": cash,
            "wallet_balance": eq,
            "unrealized_pnl": 0.0,
            "raw": acct,
        }, None
    if not client:
        return {}, "No exchange client"

    acct = client.get_account()
    if acct.get("error"):
        return {}, str(acct.get("error"))

    equity = float(acct.get("equity", 0) or 0)
    cash = float(acct.get("cash", 0) or 0)
    wallet = float(acct.get("wallet_balance", 0) or 0)
    upnl = float(acct.get("unrealized_pnl", 0) or 0)

    return {
        "exchange": label,
        "equity": equity,
        "cash": cash,
        "wallet_balance": wallet,
        "unrealized_pnl": upnl,
        "raw": acct,
    }, None


def fetch_open_positions() -> Tuple[List[Dict[str, Any]], Optional[str]]:
    _pf, client, label, err, _cfg = _connect_broker()
    if err or not client:
        return [], err
    try:
        raw = client.get_positions()
        out: List[Dict[str, Any]] = []
        for p in raw:
            try:
                q = float(p.get("qty", 0) or p.get("contracts", 0) or 0)
            except (TypeError, ValueError):
                q = 0.0
            if q <= 0:
                continue
            out.append({
                "symbol": p.get("symbol", ""),
                "side": p.get("side", ""),
                "qty": q,
                "avg_entry_price": p.get("avg_entry_price", "0"),
                "current_price": p.get("current_price", "0"),
                "unrealized_pl": p.get("unrealized_pl", "0"),
                "market_value": p.get("market_value", "0"),
            })
        return out, None
    except Exception as e:
        return [], str(e)


def recent_journal_trades(limit: int = 80) -> List[Dict[str, Any]]:
    trades = load_journal_trades_jsonl()
    slim: List[Dict[str, Any]] = []
    for t in trades[-limit:][::-1]:
        slim.append({
            "timestamp": t.get("timestamp", ""),
            "asset": t.get("asset", ""),
            "action": t.get("action", ""),
            "pnl_usd": t.get("pnl_usd", 0),
            "pnl_pct": t.get("pnl_pct", 0),
            "exit_reason": t.get("exit_reason", ""),
        })
    return slim


def config_snapshot() -> Dict[str, Any]:
    cfg = load_yaml_config()
    ex = (cfg.get("execution") or {})
    risk = (cfg.get("risk") or {})
    exchanges = cfg.get("exchanges") or []
    ex_name = exchanges[0].get("name") if exchanges else (cfg.get("exchange") or {}).get("name", "")
    try:
        from src.data.fetcher import bybit_hedge_mode_enabled
        hedge = bool(bybit_hedge_mode_enabled())
    except Exception:
        hedge = bool(ex.get("bybit_hedge_mode", False))
    return {
        "mode": cfg.get("mode", ""),
        "exchange": ex_name,
        "assets": cfg.get("assets", []),
        "poll_interval_sec": cfg.get("poll_interval", 10),
        "bybit_hedge_mode": hedge,
        "daily_loss_limit_pct": risk.get("daily_loss_limit_pct"),
        "max_drawdown_pct": risk.get("max_drawdown_pct"),
        "risk_per_trade_pct": risk.get("risk_per_trade_pct"),
    }


def _normalize_positions(raw: List[dict]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in raw:
        try:
            q = float(p.get("qty", 0) or p.get("contracts", 0) or 0)
        except (TypeError, ValueError):
            q = 0.0
        if q <= 0:
            continue
        out.append({
            "symbol": p.get("symbol", ""),
            "side": p.get("side", ""),
            "qty": q,
            "avg_entry_price": p.get("avg_entry_price", "0"),
            "current_price": p.get("current_price", "0"),
            "unrealized_pl": p.get("unrealized_pl", "0"),
            "market_value": p.get("market_value", "0"),
        })
    return out


def build_full_state() -> Dict[str, Any]:
    """One broker connection: account + open positions + journal + config."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(project_root(), ".env"), override=True)

    baseline = read_session_baseline()
    session_start = float(baseline.get("session_start_equity", 0) or 0)
    trades = load_journal_trades_jsonl()
    jsum = aggregate_journal(trades)
    cfg_snap = config_snapshot()

    pf, client, label, conn_err, _ = _connect_broker()
    broker: Dict[str, Any] = {}
    positions: List[Dict[str, Any]] = []
    positions_error = None
    err = conn_err

    if not conn_err and label == "alpaca" and getattr(pf, "alpaca", None):
        acct = pf.alpaca.get_account()
        if acct.get("error"):
            err = str(acct.get("error"))
        else:
            eq = float(acct.get("equity", 0) or 0)
            cash = float(acct.get("cash", 0) or 0)
            broker = {
                "exchange": "alpaca",
                "equity": eq,
                "cash": cash,
                "wallet_balance": eq,
                "unrealized_pnl": 0.0,
                "raw": acct,
            }
    elif not conn_err and client:
        acct = client.get_account()
        if acct.get("error"):
            err = str(acct.get("error"))
        else:
            broker = {
                "exchange": label,
                "equity": float(acct.get("equity", 0) or 0),
                "cash": float(acct.get("cash", 0) or 0),
                "wallet_balance": float(acct.get("wallet_balance", 0) or 0),
                "unrealized_pnl": float(acct.get("unrealized_pnl", 0) or 0),
                "raw": acct,
            }
        try:
            positions = _normalize_positions(client.get_positions())
        except Exception as e:
            positions_error = str(e)

    equity = float(broker.get("equity", 0) or 0) if broker else 0.0
    session_pnl = equity - session_start if session_start else 0.0
    session_pct = (session_pnl / session_start * 100.0) if session_start > 0 else None

    return {
        "broker_error": err,
        "broker": broker,
        "session_baseline": baseline,
        "session_start_equity": session_start,
        "session_pnl_usd": round(session_pnl, 2),
        "session_pnl_pct": round(session_pct, 3) if session_pct is not None else None,
        "total_trades": jsum["total_trades"],
        "wins": jsum["wins"],
        "losses": jsum["losses"],
        "win_rate": jsum["win_rate"],
        "journal_net_pnl_usd": jsum["journal_net_pnl_usd"],
        "daily_pnl": jsum["daily_pnl"],
        "cumulative_pnl": jsum["cumulative_pnl"],
        "bybit_hedge_mode": cfg_snap.get("bybit_hedge_mode", False),
        "config": cfg_snap,
        "positions": positions,
        "positions_error": positions_error,
        "recent_trades": recent_journal_trades(100),
    }


def build_summary() -> Dict[str, Any]:
    """Alias for full state (backward compatible)."""
    return build_full_state()
