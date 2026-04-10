"""
Production API Server — Port 11000
All trading system data and controls exposed as REST endpoints.
Future: Port 11001 for WebSocket streaming.
"""
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# -- Add project root to path --
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api.state import DashboardState

try:
    from src.core.component import ComponentRegistry
except Exception:
    ComponentRegistry = None  # type: ignore

logger = logging.getLogger(__name__)

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

def _load_robinhood_paper_state() -> dict:
    """Load Robinhood paper trading state (separate from dashboard_state)."""
    path = os.path.join(PROJECT_ROOT, 'logs', 'robinhood_paper_state.json')
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def _load_robinhood_paper_trades() -> list:
    """Load Robinhood paper trade log (JSONL)."""
    path = os.path.join(PROJECT_ROOT, 'logs', 'robinhood_paper.jsonl')
    if not os.path.exists(path):
        return []
    try:
        trades = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    trades.append(json.loads(line))
        return trades
    except Exception:
        return []

# -- Auth --
API_KEY = os.environ.get("DASHBOARD_API_KEY", "")
_DEV_MODE = os.environ.get("TRADE_API_DEV_MODE", "").lower() in ("1", "true", "yes")

def _require_api_key(x_api_key: Optional[str] = Header(default=None)):
    if not API_KEY:
        if _DEV_MODE:
            return True  # Explicit dev-mode opt-in only
        raise HTTPException(status_code=403,
                            detail="DASHBOARD_API_KEY not configured. Set env var or enable TRADE_API_DEV_MODE=1")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# -- App --
app = FastAPI(
    title="TradeSystem Production API",
    description="Full trading system REST API — port 11000",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:8501 http://localhost:5173 http://127.0.0.1:5173").split(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# GROUP: SYSTEM
# ─────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """Unauthenticated liveness probe."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/v1/system/status", tags=["System"])
async def system_status(_=Depends(_require_api_key)):
    """Full system status including component health."""
    state = DashboardState().get_full_state()
    components_data = {}
    if ComponentRegistry is not None:
        try:
            components_data = ComponentRegistry.get().health_report()
        except Exception:
            pass
    return {
        "trading_status": state.get("status", "UNKNOWN"),
        "last_update": state.get("last_update", ""),
        "model_version": state.get("model_version", ""),
        "components": components_data,
        "layers": state.get("layers", {}),
        "sources": state.get("sources", {}),
    }

@app.get("/api/v1/system/components", tags=["System"])
async def component_health(_=Depends(_require_api_key)):
    """Per-component initialization and health status."""
    if ComponentRegistry is None:
        return {"error": "ComponentRegistry not available"}
    try:
        return ComponentRegistry.get().health_report()
    except Exception as e:
        return {"error": str(e)}

# ─────────────────────────────────────────
# GROUP: PORTFOLIO & P&L
# ─────────────────────────────────────────

@app.get("/api/v1/portfolio", tags=["Portfolio"])
async def portfolio(_=Depends(_require_api_key)):
    """Current portfolio state: equity curve, PnL, return."""
    state = DashboardState().get_full_state()
    return state.get("portfolio", {})

@app.get("/api/v1/portfolio/positions", tags=["Portfolio"])
async def open_positions(_=Depends(_require_api_key)):
    """All currently open positions with unrealized P&L."""
    state = DashboardState().get_full_state()
    return state.get("open_positions", {})

@app.get("/api/v1/portfolio/performance", tags=["Portfolio"])
async def performance(_=Depends(_require_api_key)):
    """Performance metrics: win rate, Sharpe, drawdown."""
    state = DashboardState().get_full_state()
    return state.get("performance", {})

# ─────────────────────────────────────────
# GROUP: TRADES
# ─────────────────────────────────────────

@app.get("/api/v1/trades", tags=["Trades"])
async def trade_history(limit: int = 100, status: Optional[str] = None,
                        _=Depends(_require_api_key)):
    """Full trade history. Optional ?status=OPEN|CLOSED filter."""
    state = DashboardState().get_full_state()
    trades = state.get("trade_history", [])
    if status:
        trades = [t for t in trades if t.get("status", "").upper() == status.upper()]
    return {"count": len(trades), "trades": trades[-limit:]}

@app.get("/api/v1/trades/stats", tags=["Trades"])
async def trade_stats(_=Depends(_require_api_key)):
    """Win rate, profit factor, avg win/loss."""
    state = DashboardState().get_full_state()
    trades = state.get("trade_history", [])
    closed = [t for t in trades if t.get("status") == "CLOSED"]
    wins = [t for t in closed if t.get("pnl", 0) > 0]
    losses = [t for t in closed if t.get("pnl", 0) < 0]
    win_pnl = sum(t.get("pnl", 0) for t in wins)
    loss_pnl = abs(sum(t.get("pnl", 0) for t in losses))
    return {
        "total": len(trades),
        "closed": len(closed),
        "open": len(trades) - len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed), 4) if closed else 0,
        "total_pnl": round(sum(t.get("pnl", 0) for t in closed), 4),
        "profit_factor": round(win_pnl / loss_pnl, 4) if loss_pnl > 0 else None,
        "avg_win": round(win_pnl / len(wins), 4) if wins else 0,
        "avg_loss": round(loss_pnl / len(losses), 4) if losses else 0,
    }

# ─────────────────────────────────────────
# GROUP: LIVE PRICES (for frontend TopBar)
# ─────────────────────────────────────────

@app.get("/api/v1/prices", tags=["Market Data"])
async def live_prices():
    """Live BTC/ETH prices — tries Robinhood paper positions first, then dashboard state."""
    prices = {}

    # Try Robinhood paper state (has real bid/ask from Robinhood API)
    rh = _load_robinhood_paper_state()
    for tid, pos_data in rh.get("positions", {}).items():
        if isinstance(pos_data, dict):
            asset = pos_data.get("asset", tid.split("_")[0] if "_" in tid else tid)
            if asset not in prices:
                prices[asset] = {
                    "price": pos_data.get("current_price", pos_data.get("entry_price", 0)),
                    "change_pct": pos_data.get("current_pnl_pct", 0),
                    "bid": pos_data.get("entry_bid", 0),
                    "ask": pos_data.get("entry_ask", 0),
                    "spread_pct": pos_data.get("entry_spread_pct", 0),
                }

    # Fall back to dashboard state active_assets
    state = DashboardState().get_full_state()
    for asset_name, asset_data in state.get("active_assets", {}).items():
        if asset_name not in prices:
            prices[asset_name] = {
                "price": asset_data.get("price", asset_data.get("last_price", 0)),
                "change_pct": asset_data.get("change_pct", 0),
                "bid": asset_data.get("bid", 0),
                "ask": asset_data.get("ask", 0),
                "spread_pct": asset_data.get("spread_pct", 0),
            }

    # If prices still zero, try parsing latest live_output.log for price data
    if all(v.get("price", 0) == 0 for v in prices.values()):
        try:
            import re
            log_path = os.path.join(PROJECT_ROOT, 'logs', 'live_output.log')
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()[-50:]  # Last 50 lines
                for line in reversed(lines):
                    # Parse: [ROBINHOOD:BTC] $73,064.80
                    m = re.search(r'\[ROBINHOOD:(\w+)\]\s*\$([\d,]+\.?\d*)', line)
                    if m:
                        asset = m.group(1)
                        price = float(m.group(2).replace(',', ''))
                        if asset not in prices or prices[asset].get("price", 0) == 0:
                            prices[asset] = {"price": price, "change_pct": 0, "bid": 0, "ask": 0, "spread_pct": 1.67}
        except Exception:
            pass

    # Ensure BTC and ETH always appear
    for asset in ["BTC", "ETH"]:
        if asset not in prices:
            prices[asset] = {"price": 0, "change_pct": 0, "bid": 0, "ask": 0, "spread_pct": 0}

    return prices

# ─────────────────────────────────────────
# GROUP: DASHBOARD AGGREGATE (single call for frontend)
# ─────────────────────────────────────────

@app.get("/api/v1/dashboard", tags=["Dashboard"])
async def dashboard_aggregate(_=Depends(_require_api_key)):
    """Single aggregated endpoint for the React dashboard.
    Merges Robinhood paper state as PRIMARY source, falls back to dashboard_state."""
    state = DashboardState().get_full_state()
    rh = _load_robinhood_paper_state()
    rh_trades = _load_robinhood_paper_trades()
    overlay = state.get("agent_overlay", {})
    risk = state.get("risk_metrics", {})

    # ── ROBINHOOD PAPER as PRIMARY data source ──
    rh_equity = rh.get("equity", 0)
    rh_initial = rh.get("initial_capital", 0)
    rh_peak = rh.get("peak_equity", rh_equity)
    rh_stats = rh.get("stats", {})
    rh_pnl = rh_stats.get("total_pnl_usd", 0)
    rh_wins = rh_stats.get("wins", 0)
    rh_losses = rh_stats.get("losses", 0)
    rh_total = rh_wins + rh_losses

    # Robinhood positions (from paper state)
    positions_list = []
    for tid, pos_data in rh.get("positions", {}).items():
        if isinstance(pos_data, dict):
            positions_list.append({
                "asset": pos_data.get("asset", tid.split("_")[0] if "_" in tid else tid),
                "direction": pos_data.get("direction", "LONG"),
                "entry_price": pos_data.get("entry_price", 0),
                "current_price": pos_data.get("current_price", 0),
                "quantity": pos_data.get("quantity", 0),
                "unrealized_pnl": pos_data.get("current_pnl_usd", 0),
                "unrealized_pnl_pct": pos_data.get("current_pnl_pct", 0),
                "stop_loss": pos_data.get("sl_price", 0),
                "entry_time": pos_data.get("entry_time", ""),
                "confidence": pos_data.get("llm_confidence", 0),
                "score": pos_data.get("score", 0),
                "trade_timeframe": "4h",
            })

    # Robinhood trades (from JSONL log)
    trade_list = []
    for t in rh_trades:
        trade_list.append({
            "asset": t.get("asset", ""),
            "direction": t.get("direction", ""),
            "status": "CLOSED" if t.get("event") == "EXIT" else "OPEN",
            "event": t.get("event", ""),
            "entry_price": t.get("fill_price", t.get("entry_price", 0)),
            "exit_price": t.get("exit_price", 0),
            "pnl": t.get("pnl_usd", 0),
            "pnl_pct": t.get("pnl_pct", 0),
            "timestamp": t.get("timestamp", ""),
            "reason": t.get("reason", t.get("reasoning", "")[:100] if t.get("reasoning") else ""),
            "score": t.get("score", 0),
            "spread_pct": t.get("spread_pct", 0),
            "llm_confidence": t.get("llm_confidence", 0),
            "trade_timeframe": "4h",
        })

    # Build equity curve from Robinhood trade log (shows actual paper P&L over time)
    equity_curve = []
    running_equity = rh_initial if rh_initial > 0 else 16445.79
    for t in rh_trades:
        if t.get("event") == "EXIT":
            running_equity += t.get("pnl_usd", 0)
            equity_curve.append({"t": t.get("timestamp", ""), "v": round(running_equity, 2)})
        elif t.get("event") == "ENTRY":
            equity_curve.append({"t": t.get("timestamp", ""), "v": round(running_equity, 2)})
    # If no trades yet, show flat line at initial capital
    if not equity_curve:
        from datetime import datetime, timezone
        equity_curve = [{"t": datetime.now(tz=timezone.utc).isoformat(), "v": running_equity}]

    # Agent votes to array
    agent_votes = overlay.get("agent_votes", {})
    agent_weights = overlay.get("agent_weights", {})
    agents_list = []
    for agent_name, vote_data in agent_votes.items():
        if isinstance(vote_data, dict):
            agents_list.append({
                "id": agent_name,
                "name": agent_name.replace("_", " ").title(),
                "direction": vote_data.get("direction", 0),
                "confidence": vote_data.get("confidence", 0),
                "reasoning": vote_data.get("reasoning", ""),
                "weight": agent_weights.get(agent_name, 1.0),
            })

    return {
        "portfolio": {
            "equity": rh_equity if rh_equity > 0 else portfolio_data.get("current_total_value", 0),
            "initial_capital": rh_initial,
            "today_pnl": rh_pnl,
            "total_pnl": rh_pnl,
            "total_return_pct": round((rh_pnl / rh_initial * 100), 2) if rh_initial > 0 else 0,
            "equity_curve": equity_curve,
            "peak_equity": rh_peak,
            "sod_balance": rh_initial,
        },
        "exchange": "robinhood",
        "exchange_status": "PAPER TRADING",
        "positions": positions_list,
        "trades": trade_list[-50:],
        "trade_stats": {
            "total": rh_total,
            "wins": rh_wins,
            "losses": rh_losses,
            "win_rate": round(rh_wins / rh_total, 4) if rh_total > 0 else 0,
            "profit_factor": round(rh_stats.get("largest_win", 0) / abs(rh_stats.get("largest_loss", -1)), 2) if rh_stats.get("largest_loss", 0) != 0 else 0,
            "avg_win": round(rh_stats.get("largest_win", 0), 2),
            "avg_loss": round(abs(rh_stats.get("largest_loss", 0)), 2),
            "largest_win": rh_stats.get("largest_win", 0),
            "largest_loss": rh_stats.get("largest_loss", 0),
            "signals_seen": rh_stats.get("total_signals", 0),
        },
        "agents": {
            "list": agents_list,
            "consensus": overlay.get("consensus_level", "N/A"),
            "data_quality": overlay.get("data_quality", 0),
            "daily_pnl_mode": overlay.get("daily_pnl_mode", "NORMAL"),
            "enabled": overlay.get("enabled", False),
            "last_decision": overlay.get("last_decision", {}),
            "cycle_count": overlay.get("cycle_count", 0),
        },
        "risk": {
            "current_drawdown": risk.get("current_drawdown", 0),
            "max_drawdown": risk.get("max_drawdown", 0),
            "risk_score": risk.get("risk_score", 0),
            "vpin": risk.get("vpin_threshold", 0),
        },
        "models": state.get("benchmark", {}).get("per_model", {}),
        "status": state.get("status", "UNKNOWN"),
        "sources": state.get("sources", {}),
        "layers": state.get("layers", {}),
        "layer_logs": state.get("layer_logs", {}),
        "sentiment": state.get("sentiment", {}),
        "last_update": state.get("last_update", ""),
    }

# ─────────────────────────────────────────
# GROUP: MARKET DATA & SIGNALS
# ─────────────────────────────────────────

@app.get("/api/v1/signals", tags=["Signals"])
async def signals(_=Depends(_require_api_key)):
    """Latest signals per asset from all layers."""
    state = DashboardState().get_full_state()
    return state.get("active_assets", {})

@app.get("/api/v1/signals/{asset}", tags=["Signals"])
async def signal_for_asset(asset: str, _=Depends(_require_api_key)):
    """Signal detail for a single asset (e.g. BTC, ETH, AAVE)."""
    state = DashboardState().get_full_state()
    assets = state.get("active_assets", {})
    data = assets.get(asset.upper())
    if not data:
        raise HTTPException(status_code=404, detail=f"Asset {asset} not found")
    return data

@app.get("/api/v1/sentiment", tags=["Signals"])
async def sentiment(_=Depends(_require_api_key)):
    """Latest sentiment scores per asset."""
    state = DashboardState().get_full_state()
    assets = state.get("active_assets", {})
    return {a: v.get("sentiment", {}) for a, v in assets.items()}

# ─────────────────────────────────────────
# GROUP: MODELS
# ─────────────────────────────────────────

@app.get("/api/v1/models/status", tags=["Models"])
async def model_status(_=Depends(_require_api_key)):
    """Health and accuracy of all ML models."""
    state = DashboardState().get_full_state()
    benchmark = state.get("benchmark", {})
    components = {}
    if ComponentRegistry is not None:
        try:
            components = ComponentRegistry.get().health_report()["components"]
        except Exception:
            pass

    ml_keys = ["lightgbm", "patchtst", "rl_agent", "finbert", "strategist",
               "advanced_learning", "sentiment", "vpin", "regime_detector"]
    return {
        "models": {
            k: {
                "healthy": components.get(k, {}).get("healthy", "unknown"),
                "error": components.get(k, {}).get("error"),
                "benchmark": benchmark.get("per_model", {}).get(k, {}),
            }
            for k in ml_keys
        }
    }

@app.get("/api/v1/models/benchmark", tags=["Models"])
async def model_benchmark(_=Depends(_require_api_key)):
    """Model accuracy benchmarks vs public leaderboard."""
    state = DashboardState().get_full_state()
    return state.get("benchmark", {})

# ─────────────────────────────────────────
# GROUP: AGENTS
# ─────────────────────────────────────────

@app.get("/api/v1/agents/overlay", tags=["Agents"])
async def agent_overlay(_=Depends(_require_api_key)):
    """12-agent overlay: last decision, votes, weights, consensus."""
    state = DashboardState().get_full_state()
    return state.get("agent_overlay", {})

@app.get("/api/v1/agents/votes", tags=["Agents"])
async def agent_votes(_=Depends(_require_api_key)):
    """Individual agent votes for last decision cycle."""
    state = DashboardState().get_full_state()
    overlay = state.get("agent_overlay", {})
    return {
        "votes": overlay.get("agent_votes", {}),
        "weights": overlay.get("agent_weights", {}),
        "consensus": overlay.get("consensus_level"),
        "last_decision": overlay.get("last_decision", {}),
    }

# ─────────────────────────────────────────
# GROUP: RISK
# ─────────────────────────────────────────

@app.get("/api/v1/risk/metrics", tags=["Risk"])
async def risk_metrics(_=Depends(_require_api_key)):
    """Current risk metrics: drawdown, VaR, position limits."""
    state = DashboardState().get_full_state()
    return state.get("risk_metrics", {})

@app.get("/api/v1/risk/layers", tags=["Risk"])
async def layer_status(_=Depends(_require_api_key)):
    """9-layer pipeline status and logs."""
    state = DashboardState().get_full_state()
    return {
        "layers": state.get("layers", {}),
        "layer_logs": state.get("layer_logs", [])[-50:],
    }

# ─────────────────────────────────────────
# GROUP: LOGS (read-only, last N lines)
# ─────────────────────────────────────────

@app.get("/api/v1/logs/alerts", tags=["Logs"])
async def alerts_log(limit: int = 50, _=Depends(_require_api_key)):
    """Recent alerts from the alert system."""
    try:
        from src.core.paths import LOG_DIR
        alerts_path = LOG_DIR / "alerts.jsonl"
    except Exception:
        alerts_path = Path("logs/alerts.jsonl")
    if not alerts_path.exists():
        return {"alerts": []}
    lines = alerts_path.read_text(encoding="utf-8").splitlines()[-limit:]
    alerts = []
    for line in lines:
        try:
            alerts.append(json.loads(line))
        except Exception:
            pass
    return {"count": len(alerts), "alerts": alerts}

@app.get("/api/v1/logs/audit", tags=["Logs"])
async def audit_log(limit: int = 50, _=Depends(_require_api_key)):
    """Recent execution audit trail."""
    try:
        from src.core.paths import LOG_DIR
        audit_path = LOG_DIR / "audit_failover.jsonl"
    except Exception:
        audit_path = Path("logs/audit_failover.jsonl")
    if not audit_path.exists():
        return {"audit": []}
    lines = audit_path.read_text(encoding="utf-8").splitlines()[-limit:]
    entries = []
    for line in lines:
        try:
            entries.append(json.loads(line))
        except Exception:
            pass
    return {"count": len(entries), "audit": entries}

# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

def run_production_server(host: str = "0.0.0.0", port: int = 11000,
                          reload: bool = False):
    """Start the production API server on port 11000."""
    logger.info(f"Starting Production API on http://{host}:{port}")
    logger.info(f"  Docs: http://{host}:{port}/docs")
    logger.info(f"  Health: http://{host}:{port}/health")
    uvicorn.run("src.api.production_server:app",
                host=host, port=port, reload=reload,
                log_level="info")


if __name__ == "__main__":
    run_production_server()
