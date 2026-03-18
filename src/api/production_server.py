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
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:8501").split(),
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
