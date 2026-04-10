from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException  # noqa: F401
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
import asyncio
import json
import logging
import os
import hmac
import hashlib
import yaml
from src.api.state import DashboardState

logger = logging.getLogger(__name__)

app = FastAPI(title="Strategist Hub API")

# ── Security Configuration ──
_API_KEY = os.environ.get("DASHBOARD_API_KEY", "")

# Warn if API auth is disabled in non-development mode
try:
    _cfg_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
    with open(_cfg_path) as _f:
        _mode = yaml.safe_load(_f).get('mode', 'live')
    if _mode == 'live' and not _API_KEY:
        logger.warning(
            "SECURITY: DASHBOARD_API_KEY is not set in %s mode. "
            "API endpoints are unauthenticated. Set DASHBOARD_API_KEY env var.", _mode
        )
except Exception:
    pass

# CORS — restrict to localhost dashboards only
_allowed_origins = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:8501,http://127.0.0.1:8501,http://localhost:5173,http://127.0.0.1:5173"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)

state_manager = DashboardState()

# ── API Key Authentication ──
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _verify_api_key(api_key: str = Depends(_api_key_header)):
    """Verify API key for REST endpoints. Skip if no key configured."""
    if not _API_KEY:
        return True  # Auth disabled (dev mode)
    if not api_key or not hmac.compare_digest(api_key, _API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True


@app.get("/api/state")
async def get_state(auth: bool = Depends(_verify_api_key)):
    """Return the current full system state."""
    return state_manager.get_full_state()


@app.get("/api/history")
async def get_history(auth: bool = Depends(_verify_api_key)):
    """Read trade history from CSV."""
    import pandas as pd
    hist_path = "logs/trade_history.csv"
    if os.path.exists(hist_path):
        df = pd.read_csv(hist_path)
        return df.to_dict('records')
    return []


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time update stream with token authentication via first message."""
    await websocket.accept()
    try:
        # Authenticate via first message instead of URL query param (avoids token in logs)
        if _API_KEY:
            auth_msg = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
            if not hmac.compare_digest(auth_msg.strip(), _API_KEY):
                await websocket.close(code=4003, reason="Unauthorized")
                return
        while True:
            await websocket.send_json(state_manager.get_full_state())
            await asyncio.sleep(1)
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass


@app.get("/health")
async def health():
    """Unauthenticated health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    # Bind to localhost only — use reverse proxy for external access
    _host = os.environ.get("API_HOST", "127.0.0.1")
    uvicorn.run(app, host=_host, port=8000)
