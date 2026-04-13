"""
TradingView Webhook Receiver
==============================
Receives alerts from TradingView Pine Script strategies via webhook.
TradingView Pro -> Create Alert -> Webhook URL -> this endpoint -> trading signal.

Setup:
  1. In TradingView, create an alert on any indicator/strategy
  2. Set webhook URL to: http://your-ip:11008/webhook/tradingview?secret=YOUR_SECRET
  3. Alert message format (JSON):
     {"ticker": "BTC", "action": "buy", "price": 72000, "strategy": "LuxAlgo", "timeframe": "4H"}

Endpoints:
  POST /webhook/tradingview?secret=<shared_secret>  - Receive TradingView alert
  GET  /api/v1/tv-signals                           - Read recent signals for executor
  GET  /api/v1/tv-signals/latest?ticker=BTC         - Latest signal for a ticker
  DELETE /api/v1/tv-signals                          - Clear signal buffer
"""

import json
import os
import time
import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TV_WEBHOOK_SECRET = os.environ.get("TV_WEBHOOK_SECRET", "changeme_tradingview_secret")
SIGNAL_LOG_PATH = Path("logs/tradingview_signals.jsonl")
MAX_SIGNAL_BUFFER = 500  # In-memory ring buffer size

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TradingViewAlert(BaseModel):
    """Incoming TradingView webhook payload."""
    ticker: str
    action: str                          # buy, sell, close, strong_buy, strong_sell
    price: Optional[float] = None
    strategy: Optional[str] = None       # Name of the Pine Script strategy
    timeframe: Optional[str] = None      # e.g. "1H", "4H", "1D"
    volume: Optional[float] = None
    exchange: Optional[str] = None
    message: Optional[str] = None        # Free-text from TradingView alert
    contracts: Optional[float] = None    # Position size hint
    timestamp: Optional[str] = None      # TradingView {{time}} variable

    @field_validator("action")
    @classmethod
    def normalize_action(cls, v: str) -> str:
        v = v.strip().lower()
        valid = {"buy", "sell", "close", "strong_buy", "strong_sell", "long", "short", "flat"}
        if v not in valid:
            raise ValueError(f"action must be one of {valid}, got '{v}'")
        # Normalize aliases
        if v == "long":
            v = "buy"
        elif v == "short":
            v = "sell"
        elif v == "flat":
            v = "close"
        return v

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        return v.strip().upper().replace("USDT", "").replace("USD", "").replace("PERP", "")


class SignalRecord(BaseModel):
    """Stored signal with metadata."""
    ticker: str
    action: str
    price: Optional[float]
    strategy: Optional[str]
    timeframe: Optional[str]
    received_at: str       # ISO timestamp
    epoch_ms: int          # Unix ms for fast sorting
    signal_int: int        # Mapped to -1/0/+1 for the engine
    raw: Dict[str, Any]


# ---------------------------------------------------------------------------
# Signal Buffer (in-memory + disk persistence)
# ---------------------------------------------------------------------------
_signal_buffer: deque = deque(maxlen=MAX_SIGNAL_BUFFER)


def _action_to_signal(action: str) -> int:
    """Map TradingView action string to integer signal."""
    mapping = {
        "buy": 1,
        "strong_buy": 1,
        "sell": -1,
        "strong_sell": -1,
        "close": 0,
    }
    return mapping.get(action, 0)


def _persist_signal(record: SignalRecord) -> None:
    """Append signal to JSONL log file."""
    try:
        SIGNAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SIGNAL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")
    except Exception as e:
        logger.error(f"[TV-WEBHOOK] Failed to persist signal: {e}")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
router = APIRouter(tags=["TradingView Webhook"])


@router.post("/webhook/tradingview")
async def receive_tradingview_alert(
    request: Request,
    secret: Optional[str] = Query(None, description="Shared webhook secret"),
):
    """
    Receive a TradingView webhook alert.

    TradingView sends POST with JSON body when an alert fires.
    Authenticate via ?secret= query parameter.
    """
    # --- Authentication ---
    if secret != TV_WEBHOOK_SECRET:
        logger.warning("[TV-WEBHOOK] Rejected: invalid secret")
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    # --- Parse body ---
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    try:
        alert = TradingViewAlert(**body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Validation error: {e}")

    # --- Build record ---
    now = datetime.now(timezone.utc)
    record = SignalRecord(
        ticker=alert.ticker,
        action=alert.action,
        price=alert.price,
        strategy=alert.strategy,
        timeframe=alert.timeframe,
        received_at=now.isoformat(),
        epoch_ms=int(now.timestamp() * 1000),
        signal_int=_action_to_signal(alert.action),
        raw=body,
    )

    # --- Store ---
    _signal_buffer.append(record)
    _persist_signal(record)

    logger.info(
        f"[TV-WEBHOOK] {alert.ticker} {alert.action.upper()} "
        f"@ {alert.price} via {alert.strategy or 'unknown'} ({alert.timeframe or '?'})"
    )

    return {
        "status": "ok",
        "ticker": record.ticker,
        "signal": record.signal_int,
        "received_at": record.received_at,
    }


@router.get("/api/v1/tv-signals")
async def get_tv_signals(
    ticker: Optional[str] = Query(None, description="Filter by ticker symbol"),
    limit: int = Query(50, ge=1, le=500, description="Max records to return"),
    since_ms: Optional[int] = Query(None, description="Only signals after this epoch ms"),
):
    """
    Retrieve recent TradingView signals for the executor to consume.
    Supports filtering by ticker, limit, and time window.
    """
    signals = list(_signal_buffer)
    if ticker:
        ticker = ticker.strip().upper()
        signals = [s for s in signals if s.ticker == ticker]
    if since_ms:
        signals = [s for s in signals if s.epoch_ms > since_ms]
    # Most recent first
    signals = sorted(signals, key=lambda s: s.epoch_ms, reverse=True)[:limit]
    return {
        "count": len(signals),
        "signals": [s.model_dump() for s in signals],
    }


@router.get("/api/v1/tv-signals/latest")
async def get_latest_signal(
    ticker: str = Query(..., description="Ticker symbol (e.g. BTC, ETH)"),
):
    """
    Get the most recent signal for a specific ticker.
    Used by the executor's main loop to check for TradingView overrides.
    """
    ticker = ticker.strip().upper()
    for record in reversed(_signal_buffer):
        if record.ticker == ticker:
            age_sec = (time.time() * 1000 - record.epoch_ms) / 1000
            return {
                "found": True,
                "signal": record.model_dump(),
                "age_seconds": round(age_sec, 1),
                "stale": age_sec > 300,  # >5 min = stale
            }
    return {"found": False, "signal": None, "age_seconds": None, "stale": True}


@router.delete("/api/v1/tv-signals")
async def clear_signals(
    ticker: Optional[str] = Query(None, description="Clear only this ticker (or all)"),
):
    """Clear the in-memory signal buffer."""
    global _signal_buffer
    if ticker:
        ticker = ticker.strip().upper()
        _signal_buffer = deque(
            (s for s in _signal_buffer if s.ticker != ticker),
            maxlen=MAX_SIGNAL_BUFFER,
        )
        return {"status": "cleared", "ticker": ticker}
    else:
        _signal_buffer.clear()
        return {"status": "cleared", "ticker": "all"}


# ---------------------------------------------------------------------------
# Helper for executor integration (non-HTTP)
# ---------------------------------------------------------------------------
def get_latest_tv_signal(ticker: str, max_age_sec: float = 300.0) -> Optional[int]:
    """
    Non-HTTP helper: read latest TradingView signal from buffer.
    Returns signal_int (-1/0/+1) or None if no fresh signal exists.

    Usage in executor:
        from src.api.webhook_receiver import get_latest_tv_signal
        tv_sig = get_latest_tv_signal("BTC", max_age_sec=300)
        if tv_sig is not None:
            # Override or blend with engine signal
    """
    ticker = ticker.strip().upper()
    now_ms = time.time() * 1000
    for record in reversed(_signal_buffer):
        if record.ticker == ticker:
            age_ms = now_ms - record.epoch_ms
            if age_ms <= max_age_sec * 1000:
                return record.signal_int
            return None  # Found but stale
    return None
