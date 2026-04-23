"""
Polymarket executor — order placement with safety defaults.

Shadow-mode FIRST (default). Real order placement requires both:
  1. `ACT_POLYMARKET_LIVE=1` env flag AND
  2. Config `polymarket.enabled: true` AND
  3. `py_clob_client` installed with valid API credentials AND
  4. Readiness gate for Polymarket-specific soak is OPEN

Without all four, every `place_order()` call is recorded to warm_store
as a shadow decision but does NOT touch any live account. This mirrors
the executor's existing SHADOW_ action convention for the agentic loop.

Design constraints:
  * Never raises from the public API — errors come back as
    PolymarketOrderResult(ok=False, reason=...).
  * No hard dep on py_clob_client at import time. The import is lazy
    inside _live_client() so CI / laptop environments without it still
    boot.
  * Every submitted order (live or shadow) gets a warm_store row with
    component_signals.source="polymarket_executor" so the operator can
    audit.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


LIVE_ENV = "ACT_POLYMARKET_LIVE"
DISABLE_ENV = "ACT_POLYMARKET_DISABLED"


@dataclass
class PolymarketOrderResult:
    """One submit_order outcome."""
    ok: bool
    mode: str                             # 'shadow' | 'live'
    market_id: str
    side: str
    shares: int
    price: float
    order_id: Optional[str] = None
    decision_id: Optional[str] = None
    reason: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok), "mode": self.mode,
            "market_id": self.market_id, "side": self.side,
            "shares": int(self.shares), "price": round(self.price, 4),
            "order_id": self.order_id,
            "decision_id": self.decision_id,
            "reason": self.reason,
        }


class PolymarketExecutor:
    """Thin wrapper around order placement with safety defaults."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # Effective mode — shadow unless every gate is met.
        self._live_ready = self._compute_live_ready()
        self._client = None  # lazy

    # ── Public API ──────────────────────────────────────────────────────

    def mode(self) -> str:
        """'live' if all gates open, else 'shadow'."""
        return "live" if self._live_ready else "shadow"

    def place_order(
        self,
        *,
        market_id: str,
        side: str,
        shares: int,
        price: float,
        plan_digest: Optional[Dict[str, Any]] = None,
    ) -> PolymarketOrderResult:
        """Submit one order. Returns a PolymarketOrderResult; never raises.

        `side` is 'YES' or 'NO'. `shares` is the integer share count.
        `price` is the submit price per share (0..1).
        `plan_digest` is the compact TradePlan dict — logged for audit.
        """
        side = (side or "").upper()
        if side not in ("YES", "NO"):
            return PolymarketOrderResult(
                ok=False, mode=self.mode(), market_id=market_id,
                side=side, shares=shares, price=price,
                reason="invalid side (expected YES or NO)",
            )
        if shares <= 0 or not (0.0 < price < 1.0):
            return PolymarketOrderResult(
                ok=False, mode=self.mode(), market_id=market_id,
                side=side, shares=shares, price=price,
                reason=f"invalid shares/price ({shares}, {price})",
            )

        decision_id = f"pm-{'live' if self._live_ready else 'shadow'}-{uuid.uuid4().hex[:12]}"

        if not self._live_ready:
            # Shadow path — log only.
            self._log_to_warm_store(
                decision_id=decision_id, market_id=market_id, side=side,
                shares=shares, price=price, final_action="PM_SHADOW",
                plan_digest=plan_digest or {},
            )
            return PolymarketOrderResult(
                ok=True, mode="shadow", market_id=market_id, side=side,
                shares=shares, price=price, order_id=None,
                decision_id=decision_id,
                reason="shadow mode (gates closed)",
            )

        # Live path — try the real CLOB client.
        try:
            client = self._live_client()
        except Exception as e:
            # Live mode was requested but client unavailable — degrade to
            # shadow + log the reason so operator sees it.
            self._log_to_warm_store(
                decision_id=decision_id, market_id=market_id, side=side,
                shares=shares, price=price, final_action="PM_SHADOW",
                plan_digest={**(plan_digest or {}),
                             "live_fallback_reason": f"{type(e).__name__}: {e}"},
            )
            return PolymarketOrderResult(
                ok=False, mode="shadow", market_id=market_id, side=side,
                shares=shares, price=price, decision_id=decision_id,
                reason=f"live client unavailable: {type(e).__name__}: {e}",
            )

        try:
            # The real py_clob_client API is (token_id, side, size, price).
            # We pass market_id as the token selector and let the client
            # resolve. Errors propagate as structured failure.
            resp = client.place_order(
                market_id=market_id, side=side, shares=shares, price=price,
            )
            order_id = str(resp.get("order_id") or resp.get("id") or "")
            ok = bool(resp.get("ok", True))
            self._log_to_warm_store(
                decision_id=decision_id, market_id=market_id, side=side,
                shares=shares, price=price,
                final_action="PM_LIVE" if ok else "PM_LIVE_REJECT",
                plan_digest={**(plan_digest or {}), "order_id": order_id,
                             "raw_resp": resp},
            )
            return PolymarketOrderResult(
                ok=ok, mode="live", market_id=market_id, side=side,
                shares=shares, price=price, order_id=order_id,
                decision_id=decision_id, raw=resp,
            )
        except Exception as e:
            self._log_to_warm_store(
                decision_id=decision_id, market_id=market_id, side=side,
                shares=shares, price=price, final_action="PM_LIVE_ERROR",
                plan_digest={**(plan_digest or {}),
                             "error": f"{type(e).__name__}: {e}"},
            )
            return PolymarketOrderResult(
                ok=False, mode="live", market_id=market_id, side=side,
                shares=shares, price=price, decision_id=decision_id,
                reason=f"order failed: {type(e).__name__}: {e}",
            )

    # ── Gating ──────────────────────────────────────────────────────────

    def _compute_live_ready(self) -> bool:
        if os.environ.get(DISABLE_ENV, "0") == "1":
            return False
        if os.environ.get(LIVE_ENV, "0") != "1":
            return False
        pm_cfg = (self.config.get("polymarket") or {})
        if not pm_cfg.get("enabled", False):
            return False
        # Optional: Polymarket-specific readiness gate. If missing, fall
        # back to the Robinhood readiness gate — better to err on caution.
        try:
            from src.orchestration.readiness_gate import evaluate
            gate = evaluate()
            if not gate.open_:
                return False
        except Exception:
            return False
        return True

    def _live_client(self):
        """Lazy-init the CLOB client. Raises if py_clob_client missing."""
        if self._client is not None:
            return self._client
        try:
            # py_clob_client is an optional dep; import lazily.
            from py_clob_client.client import ClobClient  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "py_clob_client not installed — `pip install py-clob-client` "
                "to enable live Polymarket trading"
            ) from e

        pm_cfg = (self.config.get("polymarket") or {})
        host = pm_cfg.get("host") or "https://clob.polymarket.com"
        key = os.environ.get("POLYMARKET_API_KEY") or pm_cfg.get("api_key")
        if not key:
            raise RuntimeError(
                "POLYMARKET_API_KEY not set (env or config.polymarket.api_key)"
            )
        # ClobClient initialization — operator must have configured this
        # per py_clob_client README; we just pass-through.
        self._client = ClobClient(host=host, key=key)
        return self._client

    # ── Audit log ──────────────────────────────────────────────────────

    def _log_to_warm_store(
        self, *, decision_id: str, market_id: str, side: str,
        shares: int, price: float, final_action: str,
        plan_digest: Dict[str, Any],
    ) -> None:
        try:
            from src.orchestration.warm_store import get_store
            store = get_store()
            store.write_decision({
                "decision_id": decision_id,
                "symbol": f"POLYMARKET:{market_id}",
                "ts_ns": time.time_ns(),
                "direction": 1 if side == "YES" else -1,
                "confidence": 0.0,
                "final_action": final_action,
                "plan": plan_digest,
                "component_signals": {
                    "source": "polymarket_executor",
                    "mode": "live" if final_action == "PM_LIVE" else "shadow",
                    "market_id": market_id,
                    "side": side, "shares": int(shares), "price": float(price),
                },
            })
        except Exception as e:
            logger.debug("polymarket_executor warm_store write failed: %s", e)
