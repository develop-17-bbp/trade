"""Alpaca paper-stocks executor.

Mirror of `src/trading/robinhood_executor.py` shape but for Alpaca's
equity API. The faithful-copy principle from the dual-asset plan
applies: same conviction/authority/readiness gates upstream, just a
different concrete venue at the bottom.

Hard rules baked in (operator basket: SPY/QQQ/TQQQ/SOXL):
  * RTH-only via `src/utils/market_hours.is_us_market_open` — orders
    outside session refused with reason='market_closed'.
  * Leveraged ETFs (TQQQ/SOXL) refused inside the 30-min pre-close
    blackout (decay + overnight gap risk).
  * Position-size cap from `SymbolMeta.intraday_pct_max()` (5% leveraged,
    15% non-leveraged) — second hard cap on top of conviction-tier sizing.
  * Long-only fractional (Alpaca limitation — no fractional shorts).
    Short-sell only on Easy-To-Borrow ('ETB') symbols; SOXS / SQQQ etc.
    handle inverse exposure instead.

The executor writes warm_store rows tagged `asset_class='STOCK'`,
`venue='alpaca'` so the per-class readiness gate, finetune corpus
filter, and warm_store_sync replication all see the new venue cleanly.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class StocksOrderResult:
    submitted: bool
    reason:    str = ""
    order_id:  Optional[str] = None
    venue:     str = "alpaca"
    paper:     bool = True
    asset_class: str = "STOCK"
    decision_id: Optional[str] = None


class AlpacaExecutor:
    """Thin wrapper around AlpacaClient.place_order with stock-aware gates.

    Construct once at startup; share across the agentic loop.

    Side effects:
      * Reads `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` env vars.
      * Honours `ACT_DISABLE_AGENTIC_LOOP=1` (returns 'kill_switch' reason).
      * Honours `ACT_STOCKS_REAL_CAPITAL_ENABLED=1` AND `ACT_REAL_CAPITAL_ENABLED=1`
        for any non-paper order — both must be set.
      * Writes warm_store decision rows on every submitted order, tagged
        `asset_class=STOCK`, `venue=alpaca`.
    """

    def __init__(self, paper: bool = True):
        from src.data.fetcher import AlpacaClient
        self._client = AlpacaClient(paper=paper)
        self.available = self._client.available
        self.paper = paper

    # ── Public API ─────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        if not self.available:
            return {"available": False, "reason": "no_credentials"}
        try:
            acct = self._client.get_account()
            return {
                "available":    True,
                "paper":        self.paper,
                "equity":       float(acct.get("equity", 0) or 0),
                "buying_power": float(acct.get("buying_power", 0) or 0),
                "status":       acct.get("status"),
            }
        except Exception as e:
            return {"available": False, "reason": f"probe_failed: {e}"}

    def submit_order(self, symbol: str, side: str, qty: float,
                     *,
                     limit_price: Optional[float] = None,
                     conviction_tier: str = "normal",
                     trace_id: Optional[str] = None,
                     plan: Optional[Dict[str, Any]] = None) -> StocksOrderResult:
        """Place a stocks order with all stocks-specific gates.

        Returns StocksOrderResult — the executor never raises; gate
        rejections come back as `submitted=False, reason='<why>'`."""
        from src.models.asset_class import classify
        from src.utils.market_hours import (
            is_us_market_open, is_pre_close_leveraged_blackout, is_pre_close_blackout,
        )

        # ── Kill switches ─────────────────────────────────────────
        if os.getenv("ACT_DISABLE_AGENTIC_LOOP") == "1":
            return self._reject(symbol, side, qty, reason="kill_switch", plan=plan)

        # ── Real-capital double-flag ─────────────────────────────
        is_real_capital = (
            os.getenv("ACT_REAL_CAPITAL_ENABLED") == "1"
            and os.getenv("ACT_STOCKS_REAL_CAPITAL_ENABLED") == "1"
        )
        if not self.paper and not is_real_capital:
            return self._reject(symbol, side, qty,
                                reason="real_capital_flags_not_set", plan=plan)

        # ── Symbol classification + leveraged-ETF rules ──────────
        meta = classify(symbol, venue_hint="alpaca")
        if not meta.asset_class.is_stock():
            return self._reject(symbol, side, qty,
                                reason=f"not_a_stock:{meta.asset_class}", plan=plan)
        if meta.is_leveraged_etf and is_pre_close_leveraged_blackout():
            return self._reject(symbol, side, qty,
                                reason="leveraged_etf_pre_close_blackout", plan=plan)
        if is_pre_close_blackout(minutes=5.0):
            # 5-min blackout for ALL symbols — last-minute liquidity
            # tightens too much for any new entry.
            return self._reject(symbol, side, qty,
                                reason="pre_close_blackout_5min", plan=plan)

        # ── Market-hours gate ────────────────────────────────────
        if not is_us_market_open():
            return self._reject(symbol, side, qty,
                                reason="market_closed", plan=plan)

        # ── Account / connectivity ───────────────────────────────
        if not self.available:
            return self._reject(symbol, side, qty,
                                reason="alpaca_unavailable", plan=plan)

        # ── Place order ──────────────────────────────────────────
        side_lower = side.lower()
        if side_lower not in ("buy", "sell"):
            return self._reject(symbol, side, qty,
                                reason=f"unknown_side:{side}", plan=plan)

        order_type = "limit" if limit_price else "market"
        try:
            resp = self._client.place_order(
                symbol=symbol.upper(),
                side=side_lower,
                qty=float(qty),
                order_type=order_type,
                limit_price=limit_price,
                # Stocks use 'day' TIF — Alpaca's 'gtc' isn't allowed
                # outside of a few extended-hours order types.
                time_in_force="day",
            )
        except Exception as e:
            return self._reject(symbol, side, qty,
                                reason=f"alpaca_error:{e}", plan=plan)

        if resp.get("status") != "success":
            return self._reject(symbol, side, qty,
                                reason=f"alpaca_rejected:{resp.get('message') or resp.get('error') or 'unknown'}",
                                plan=plan)

        decision_id = f"stocks-alpaca-{uuid.uuid4().hex}"
        self._log_decision(
            decision_id=decision_id,
            symbol=symbol.upper(),
            side=side_lower,
            qty=qty,
            limit_price=limit_price,
            conviction_tier=conviction_tier,
            order_resp=resp,
            trace_id=trace_id,
            plan=plan,
            meta=meta,
        )
        logger.info(
            "[ALPACA-EXEC] submitted %s %.4f %s tier=%s order_id=%s",
            side_lower.upper(), qty, symbol.upper(), conviction_tier, resp.get("order_id"),
        )
        return StocksOrderResult(
            submitted=True,
            order_id=resp.get("order_id"),
            paper=self.paper,
            decision_id=decision_id,
        )

    # ── Internals ──────────────────────────────────────────────────

    def _reject(self, symbol: str, side: str, qty: float, *,
                reason: str, plan: Optional[Dict[str, Any]] = None) -> StocksOrderResult:
        """Log a structured rejection so /diagnose-noop can count by reason."""
        decision_id = f"stocks-alpaca-reject-{uuid.uuid4().hex}"
        try:
            self._log_decision(
                decision_id=decision_id, symbol=symbol.upper(),
                side=str(side).lower(), qty=float(qty),
                limit_price=None, conviction_tier="rejected",
                order_resp={"status": "rejected", "reason": reason},
                trace_id=None, plan=plan, meta=None,
                final_action_override="STOCKS_REJECT",
            )
        except Exception:
            pass
        logger.info("[ALPACA-EXEC] reject %s %s qty=%.4f reason=%s",
                    side, symbol, qty, reason)
        return StocksOrderResult(submitted=False, reason=reason,
                                 paper=self.paper, decision_id=decision_id)

    def _log_decision(self, *, decision_id: str, symbol: str, side: str,
                      qty: float, limit_price: Optional[float],
                      conviction_tier: str, order_resp: Dict[str, Any],
                      trace_id: Optional[str], plan: Optional[Dict[str, Any]],
                      meta: Any, final_action_override: Optional[str] = None) -> None:
        try:
            from src.orchestration.warm_store import get_store
            store = get_store()
            direction = 1 if side == "buy" else -1
            final_action = final_action_override or ("BUY" if side == "buy" else "SELL")
            comp: Dict[str, Any] = {
                "source": "alpaca_executor",
                "qty": qty,
                "limit_price": limit_price,
                "tier": conviction_tier,
                "order_id": order_resp.get("order_id"),
                "alpaca_status": order_resp.get("status"),
                "alpaca_reason": order_resp.get("reason") or order_resp.get("message"),
            }
            if meta is not None:
                comp["is_leveraged_etf"] = bool(getattr(meta, "is_leveraged_etf", False))
                comp["is_index_etf"]     = bool(getattr(meta, "is_index_etf", False))
            store.write_decision({
                "decision_id":  decision_id,
                "trace_id":     trace_id,
                "symbol":       symbol,
                "ts_ns":        time.time_ns(),
                "direction":    direction,
                "final_action": final_action,
                "consensus":    "alpaca_exec",
                "asset_class":  "STOCK",
                "venue":        "alpaca",
                "plan":         plan or {},
                "component_signals": comp,
            })
        except Exception as e:
            logger.debug("[ALPACA-EXEC] warm_store write failed: %s", e)
