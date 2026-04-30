"""Alpaca options paper executor — minimal directional support.

Mirrors `alpaca_executor.py` shape for stocks but submits option
orders against Alpaca's options Level 3 endpoints. The single-leg
long_call / long_put paths are wired here; multi-leg spreads
(vertical_call, vertical_put, iron_condor) need the analyst-side
option-plan generator + conviction-gate retune that's NOT in this
commit.

Hard rules baked in:
  * RTH-only via `src/utils/market_hours.is_us_market_open` — Alpaca
    rejects option orders outside session anyway; this catches it
    upstream so the warm_store reject row records "market_closed"
    instead of an opaque Alpaca 4xx.
  * `min_dte` / `max_dte` from the alpaca_options exchange config —
    refuse contracts expiring sooner / later than the operator window.
  * Greek caps (delta_per_position, vega_total) — refuse if the
    proposed position breaches the per-position delta or pushes the
    portfolio's combined vega over the cap.
  * Paper-only by default; live requires both ACT_REAL_CAPITAL_ENABLED=1
    AND ACT_OPTIONS_REAL_CAPITAL_ENABLED=1.

The executor writes warm_store rows tagged `asset_class='OPTIONS'`,
`venue='alpaca'` so the per-class readiness gate, finetune corpus
filter, and warm_store_sync replication see options decisions cleanly
alongside stocks + crypto.

Operator pre-req: Alpaca paper account must have options Level 3
cleared (one-time application via the dashboard). Without it the
order POSTs return 403; this executor returns the error verbatim.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class OptionsOrderResult:
    submitted: bool
    reason:    str = ""
    order_id:  Optional[str] = None
    venue:     str = "alpaca"
    paper:     bool = True
    asset_class: str = "OPTIONS"
    decision_id: Optional[str] = None
    occ_symbol: Optional[str] = None  # OCC option symbol (e.g. SPY250620C00500000)


class AlpacaOptionsExecutor:
    """Minimal options executor — single-leg long_call / long_put.

    Construct once at startup. Share across the agentic loop. Errors
    return as `OptionsOrderResult(submitted=False, reason=...)` —
    never raises.

    Side effects:
      * Reads `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` env.
      * Honors `ACT_DISABLE_AGENTIC_LOOP=1`.
      * Honors `ACT_REAL_CAPITAL_ENABLED=1 + ACT_OPTIONS_REAL_CAPITAL_ENABLED=1`
        for non-paper orders.
      * Writes warm_store decision rows tagged `asset_class=OPTIONS`,
        `venue=alpaca`.
    """

    def __init__(self, paper: bool = True,
                 min_dte: int = 7, max_dte: int = 45,
                 max_per_position_pct: float = 5.0,
                 max_total_exposure_pct: float = 20.0,
                 max_delta_per_position: float = 0.5,
                 max_vega_total: float = 50.0):
        from src.data.fetcher import AlpacaClient
        self._client = AlpacaClient(paper=paper)
        self.available = self._client.available
        self.paper = paper
        self.min_dte = int(min_dte)
        self.max_dte = int(max_dte)
        self.max_per_position_pct = float(max_per_position_pct)
        self.max_total_exposure_pct = float(max_total_exposure_pct)
        self.max_delta_per_position = float(max_delta_per_position)
        self.max_vega_total = float(max_vega_total)
        # Track open positions' aggregate vega/delta so we can
        # reject before submission. Updated post-fill by the executor's
        # position bookkeeping.
        self._open_vega_total: float = 0.0

    # ── Public API ─────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        if not self.available:
            return {"available": False, "reason": "no_credentials"}
        try:
            acct = self._client.get_account()
            opt_level = (acct.get("options_trading_level") or
                         acct.get("options_approved_level"))
            return {
                "available":         True,
                "paper":             self.paper,
                "options_level":     opt_level,
                "options_buying_power": float(acct.get("options_buying_power", 0) or 0),
                "equity":            float(acct.get("equity", 0) or 0),
            }
        except Exception as e:
            return {"available": False, "reason": f"probe_failed: {e}"}

    def submit_long_directional(
        self,
        underlying: str,
        side: str,                      # 'call' or 'put'
        strike: float,
        expiration: str,                # ISO date 'YYYY-MM-DD'
        qty: int = 1,
        *,
        limit_price: Optional[float] = None,
        delta_estimate: Optional[float] = None,
        vega_estimate: Optional[float] = None,
        conviction_tier: str = "normal",
        trace_id: Optional[str] = None,
        plan: Optional[Dict[str, Any]] = None,
    ) -> OptionsOrderResult:
        """Single-leg long call or long put."""
        from src.utils.market_hours import is_us_market_open

        underlying_u = underlying.upper().strip()
        side_l = (side or "").lower().strip()
        if side_l not in ("call", "put"):
            return self._reject(underlying_u, side, qty, occ=None,
                                reason=f"unknown_side:{side}", plan=plan)

        # ── Kill switches ─────────────────────────────────────────
        if os.getenv("ACT_DISABLE_AGENTIC_LOOP") == "1":
            return self._reject(underlying_u, side_l, qty, occ=None,
                                reason="kill_switch", plan=plan)

        # ── Real-capital double-flag ─────────────────────────────
        is_real = (
            os.getenv("ACT_REAL_CAPITAL_ENABLED") == "1"
            and os.getenv("ACT_OPTIONS_REAL_CAPITAL_ENABLED") == "1"
        )
        if not self.paper and not is_real:
            return self._reject(underlying_u, side_l, qty, occ=None,
                                reason="real_capital_flags_not_set", plan=plan)

        # ── DTE check ────────────────────────────────────────────
        try:
            exp_date = _dt.date.fromisoformat(expiration)
        except Exception:
            return self._reject(underlying_u, side_l, qty, occ=None,
                                reason=f"bad_expiration:{expiration}", plan=plan)
        dte = (exp_date - _dt.date.today()).days
        if dte < self.min_dte:
            return self._reject(underlying_u, side_l, qty, occ=None,
                                reason=f"dte_too_short:{dte}<{self.min_dte}", plan=plan)
        if dte > self.max_dte:
            return self._reject(underlying_u, side_l, qty, occ=None,
                                reason=f"dte_too_long:{dte}>{self.max_dte}", plan=plan)

        # ── Greek caps ───────────────────────────────────────────
        if delta_estimate is not None:
            if abs(float(delta_estimate)) > self.max_delta_per_position:
                return self._reject(
                    underlying_u, side_l, qty, occ=None,
                    reason=f"delta_too_high:{delta_estimate:+.2f}>{self.max_delta_per_position}",
                    plan=plan,
                )
        if vega_estimate is not None:
            new_vega = self._open_vega_total + abs(float(vega_estimate)) * qty
            if new_vega > self.max_vega_total:
                return self._reject(
                    underlying_u, side_l, qty, occ=None,
                    reason=f"vega_cap_breach:{new_vega:.1f}>{self.max_vega_total}",
                    plan=plan,
                )

        # ── Market-hours gate ────────────────────────────────────
        if not is_us_market_open():
            return self._reject(underlying_u, side_l, qty, occ=None,
                                reason="market_closed", plan=plan)

        if not self.available:
            return self._reject(underlying_u, side_l, qty, occ=None,
                                reason="alpaca_unavailable", plan=plan)

        # ── Build OCC symbol + submit ────────────────────────────
        occ = _build_occ_symbol(underlying_u, exp_date, side_l, float(strike))
        order_type = "limit" if limit_price else "market"
        try:
            resp = self._client.place_order(
                symbol=occ,
                side="buy",                    # long-only path
                qty=int(qty),
                order_type=order_type,
                limit_price=limit_price,
                time_in_force="day",
            )
        except Exception as e:
            return self._reject(underlying_u, side_l, qty, occ=occ,
                                reason=f"alpaca_error:{e}", plan=plan)

        if resp.get("status") != "success":
            return self._reject(
                underlying_u, side_l, qty, occ=occ,
                reason=f"alpaca_rejected:{resp.get('message') or resp.get('error') or 'unknown'}",
                plan=plan,
            )

        decision_id = f"options-alpaca-{uuid.uuid4().hex}"
        # Update aggregate vega bookkeeping on success.
        if vega_estimate is not None:
            self._open_vega_total += abs(float(vega_estimate)) * qty
        logger.info(
            "[ALPACA-OPT] submitted long %s %dx %s @ %s order_id=%s tier=%s",
            side_l.upper(), qty, occ, limit_price or "MKT",
            resp.get("order_id"), conviction_tier,
        )
        # Best-effort warm_store row — same shape as stocks executor.
        try:
            self._log_decision(
                decision_id=decision_id, underlying=underlying_u,
                occ=occ, side=side_l, qty=qty, conviction_tier=conviction_tier,
                order_resp=resp, trace_id=trace_id, plan=plan,
            )
        except Exception as e:
            logger.debug("[ALPACA-OPT] _log_decision soft-fail: %s", e)
        return OptionsOrderResult(
            submitted=True, order_id=resp.get("order_id"),
            paper=self.paper, decision_id=decision_id, occ_symbol=occ,
        )

    # ── Internals ──────────────────────────────────────────────────

    def _reject(self, underlying: str, side: str, qty: int, *,
                occ: Optional[str], reason: str,
                plan: Optional[Dict[str, Any]] = None) -> OptionsOrderResult:
        decision_id = f"options-alpaca-reject-{uuid.uuid4().hex}"
        try:
            self._log_decision(
                decision_id=decision_id, underlying=underlying.upper(),
                occ=occ, side=str(side).lower(), qty=int(qty),
                conviction_tier="rejected",
                order_resp={"status": "rejected", "reason": reason},
                trace_id=None, plan=plan,
                final_action_override="OPTIONS_REJECT",
            )
        except Exception as e:
            logger.debug("[ALPACA-OPT] reject _log_decision soft-fail: %s", e)
        return OptionsOrderResult(
            submitted=False, reason=reason,
            paper=self.paper, decision_id=decision_id, occ_symbol=occ,
        )

    def _log_decision(self, *, decision_id: str, underlying: str,
                      occ: Optional[str], side: str, qty: int,
                      conviction_tier: str, order_resp: Dict[str, Any],
                      trace_id: Optional[str], plan: Optional[Dict[str, Any]],
                      final_action_override: Optional[str] = None) -> None:
        """Best-effort warm_store row write."""
        try:
            from src.orchestration.warm_store import get_store
        except Exception:
            return
        try:
            store = get_store()
            ts_ns = time.time_ns()
            row: Dict[str, Any] = {
                "decision_id":    decision_id,
                "ts_ns":          ts_ns,
                "symbol":         underlying,
                "asset_class":    "OPTIONS",
                "venue":          "alpaca",
                "occ_symbol":     occ,
                "side":           side,
                "qty":            int(qty),
                "conviction_tier": conviction_tier,
                "final_action":   (final_action_override or "OPTIONS_SUBMIT"),
                "trace_id":       trace_id,
                "plan_json":      plan,
                "order_resp":     order_resp,
            }
            try:
                store.write_decision(row)
            except Exception:
                store.write(row)
        except Exception:
            pass


def _build_occ_symbol(underlying: str, exp_date: _dt.date,
                      side: str, strike: float) -> str:
    """Build the OCC option symbol (21 chars).

    Format: ROOT (1-6 chars) + YYMMDD + C/P + STRIKE (8-digit, ×1000).
    Example: SPY 250620 C 00500000 -> SPY250620C00500000
    Alpaca, OCC, and the major data feeds all accept this format.
    """
    root = underlying.upper().rstrip()
    yymmdd = exp_date.strftime("%y%m%d")
    cp = "C" if side.lower() == "call" else "P"
    # Strike encoded as integer cents × 10 (8 digits, zero-padded).
    # Example: $500.00 strike -> 00500000 (500000 cents → padded to 8).
    strike_int = int(round(float(strike) * 1000))
    strike_str = f"{strike_int:08d}"
    return f"{root}{yymmdd}{cp}{strike_str}"
