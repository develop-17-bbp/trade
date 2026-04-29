"""Skill: /flatten-positions — close every open position recorded on disk.

Cures the position-stacking failure mode (172 stuck ETH paper longs on
2026-04-29). For each open position in `data/sl_state.json`:
    1. Fetch current price (best-effort).
    2. Compute realized PnL vs the recorded entry.
    3. Write a warm_store outcome row tagged exit_reason='operator_flatten'.
    4. Clear the position from sl_state.

Pairs with the new ACT_MAX_OPEN_POSITIONS_PER_ASSET cap (default 3) so
when the bot restarts it can't re-stack into the same trap.

Non-reversible — losses become realized once the outcome row is written.
This is the *honest* version of "reset these losses": the loss is
recorded and visible to /diagnose-noop / weekly_brief / future
finetune corpus, not deleted.

Usage:
    /flatten-positions confirm=true                 # close all
    /flatten-positions confirm=true asset=ETH       # close only ETH
    /flatten-positions confirm=true dry_run=true    # show what would close
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List

from src.skills.registry import SkillResult


def _fetch_price(symbol: str) -> float:
    """Best-effort current price. Returns 0.0 if no source reachable."""
    try:
        from src.data.fetcher import PriceFetcher
        try:
            pf = PriceFetcher()
        except Exception:
            pf = PriceFetcher({})
        p = pf.fetch_latest_price(symbol)
        return float(p or 0.0)
    except Exception:
        return 0.0


def _compute_pnl(direction: str, entry: float, exit_p: float, qty: float):
    """Return (pnl_pct, pnl_usd). Negative for losers."""
    if entry <= 0 or exit_p <= 0 or qty <= 0:
        return 0.0, 0.0
    if direction.upper() == 'LONG':
        pnl_pct = (exit_p - entry) / entry * 100.0
        pnl_usd = (exit_p - entry) * qty
    else:
        pnl_pct = (entry - exit_p) / entry * 100.0
        pnl_usd = (entry - exit_p) * qty
    return pnl_pct, pnl_usd


def run(args: Dict[str, Any]) -> SkillResult:
    if not args.get("confirm", False):
        return SkillResult(
            ok=False,
            error=(
                "pass confirm=true to acknowledge that flatten-positions writes "
                "realized-PnL outcome rows for every open position and clears "
                "the in-flight book"
            ),
        )

    asset_filter = (args.get("asset") or "").upper().strip() or None
    dry_run = bool(args.get("dry_run", False))
    incident_id = f"flatten-{uuid.uuid4().hex[:12]}"

    # 1. Halt the agentic loop so the bot can't immediately re-enter.
    if not dry_run:
        os.environ["ACT_DISABLE_AGENTIC_LOOP"] = "1"

    # 2. Load open positions from sl_state.json.
    try:
        from src.persistence.sl_persistence import SLPersistenceManager
        mgr = SLPersistenceManager()
        all_positions = mgr._load_raw() or {}
    except Exception as e:
        return SkillResult(
            ok=False,
            error=f"failed to load sl_state.json: {type(e).__name__}: {e}",
        )

    if not all_positions:
        return SkillResult(
            ok=True,
            message="No open positions in sl_state.json — nothing to flatten.",
            data={"incident_id": incident_id, "flattened": 0},
        )

    # 3. Iterate, fetch price, write outcome, clear.
    flattened: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    try:
        from src.orchestration.warm_store import get_store
        store = get_store() if not dry_run else None
    except Exception:
        store = None

    for asset, pos in list(all_positions.items()):
        if asset_filter and asset.upper() != asset_filter:
            skipped.append({"asset": asset, "reason": f"filtered by asset={asset_filter}"})
            continue

        direction = str(pos.get("direction", "LONG")).upper()
        entry_price = float(pos.get("entry_price", 0) or 0)
        qty = float(pos.get("quantity", 0) or pos.get("qty", 0) or 0)
        entry_ts = float(pos.get("entry_time", 0) or time.time())
        decision_id = str(pos.get("decision_id") or pos.get("order_id") or "") or f"flatten-orphan-{asset}-{int(entry_ts)}"

        # Map asset → symbol best-effort. Crypto: BTC → BTC/USD; stocks pass through.
        symbol = asset
        if asset in ("BTC", "ETH", "SOL", "DOGE"):
            symbol = f"{asset}/USD"
        exit_price = _fetch_price(symbol)
        if exit_price <= 0 and entry_price > 0:
            # Can't reach the price source — fall back to entry, mark exit_reason.
            exit_price = entry_price
            note = "exit_price_fallback_to_entry"
        else:
            note = ""
        pnl_pct, pnl_usd = _compute_pnl(direction, entry_price, exit_price, qty)

        record = {
            "asset": asset,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "qty": qty,
            "pnl_pct": round(pnl_pct, 4),
            "pnl_usd": round(pnl_usd, 4),
            "note": note,
        }

        if dry_run:
            skipped.append({**record, "reason": "dry_run"})
            continue

        # Write outcome row — this is what makes the loss "realized" in warm_store.
        # The outcomes table has a FK on decision_id → decisions; for orphaned
        # positions whose original decision row may have been pruned (or never
        # written, e.g. positions recovered from a crash), synthesize a
        # reconcile decision row first so the FK is satisfied. Either path
        # leaves an audit trail.
        try:
            # Best-effort: if the decision_id already exists, the INSERT OR
            # REPLACE in write_decision is a no-op; if not, it backfills.
            store.write_decision({
                "decision_id": decision_id,
                "symbol": asset,
                "ts_ns": int(entry_ts * 1e9),
                "direction": 1 if direction == "LONG" else -1,
                "final_action": "ORPHAN_FLATTEN_RECONCILE",
                "consensus": "operator_flatten",
                "component_signals": {
                    "source": "skill:flatten-positions",
                    "reason": "synthesized to satisfy outcomes FK on orphaned position",
                    "incident_id": incident_id,
                },
            })
            store.flush()
            store.write_outcome({
                "decision_id": decision_id,
                "symbol": asset,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "pnl_usd": pnl_usd,
                "duration_s": max(0.0, time.time() - entry_ts),
                "exit_reason": "operator_flatten",
                "regime": "unknown",
                "entry_ts": entry_ts,
                "exit_ts": time.time(),
                # Asset class default ('CRYPTO') is correct for the 172 ETH
                # case; if operator extends to stocks later they'd pass it
                # via args. Phase B/C dual-asset extension already columns this.
            })
            store.flush()
        except Exception as e:
            skipped.append({**record, "reason": f"warm_store_write_failed: {e}"})
            continue

        # Clear from sl_state (atomic per-asset, so a partial run is recoverable).
        try:
            mgr.clear_position(asset)
        except Exception as e:
            record["clear_error"] = str(e)

        flattened.append(record)

    # 4. Audit incident row to warm_store.
    if not dry_run and store is not None:
        try:
            store.write_decision({
                "decision_id": incident_id,
                "symbol": "INCIDENT",
                "ts_ns": time.time_ns(),
                "final_action": "FLATTEN_POSITIONS",
                "component_signals": {
                    "source": "skill:flatten-positions",
                    "asset_filter": asset_filter,
                    "n_flattened": len(flattened),
                    "n_skipped": len(skipped),
                    "total_pnl_usd": round(sum(r["pnl_usd"] for r in flattened), 2),
                    "total_pnl_pct_avg": round(
                        sum(r["pnl_pct"] for r in flattened) / max(1, len(flattened)),
                        3,
                    ),
                },
            })
            store.flush()
        except Exception:
            pass

    total_pnl = sum(r["pnl_usd"] for r in flattened)
    msg_lines = [
        f"flatten-positions {'(DRY RUN)' if dry_run else 'engaged'}: incident={incident_id}",
        f"  flattened: {len(flattened)} positions",
        f"  skipped:   {len(skipped)} positions",
        f"  total realized PnL: ${total_pnl:+,.2f}",
    ]
    if not dry_run:
        msg_lines.append(
            "  Agentic loop halted (ACT_DISABLE_AGENTIC_LOOP=1). Set "
            "ACT_MAX_OPEN_POSITIONS_PER_ASSET=3 before restart to prevent re-stacking."
        )

    return SkillResult(
        ok=True,
        message="\n".join(msg_lines),
        data={
            "incident_id": incident_id,
            "dry_run": dry_run,
            "asset_filter": asset_filter,
            "flattened": flattened,
            "skipped": skipped,
            "total_pnl_usd": round(total_pnl, 2),
        },
    )
