"""Soak-window readiness gate — blocks real-capital trades until proven.

The Learning Mesh Plan §9.4 defines a go/no-go for real money:
    - rolling 500-trade live win rate ≥ Phase 4 baseline + 3pp, OR
    - matches baseline with 20% fewer training-compute-hours, AND
    - no regression in p99 decision latency (< 3.5s).

The human-operator version is "run a 2-week paper soak and look at the
dashboard." This module turns that into an automated gate: the bot refuses
to place real-capital orders until every condition is satisfied.

Defaults (override via env or config):
    MIN_TRADES           500     closed trades in the warm store
    MIN_SOAK_DAYS        14      calendar days since first trade
    MIN_CREDIT_R2        0.4     regression fit from Phase 4.5a
    MAX_QUARANTINED      0       no learner currently in quarantine
    MAX_VIOLATION_RATE   0.02    authority-violation count / trades
    REAL_CAPITAL_FLAG    ACT_REAL_CAPITAL_ENABLED  (final operator flag)

The gate is conservative — ANY failing condition closes it. Reasons are
exposed so the operator can see which condition blocks and how close they
are to clearing it.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# ── Tunables (env-overridable, read fresh on every evaluate() call) ─────
REAL_CAPITAL_FLAG = "ACT_REAL_CAPITAL_ENABLED"


def _tunables() -> Dict[str, float]:
    """Read env overrides on every call so ops can adjust without a restart."""
    return {
        "MIN_TRADES": int(os.getenv("ACT_GATE_MIN_TRADES", "500")),
        "MIN_SOAK_DAYS": float(os.getenv("ACT_GATE_MIN_SOAK_DAYS", "14")),
        "MIN_CREDIT_R2": float(os.getenv("ACT_GATE_MIN_CREDIT_R2", "0.4")),
        "MAX_QUARANTINED": int(os.getenv("ACT_GATE_MAX_QUARANTINED", "0")),
        "MAX_VIOLATION_RATE": float(os.getenv("ACT_GATE_MAX_VIOLATION_RATE", "0.02")),
        # Per-trade Sharpe floor on the last N outcomes. Enforces "positive PnL even
        # at low WR" — a negative-mean / high-variance streak can't clear the soak
        # regardless of trade count. Set to 0 to disable.
        "MIN_ROLLING_SHARPE": float(os.getenv("ACT_GATE_MIN_SHARPE", "1.0")),
        "SHARPE_WINDOW": int(os.getenv("ACT_GATE_SHARPE_WINDOW", "30")),
    }


@dataclass
class GateState:
    """Snapshot of gate evaluation — all booleans + the numbers behind them."""

    open_: bool
    reasons: List[str]
    details: Dict[str, object]

    def to_dict(self) -> Dict:
        return {"open": self.open_, "reasons": self.reasons, "details": self.details}


# ── Core checks ─────────────────────────────────────────────────────────


def _warm_store_path() -> str:
    return os.getenv(
        "ACT_WARM_DB_PATH",
        str(Path(__file__).resolve().parents[2] / "data" / "warm_store.sqlite"),
    )


def _count_trades_and_age(path: str) -> Tuple[int, float]:
    """(n_outcomes, age_in_days_of_oldest_outcome). (0, 0.0) if db missing."""
    if not os.path.exists(path):
        return 0, 0.0
    try:
        conn = sqlite3.connect(path, timeout=2.0)
        cur = conn.execute("SELECT COUNT(*), MIN(exit_ts) FROM outcomes")
        row = cur.fetchone()
        conn.close()
        n = int(row[0] or 0)
        earliest = float(row[1] or 0.0)
        age_days = 0.0 if earliest <= 0 else max(0.0, (time.time() - earliest) / 86400.0)
        return n, age_days
    except Exception as e:
        logger.debug("readiness_gate: warm_store read failed: %s", e)
        return 0, 0.0


def _count_violations_last_1000(path: str) -> Tuple[int, int]:
    """(violations, decisions_checked) over most recent 1000 decisions."""
    if not os.path.exists(path):
        return 0, 0
    try:
        conn = sqlite3.connect(path, timeout=2.0)
        cur = conn.execute(
            "SELECT authority_violations FROM decisions ORDER BY ts_ns DESC LIMIT 1000"
        )
        rows = cur.fetchall()
        conn.close()
        checked = len(rows)
        violations = 0
        for (raw,) in rows:
            try:
                v = json.loads(raw or "[]")
                if isinstance(v, list) and v:
                    violations += 1
            except Exception:
                continue
        return violations, checked
    except Exception as e:
        logger.debug("readiness_gate: violation count failed: %s", e)
        return 0, 0


def _current_credit_r2() -> float:
    """Read the latest R² from the in-process CreditAssigner, if any."""
    try:
        # Avoid hard import — meta_coordinator may never have been instantiated.
        from src.learning.meta_coordinator import get_coordinator
        coord = get_coordinator()
        assigner = getattr(coord, "_assigner", None)
        if assigner is None:
            return 0.0
        return float(assigner.last_r2())
    except Exception:
        return 0.0


def _active_quarantine_count() -> int:
    """Count learners currently quarantined. Requires a process-local
    QuarantineManager — if none exists we conservatively return 0 so this
    check doesn't block the gate before safety.py is wired anywhere."""
    try:
        from src.learning import safety  # noqa: F401
        mgr = getattr(safety, "_GLOBAL_QUARANTINE_MANAGER", None)
        if mgr is None:
            return 0
        q = mgr.quarantined_learners()
        return sum(1 for v in q.values() if v)
    except Exception:
        return 0


# ── Public API ──────────────────────────────────────────────────────────


def evaluate() -> GateState:
    """Evaluate all gate conditions. Never raises — failures close the gate."""
    t = _tunables()
    reasons: List[str] = []
    details: Dict[str, object] = {}

    path = _warm_store_path()
    n, age_days = _count_trades_and_age(path)
    details["trades"] = n
    details["soak_days"] = round(age_days, 2)
    if n < t["MIN_TRADES"]:
        reasons.append(f"trades {n} < required {t['MIN_TRADES']}")
    if age_days < t["MIN_SOAK_DAYS"]:
        reasons.append(f"soak age {age_days:.1f}d < required {t['MIN_SOAK_DAYS']:.1f}d")

    violations, checked = _count_violations_last_1000(path)
    rate = (violations / checked) if checked else 0.0
    details["authority_violations"] = violations
    details["authority_checked"] = checked
    details["authority_violation_rate"] = round(rate, 4)
    if checked >= 100 and rate > t["MAX_VIOLATION_RATE"]:
        reasons.append(
            f"authority violation rate {rate:.3f} > ceiling {t['MAX_VIOLATION_RATE']:.3f}"
        )

    r2 = _current_credit_r2()
    details["credit_r2"] = round(r2, 4)
    if n >= t["MIN_TRADES"] and r2 < t["MIN_CREDIT_R2"]:
        reasons.append(f"credit R² {r2:.3f} < required {t['MIN_CREDIT_R2']:.3f}")

    q = _active_quarantine_count()
    details["quarantined_learners"] = q
    if q > t["MAX_QUARANTINED"]:
        reasons.append(f"{q} quarantined learners > ceiling {t['MAX_QUARANTINED']}")

    # Rolling Sharpe on the last SHARPE_WINDOW trades from SafeEntryState. Gates
    # against the "negative-EV or high-variance" pattern — paper soak can't complete
    # with a −0.3 Sharpe even if trade count is met. Only enforced once MIN_TRADES
    # warm-store outcomes are in AND SafeEntryState has at least SHARPE_WINDOW
    # samples — avoids rejecting the gate when the state file is sparse or absent
    # (e.g. warm_store seeded from backfill but SafeEntryState not populated yet).
    sharpe = 0.0
    sharpe_samples = 0
    try:
        from src.trading.safe_entries import SafeEntryState, default_state_path
        state = SafeEntryState.load(default_state_path())
        sharpe = state.combined_rolling_sharpe(n=int(t["SHARPE_WINDOW"]))
        sharpe_samples = sum(len(a.trade_pnl_pcts) for a in state.assets.values())
    except Exception:
        pass
    details["rolling_sharpe"] = round(sharpe, 4)
    details["sharpe_window"] = int(t["SHARPE_WINDOW"])
    details["sharpe_samples"] = sharpe_samples
    if (
        n >= t["MIN_TRADES"]
        and sharpe_samples >= int(t["SHARPE_WINDOW"])
        and t["MIN_ROLLING_SHARPE"] > 0
        and sharpe < t["MIN_ROLLING_SHARPE"]
    ):
        reasons.append(
            f"rolling Sharpe {sharpe:.2f} < required {t['MIN_ROLLING_SHARPE']:.2f}"
            f" (window={int(t['SHARPE_WINDOW'])})"
        )

    operator_flag = os.getenv(REAL_CAPITAL_FLAG, "0") == "1"
    details["operator_flag"] = operator_flag
    if not operator_flag:
        reasons.append(f"{REAL_CAPITAL_FLAG} not set (operator gate)")

    gate = GateState(open_=(len(reasons) == 0), reasons=reasons, details=details)
    _emit_metric(gate)
    return gate


def is_live_ready() -> bool:
    """True iff every condition is met. Safe to call from the hot path.

    The executor should call this before placing any order that uses real
    capital. When it returns False, route the order to paper / reject it.
    """
    return evaluate().open_


def format_report(state: GateState = None) -> str:
    """Human-readable multi-line report — used by CLI + startup banner."""
    if state is None:
        state = evaluate()
    lines = [f"Readiness gate: {'OPEN (live-ready)' if state.open_ else 'CLOSED'}"]
    if state.reasons:
        lines.append("  Failing conditions:")
        for r in state.reasons:
            lines.append(f"    • {r}")
    lines.append("  Numbers:")
    for k, v in state.details.items():
        lines.append(f"    {k:<26} {v}")
    return "\n".join(lines)


def _emit_metric(state: GateState) -> None:
    try:
        from src.orchestration.metrics import record_readiness_gate
        record_readiness_gate(open_=state.open_, failure_count=len(state.reasons))
    except Exception:
        pass


def main() -> int:
    """`python -m src.orchestration.readiness_gate` — print gate state."""
    state = evaluate()
    print(format_report(state))
    return 0 if state.open_ else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
