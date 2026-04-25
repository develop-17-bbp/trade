"""Skill: /trade-path-trace -- per-decision gate-by-gate breakdown.

Reads the most recent N rows from warm_store.decisions and shows for
each one:

  * decision_id                — agentic-* (real) or shadow-* (audit)
  * scanner.opportunity_score  — from component_signals JSON
  * analyst.proceed/confidence — from plan_json
  * authority.violations       — list (empty = passed)
  * conviction.tier + reason   — sniper/normal/reject + why
  * cost_gate.margin           — expected_return - frictional cost
  * readiness.open             — paper-mode bypass shown explicitly
  * final_action               — what the executor actually did
  * thesis                     — analyst's one-liner

Use it after START_ALL settles to confirm trades are firing for the
right reasons, OR to identify which gate is the actual bottleneck
when nothing fires.

CLI:
    python -m src.skills.cli run trade-path-trace
    python -m src.skills.cli run trade-path-trace asset=BTC limit=10
    python -m src.skills.cli run trade-path-trace only_real=true
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.skills.registry import SkillResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WARM_STORE = PROJECT_ROOT / "data" / "warm_store.sqlite"


def _safe_load(s: Any) -> Dict[str, Any]:
    if not s:
        return {}
    if isinstance(s, dict):
        return s
    try:
        v = json.loads(str(s))
        return v if isinstance(v, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _format_row(row: sqlite3.Row) -> List[str]:
    """Render one decision row into a printable list of lines."""
    out: List[str] = []

    decision_id = row["decision_id"] or ""
    is_shadow = decision_id.startswith("shadow-")
    kind = "SHADOW" if is_shadow else "AGENTIC"
    ts_ns = row["ts_ns"] or 0
    ts_s = ts_ns / 1_000_000_000 if ts_ns else 0
    age_min = (
        (int(__import__("time").time()) - int(ts_s)) / 60.0 if ts_s else -1.0
    )

    plan = _safe_load(row["plan_json"])
    sigs = _safe_load(row["component_signals"])
    crit = _safe_load(row["self_critique"])

    direction = row["direction"] or 0
    final_action = row["final_action"] or ""
    confidence = row["confidence"] or 0.0
    veto = bool(row["veto"])
    auth_viol = row["authority_violations"] or "[]"
    try:
        auth_viol_list = json.loads(auth_viol) if isinstance(auth_viol, str) else auth_viol
    except json.JSONDecodeError:
        auth_viol_list = []

    out.append(
        f"  [{kind}] {decision_id}  "
        f"asset={row['symbol']}  "
        f"age={age_min:.1f}min  "
        f"dir={direction:+d}  "
        f"final={final_action or '(unset)'}"
    )

    # Scanner block
    scanner = sigs.get("scanner") or {}
    if scanner:
        out.append(
            f"    scanner: score={scanner.get('opportunity_score', '?')}  "
            f"dir={scanner.get('proposed_direction', '?')}  "
            f"signals={scanner.get('top_signals') or []}"
        )

    # Analyst / TradePlan
    if plan:
        proceed = plan.get("proceed", plan.get("direction") not in (None, "FLAT", "SKIP"))
        thesis = (plan.get("thesis") or plan.get("rationale") or "")[:120]
        out.append(
            f"    analyst: proceed={proceed}  "
            f"conf={confidence:.2f}  "
            f"size_pct={plan.get('size_pct', '?')}  "
            f"tier={plan.get('entry_tier', '?')}"
        )
        if thesis:
            out.append(f"    thesis : {thesis}")

    # Gate stack
    gates = sigs.get("gates") or {}
    if gates:
        for name in ("authority", "conviction", "cost", "readiness", "safe_entries"):
            g = gates.get(name)
            if g is None:
                continue
            passed = g.get("passed", g.get("ok"))
            reason = g.get("reason", "")
            out.append(f"    gate.{name}: passed={passed}  {reason}")
    elif auth_viol_list:
        out.append(f"    authority_violations: {auth_viol_list}")

    if veto:
        out.append(f"    VETO: {sigs.get('veto_reason', '(unspecified)')}")

    # Self-critique (post-close)
    if crit:
        out.append(
            f"    critique: matched_thesis={crit.get('matched_thesis')}  "
            f"realized_pct={crit.get('realized_pct', '?')}  "
            f"verdict={crit.get('verdict', '?')}"
        )

    return out


def _summarize(rows: List[sqlite3.Row]) -> Dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "by_kind": {"agentic": 0, "shadow": 0},
            "by_action": {},
            "rejected_by_gate": {},
            "fired": 0,
        }
    by_kind = {"agentic": 0, "shadow": 0}
    by_action: Dict[str, int] = {}
    rejected_by_gate: Dict[str, int] = {}
    fired = 0
    for r in rows:
        kind = "shadow" if (r["decision_id"] or "").startswith("shadow-") else "agentic"
        by_kind[kind] = by_kind.get(kind, 0) + 1
        action = r["final_action"] or "(unset)"
        by_action[action] = by_action.get(action, 0) + 1
        if action.upper() in ("LONG", "SHORT", "BUY", "SELL"):
            fired += 1
        sigs = _safe_load(r["component_signals"])
        for gname, g in (sigs.get("gates") or {}).items():
            if isinstance(g, dict) and g.get("passed") is False:
                rejected_by_gate[gname] = rejected_by_gate.get(gname, 0) + 1
    return {
        "count": len(rows),
        "by_kind": by_kind,
        "by_action": by_action,
        "rejected_by_gate": rejected_by_gate,
        "fired": fired,
    }


def run(args: Optional[Dict[str, Any]] = None) -> SkillResult:
    args = dict(args or {})
    asset = str(args.get("asset", "")).upper().strip()
    try:
        limit = max(1, min(50, int(args.get("limit", 5))))
    except (TypeError, ValueError):
        limit = 5
    only_real_raw = args.get("only_real", False)
    if isinstance(only_real_raw, str):
        only_real = only_real_raw.strip().lower() in ("1", "true", "yes", "on")
    else:
        only_real = bool(only_real_raw)

    db_path = os.getenv("ACT_WARM_DB_PATH", str(WARM_STORE))
    if not os.path.exists(db_path):
        return SkillResult(
            ok=False,
            error=(
                f"warm_store not found at {db_path}. "
                "Bot has not run yet, or ACT_WARM_DB_PATH is wrong."
            ),
        )

    where = []
    params: List[Any] = []
    if asset:
        where.append("symbol = ?")
        params.append(asset)
    if only_real:
        where.append("decision_id NOT LIKE 'shadow-%'")
    sql = (
        "SELECT decision_id, trace_id, symbol, ts_ns, direction, confidence, "
        "consensus, veto, raw_signal, final_action, authority_violations, "
        "payload_json, component_signals, plan_json, self_critique "
        "FROM decisions"
    )
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY ts_ns DESC LIMIT ?"
    params.append(limit)

    try:
        conn = sqlite3.connect(db_path, timeout=3.0)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.close()
    except sqlite3.OperationalError as e:
        return SkillResult(
            ok=False, error=f"warm_store read failed: {e}",
        )

    summary = _summarize(rows)
    lines: List[str] = []
    lines.append("=" * 60)
    filt = []
    if asset:
        filt.append(f"asset={asset}")
    if only_real:
        filt.append("only_real=true")
    lines.append(f" trade-path-trace  limit={limit}  " + " ".join(filt))
    lines.append("=" * 60)
    lines.append(
        f" Summary: {summary['count']} rows  "
        f"agentic={summary['by_kind']['agentic']}  "
        f"shadow={summary['by_kind']['shadow']}  "
        f"fired={summary['fired']}"
    )
    if summary["by_action"]:
        lines.append(f" Actions: {summary['by_action']}")
    if summary["rejected_by_gate"]:
        lines.append(f" Rejected-by-gate: {summary['rejected_by_gate']}")
    lines.append("")
    if not rows:
        lines.append(" (no decisions match the filter)")
    else:
        for row in rows:
            lines.extend(_format_row(row))
            lines.append("")

    return SkillResult(
        ok=True,
        message="\n".join(lines),
        data={
            "summary": summary,
            "limit": limit,
            "asset": asset or None,
            "only_real": only_real,
            "warm_store_path": db_path,
        },
    )
