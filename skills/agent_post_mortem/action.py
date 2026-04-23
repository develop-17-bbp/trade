"""
Skill: /agent-post-mortem — counterfactual chat about one decision.

Loads one decision_id's full context from warm_store + brain_memory,
asks the Analyst brain to speak as each agent and explain its vote.
Also supports `what_if` rewind: re-seed the analyst with the brain_memory
state from N seconds before the actual decision and ask "would you have
traded?"

No LLM call is fatal — missing brains fall back to a structured summary
of the trace itself (still useful for debugging).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.skills.registry import SkillResult

logger = logging.getLogger(__name__)


def _warm_db_path() -> str:
    import os
    return os.getenv(
        "ACT_WARM_DB_PATH",
        str(Path(__file__).resolve().parents[3] / "data" / "warm_store.sqlite"),
    )


def _load_decision(decision_id: str) -> Optional[Dict[str, Any]]:
    """Pull the decision row + matching outcome (if any) from warm_store."""
    db = _warm_db_path()
    try:
        conn = sqlite3.connect(db, timeout=3.0)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
    except Exception as e:
        logger.debug("post-mortem: cannot open warm_store: %s", e)
        return None
    try:
        row = cur.execute(
            "SELECT decision_id, symbol, ts_ns, direction, confidence, "
            "final_action, authority_violations, payload_json, "
            "component_signals, plan_json, self_critique "
            "FROM decisions WHERE decision_id=?", (decision_id,),
        ).fetchone()
        if not row:
            return None
        outcome = cur.execute(
            "SELECT pnl_pct, exit_price, duration_s, exit_reason, regime, "
            "entry_ts, exit_ts, payload_json FROM outcomes "
            "WHERE decision_id=?", (decision_id,),
        ).fetchone()
    except Exception as e:
        logger.debug("post-mortem: query failed: %s", e)
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass

    def _load_json(s):
        try:
            return json.loads(s or "{}")
        except Exception:
            return {}

    return {
        "decision_id": row["decision_id"],
        "symbol": row["symbol"],
        "ts_ns": int(row["ts_ns"] or 0),
        "direction": int(row["direction"] or 0),
        "confidence": float(row["confidence"] or 0.0),
        "final_action": row["final_action"],
        "authority_violations": _load_json(row["authority_violations"]),
        "payload": _load_json(row["payload_json"]),
        "component_signals": _load_json(row["component_signals"]),
        "plan": _load_json(row["plan_json"]),
        "self_critique": _load_json(row["self_critique"]),
        "outcome": ({
            "pnl_pct": outcome["pnl_pct"],
            "exit_price": outcome["exit_price"],
            "duration_s": outcome["duration_s"],
            "exit_reason": outcome["exit_reason"],
            "regime": outcome["regime"],
            "entry_ts": outcome["entry_ts"],
            "exit_ts": outcome["exit_ts"],
            "payload": _load_json(outcome["payload_json"]),
        } if outcome else None),
    }


def _summarize_trace(decision: Dict[str, Any]) -> str:
    """Fallback non-LLM digest — always works even if no LLM is reachable."""
    dm = decision["plan"] or {}
    sc = decision["self_critique"] or {}
    oc = decision["outcome"] or {}
    lines = [
        f"Decision {decision['decision_id']}  symbol={decision['symbol']}",
        f"  Compiled plan: direction={dm.get('direction', '?')} "
        f"tier={dm.get('entry_tier', '?')} size_pct={dm.get('size_pct', '?')}",
        f"  Entry: {dm.get('entry_price', '?')}  SL: {dm.get('sl_price', '?')}",
        f"  Thesis: {(dm.get('thesis') or '')[:200]}",
    ]
    if oc:
        lines.append(
            f"  Outcome: pnl_pct={oc.get('pnl_pct'):+.2f}% "
            f"exit_reason={oc.get('exit_reason')} "
            f"duration={oc.get('duration_s'):.0f}s"
            if oc.get('pnl_pct') is not None else
            f"  Outcome: open, no close yet"
        )
    if sc:
        lines.append(
            f"  Self-critique: matched={sc.get('matched_thesis')} "
            f"miss={sc.get('miss_reason', '')[:120]}"
        )
    av = decision.get("authority_violations") or []
    if av:
        lines.append(f"  Authority violations: {av}")
    cs = decision.get("component_signals") or {}
    if cs:
        lines.append(f"  Components: {', '.join(cs.keys())[:200]}")
    return "\n".join(lines)


def _llm_post_mortem(
    decision: Dict[str, Any], agent_name: str,
) -> Optional[str]:
    """Ask the analyst brain to answer as `agent_name`. Returns the
    model's verdict string, or None if no brain is reachable."""
    try:
        from src.ai.dual_brain import analyze
    except Exception:
        return None
    prompt = (
        f"You are ACT's {agent_name} agent speaking in post-mortem.\n\n"
        f"Plan you voted on:\n{json.dumps(decision.get('plan') or {}, default=str, indent=2)[:1500]}\n\n"
        f"Outcome:\n{json.dumps(decision.get('outcome') or {}, default=str, indent=2)[:800]}\n\n"
        f"Self-critique that was written at close:\n"
        f"{json.dumps(decision.get('self_critique') or {}, default=str)[:600]}\n\n"
        f"Answer in <= 3 sentences: (a) why you voted the way you did, "
        f"(b) given the outcome, would you vote the same today, "
        f"(c) what single signal would have changed your vote?"
    )
    try:
        resp = analyze(prompt)
        if resp and resp.ok and resp.text:
            return resp.text.strip()
    except Exception as e:
        logger.debug("post-mortem: llm call failed: %s", e)
    return None


def _what_if_rewind(
    decision: Dict[str, Any], shift_seconds: float,
) -> Optional[str]:
    """Re-seed the analyst with the brain_memory state from
    `shift_seconds` before the decision's timestamp and ask whether
    it would have compiled the same plan."""
    try:
        from src.ai.agentic_bridge import compile_agentic_plan
        from src.ai.brain_memory import get_brain_memory
    except Exception as e:
        return f"what_if unavailable: {type(e).__name__}: {e}"

    asset = (decision.get("symbol") or "BTC").split(":")[-1].upper()
    decision_ts = float(decision.get("ts_ns", 0)) / 1_000_000_000
    target_ts = decision_ts - float(shift_seconds)
    try:
        mem = get_brain_memory()
        # Reach back for a scan report near the target timestamp.
        scan = mem.read_latest_scan(asset, max_age_s=86400.0)
        if scan is None or abs(scan.ts - target_ts) > 3600:
            return (
                f"what_if at T-{int(shift_seconds)}s: no brain_memory "
                f"scan within 1h of target timestamp; rewind unavailable"
            )
    except Exception as e:
        return f"what_if brain_memory error: {type(e).__name__}: {e}"

    # Compile a plan with the rewound seed context via stub-safe path.
    try:
        loop_result = compile_agentic_plan(
            asset=asset,
            regime=scan.proposed_direction or "UNKNOWN",
            quant_data=f"[REWIND T-{int(shift_seconds)}s] scanner score "
                       f"{scan.opportunity_score:.0f}, direction {scan.proposed_direction}",
            # similar_trades / recent_critiques default to recent — good enough.
            max_steps=3,
        )
    except Exception as e:
        return f"what_if compile failed: {type(e).__name__}: {e}"

    plan = loop_result.plan
    return (
        f"what_if T-{int(shift_seconds)}s: analyst would have "
        f"{plan.direction} tier={plan.entry_tier} size={plan.size_pct}% "
        f"(terminated_reason={loop_result.terminated_reason})"
    )


def run(args: Dict[str, Any]) -> SkillResult:
    decision_id = str(args.get("decision_id") or "").strip()
    if not decision_id:
        return SkillResult(
            ok=False,
            error="decision_id is required (find one via `query_recent_trades` tool or warm_store)",
        )

    decision = _load_decision(decision_id)
    if decision is None:
        return SkillResult(
            ok=False, error=f"decision {decision_id!r} not found in warm_store",
        )

    parts: List[str] = [_summarize_trace(decision)]

    # LLM post-mortem — optional per requested agents.
    agents = args.get("agents") or ["risk_guardian", "trend_momentum"]
    if isinstance(agents, str):
        agents = [a.strip() for a in agents.split(",") if a.strip()]
    llm_parts: Dict[str, Any] = {}
    for name in agents[:5]:
        ans = _llm_post_mortem(decision, name)
        if ans:
            parts.append(f"\n[{name}]\n{ans}")
            llm_parts[name] = ans

    # Optional what-if rewind.
    what_if_s = args.get("what_if_seconds")
    what_if_answer: Optional[str] = None
    if what_if_s is not None:
        try:
            what_if_answer = _what_if_rewind(decision, float(what_if_s))
        except Exception as e:
            what_if_answer = f"what_if failed: {type(e).__name__}: {e}"
        if what_if_answer:
            parts.append(f"\n{what_if_answer}")

    return SkillResult(
        ok=True,
        message="\n".join(parts),
        data={
            "decision": decision,
            "agent_post_mortems": llm_parts,
            "what_if": what_if_answer,
        },
    )
