"""Skill: /emergency-flatten — halt agentic loop + raise emergency mode."""
from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict

from src.skills.registry import SkillResult


def run(args: Dict[str, Any]) -> SkillResult:
    # Operator sanity check — this is destructive, so requires an explicit
    # confirm=true in args. The registry enforces this at dispatch for
    # LLM-invoked calls, but we double-check here for operator calls too.
    if not args.get("confirm", False):
        return SkillResult(
            ok=False,
            error="pass confirm=true to acknowledge emergency-flatten side effects",
        )

    reason = str(args.get("reason") or "operator-triggered")
    incident_id = f"incident-{uuid.uuid4().hex[:12]}"
    actions_taken = []

    # 1. Halt the agentic loop immediately.
    os.environ["ACT_DISABLE_AGENTIC_LOOP"] = "1"
    actions_taken.append("ACT_DISABLE_AGENTIC_LOOP=1")

    # 2. Raise emergency mode so scheduler halves retrain intervals.
    try:
        from src.orchestration.readiness_gate import publish_emergency_mode
        publish_emergency_mode(True)
        actions_taken.append("emergency_mode=on")
    except Exception as e:
        actions_taken.append(f"emergency_mode_failed: {type(e).__name__}")

    # 3. Audit: write an incident row to warm_store.
    try:
        from src.orchestration.warm_store import get_store
        store = get_store()
        store.write_decision({
            "decision_id": incident_id,
            "symbol": "INCIDENT",
            "ts_ns": time.time_ns(),
            "final_action": "EMERGENCY_FLATTEN",
            "component_signals": {
                "source": "skill:emergency-flatten",
                "reason": reason,
                "actions_taken": actions_taken,
            },
        })
        actions_taken.append(f"warm_store_logged={incident_id}")
    except Exception as e:
        actions_taken.append(f"warm_store_failed: {type(e).__name__}")

    return SkillResult(
        ok=True,
        message=(
            f"Emergency flatten engaged. incident={incident_id} reason={reason}. "
            "Agentic loop halted + emergency mode on. "
            "Open positions continue to be managed by the executor's "
            "stops/TPs — manually close via the broker if needed."
        ),
        data={
            "incident_id": incident_id,
            "reason": reason,
            "actions_taken": actions_taken,
        },
    )
