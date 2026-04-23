"""Skill: /readiness — print the readiness-gate state + emergency flag."""
from __future__ import annotations

from typing import Any, Dict

from src.skills.registry import SkillResult


def run(args: Dict[str, Any]) -> SkillResult:
    try:
        from src.orchestration.readiness_gate import (
            evaluate,
            format_report,
            is_emergency_mode,
        )
        state = evaluate()
        report = format_report(state)
        emergency = is_emergency_mode(state)
    except Exception as e:
        return SkillResult(ok=False, error=f"{type(e).__name__}: {e}")

    extra = f"Emergency mode: {'ON' if emergency else 'off'}"
    return SkillResult(
        ok=True,
        message=f"{report}\n  {extra}",
        data={
            "gate": state.to_dict(),
            "emergency_mode": bool(emergency),
        },
    )
