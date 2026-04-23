"""Skill: /fine-tune-brain — operator-triggered dual-brain fine-tune cycle."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.skills.registry import SkillResult

logger = logging.getLogger(__name__)


def _resolve_backend(requested: Optional[str]):
    """Pick a TrainerBackend. Default: try Unsloth; fall back to stub."""
    if requested == "stub":
        from src.ai.dual_brain_trainer import StubBackend
        return StubBackend(), "stub"
    # Try to load a real backend. This module is optional — if unsloth
    # isn't installed (CI / laptop) we fall back to stub with a warning.
    try:
        from src.ai.unsloth_backend import UnslothQLoRABackend   # type: ignore
        return UnslothQLoRABackend(), "unsloth"
    except Exception as e:
        logger.info("unsloth backend unavailable (%s) — falling back to stub", e)
        from src.ai.dual_brain_trainer import StubBackend
        return StubBackend(), "stub"


def run(args: Dict[str, Any]) -> SkillResult:
    # Guard — destructive skill (hot-swaps active adapters) must confirm.
    if not args.get("confirm", False):
        return SkillResult(
            ok=False,
            error=(
                "pass confirm=true to run fine-tune-brain. "
                "This will pause the agentic loop, train new adapters, "
                "and hot-swap them if they pass the champion gate."
            ),
        )

    asset = args.get("asset")
    min_samples = int(args.get("min_samples") or 100)
    min_improvement_pct = float(args.get("min_improvement_pct") or 2.0)
    pause = bool(args.get("pause_agentic", True))
    backend_choice = args.get("backend")          # 'stub' to force dry-run
    dry_run = bool(args.get("dry_run", False))
    if dry_run:
        backend_choice = "stub"

    try:
        from src.ai.dual_brain_trainer import persist_report, run_cycle
        backend, backend_name = _resolve_backend(backend_choice)
        report = run_cycle(
            backend,
            asset=asset,
            min_samples=min_samples,
            min_improvement_pct=min_improvement_pct,
            pause_agentic=pause,
        )
    except Exception as e:
        return SkillResult(ok=False, error=f"{type(e).__name__}: {e}")

    path = persist_report(report)
    d = report.to_dict()
    d["backend"] = backend_name
    d["log_path"] = path

    parts = [
        f"Fine-tune cycle ({backend_name} backend) — duration {d['duration_s']}s",
        f"  filter_stats={report.filter_stats}",
    ]
    if d.get("error"):
        parts.append(f"  error: {d['error']}")
    if report.analyst:
        a = report.analyst
        verdict = ("promoted" if a.promoted else
                   ("rejected" if a.gate else "train_failed"))
        parts.append(f"  analyst:  {a.incumbent_model} -> {a.challenger_tag}  [{verdict}]")
        if a.gate and a.gate.reason:
            parts.append(f"            {a.gate.reason}")
    if report.scanner:
        s = report.scanner
        verdict = ("promoted" if s.promoted else
                   ("rejected" if s.gate else "train_failed"))
        parts.append(f"  scanner:  {s.incumbent_model} -> {s.challenger_tag}  [{verdict}]")
        if s.gate and s.gate.reason:
            parts.append(f"            {s.gate.reason}")
    if path:
        parts.append(f"  report:   {path}")

    return SkillResult(
        ok=not bool(d.get("error")), message="\n".join(parts), data=d,
    )
