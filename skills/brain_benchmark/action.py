"""Skill: /brain-benchmark — run Brain Quality Score measurement.

C26 Step 4. Loads labeled scenarios from warm_store + optional
curated file, runs ACT's unified brain + reference LLM on each,
scores both against the known outcome, writes a markdown report.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.skills.registry import SkillResult

logger = logging.getLogger(__name__)


def run(args: Optional[Dict[str, Any]] = None) -> SkillResult:
    args = dict(args or {})
    try:
        from src.evaluation.brain_benchmark import (
            run_brain_benchmark, write_benchmark_report,
            DEFAULT_REF_MODEL, DEFAULT_TARGET_SCORE,
        )
    except Exception as e:
        return SkillResult(ok=False, error=f"benchmark module import failed: {e}")

    ref_model = str(args.get("ref_model") or DEFAULT_REF_MODEL)
    try:
        max_scenarios = int(args.get("max_scenarios") or 50)
    except (TypeError, ValueError):
        max_scenarios = 50
    max_scenarios = max(1, min(500, max_scenarios))

    result = run_brain_benchmark(ref_model=ref_model, max_scenarios=max_scenarios)
    report_path = write_benchmark_report(result)

    return SkillResult(
        ok=True,
        message=(
            f"Brain Quality Score: {result.brain_quality_score:.4f} "
            f"(target: {DEFAULT_TARGET_SCORE}) — "
            f"{'✅ BEATS target' if result.beats_target() else 'below target'}. "
            f"{result.n_scenarios} scenarios scored. Report: {report_path.name}"
        ),
        data={
            "brain_quality_score": round(result.brain_quality_score, 4),
            "target": DEFAULT_TARGET_SCORE,
            "beats_target": result.beats_target(),
            "n_scenarios": result.n_scenarios,
            "local_wins": result.local_wins,
            "ref_wins": result.ref_wins,
            "ties": result.ties,
            "ref_model": ref_model,
            "by_regime": result.by_regime,
            "report_path": str(report_path),
        },
    )
