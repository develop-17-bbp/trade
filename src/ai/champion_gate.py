"""
Champion gate — "only the best gets promoted" validation for fine-tunes.

After training a new LoRA adapter on quality-filtered experience
(src/ai/training_data_filter.py), we do NOT hot-swap it into production
without first proving it beats the incumbent on a held-out validation
set. This module is the gatekeeper.

Design:
  * Inputs: (incumbent_model_id, challenger_model_id, validation_set)
    plus a callable that runs a model against one validation item and
    returns a structured output.
  * Outputs: a ChampionGateResult with per-metric deltas + pass/fail.
  * Metrics (for analyst):
      - schema_valid_rate — did the output parse as TradePlan?
      - direction_agreement — did the proposed direction match the
        profitable direction from the validation-set outcome?
      - size_reasonableness — was proposed size within 2x of the
        validation trade's size?
      - no_worse_on_any_metric_by_more_than 5% — sanity check
  * Metrics (for scanner):
      - schema_valid_rate
      - direction_agreement (scanner's proposed_direction vs outcome)
      - opportunity_score_calibration — correlation between score and
        realized PnL on validation set (Spearman ρ)

A challenger must beat incumbent by `min_improvement_pct` (default 2%)
on the primary metric AND not lose by > 5% on any single metric. If
either condition fails, the challenger is rejected; the incumbent
stays. Rejected challengers aren't thrown away — they're kept as
`<name>:act-<ts>-rejected` tags in Ollama for post-hoc analysis.

Zero ML deps — this module is pure aggregation logic. The actual
inference call is injected so tests don't need a GPU.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


DEFAULT_MIN_IMPROVEMENT_PCT = 2.0
DEFAULT_MAX_REGRESSION_PCT = 5.0


@dataclass
class MetricScore:
    """One metric's score + confidence."""
    name: str
    incumbent: float
    challenger: float

    @property
    def delta_pct(self) -> float:
        if abs(self.incumbent) < 1e-9:
            return 0.0 if abs(self.challenger) < 1e-9 else 100.0
        return 100.0 * (self.challenger - self.incumbent) / abs(self.incumbent)

    def to_dict(self) -> Dict[str, float]:
        return {
            "name": self.name,
            "incumbent": round(self.incumbent, 4),
            "challenger": round(self.challenger, 4),
            "delta_pct": round(self.delta_pct, 2),
        }


@dataclass
class ChampionGateResult:
    """Full verdict: promote or reject, with numbers behind the decision."""
    brain: str                                    # 'analyst' | 'scanner'
    incumbent_id: str
    challenger_id: str
    primary_metric: str                           # the metric used as the pass/fail driver
    promote: bool
    reason: str                                   # one-line explanation
    metrics: List[MetricScore] = field(default_factory=list)
    min_improvement_pct: float = DEFAULT_MIN_IMPROVEMENT_PCT
    max_regression_pct: float = DEFAULT_MAX_REGRESSION_PCT
    validation_size: int = 0
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brain": self.brain,
            "incumbent_id": self.incumbent_id,
            "challenger_id": self.challenger_id,
            "primary_metric": self.primary_metric,
            "promote": bool(self.promote),
            "reason": self.reason,
            "metrics": [m.to_dict() for m in self.metrics],
            "min_improvement_pct": self.min_improvement_pct,
            "max_regression_pct": self.max_regression_pct,
            "validation_size": int(self.validation_size),
            "ts": self.ts,
        }


# ── Per-brain scoring (pure; no ML) ─────────────────────────────────────


def score_analyst_output(
    model_output: str,
    ground_truth_sample: Dict[str, Any],
) -> Dict[str, float]:
    """Score one analyst output against one held-out sample.

    Returns per-metric floats in [0.0, 1.0]. Callers accumulate over the
    full validation set and compute means.

    `ground_truth_sample` is a dict shaped like
    `ExperienceSample.to_dict()` (see training_data_filter.py).
    """
    metrics = {
        "schema_valid": 0.0,
        "direction_agreement": 0.0,
        "size_reasonableness": 0.0,
    }
    try:
        obj = json.loads(model_output) if isinstance(model_output, str) else model_output
        if not isinstance(obj, dict):
            return metrics
    except Exception:
        return metrics

    # Extract the plan (support both bare plan + envelope {"plan": {...}}).
    plan = obj.get("plan") if isinstance(obj.get("plan"), dict) else obj
    if not isinstance(plan, dict) or "direction" not in plan:
        return metrics

    metrics["schema_valid"] = 1.0
    gt_direction = str(ground_truth_sample.get("direction", "")).upper()
    predicted = str(plan.get("direction", "")).upper()
    if gt_direction and predicted == gt_direction:
        metrics["direction_agreement"] = 1.0

    gt_size = (ground_truth_sample.get("plan") or {}).get("size_pct") or 0.0
    predicted_size = plan.get("size_pct") or 0.0
    try:
        gt_size_f = float(gt_size)
        pred_size_f = float(predicted_size)
        if gt_size_f > 0 and pred_size_f > 0:
            ratio = pred_size_f / gt_size_f
            # Reasonable window: 0.5× to 2× the ground-truth size.
            if 0.5 <= ratio <= 2.0:
                metrics["size_reasonableness"] = 1.0
    except Exception:
        pass
    return metrics


def score_scanner_output(
    model_output: str,
    ground_truth_sample: Dict[str, Any],
) -> Dict[str, float]:
    """Score one scanner output against one sample.

    Returns per-metric floats. We test direction-agreement plus, if
    multiple samples are later aggregated, the caller can compute
    opportunity_score / pnl rank correlation.
    """
    metrics = {"schema_valid": 0.0, "direction_agreement": 0.0}
    try:
        obj = json.loads(model_output) if isinstance(model_output, str) else model_output
        if not isinstance(obj, dict):
            return metrics
    except Exception:
        return metrics
    metrics["schema_valid"] = 1.0
    gt_direction = str(ground_truth_sample.get("direction", "")).upper()
    predicted = str(obj.get("proposed_direction", "")).upper()
    if gt_direction and predicted == gt_direction:
        metrics["direction_agreement"] = 1.0
    return metrics


# ── Validation runner ───────────────────────────────────────────────────


def run_validation(
    model_id: str,
    samples: List[Dict[str, Any]],
    *,
    brain: str,
    inference_fn: Callable[[str, Dict[str, Any]], str],
) -> Dict[str, float]:
    """Run `model_id` against each validation sample, accumulate scores.

    `inference_fn(model_id, sample)` → the model's raw string output.
    Tests inject a stub; production backend calls Ollama / vLLM.

    Returns averaged metrics {name -> [0.0, 1.0]}.
    """
    if not samples:
        return {}
    scorer = score_analyst_output if brain == "analyst" else score_scanner_output
    accum: Dict[str, float] = {}
    for s in samples:
        try:
            out = inference_fn(model_id, s) or ""
        except Exception as e:
            logger.debug("champion_gate: inference %s sample error: %s", model_id, e)
            out = ""
        per = scorer(out, s)
        for k, v in per.items():
            accum[k] = accum.get(k, 0.0) + float(v)
    n = float(len(samples))
    return {k: v / n for k, v in accum.items()}


# ── Public: the gate itself ─────────────────────────────────────────────


def evaluate_gate(
    brain: str,
    incumbent_id: str,
    challenger_id: str,
    validation_samples: List[Dict[str, Any]],
    *,
    inference_fn: Callable[[str, Dict[str, Any]], str],
    primary_metric: Optional[str] = None,
    min_improvement_pct: float = DEFAULT_MIN_IMPROVEMENT_PCT,
    max_regression_pct: float = DEFAULT_MAX_REGRESSION_PCT,
) -> ChampionGateResult:
    """Run validation for both models, apply gate rules, return verdict.

    Never raises — a failed inference for one model just produces
    lower scores for that model, not an exception.
    """
    primary = primary_metric or (
        "direction_agreement" if brain == "analyst" else "direction_agreement"
    )

    inc = run_validation(incumbent_id, validation_samples,
                         brain=brain, inference_fn=inference_fn)
    cha = run_validation(challenger_id, validation_samples,
                         brain=brain, inference_fn=inference_fn)

    # Merge into MetricScore list; sort so primary metric appears first.
    all_names = sorted(set(inc.keys()) | set(cha.keys()),
                       key=lambda n: (n != primary, n))
    metrics = [
        MetricScore(name=n, incumbent=inc.get(n, 0.0), challenger=cha.get(n, 0.0))
        for n in all_names
    ]

    if not metrics:
        return ChampionGateResult(
            brain=brain, incumbent_id=incumbent_id, challenger_id=challenger_id,
            primary_metric=primary, promote=False,
            reason="no metrics computed (empty validation set?)",
            validation_size=len(validation_samples),
            min_improvement_pct=min_improvement_pct,
            max_regression_pct=max_regression_pct,
        )

    primary_score = next((m for m in metrics if m.name == primary), metrics[0])

    # Rule 1: challenger must beat incumbent by min_improvement_pct on primary.
    if primary_score.delta_pct < min_improvement_pct:
        return ChampionGateResult(
            brain=brain, incumbent_id=incumbent_id, challenger_id=challenger_id,
            primary_metric=primary, promote=False,
            reason=(f"challenger delta {primary_score.delta_pct:.1f}% on "
                    f"{primary} < required {min_improvement_pct:.1f}%"),
            metrics=metrics,
            validation_size=len(validation_samples),
            min_improvement_pct=min_improvement_pct,
            max_regression_pct=max_regression_pct,
        )

    # Rule 2: no single metric regresses by more than max_regression_pct.
    for m in metrics:
        if m.delta_pct < -max_regression_pct:
            return ChampionGateResult(
                brain=brain, incumbent_id=incumbent_id, challenger_id=challenger_id,
                primary_metric=primary, promote=False,
                reason=(f"challenger regresses {m.delta_pct:.1f}% on {m.name} "
                        f"(max allowed {-max_regression_pct:.1f}%)"),
                metrics=metrics,
                validation_size=len(validation_samples),
                min_improvement_pct=min_improvement_pct,
                max_regression_pct=max_regression_pct,
            )

    # All rules passed — promote.
    return ChampionGateResult(
        brain=brain, incumbent_id=incumbent_id, challenger_id=challenger_id,
        primary_metric=primary, promote=True,
        reason=(f"challenger beats incumbent by {primary_score.delta_pct:.1f}% "
                f"on {primary} with no regression > {max_regression_pct:.1f}%"),
        metrics=metrics,
        validation_size=len(validation_samples),
        min_improvement_pct=min_improvement_pct,
        max_regression_pct=max_regression_pct,
    )
