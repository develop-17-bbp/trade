"""
Dual-brain trainer orchestrator — C10 wiring.

Coordinates the end-to-end fine-tune pipeline:

  1. training_data_filter.load_experience_samples()  — pull + filter
  2. split into train / validation (80/20 by timestamp)
  3. pause the agentic loop (ACT_DISABLE_AGENTIC_LOOP=1)
  4. train analyst adapter (heavy; ~30-45 min)
  5. train scanner adapter (~5-10 min)
  6. champion_gate.evaluate_gate for each brain
  7. promote or reject; hot-swap via Ollama if promoted
  8. resume agentic loop

The actual QLoRA training call is abstracted behind a TrainerBackend
protocol. Tests inject a stub; production code plugs in Unsloth.
That way this module is CPU-testable + GPU-only work isolated.

VRAM sequencing enforced: analyst and scanner train ONE AT A TIME.
Never attempt to load both 32B-class models for training simultaneously
on a 32 GB card.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from src.ai.champion_gate import (
    ChampionGateResult,
    DEFAULT_MAX_REGRESSION_PCT,
    DEFAULT_MIN_IMPROVEMENT_PCT,
    evaluate_gate,
)
from src.ai.training_data_filter import (
    DEFAULT_MAX_AGE_DAYS,
    DEFAULT_MIN_PNL_ABS_PCT,
    ExperienceSample,
    FilterStats,
    format_analyst_sft_example,
    format_scanner_sft_example,
    load_experience_samples,
)

logger = logging.getLogger(__name__)


DEFAULT_MIN_SAMPLES = int(os.getenv("ACT_FT_MIN_SAMPLES", "100"))
DEFAULT_VALIDATION_SPLIT = 0.2
DISABLE_AGENTIC_ENV = "ACT_DISABLE_AGENTIC_LOOP"


@dataclass
class TrainingJob:
    """Config for one per-brain training run."""
    brain: str                                # 'analyst' | 'scanner'
    incumbent_model: str                      # e.g. 'deepseek-r1:32b'
    challenger_tag: str                       # e.g. 'deepseek-r1:32b-act-1712345'
    train_samples: List[Dict[str, Any]]       # SFT rows
    validation_samples: List[Dict[str, Any]]  # held-out


@dataclass
class TrainerResult:
    """Outcome of one brain's training + gate evaluation."""
    brain: str
    incumbent_model: str
    challenger_tag: str
    training_ok: bool
    gate: Optional[ChampionGateResult] = None
    promoted: bool = False
    swap_error: Optional[str] = None
    train_samples: int = 0
    validation_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brain": self.brain,
            "incumbent_model": self.incumbent_model,
            "challenger_tag": self.challenger_tag,
            "training_ok": bool(self.training_ok),
            "promoted": bool(self.promoted),
            "swap_error": self.swap_error,
            "gate": (self.gate.to_dict() if self.gate else None),
            "train_samples": int(self.train_samples),
            "validation_samples": int(self.validation_samples),
        }


@dataclass
class CycleReport:
    """Full fine-tune cycle result — one analyst + one scanner run."""
    started_at: float
    finished_at: float
    filter_stats: Dict[str, Any]
    analyst: Optional[TrainerResult] = None
    scanner: Optional[TrainerResult] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round(self.finished_at - self.started_at, 1),
            "filter_stats": dict(self.filter_stats),
            "analyst": self.analyst.to_dict() if self.analyst else None,
            "scanner": self.scanner.to_dict() if self.scanner else None,
            "error": self.error,
        }


# ── Backend protocol ───────────────────────────────────────────────────


class TrainerBackend(Protocol):
    """Pluggable backend for the actual ML call. Implementations:

    * `UnslothQLoRABackend` (production; requires GPU + unsloth).
    * `StubBackend` (tests; returns canned outputs).
    """

    def train(self, base_model: str, sft_rows: List[Dict[str, str]],
              out_tag: str) -> bool:
        """Fine-tune `base_model` on `sft_rows`, save as `out_tag` in
        the local Ollama registry. Returns True on success. Never
        raises — exceptions are caught and reported as False."""
        ...

    def infer(self, model_id: str, sample: Dict[str, Any]) -> str:
        """Run one validation inference. Used by champion_gate."""
        ...


# ── Stub for tests (and for dry-run without a GPU) ─────────────────────


class StubBackend:
    """Deterministic stub — records calls, returns canned outputs.

    Produces DIFFERENT outputs for incumbent vs challenger so
    champion-gate tests have something to distinguish.
    """

    def __init__(self, train_ok: bool = True,
                 incumbent_direction_match: bool = False,
                 challenger_direction_match: bool = True):
        self.train_ok = train_ok
        self.incumbent_direction_match = incumbent_direction_match
        self.challenger_direction_match = challenger_direction_match
        self.trained_tags: List[str] = []

    def train(self, base_model: str, sft_rows, out_tag: str) -> bool:
        self.trained_tags.append(out_tag)
        return self.train_ok

    def infer(self, model_id: str, sample: Dict[str, Any]) -> str:
        want = str(sample.get("direction", "LONG")).upper()
        match = self.challenger_direction_match if model_id in self.trained_tags \
            else self.incumbent_direction_match
        direction = want if match else ("SHORT" if want == "LONG" else "LONG")
        return json.dumps({"direction": direction, "size_pct": 5.0,
                           "proposed_direction": direction})


# ── Validation-set split ───────────────────────────────────────────────


def split_samples(
    samples: List[ExperienceSample],
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
) -> tuple[List[ExperienceSample], List[ExperienceSample]]:
    """Split by timestamp — newest N% become validation (the most
    market-relevant holdout)."""
    if not samples:
        return [], []
    # Sort oldest-first so the last slice is the most recent.
    ordered = sorted(samples, key=lambda s: s.ts)
    split_idx = max(1, int(len(ordered) * (1.0 - validation_split)))
    return ordered[:split_idx], ordered[split_idx:]


# ── Core cycle ─────────────────────────────────────────────────────────


def run_cycle(
    backend: TrainerBackend,
    *,
    asset: Optional[str] = None,
    max_age_days: float = DEFAULT_MAX_AGE_DAYS,
    min_pnl_abs_pct: float = DEFAULT_MIN_PNL_ABS_PCT,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    min_improvement_pct: float = DEFAULT_MIN_IMPROVEMENT_PCT,
    max_regression_pct: float = DEFAULT_MAX_REGRESSION_PCT,
    pause_agentic: bool = True,
    analyst_incumbent: Optional[str] = None,
    scanner_incumbent: Optional[str] = None,
    brains: Optional[List[str]] = None,
) -> CycleReport:
    """Run one full fine-tune cycle: filter → split → train analyst →
    train scanner → gate both → promote or reject. Never raises.

    `brains` filters which brains to train. None = both. Used by the
    two-box deploy: 5090 nightly passes ['analyst'], 4060 every-3h
    passes ['scanner'].
    """
    do_analyst = (brains is None) or ("analyst" in brains)
    do_scanner = (brains is None) or ("scanner" in brains)
    started = time.time()
    prev_disable_env = os.environ.get(DISABLE_AGENTIC_ENV)
    try:
        if pause_agentic:
            os.environ[DISABLE_AGENTIC_ENV] = "1"

        samples, stats = load_experience_samples(
            asset=asset, max_age_days=max_age_days,
            min_pnl_abs_pct=min_pnl_abs_pct,
        )
        filter_stats = stats.to_dict() if isinstance(stats, FilterStats) else {}
        if len(samples) < min_samples:
            return CycleReport(
                started_at=started, finished_at=time.time(),
                filter_stats=filter_stats,
                error=(f"only {len(samples)} filtered samples "
                       f"< min_samples {min_samples}; skipping cycle"),
            )

        train_set, val_set = split_samples(samples)

        # Resolve per-brain incumbent if caller didn't specify.
        if analyst_incumbent is None or scanner_incumbent is None:
            try:
                from src.ai.dual_brain import ANALYST, SCANNER, _resolve
                analyst_incumbent = analyst_incumbent or _resolve(None, ANALYST).model
                scanner_incumbent = scanner_incumbent or _resolve(None, SCANNER).model
            except Exception as e:
                return CycleReport(
                    started_at=started, finished_at=time.time(),
                    filter_stats=filter_stats,
                    error=f"dual_brain resolve failed: {e}",
                )

        ts = int(time.time())

        analyst_result: Optional[TrainerResult] = None
        scanner_result: Optional[TrainerResult] = None

        # ── Analyst first (heavier, more important) ────────────────────
        if do_analyst:
            analyst_result = _train_one_brain(
                backend, brain="analyst",
                incumbent=analyst_incumbent,
                challenger_tag=f"{analyst_incumbent}-act-{ts}",
                train_set=train_set, val_set=val_set,
                format_fn=format_analyst_sft_example,
                min_improvement_pct=min_improvement_pct,
                max_regression_pct=max_regression_pct,
            )

        # ── Scanner second ────────────────────────────────────────────
        if do_scanner:
            scanner_result = _train_one_brain(
                backend, brain="scanner",
                incumbent=scanner_incumbent,
                challenger_tag=f"{scanner_incumbent}-act-{ts}",
                train_set=train_set, val_set=val_set,
                format_fn=_format_scanner_safe,
                min_improvement_pct=min_improvement_pct,
                max_regression_pct=max_regression_pct,
            )

        return CycleReport(
            started_at=started, finished_at=time.time(),
            filter_stats=filter_stats,
            analyst=analyst_result, scanner=scanner_result,
        )
    finally:
        if pause_agentic:
            if prev_disable_env is None:
                os.environ.pop(DISABLE_AGENTIC_ENV, None)
            else:
                os.environ[DISABLE_AGENTIC_ENV] = prev_disable_env


def _format_scanner_safe(sample: ExperienceSample) -> Optional[Dict[str, str]]:
    """Scanner format returns None on non-positive samples; caller skips."""
    return format_scanner_sft_example(sample)


def _train_one_brain(
    backend: TrainerBackend,
    *,
    brain: str,
    incumbent: str,
    challenger_tag: str,
    train_set: List[ExperienceSample],
    val_set: List[ExperienceSample],
    format_fn: Callable[[ExperienceSample], Optional[Dict[str, str]]],
    min_improvement_pct: float,
    max_regression_pct: float,
) -> TrainerResult:
    """Run one brain's full train + gate + optional promote."""
    sft_rows: List[Dict[str, str]] = []
    for s in train_set:
        row = format_fn(s)
        if row is not None:
            sft_rows.append(row)

    val_dicts = [s.to_dict() for s in val_set]
    result = TrainerResult(
        brain=brain, incumbent_model=incumbent,
        challenger_tag=challenger_tag, training_ok=False,
        train_samples=len(sft_rows), validation_samples=len(val_dicts),
    )

    if len(sft_rows) < 10:
        result.swap_error = f"insufficient train rows after format ({len(sft_rows)})"
        return result

    try:
        result.training_ok = bool(backend.train(incumbent, sft_rows, challenger_tag))
    except Exception as e:
        logger.debug("%s backend.train raised: %s", brain, e)
        result.training_ok = False
    if not result.training_ok:
        return result

    try:
        result.gate = evaluate_gate(
            brain=brain, incumbent_id=incumbent, challenger_id=challenger_tag,
            validation_samples=val_dicts,
            inference_fn=backend.infer,
            min_improvement_pct=min_improvement_pct,
            max_regression_pct=max_regression_pct,
        )
    except Exception as e:
        logger.debug("%s champion_gate raised: %s", brain, e)
        result.swap_error = f"gate_error: {type(e).__name__}: {e}"
        return result

    if result.gate and result.gate.promote:
        swap_err = _hot_swap(brain, challenger_tag)
        if swap_err:
            result.swap_error = swap_err
        else:
            result.promoted = True
    return result


def _hot_swap(brain: str, challenger_tag: str) -> Optional[str]:
    """Set the env var so the next agentic cycle uses the new adapter.

    Does NOT mutate config.yaml — env override wins. Operator can pin
    via config.yaml later after confirming the swap behaves in paper.
    Returns error string on failure (caller records it), None on OK.
    """
    env_key = "ACT_ANALYST_MODEL" if brain == "analyst" else "ACT_SCANNER_MODEL"
    try:
        os.environ[env_key] = challenger_tag
    except Exception as e:
        return f"env_set_failed: {e}"
    # setx for Windows persistence (so the flag survives restart).
    try:
        if os.name == "nt":
            subprocess.run(["setx", env_key, challenger_tag],
                           check=False, capture_output=True, timeout=10.0)
    except Exception as e:
        logger.debug("setx %s failed (in-process env still set): %s", env_key, e)
    return None


# ── Audit log ──────────────────────────────────────────────────────────


def persist_report(report: CycleReport, out_dir: Optional[str] = None) -> Optional[str]:
    """Write the cycle report to `logs/fine_tune/<timestamp>.json`.
    Returns the path, or None on failure."""
    try:
        out_dir = out_dir or str(
            Path(__file__).resolve().parents[2] / "logs" / "fine_tune"
        )
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        path = Path(out_dir) / f"cycle-{int(report.started_at)}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        return str(path)
    except Exception as e:
        logger.debug("persist_report failed: %s", e)
        return None
