"""
Training-data quality filter — "only the best gets fine-tuned".

For fine-tuning the dual-brain LLMs (C10), we do NOT want to train on
every trade ACT has ever made. Training on losers teaches the model to
reproduce losing behavior. Training on lucky wins teaches it to chase
noise. The filter here picks experience samples that:

  * The analyst's self-critique flagged as `matched_thesis=True`
    (the plan's prediction matched reality — not a lucky accident).
  * Had realized PnL above a minimum absolute threshold (drop noise).
  * For POSITIVE examples: pnl_pct > 0 — trained on as "do more of this".
  * For NEGATIVE examples (optional, DPO-style): pnl_pct < 0 with
    matched_thesis=False — trained on as "avoid this path".
  * Are recent (default: last 30 days) — stale market regimes have
    different dynamics.

Reads from the existing warm_store (decisions + outcomes + plan_json +
self_critique JSON) so no new storage. Outputs plain dicts ready for
either SFT (scanner+analyst) or DPO (preference-pair training).

Scanner training data — the input is the scanner's snapshot at tick T,
the target is its (cleaned-of-<think>) scan report. Filter to samples
where the analyst's downstream TradePlan led to a profitable outcome;
that way the scanner learns to flag setups that actually deliver.

Analyst training data — the input is {scanner report + quant data +
context}, the target is the TradePlan JSON the analyst emitted. Same
quality filter applied.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


DEFAULT_MAX_AGE_DAYS = float(os.getenv("ACT_TRAIN_MAX_AGE_DAYS", "30"))
DEFAULT_MIN_PNL_ABS_PCT = float(os.getenv("ACT_TRAIN_MIN_PNL_PCT", "0.3"))
DEFAULT_REQUIRE_MATCHED_THESIS = True


@dataclass
class ExperienceSample:
    """One labeled trade outcome, ready for fine-tune prep."""
    decision_id: str
    asset: str
    ts: float                           # exit_ts epoch seconds
    pnl_pct: float
    direction: str                      # 'LONG' | 'SHORT' | 'FLAT'
    matched_thesis: bool
    label: str                          # 'positive' | 'negative'
    plan: Dict[str, Any]                # full TradePlan dict
    outcome: Dict[str, Any]             # exit_price, duration_s, exit_reason, etc.
    self_critique: Dict[str, Any]       # verifier output
    scanner_tag: Dict[str, Any] = field(default_factory=dict)  # from component_signals

    def is_profit(self) -> bool:
        return self.pnl_pct > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "asset": self.asset,
            "ts": self.ts,
            "pnl_pct": round(self.pnl_pct, 4),
            "direction": self.direction,
            "matched_thesis": self.matched_thesis,
            "label": self.label,
            "plan": dict(self.plan),
            "outcome": dict(self.outcome),
            "self_critique": dict(self.self_critique),
            "scanner_tag": dict(self.scanner_tag),
        }


@dataclass
class FilterStats:
    """Summary of how much data passed each filter."""
    total: int = 0
    missing_plan: int = 0
    missing_critique: int = 0
    stale: int = 0
    below_min_pnl: int = 0
    thesis_unmatched_excluded: int = 0
    kept_positive: int = 0
    kept_negative: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "total": self.total,
            "missing_plan": self.missing_plan,
            "missing_critique": self.missing_critique,
            "stale": self.stale,
            "below_min_pnl": self.below_min_pnl,
            "thesis_unmatched_excluded": self.thesis_unmatched_excluded,
            "kept_positive": self.kept_positive,
            "kept_negative": self.kept_negative,
        }


# ── Core filter ─────────────────────────────────────────────────────────


def load_experience_samples(
    *,
    asset: Optional[str] = None,
    max_age_days: float = DEFAULT_MAX_AGE_DAYS,
    min_pnl_abs_pct: float = DEFAULT_MIN_PNL_ABS_PCT,
    require_matched_thesis: bool = DEFAULT_REQUIRE_MATCHED_THESIS,
    include_negatives: bool = True,
    db_path: Optional[str] = None,
    limit: int = 2000,
) -> Tuple[List[ExperienceSample], FilterStats]:
    """Join decisions + outcomes, filter, and return labeled samples.

    Shadow-mode rows (decision_id starts with 'shadow-') are skipped
    because they weren't executed — no ground-truth outcome to train on.
    """
    path = db_path or os.getenv(
        "ACT_WARM_DB_PATH",
        str(Path(__file__).resolve().parents[2] / "data" / "warm_store.sqlite"),
    )
    samples: List[ExperienceSample] = []
    stats = FilterStats()

    if not os.path.exists(path):
        logger.debug("training_data_filter: warm_store missing at %s", path)
        return samples, stats

    try:
        conn = sqlite3.connect(path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
    except Exception as e:
        logger.debug("training_data_filter: cannot open %s: %s", path, e)
        return samples, stats

    try:
        query = (
            "SELECT d.decision_id, d.symbol, d.plan_json, d.self_critique, "
            "d.component_signals, o.pnl_pct, o.exit_ts, o.direction, "
            "o.payload_json AS outcome_payload "
            "FROM decisions d JOIN outcomes o USING(decision_id) "
            "WHERE d.decision_id NOT LIKE 'shadow-%' "
        )
        params: List[Any] = []
        if asset:
            query += " AND d.symbol = ? "
            params.append(asset.upper())
        cutoff = time.time() - (max_age_days * 86400.0)
        query += " AND o.exit_ts >= ? ORDER BY o.exit_ts DESC LIMIT ?"
        params.extend([cutoff, int(limit)])
        rows = cur.execute(query, tuple(params)).fetchall()
    except Exception as e:
        logger.debug("training_data_filter: query failed: %s", e)
        try:
            conn.close()
        except Exception:
            pass
        return samples, stats
    finally:
        try:
            conn.close()
        except Exception:
            pass

    now = time.time()
    for row in rows:
        stats.total += 1
        try:
            plan = json.loads(row["plan_json"] or "{}") if row["plan_json"] else {}
        except Exception:
            plan = {}
        if not plan:
            stats.missing_plan += 1
            continue

        try:
            critique = json.loads(row["self_critique"] or "{}") if row["self_critique"] else {}
        except Exception:
            critique = {}
        # If there's no critique at all, we can't be sure the trade
        # matched thesis — skip unless the caller explicitly opts out.
        if not critique:
            stats.missing_critique += 1
            if require_matched_thesis:
                continue

        pnl = float(row["pnl_pct"] or 0.0)
        if abs(pnl) < min_pnl_abs_pct:
            stats.below_min_pnl += 1
            continue

        matched = bool(critique.get("matched_thesis", False))
        if require_matched_thesis and not matched:
            stats.thesis_unmatched_excluded += 1
            # Unmatched thesis CAN still be a useful negative example:
            # "the analyst thought X, but reality was Y". Keep only if
            # include_negatives AND pnl < 0 so the training loop can
            # use it as "avoid this path".
            if not (include_negatives and pnl < 0):
                continue

        label = "positive" if pnl > 0 and matched else "negative"
        if label == "negative" and not include_negatives:
            continue

        try:
            outcome_payload = json.loads(row["outcome_payload"] or "{}")
        except Exception:
            outcome_payload = {}
        try:
            comp_signals = json.loads(row["component_signals"] or "{}")
        except Exception:
            comp_signals = {}

        sample = ExperienceSample(
            decision_id=row["decision_id"],
            asset=(row["symbol"] or "").upper(),
            ts=float(row["exit_ts"] or now),
            pnl_pct=pnl,
            direction=str(row["direction"] or "").upper(),
            matched_thesis=matched,
            label=label,
            plan=plan,
            outcome=outcome_payload,
            self_critique=critique,
            scanner_tag=comp_signals,
        )
        samples.append(sample)
        if label == "positive":
            stats.kept_positive += 1
        else:
            stats.kept_negative += 1

    logger.info(
        "training_data_filter: kept %d positive + %d negative of %d total",
        stats.kept_positive, stats.kept_negative, stats.total,
    )
    return samples, stats


# ── Prompt-target formatting ────────────────────────────────────────────


def format_analyst_sft_example(sample: ExperienceSample) -> Dict[str, str]:
    """Shape one sample as {prompt, completion} for analyst SFT.

    Prompt: the seed context the analyst would have seen
    (scanner tag + asset + direction proposed).
    Completion: the TradePlan JSON the analyst emitted.
    """
    prompt_bits = [
        f"## Asset\n{sample.asset}",
        f"## Scanner tag\n{json.dumps(sample.scanner_tag, default=str)[:400]}",
        "## Task\nCompile a TradePlan (JSON). Ground every field in the context.",
    ]
    completion = json.dumps(sample.plan, default=str)
    return {"prompt": "\n\n".join(prompt_bits), "completion": completion}


def format_scanner_sft_example(sample: ExperienceSample) -> Optional[Dict[str, str]]:
    """Shape one sample as {prompt, completion} for scanner SFT.

    Only returns a row for positive samples — training a scanner to
    predict opportunities on losing trades teaches it the wrong thing.
    Returns None if the scanner_tag is missing (no scan report linked).
    """
    if sample.label != "positive":
        return None
    scan = sample.scanner_tag or {}
    if not scan:
        return None
    prompt = (
        f"## Asset\n{sample.asset}\n\n"
        f"## Task\nEmit a compact JSON scan report with fields: "
        f"opportunity_score (0-100), proposed_direction "
        f"(LONG/SHORT/FLAT), top_signals, rationale."
    )
    completion = json.dumps(
        {
            "opportunity_score": int(min(100, max(0, 50 + sample.pnl_pct * 10))),
            "proposed_direction": sample.direction,
            "top_signals": list(scan.get("top_signals") or [])[:5],
            "rationale": str(scan.get("rationale") or "")[:200],
        },
        default=str,
    )
    return {"prompt": prompt, "completion": completion}


def format_dpo_pairs(
    samples: Iterable[ExperienceSample],
) -> List[Dict[str, Any]]:
    """Build DPO preference pairs: positive as chosen, nearest matching
    negative (same asset, nearest timestamp) as rejected.

    DPO-style training is optional (Unsloth supports it). If no negative
    is available for a positive sample, we skip — no pair, no gradient.
    """
    pos: List[ExperienceSample] = []
    neg_by_asset: Dict[str, List[ExperienceSample]] = {}
    for s in samples:
        if s.label == "positive":
            pos.append(s)
        else:
            neg_by_asset.setdefault(s.asset, []).append(s)

    pairs: List[Dict[str, Any]] = []
    for p in pos:
        bucket = neg_by_asset.get(p.asset) or []
        if not bucket:
            continue
        nearest = min(bucket, key=lambda n: abs(n.ts - p.ts))
        pairs.append({
            "asset": p.asset,
            "prompt": format_analyst_sft_example(p)["prompt"],
            "chosen": json.dumps(p.plan, default=str),
            "rejected": json.dumps(nearest.plan, default=str),
            "chosen_pnl_pct": p.pnl_pct,
            "rejected_pnl_pct": nearest.pnl_pct,
        })
    return pairs
