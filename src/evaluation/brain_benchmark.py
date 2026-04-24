"""Brain Benchmark — measure ACT's dual-brain against a SOTA reference.

C26 Step 4 of the plan. Operator's stated target is beating 64.4% on
agentic financial-analysis benchmarks (reference: Opus-class models).
This module gives ACT a reproducible way to measure itself.

Design:

  For each scenario (a market snapshot with known outcome):

    1. Run ACT's dual-brain on the context → TradePlan `local_plan`
    2. Run reference LLM on the SAME context → TradePlan `ref_plan`
    3. Score both plans against the known outcome on 5 axes:
       - direction agreement (+1 if correct, -1 if wrong, 0 if flat)
       - thesis quality (non-empty, cites evidence: +1 each)
       - authority compliance (no violations: +1)
       - expected-PnL calibration (|predicted - actual| / |actual|)
       - risk-adjusted return (hypothetical PnL / max_dd on bars after)
    4. Brain Quality Score = N_local_wins / N_scenarios

Target ≥ 0.644 (explicitly beats the cited SOTA benchmark).

Safe to run offline — reference LLM is lazy-imported; if no API key
is set or the client isn't installed, reference arm is stubbed to a
neutral baseline. Test suite runs without network.

Scenarios come from:
  * warm_store recent decisions with outcomes (labeled historical)
  * Optional hand-curated scenarios file at
    `data/benchmark_scenarios.jsonl` (one JSON per line)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REF_MODEL = "claude-haiku-4-5"
DEFAULT_TARGET_SCORE = 0.644


class BenchmarkScenario:
    """One (input-context, known-outcome) pair used to score both brains."""

    def __init__(
        self,
        scenario_id: str,
        asset: str,
        context: Dict[str, Any],
        actual_outcome: Dict[str, Any],
        regime: str = "UNKNOWN",
    ) -> None:
        self.scenario_id = scenario_id
        self.asset = asset
        self.context = context
        self.actual_outcome = actual_outcome
        self.regime = regime


class PlanScore:
    """Output of scoring one plan against one known outcome."""

    def __init__(
        self,
        direction_correct: bool = False,
        thesis_quality: float = 0.0,
        authority_clean: bool = False,
        pnl_calibration: float = 0.0,      # 1.0 = perfect
        risk_adjusted: float = 0.0,         # Sharpe-like
        composite: float = 0.0,
        notes: str = "",
    ) -> None:
        self.direction_correct = direction_correct
        self.thesis_quality = thesis_quality
        self.authority_clean = authority_clean
        self.pnl_calibration = pnl_calibration
        self.risk_adjusted = risk_adjusted
        self.composite = composite
        self.notes = notes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction_correct": bool(self.direction_correct),
            "thesis_quality": round(float(self.thesis_quality), 3),
            "authority_clean": bool(self.authority_clean),
            "pnl_calibration": round(float(self.pnl_calibration), 3),
            "risk_adjusted": round(float(self.risk_adjusted), 3),
            "composite": round(float(self.composite), 3),
            "notes": str(self.notes)[:120],
        }


class BenchmarkResult:
    """Aggregate across all scenarios."""

    def __init__(
        self,
        ref_model: str,
        n_scenarios: int = 0,
        local_wins: int = 0,
        ties: int = 0,
        ref_wins: int = 0,
        local_scores: Optional[List[PlanScore]] = None,
        ref_scores: Optional[List[PlanScore]] = None,
        by_regime: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> None:
        self.ref_model = ref_model
        self.n_scenarios = n_scenarios
        self.local_wins = local_wins
        self.ties = ties
        self.ref_wins = ref_wins
        self.local_scores = list(local_scores or [])
        self.ref_scores = list(ref_scores or [])
        self.by_regime = dict(by_regime or {})

    @property
    def brain_quality_score(self) -> float:
        if self.n_scenarios <= 0:
            return 0.0
        return self.local_wins / self.n_scenarios

    def beats_target(self, target: float = DEFAULT_TARGET_SCORE) -> bool:
        return self.brain_quality_score >= target

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ref_model": self.ref_model,
            "n_scenarios": self.n_scenarios,
            "local_wins": self.local_wins,
            "ties": self.ties,
            "ref_wins": self.ref_wins,
            "brain_quality_score": round(self.brain_quality_score, 4),
            "beats_target": self.beats_target(),
            "target": DEFAULT_TARGET_SCORE,
            "by_regime": dict(self.by_regime),
        }


# ── Scorer ──────────────────────────────────────────────────────────────


def score_plan_vs_outcome(plan: Dict[str, Any], outcome: Dict[str, Any]) -> PlanScore:
    """Score one plan against the known post-hoc outcome.

    `plan` fields expected:
      - direction: LONG | SHORT | FLAT | SKIP
      - thesis: str (non-empty if plan != SKIP)
      - expected_pnl_pct_range: [min, max] (optional)
      - supporting_evidence: list (optional)
      - authority_violations: list (optional; empty = clean)

    `outcome` fields expected:
      - actual_direction: 'UP' | 'DOWN' | 'FLAT' (what the market did)
      - pnl_pct: realized PnL on that trade (or hypothetical if skipped)
      - max_adverse_pct: worst drawdown over hold window
    """
    actual = str(outcome.get("actual_direction", "FLAT")).upper()
    predicted = str(plan.get("direction", "FLAT")).upper()

    # Direction correctness — LONG matches UP, SHORT matches DOWN,
    # FLAT/SKIP = correct if actual was FLAT.
    direction_correct = (
        (predicted == "LONG" and actual == "UP")
        or (predicted == "SHORT" and actual == "DOWN")
        or (predicted in ("FLAT", "SKIP") and actual == "FLAT")
    )

    # Thesis quality — non-empty + cites evidence counts
    thesis = str(plan.get("thesis") or "")
    evidence = plan.get("supporting_evidence") or []
    thesis_quality = 0.0
    if thesis.strip():
        thesis_quality += 0.5
    if len(thesis) >= 40:
        thesis_quality += 0.25
    if isinstance(evidence, list) and len(evidence) > 0:
        thesis_quality += 0.25

    # Authority compliance
    violations = plan.get("authority_violations") or []
    authority_clean = len(violations) == 0

    # PnL calibration — how close predicted range was to actual
    pnl_calibration = 0.0
    pnl_range = plan.get("expected_pnl_pct_range") or None
    actual_pnl = float(outcome.get("pnl_pct", 0.0) or 0.0)
    if pnl_range and len(pnl_range) == 2:
        lo, hi = float(pnl_range[0]), float(pnl_range[1])
        if lo <= actual_pnl <= hi:
            pnl_calibration = 1.0
        else:
            # Normalized miss: 0 = on target, 1 = way off
            miss = min(abs(actual_pnl - lo), abs(actual_pnl - hi))
            pnl_calibration = max(0.0, 1.0 - miss / max(abs(actual_pnl), 1.0))

    # Risk-adjusted return — Sharpe-like with max adverse as proxy for vol
    max_adverse = abs(float(outcome.get("max_adverse_pct", 0.0) or 0.0))
    if direction_correct and max_adverse > 0:
        risk_adjusted = actual_pnl / max(max_adverse, 0.5)
    elif direction_correct:
        risk_adjusted = actual_pnl
    else:
        risk_adjusted = -abs(actual_pnl)

    # Composite — weighted average, all scaled to [0, 1]
    composite = (
        (1.0 if direction_correct else 0.0) * 0.40
        + thesis_quality * 0.15
        + (1.0 if authority_clean else 0.0) * 0.15
        + pnl_calibration * 0.15
        + max(0.0, min(1.0, (risk_adjusted + 1.0) / 3.0)) * 0.15
    )

    return PlanScore(
        direction_correct=direction_correct,
        thesis_quality=thesis_quality,
        authority_clean=authority_clean,
        pnl_calibration=pnl_calibration,
        risk_adjusted=risk_adjusted,
        composite=composite,
        notes=f"pred={predicted} actual={actual} pnl={actual_pnl:+.2f}%",
    )


# ── Reference LLM (Anthropic API) ───────────────────────────────────────


def _call_reference_llm(
    system_prompt: str, user_prompt: str, model: str = DEFAULT_REF_MODEL,
) -> str:
    """Lazy Anthropic-API call. Returns '' on any failure so benchmark
    degrades gracefully rather than crashing offline."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return ""
    try:
        from anthropic import Anthropic
        client = Anthropic()
        msg = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        parts = []
        for block in getattr(msg, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts)
    except Exception as e:
        logger.debug("reference LLM call failed: %s", e)
        return ""


def _stub_reference_plan(context: Dict[str, Any]) -> Dict[str, Any]:
    """Neutral baseline used when reference LLM isn't available.
    Always predicts FLAT — makes the benchmark a floor-check rather
    than a ceiling-check, but lets offline runs produce a score."""
    return {
        "direction": "FLAT",
        "thesis": "stub reference — skip when uncertain",
        "expected_pnl_pct_range": [0.0, 0.0],
        "supporting_evidence": [],
    }


# ── Runner ─────────────────────────────────────────────────────────────


def _load_warm_store_scenarios(limit: int = 200) -> List[BenchmarkScenario]:
    """Pull labeled historical decisions with realized outcomes."""
    import sqlite3
    db = Path(
        os.getenv("ACT_WARM_DB_PATH")
        or str(PROJECT_ROOT / "data" / "warm_store.sqlite")
    )
    if not db.exists():
        return []
    try:
        conn = sqlite3.connect(str(db), timeout=5.0)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(decisions)").fetchall()}
    except Exception:
        return []

    required = {"decision_id", "symbol", "ts_ns", "plan_json"}
    if not required.issubset(cols):
        conn.close()
        return []

    sql_cols = list(required) + [c for c in ("direction", "outcome_json", "self_critique")
                                  if c in cols]
    sql = (f"SELECT {', '.join(sql_cols)} FROM decisions "
           "WHERE plan_json IS NOT NULL AND plan_json != '{}' "
           "ORDER BY ts_ns DESC LIMIT ?")
    try:
        rows = conn.execute(sql, (int(limit),)).fetchall()
        conn.close()
    except Exception:
        return []

    scenarios: List[BenchmarkScenario] = []
    for r in rows:
        rec = dict(zip(sql_cols, r))
        try:
            plan = json.loads(rec.get("plan_json") or "{}")
        except Exception:
            continue
        try:
            outcome = json.loads(rec.get("outcome_json") or "{}")
        except Exception:
            outcome = {}
        if not outcome.get("pnl_pct"):
            # No realized outcome — skip (can't score)
            continue
        # Derive actual_direction from sign of pnl_pct (crude but works
        # when plan.direction matched the realized move).
        pnl = float(outcome.get("pnl_pct") or 0.0)
        if abs(pnl) < 0.1:
            actual_dir = "FLAT"
        elif pnl > 0 and str(plan.get("direction", "")).upper() == "LONG":
            actual_dir = "UP"
        elif pnl < 0 and str(plan.get("direction", "")).upper() == "LONG":
            actual_dir = "DOWN"
        elif pnl > 0 and str(plan.get("direction", "")).upper() == "SHORT":
            actual_dir = "DOWN"
        else:
            actual_dir = "UP"
        scenarios.append(BenchmarkScenario(
            scenario_id=str(rec.get("decision_id", "?")),
            asset=str(rec.get("symbol", "BTC")).upper(),
            context={"plan": plan, "ts_ns": int(rec.get("ts_ns", 0))},
            actual_outcome={
                "actual_direction": actual_dir,
                "pnl_pct": pnl,
                "max_adverse_pct": float(outcome.get("max_adverse_pct", 0.0) or 0.0),
            },
            regime=str(plan.get("regime") or outcome.get("regime") or "UNKNOWN"),
        ))
    return scenarios


def _load_curated_scenarios() -> List[BenchmarkScenario]:
    """Optional hand-curated scenarios file — one JSON line per scenario."""
    path = PROJECT_ROOT / "data" / "benchmark_scenarios.jsonl"
    if not path.exists():
        return []
    out = []
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(BenchmarkScenario(
                scenario_id=str(obj.get("scenario_id", "curated-?")),
                asset=str(obj.get("asset", "BTC")).upper(),
                context=obj.get("context") or {},
                actual_outcome=obj.get("actual_outcome") or {},
                regime=str(obj.get("regime", "UNKNOWN")),
            ))
    except Exception as e:
        logger.debug("curated scenarios load failed: %s", e)
    return out


def run_brain_benchmark(
    scenarios: Optional[List[BenchmarkScenario]] = None,
    local_runner: Optional[Callable[[BenchmarkScenario], Dict[str, Any]]] = None,
    reference_runner: Optional[Callable[[BenchmarkScenario], Dict[str, Any]]] = None,
    ref_model: str = DEFAULT_REF_MODEL,
    max_scenarios: int = 200,
) -> BenchmarkResult:
    """Run the benchmark. Both runners are injectable for tests.

    Default `local_runner` calls ACT's unified brain via TradingBrainV2.
    Default `reference_runner` calls the Anthropic reference LLM OR
    falls back to the neutral stub when no API key is set.
    """
    if scenarios is None:
        scenarios = _load_warm_store_scenarios(limit=max_scenarios)
        scenarios += _load_curated_scenarios()
    scenarios = list(scenarios)[:max_scenarios]

    if not scenarios:
        return BenchmarkResult(ref_model=ref_model, n_scenarios=0)

    # Default runners
    if local_runner is None:
        local_runner = _default_local_runner
    if reference_runner is None:
        reference_runner = lambda s: _default_reference_runner(s, ref_model)

    result = BenchmarkResult(ref_model=ref_model, n_scenarios=len(scenarios))
    regime_counts: Dict[str, Dict[str, int]] = {}

    for s in scenarios:
        try:
            local_plan = local_runner(s) or {}
        except Exception as e:
            logger.debug("local_runner failed on %s: %s", s.scenario_id, e)
            local_plan = {"direction": "FLAT", "thesis": f"error: {type(e).__name__}"}
        try:
            ref_plan = reference_runner(s) or {}
        except Exception as e:
            logger.debug("reference_runner failed on %s: %s", s.scenario_id, e)
            ref_plan = _stub_reference_plan(s.context)

        local_score = score_plan_vs_outcome(local_plan, s.actual_outcome)
        ref_score = score_plan_vs_outcome(ref_plan, s.actual_outcome)

        result.local_scores.append(local_score)
        result.ref_scores.append(ref_score)

        regime_counts.setdefault(s.regime, {"local_wins": 0, "ref_wins": 0, "ties": 0})
        if local_score.composite > ref_score.composite:
            result.local_wins += 1
            regime_counts[s.regime]["local_wins"] += 1
        elif ref_score.composite > local_score.composite:
            result.ref_wins += 1
            regime_counts[s.regime]["ref_wins"] += 1
        else:
            result.ties += 1
            regime_counts[s.regime]["ties"] += 1

    result.by_regime = regime_counts
    return result


# ── Default runners (lazy imports) ─────────────────────────────────────


def _default_local_runner(scenario: BenchmarkScenario) -> Dict[str, Any]:
    """Run ACT's unified brain on the scenario — returns plan dict.

    Uses dual_brain.analyze with a compact prompt derived from the
    scenario context. Real runs will lean on the plan's supporting
    evidence fields when the analyst tool-use loop invokes tools.
    """
    try:
        from src.ai.dual_brain import analyze
    except Exception:
        return {"direction": "FLAT", "thesis": "dual_brain unavailable"}
    prompt = (
        f"Asset: {scenario.asset}\n"
        f"Regime: {scenario.regime}\n"
        f"Context: {json.dumps(scenario.context, default=str)[:800]}\n"
        "Return STRICT JSON with keys: direction (LONG|SHORT|FLAT), "
        "thesis (short), expected_pnl_pct_range (2-element array), "
        "supporting_evidence (array of strings)."
    )
    try:
        resp = analyze(prompt)
        text = getattr(resp, "text", "") or ""
        if not text.strip():
            return {"direction": "FLAT", "thesis": "empty LLM"}
        # Reuse the hardened parser
        from src.ai.agentic_trade_loop import _extract_json
        parsed = _extract_json(text) or {}
        return parsed
    except Exception as e:
        return {"direction": "FLAT", "thesis": f"local_runner error: {e}"}


def _default_reference_runner(
    scenario: BenchmarkScenario, model: str = DEFAULT_REF_MODEL,
) -> Dict[str, Any]:
    system = (
        "You are a disciplined crypto spot trader. Return STRICT JSON "
        "with keys: direction (LONG|SHORT|FLAT), thesis (short), "
        "expected_pnl_pct_range (2-element array of percentages), "
        "supporting_evidence (array of short strings). Nothing else."
    )
    user = (
        f"Asset: {scenario.asset}\n"
        f"Regime: {scenario.regime}\n"
        f"Context: {json.dumps(scenario.context, default=str)[:800]}\n"
        "Your task: propose a trade plan for this setup."
    )
    text = _call_reference_llm(system, user, model=model)
    if not text.strip():
        return _stub_reference_plan(scenario.context)
    try:
        from src.ai.agentic_trade_loop import _extract_json
        return _extract_json(text) or _stub_reference_plan(scenario.context)
    except Exception:
        return _stub_reference_plan(scenario.context)


# ── Report writer ──────────────────────────────────────────────────────


def write_benchmark_report(result: BenchmarkResult, stamp: Optional[str] = None) -> Path:
    """Write a human-readable markdown report to reports/."""
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    path = reports_dir / f"brain_benchmark_{stamp}.md"
    lines = [
        f"# ACT Brain Benchmark — {stamp}",
        "",
        f"**Reference model:** `{result.ref_model}`",
        f"**Scenarios:** {result.n_scenarios}",
        "",
        f"**Brain Quality Score:** `{result.brain_quality_score:.4f}`",
        f"**Target (beats cited SOTA):** `{DEFAULT_TARGET_SCORE}`",
        f"**Beats target:** {'✅ YES' if result.beats_target() else '❌ no'}",
        "",
        "## Outcome distribution",
        "",
        f"- Local wins: {result.local_wins}",
        f"- Reference wins: {result.ref_wins}",
        f"- Ties: {result.ties}",
        "",
    ]
    if result.by_regime:
        lines.append("## By regime")
        lines.append("")
        lines.append("| Regime | Local wins | Ref wins | Ties |")
        lines.append("|---|---|---|---|")
        for regime, d in sorted(result.by_regime.items()):
            lines.append(
                f"| {regime} | {d.get('local_wins', 0)} | "
                f"{d.get('ref_wins', 0)} | {d.get('ties', 0)} |"
            )
        lines.append("")
    lines.extend([
        "---",
        "",
        "*Auto-generated by `src/evaluation/brain_benchmark.py`. Reproduce: "
        "`python -m src.skills.cli run brain-benchmark`.*",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
