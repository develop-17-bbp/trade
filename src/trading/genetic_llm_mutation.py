"""
LLM-Guided Mutation Operator (P1 of genetic-loop audit)
========================================================
Uses a local LLM (the Analyst brain) as a *mutation operator* — given
the current population's profile + recent regime + live outcomes, the
LLM proposes new (entry_rule, exit_rule, gene) combinations rather
than the random mutation default.

This is the 2025-2026 evolutionary-trading state-of-the-art pattern:
"Evolution of Thought" / "Language Model Crossover" applied to
trading-strategy DNA. Random mutation is sample-inefficient; an LLM
that has seen orders of magnitude more strategy literature can propose
novel combinations far faster.

Design:
  * Soft-fail: if Ollama is unreachable or returns garbage, the call
    silently degrades to standard random mutation. This is critical —
    the genetic loop must never block on LLM downtime.
  * Whitelist: LLM proposals are validated against the existing
    `ENTRY_TEMPLATES` / `EXIT_TEMPLATES` lists. Unknown templates are
    rejected (we can't execute them yet) and counted as proposal_rejected.
  * Budget: max 1 LLM call per generation by default; the operator can
    raise this via `llm_calls_per_gen` if API budget allows.
  * Determinism: at temperature 0.2 with a seeded prompt, repeated
    runs converge on the same proposal — useful for reproducibility.

Anti-overfit design:
  * LLM gets recent OOS outcomes, NOT in-sample fitness — so it
    suggests strategies that have *worked live*, not just backtested.
  * The proposed DNA still has to pass walk-forward validation (P0)
    + Deflated-Sharpe gate before being elevated.
"""
from __future__ import annotations

import json
import logging
import os
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.trading.genetic_strategy_engine import (
    ENTRY_TEMPLATES,
    EXIT_TEMPLATES,
    INDICATOR_GENES,
    StrategyDNA,
    _clamp_gene,
)

logger = logging.getLogger(__name__)

_PROMPT_HEADER = """You are an evolutionary mutation operator for a trading-strategy genetic algorithm.
Your job: propose ONE high-quality strategy DNA that is novel relative to the current population
but plausible given the recent market regime and live trade outcomes.

Hard rules:
  - Output STRICT JSON only, no prose.
  - entry_rule MUST be one of: {entry_list}
  - exit_rule MUST be one of: {exit_list}
  - genes is a dict mapping gene_name -> numeric value, with these names available:
    {gene_list}
  - Numeric values MUST be inside the range shown in the gene catalog (clamped if not).
  - Do NOT invent new entry_rule, exit_rule, or gene names.
"""

_PROMPT_TAIL = """
Current top-3 population profile (entry/exit/fitness):
{population_profile}

Recent regime: {regime}
Recent live outcomes (last {n_live}, win_rate={win_rate:.1%}):
{live_outcomes}

Output a JSON object with keys: entry_rule, exit_rule, genes, rationale.
Rationale should be one sentence on why this combination is novel + edge-aware.
"""


@dataclass
class LLMMutationResult:
    proposed: bool
    accepted: bool
    rejection_reason: str = ""
    raw_response: str = ""
    new_dna: Optional[StrategyDNA] = None
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposed": self.proposed,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "rationale": self.rationale,
            "new_dna_name": self.new_dna.name if self.new_dna else None,
            "new_entry_rule": self.new_dna.entry_rule if self.new_dna else None,
            "new_exit_rule": self.new_dna.exit_rule if self.new_dna else None,
        }


def _format_population_profile(population: List[StrategyDNA], k: int = 3) -> str:
    if not population:
        return "  (empty)"
    sorted_pop = sorted(population, key=lambda d: getattr(d, "fitness", 0.0),
                         reverse=True)[:k]
    lines = []
    for d in sorted_pop:
        lines.append(f"  - {d.entry_rule} / {d.exit_rule} fitness={d.fitness:.3f} "
                      f"win_rate={d.win_rate:.0%} pnl={d.total_pnl:+.1f}%")
    return "\n".join(lines)


def _format_gene_catalog() -> str:
    lines = []
    for name, info in INDICATOR_GENES.items():
        lo, hi = info["range"]
        lines.append(f"    {name}: range=[{lo}, {hi}], default={info['default']}")
    return "\n".join(lines)


def _format_live_outcomes(outcomes: List[Dict[str, Any]], k: int = 5) -> str:
    if not outcomes:
        return "  (no recent outcomes)"
    lines = []
    for out in outcomes[-k:]:
        lines.append(f"  - {out.get('asset', '?')} {out.get('direction', '?')} "
                      f"pnl={out.get('pnl_pct', 0):+.2f}% "
                      f"won={out.get('won', False)}")
    return "\n".join(lines)


def _call_local_llm(
    prompt: str, system: str = "",
    model: Optional[str] = None,
    timeout_s: float = 30.0,
) -> Optional[str]:
    """Best-effort local Ollama call. Returns None on any failure."""
    try:
        import requests  # noqa: F401
    except ImportError:
        return None
    try:
        import requests
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        analyst_model = (
            model
            or os.environ.get("ACT_ANALYST_MODEL", "")
            or os.environ.get("OLLAMA_REMOTE_MODEL", "")
            or "qwen3:8b"
        )
        payload = {
            "model": analyst_model,
            "messages": [
                {"role": "system", "content": system or "You are a precise JSON-only assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 4096},
            "format": "json",
        }
        resp = requests.post(f"{host}/api/chat", json=payload, timeout=timeout_s)
        if resp.status_code != 200:
            logger.debug("LLM mutation: ollama %s -> %d", host, resp.status_code)
            return None
        data = resp.json()
        return data.get("message", {}).get("content", "")
    except Exception as exc:
        logger.debug("LLM mutation: call failed: %s", exc)
        return None


def _validate_llm_proposal(
    raw: str,
) -> Optional[Dict[str, Any]]:
    """Parse + whitelist-validate LLM output. Returns proposal dict or None."""
    if not raw:
        return None
    try:
        # If wrapped in code fences, strip them.
        s = raw.strip()
        if s.startswith("```"):
            s = s.split("\n", 1)[1] if "\n" in s else s[3:]
            if s.endswith("```"):
                s = s[:-3]
        data = json.loads(s)
    except Exception:
        # Try locating embedded JSON object.
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(raw[start:end + 1])
        except Exception:
            return None

    entry = data.get("entry_rule")
    exit_ = data.get("exit_rule")
    genes = data.get("genes") or {}
    if entry not in ENTRY_TEMPLATES:
        return None
    if exit_ not in EXIT_TEMPLATES:
        return None
    if not isinstance(genes, dict):
        return None

    clean_genes: Dict[str, float] = {}
    for k, v in genes.items():
        if k not in INDICATOR_GENES:
            continue  # silently drop unknown
        try:
            clean_genes[k] = float(v)
        except (TypeError, ValueError):
            continue

    return {
        "entry_rule": entry,
        "exit_rule": exit_,
        "genes": clean_genes,
        "rationale": str(data.get("rationale", ""))[:300],
    }


def llm_mutate_population(
    population: List[StrategyDNA],
    regime: str = "unknown",
    live_outcomes: Optional[List[Dict[str, Any]]] = None,
    llm_calls_per_gen: int = 1,
    model: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> LLMMutationResult:
    """Make a single LLM-mutation call and inject the proposal into population."""
    rng = rng or random.Random()
    live_outcomes = live_outcomes or []
    win_count = sum(1 for o in live_outcomes if o.get("won"))
    win_rate = win_count / max(1, len(live_outcomes))

    prompt = (
        _PROMPT_HEADER.format(
            entry_list=", ".join(ENTRY_TEMPLATES),
            exit_list=", ".join(EXIT_TEMPLATES),
            gene_list=_format_gene_catalog(),
        )
        + _PROMPT_TAIL.format(
            population_profile=_format_population_profile(population),
            regime=regime,
            n_live=len(live_outcomes),
            win_rate=win_rate,
            live_outcomes=_format_live_outcomes(live_outcomes),
        )
    )

    raw = _call_local_llm(prompt, model=model)
    if not raw:
        return LLMMutationResult(
            proposed=False, accepted=False,
            rejection_reason="llm_unreachable",
        )
    proposal = _validate_llm_proposal(raw)
    if not proposal:
        return LLMMutationResult(
            proposed=True, accepted=False,
            rejection_reason="invalid_json_or_whitelist_failed",
            raw_response=raw[:400],
        )

    # Build a new DNA from the proposal.
    base = deepcopy(rng.choice(population)) if population else StrategyDNA()
    base.entry_rule = proposal["entry_rule"]
    base.exit_rule = proposal["exit_rule"]
    for k, v in proposal["genes"].items():
        base.genes[k] = _clamp_gene(k, v)
    base.name = f"LLM_{rng.randint(10000, 99999)}"
    base.fitness = 0.0
    base.parents = []
    base.generation = max((d.generation for d in population), default=0) + 1
    setattr(base, "llm_proposed", True)
    setattr(base, "llm_rationale", proposal["rationale"])

    return LLMMutationResult(
        proposed=True, accepted=True,
        new_dna=base,
        rationale=proposal["rationale"],
        raw_response=raw[:200],
    )


def inject_llm_mutations_into_population(
    engine: Any,
    n_calls: int = 1,
    regime: str = "unknown",
    live_outcomes: Optional[List[Dict[str, Any]]] = None,
    replace_worst: bool = True,
) -> List[LLMMutationResult]:
    """Convenience: run N LLM mutations, replacing worst-N in the population."""
    if not engine.population:
        return []
    rng = random.Random()
    results: List[LLMMutationResult] = []
    for _ in range(max(0, n_calls)):
        r = llm_mutate_population(
            engine.population,
            regime=regime,
            live_outcomes=live_outcomes,
            rng=rng,
        )
        results.append(r)
        if r.accepted and r.new_dna and replace_worst:
            engine.population.sort(key=lambda d: getattr(d, "fitness", 0.0))
            engine.population[0] = r.new_dna  # replace worst
    return results


__all__ = [
    "llm_mutate_population",
    "inject_llm_mutations_into_population",
    "LLMMutationResult",
]
