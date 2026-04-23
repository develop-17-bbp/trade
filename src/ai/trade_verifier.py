"""
Post-close self-critique — the "verify" step of the Claude-Code-pattern loop.

When a trade closes, this module runs a second, isolated LLM pass that
compares what the parent agent predicted (stored in `plan_json` on the
warm_store decisions row) against what actually happened (stored in the
outcomes row). The verdict is written back to
`decisions.self_critique` so the NEXT cycle's AgenticContext can include
it in the seed — tightening confidence calibration over time.

Sub-agent pattern (per operator direction):
  * Runs in its own isolated LLM call, independent of the trade loop's
    context. Only the plan + outcome + a focused system prompt go in.
  * Output is a compact structured JSON (matched_thesis bool, miss_reason
    string, confidence_calibration_delta float). Parent never sees the raw
    verifier transcript.
  * Falls back to a deterministic mechanical verdict if the LLM is
    unreachable — trading never blocks on verification being available.

This module is INTENTIONALLY small. It reuses:
  * warm_store for read (plan_json) and write (update_self_critique)
  * llm_provider router if available
  * trade_plan.TradePlan for the predicted side

No new storage, no new config keys.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


DISABLE_ENV = "ACT_DISABLE_TRADE_VERIFIER"


@dataclass
class SelfCritique:
    """The structured post-trade verdict."""
    matched_thesis: bool
    miss_reason: str
    updated_belief: str
    next_time_do: str
    confidence_calibration_delta: float       # signed, roughly [-1, +1]
    verifier_source: str = "llm"              # 'llm' | 'mechanical' | 'stub'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched_thesis": bool(self.matched_thesis),
            "miss_reason": self.miss_reason[:400],
            "updated_belief": self.updated_belief[:400],
            "next_time_do": self.next_time_do[:400],
            "confidence_calibration_delta": round(float(self.confidence_calibration_delta), 3),
            "verifier_source": self.verifier_source,
        }


SYSTEM_PROMPT = (
    "You are ACT's TradeVerifier. Given a TradePlan the parent agent "
    "compiled and the actual trade outcome, judge whether the thesis "
    "matched reality. Respond with ONE JSON object containing: "
    "matched_thesis (bool), miss_reason (short string), "
    "updated_belief (short string), next_time_do (short string), "
    "confidence_calibration_delta (float in [-1, 1]). "
    "Be terse: no preamble, no closing remarks, JSON only."
)


def _build_user_prompt(plan: Dict[str, Any], outcome: Dict[str, Any]) -> str:
    return (
        "## PLAN (what the parent predicted)\n"
        + json.dumps(plan, default=str, indent=2)[:1500]
        + "\n\n## OUTCOME (what actually happened)\n"
        + json.dumps(outcome, default=str, indent=2)[:800]
        + "\n\nEmit the JSON verdict."
    )


def _mechanical_verdict(plan: Dict[str, Any], outcome: Dict[str, Any]) -> SelfCritique:
    """Deterministic fallback used when no LLM is available.

    Doesn't try to be clever — just records whether PnL sign matched the
    predicted direction, flags "exit_reason=sl" as thesis-miss, and
    returns a small calibration nudge toward humility.
    """
    pnl = float(outcome.get("pnl_pct") or 0.0)
    direction = str(plan.get("direction") or "").upper()
    hit_sl = str(outcome.get("exit_reason") or "").lower() in ("sl", "stop", "stop_loss")
    matched = (pnl > 0) and direction in ("LONG", "SHORT")
    delta = 0.05 if matched else (-0.10 if hit_sl else -0.03)
    reason = "SL hit — thesis inverted" if hit_sl else ("profit taken as predicted" if matched else "drift exit")
    return SelfCritique(
        matched_thesis=matched,
        miss_reason=reason if not matched else "",
        updated_belief=f"pnl={pnl:+.2f}% direction={direction}",
        next_time_do="no change" if matched else "tighten evidence before next similar setup",
        confidence_calibration_delta=delta,
        verifier_source="mechanical",
    )


def verify_outcome(
    plan: Dict[str, Any],
    outcome: Dict[str, Any],
    llm_call: Optional[Callable[[str, str], str]] = None,
) -> SelfCritique:
    """Produce a SelfCritique for one closed trade.

    `llm_call(system_prompt, user_prompt) -> str` is injectable for tests.
    If None, we try src.ai.llm_provider.LLMRouter and fall back to the
    mechanical verdict if anything errors or if the response can't be
    parsed into the expected schema.
    """
    if os.getenv(DISABLE_ENV, "0") == "1":
        return _mechanical_verdict(plan, outcome)

    user_prompt = _build_user_prompt(plan, outcome)

    if llm_call is None:
        llm_call = _default_llm_call

    try:
        raw = (llm_call(SYSTEM_PROMPT, user_prompt) or "").strip()
    except Exception as e:
        logger.debug("trade_verifier llm_call failed: %s", e)
        return _mechanical_verdict(plan, outcome)

    parsed = _parse_verdict(raw)
    if parsed is None:
        return _mechanical_verdict(plan, outcome)
    return parsed


def _default_llm_call(system_prompt: str, user_prompt: str) -> str:
    """Best-effort dispatch via LLMRouter.generate. Never raises."""
    try:
        from src.ai.llm_provider import LLMRouter
        router = LLMRouter()
        router.add_from_env()
        # Pick any available provider; generate returns {response, provider, ...}.
        for name, provider in list(router.providers.items()):
            try:
                resp = provider.generate(prompt=user_prompt, system_prompt=system_prompt)
                text = resp.get("response") if isinstance(resp, dict) else str(resp)
                if text:
                    return text
            except Exception:
                continue
    except Exception:
        pass
    return ""


def _parse_verdict(raw: str) -> Optional[SelfCritique]:
    """Extract the first JSON object from `raw` and shape it into SelfCritique."""
    if not raw:
        return None
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = raw[start : end + 1]
    try:
        obj = json.loads(candidate)
    except Exception:
        return None
    try:
        return SelfCritique(
            matched_thesis=bool(obj.get("matched_thesis", False)),
            miss_reason=str(obj.get("miss_reason", "") or ""),
            updated_belief=str(obj.get("updated_belief", "") or ""),
            next_time_do=str(obj.get("next_time_do", "") or ""),
            confidence_calibration_delta=float(obj.get("confidence_calibration_delta", 0.0)),
            verifier_source="llm",
        )
    except Exception:
        return None


# ── warm_store integration ──────────────────────────────────────────────


def verify_and_persist(decision_id: str, outcome: Dict[str, Any]) -> Optional[SelfCritique]:
    """Load the plan for `decision_id`, run the verifier, persist the
    result back onto the decision row.

    Returns the SelfCritique (or None if no plan was on file). Safe to
    call from the executor's close-callback — swallows all errors.
    """
    try:
        from src.orchestration.warm_store import get_store
        store = get_store()
        # Quick fetch — we keep this small so we don't need a new API.
        import sqlite3
        conn = sqlite3.connect(store.db_path, timeout=2.0)
        try:
            row = conn.execute(
                "SELECT plan_json FROM decisions WHERE decision_id=?", (decision_id,)
            ).fetchone()
        finally:
            conn.close()
        if not row or not row[0]:
            return None
        plan = json.loads(row[0])
        if not plan:
            return None
    except Exception as e:
        logger.debug("trade_verifier: failed to load plan for %s: %s", decision_id, e)
        return None

    critique = verify_outcome(plan, outcome)
    try:
        get_store().update_self_critique(decision_id, critique.to_dict())
    except Exception as e:
        logger.debug("trade_verifier: failed to persist critique for %s: %s", decision_id, e)
    return critique
