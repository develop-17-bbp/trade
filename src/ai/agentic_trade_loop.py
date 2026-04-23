"""
Agentic trade loop — multi-turn ReAct driver that compiles a TradePlan.

Mirrors Claude Code's loop: LLM + tools + context + verify. Each tick:

  1. Seed context via AgenticContext (RAG + quant + last critiques).
  2. Up to MAX_STEPS of: LLM → tool_call OR plan OR skip.
     - On tool_call: dispatch via ToolRegistry, append result, iterate.
     - On plan: validate as TradePlan, return it.
     - On skip: return a skip-plan for audit.
  3. Self-verification hook registered when a non-skip plan is emitted
     (trade_verifier.py reads warm_store at trade close).

Provider-agnostic JSON protocol so this works on Ollama, Anthropic,
Gemini, or any provider that can produce JSON. The LLM is instructed
to emit exactly ONE of three envelopes per turn:

  {"tool_call": {"name": "<tool>", "args": {...}}}
  {"plan": { <TradePlan fields> }}
  {"skip": "<reason>"}

The loop parses the first top-level JSON object it finds in the
response. If parsing fails for MAX_PARSE_FAILURES consecutive turns,
the loop bails with a skip-plan rather than spinning.

Kill switch: env `ACT_DISABLE_AGENTIC_LOOP=1` makes `run()` return a
skip-plan immediately. Integrated with the existing authority-gate
pattern — nothing submitted through this loop bypasses conviction_gate
or readiness_gate; see `submit_trade_plan` tool in trade_tools.py.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.ai.agentic_context import AgenticContext
from src.ai.trade_tools import ToolRegistry, build_default_registry
from src.trading.trade_plan import TradePlan

logger = logging.getLogger(__name__)


DEFAULT_MAX_STEPS = int(os.getenv("ACT_AGENTIC_MAX_STEPS", "8"))
MAX_PARSE_FAILURES = 2
DISABLE_ENV = "ACT_DISABLE_AGENTIC_LOOP"


@dataclass
class LoopResult:
    plan: TradePlan
    steps_taken: int
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    terminated_reason: str = ""                 # 'plan' | 'skip' | 'max_steps' | 'disabled' | 'parse_failures'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "steps_taken": self.steps_taken,
            "tool_calls": list(self.tool_calls),
            "terminated_reason": self.terminated_reason,
        }


# ── JSON extraction ─────────────────────────────────────────────────────

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Find the first balanced top-level JSON object in the LLM response.

    Handles: ```json ... ``` fences, raw JSON, JSON embedded in prose.
    """
    if not text:
        return None
    # Strip ```json fence if present
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass
    # Fall through to greedy match; try progressively shrinking from the right
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return None
    candidate = m.group(0)
    for end in range(len(candidate), 0, -1):
        try:
            return json.loads(candidate[:end])
        except Exception:
            continue
    return None


# ── Loop ────────────────────────────────────────────────────────────────


class AgenticTradeLoop:
    """One instance per asset per tick. Not thread-safe — don't share."""

    def __init__(
        self,
        asset: str,
        llm_call: Callable[[List[Dict[str, Any]]], str],
        registry: Optional[ToolRegistry] = None,
        context: Optional[AgenticContext] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
    ):
        self.asset = asset
        self.llm_call = llm_call
        self.registry = registry or build_default_registry()
        self.context = context or AgenticContext(asset=asset)
        self.max_steps = max_steps

    # ── Seed ────────────────────────────────────────────────────────────

    def seed(
        self,
        regime: str = "UNKNOWN",
        quant_data: str = "",
        recent_critiques: Optional[List[Dict[str, Any]]] = None,
        similar_trades: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.context.build_seed_context(
            regime=regime,
            quant_data=quant_data,
            recent_critiques=recent_critiques,
            similar_trades=similar_trades,
        )
        # Append the tool-call envelope contract as a user message so the
        # LLM knows the JSON schema it must emit each turn.
        self.context.append({
            "role": "user",
            "content": (
                "## RESPONSE FORMAT\n"
                "Emit ONE JSON object per turn. Options:\n"
                "  {\"tool_call\": {\"name\": <tool_name>, \"args\": {...}}}\n"
                "  {\"plan\": { <TradePlan fields> }}\n"
                "  {\"skip\": \"<one-line reason>\"}\n"
                f"Available tools: {', '.join(self.registry.list_names())}."
            ),
        })

    # ── Main run ────────────────────────────────────────────────────────

    def run(self) -> LoopResult:
        # Kill switch — immediate skip, no LLM call.
        if os.getenv(DISABLE_ENV, "0") == "1":
            return LoopResult(
                plan=TradePlan.skip(self.asset, thesis="agentic loop disabled by env"),
                steps_taken=0,
                terminated_reason="disabled",
            )

        tool_calls: List[Dict[str, Any]] = []
        parse_failures = 0

        for step in range(self.max_steps):
            raw = self._call_llm()
            envelope = _extract_json(raw or "")
            if envelope is None:
                parse_failures += 1
                if parse_failures >= MAX_PARSE_FAILURES:
                    return LoopResult(
                        plan=TradePlan.skip(self.asset, thesis="LLM produced unparseable output"),
                        steps_taken=step + 1,
                        tool_calls=tool_calls,
                        terminated_reason="parse_failures",
                    )
                # Nudge the model with a reminder and keep going.
                self.context.append({
                    "role": "user",
                    "content": "Your last response did not include a valid JSON envelope. Emit one of {tool_call, plan, skip}.",
                })
                continue

            # Reset on successful parse.
            parse_failures = 0

            # ── Dispatch envelope ─────────────────────────────────────
            if "tool_call" in envelope:
                tc = envelope.get("tool_call") or {}
                name = str(tc.get("name") or "")
                args = tc.get("args") or {}
                result = self.registry.dispatch(name, args)
                tool_calls.append({"name": name, "args": args, "result_preview": result[:200]})
                self.context.append({"role": "assistant", "content": raw})
                self.context.append({
                    "role": "user",
                    "content": f"## TOOL RESULT ({name})\n{result}",
                })
                if self.context.should_warn():
                    self.context.summarize_older_rounds()
                continue

            if "plan" in envelope:
                try:
                    plan = TradePlan(asset=self.asset, **(envelope.get("plan") or {}))
                except Exception as e:
                    # Plan didn't validate; tell the LLM and iterate.
                    self.context.append({"role": "assistant", "content": raw})
                    self.context.append({
                        "role": "user",
                        "content": f"Plan validation failed: {type(e).__name__}: {e}. Fix and re-emit.",
                    })
                    continue
                return LoopResult(
                    plan=plan, steps_taken=step + 1,
                    tool_calls=tool_calls, terminated_reason="plan",
                )

            if "skip" in envelope:
                reason = str(envelope.get("skip") or "")[:200]
                return LoopResult(
                    plan=TradePlan.skip(self.asset, thesis=reason or "LLM skipped"),
                    steps_taken=step + 1,
                    tool_calls=tool_calls,
                    terminated_reason="skip",
                )

            # Unknown envelope — nudge.
            self.context.append({
                "role": "user",
                "content": "Envelope unrecognised. Use exactly one of {tool_call, plan, skip}.",
            })

        # Out of budget — emit skip rather than an unbounded loop.
        return LoopResult(
            plan=TradePlan.skip(self.asset, thesis="max_steps reached"),
            steps_taken=self.max_steps,
            tool_calls=tool_calls,
            terminated_reason="max_steps",
        )

    def _call_llm(self) -> str:
        try:
            return str(self.llm_call(list(self.context.messages)) or "")
        except Exception as e:
            logger.debug("agentic_trade_loop LLM call failed: %s", e)
            return ""
