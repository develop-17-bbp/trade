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
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*$", re.IGNORECASE | re.DOTALL)
# JS-style comments that appear in LLM JSON output. Don't match inside strings
# (approximation — not perfect but good enough for well-formed LLM output).
_JS_LINE_COMMENT_RE = re.compile(r"(?<!:)//[^\n\r]*")
_JS_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
# Trailing commas before } or ]
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _find_balanced_json(text: str, start: int = 0) -> Optional[str]:
    """Find the first balanced {...} substring using a depth counter.

    Tolerant to strings containing braces, escaped quotes. Returns None
    if no balanced object is found.
    """
    i = text.find("{", start)
    if i < 0:
        return None
    depth = 0
    in_str = False
    escaped = False
    for j in range(i, len(text)):
        ch = text[j]
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_str:
            escaped = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[i:j + 1]
    return None


def _sanitize_json_text(raw: str) -> str:
    """Light cleanup for common LLM JSON output quirks.

    * strip JS-style line comments (//) and block comments (/*...*/)
    * strip trailing commas before } or ]
    * strip leading/trailing backticks
    """
    s = raw.strip().strip("`")
    s = _JS_BLOCK_COMMENT_RE.sub("", s)
    s = _JS_LINE_COMMENT_RE.sub("", s)
    s = _TRAILING_COMMA_RE.sub(r"\1", s)
    return s


def _try_single_quoted_repair(raw: str) -> Optional[Dict[str, Any]]:
    """Last-resort: LLM emitted Python-dict-style with single quotes.

    ast.literal_eval handles {'key': 'value', 'nums': [1, 2]} where
    json.loads fails. Safe — literal_eval refuses anything other than
    primitives / lists / tuples / dicts / bool / None.
    """
    import ast
    try:
        obj = ast.literal_eval(raw.strip())
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Find the first balanced top-level JSON object in the LLM response.

    Tolerant to:
      * <think>...</think> reasoning-model prefixes (deepseek-r1, qwen3)
      * ```json ... ``` fences
      * JS-style comments inside JSON (// and /* */)
      * trailing commas before } or ]
      * prose before/after the JSON block
      * single-line JSON with no formatting
    """
    if not text:
        return None
    # 1. Strip any <think>...</think> reasoning trace first — these can
    # contain braces that would confuse the balanced-brace scanner.
    cleaned = _THINK_RE.sub("", text)
    cleaned = _UNCLOSED_THINK_RE.sub("", cleaned)

    # 2. Try fenced code block first — the cleanest signal of intent.
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", cleaned, re.DOTALL)
    if fenced:
        try:
            return json.loads(_sanitize_json_text(fenced.group(1)))
        except Exception:
            pass   # fall through to balanced-brace search

    # 3. Balanced-brace scan. Walk forward — the LAST top-level object
    # often wins when the LLM includes examples followed by the real
    # answer. Collect candidates; try largest first.
    candidates: list = []
    start = 0
    while start < len(cleaned):
        found = _find_balanced_json(cleaned, start)
        if not found:
            break
        candidates.append(found)
        start = cleaned.find(found, start) + len(found)
    # Try longest first — most specific likely to be the real payload.
    for cand in sorted(candidates, key=len, reverse=True):
        sanitized = _sanitize_json_text(cand)
        try:
            return json.loads(sanitized)
        except Exception:
            # Python-dict-style single-quoted? ast.literal_eval handles
            # {'key': 'value'} that json.loads rejects.
            repaired = _try_single_quoted_repair(sanitized)
            if repaired is not None:
                return repaired
            continue

    # 4. Last-resort: greedy single-match + progressive shrink-from-right
    # (handles cases where balanced search failed due to quoted braces).
    m = _JSON_BLOCK_RE.search(cleaned)
    if not m:
        return None
    candidate = _sanitize_json_text(m.group(0))
    for end in range(len(candidate), 0, -1):
        piece = candidate[:end]
        try:
            return json.loads(piece)
        except Exception:
            repaired = _try_single_quoted_repair(piece)
            if repaired is not None:
                return repaired
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
                "  Single tool:    {\"tool_call\":  {\"name\": <tool_name>, \"args\": {...}}}\n"
                "  Parallel tools: {\"tool_calls\": [{\"name\": <tool>, \"args\": {...}}, ...]}\n"
                "  Plan:           {\"plan\": { <TradePlan fields> }}\n"
                "  Skip:           {\"skip\": \"<one-line reason>\"}\n\n"
                "Use 'tool_calls' (plural list) when fetching MULTIPLE INDEPENDENT "
                "pieces of evidence — saves ReAct steps. Example: query news, macro, "
                "and on-chain in parallel rather than three sequential turns. "
                "Per-tick memoization caches results, so duplicate calls in the "
                "same list are folded into one fetch.\n"
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
                # Log the first 200 chars of the unparseable output so
                # operators diagnosing zero-trade cycles can SEE what the
                # model actually said. Scrub before logging in case the
                # model accidentally echoed secrets.
                try:
                    from src.ai.output_scrubber import scrub
                    preview = scrub(str(raw)[:400]).text.replace("\n", " ")
                except Exception:
                    preview = str(raw)[:400].replace("\n", " ")
                logger.warning(
                    "[agentic_loop:%s] parse_failure %d/%d — step=%d preview=%r",
                    self.asset, parse_failures, MAX_PARSE_FAILURES, step + 1, preview,
                )
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

            # Parallel tool calls — list form. Resolves all in this single
            # ReAct step rather than chaining N sequential steps. Per-tick
            # memoization on the registry de-dupes any repeats.
            if "tool_calls" in envelope:
                calls = envelope.get("tool_calls") or []
                if not isinstance(calls, list) or not calls:
                    self.context.append({
                        "role": "user",
                        "content": "tool_calls must be a non-empty list of {name, args}",
                    })
                    continue
                # Cap concurrent calls to prevent the LLM from emitting a
                # 50-tool burst that floods context.
                MAX_PARALLEL = 6
                calls = calls[:MAX_PARALLEL]
                results: List[str] = []
                for tc in calls:
                    if not isinstance(tc, dict):
                        continue
                    name = str(tc.get("name") or "")
                    args = tc.get("args") or {}
                    result = self.registry.dispatch(name, args)
                    tool_calls.append({"name": name, "args": args,
                                        "result_preview": result[:200]})
                    results.append(f"### {name}\n{result}")
                self.context.append({"role": "assistant", "content": raw})
                self.context.append({
                    "role": "user",
                    "content": "## TOOL RESULTS (parallel)\n" + "\n\n".join(results),
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
