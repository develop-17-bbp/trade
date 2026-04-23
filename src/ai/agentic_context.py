"""
Agentic-loop context manager — thin composition over existing memory infra.

What this module is:
  A multi-turn-aware helper for the agentic trade loop. Tracks the growing
  message list across tool-use rounds, enforces a token budget, summarizes
  older rounds when the budget tightens, and seeds each new cycle with
  cross-session RAG + verified quant data + the last few self-critiques.

What this module is NOT:
  * A replacement for src/ai/memory_vault.py — that already handles
    cross-session RAG (find_similar_trades + get_regime_stats). Reused here.
  * A replacement for src/ai/prompt_constraints.py — that already handles
    math injection and the ALLOWED_CONFIG_RANGES whitelist. Reused here.
  * A new embedding store — MemoryVault's existing store is the only one.

It exists because the agentic loop is multi-turn (up to 8 tool-use rounds
per cycle), whereas the existing modules were designed for single-shot
analyst prompts. The multi-turn specifics — token accounting, sliding
window, summarization of older rounds — live here.

Contract:
  ctx = AgenticContext(asset='BTC')
  messages = ctx.build_seed_context()              # Phase 1: seed from RAG + quant
  for step in range(MAX_STEPS):
      messages = ctx.append(msg)                   # grow window
      if ctx.token_budget_exceeded():
          messages = ctx.summarize_older_rounds()  # compact in place
      response = llm.call(messages, tools=...)
      ...

Token counting: uses tiktoken if installed, else approximate (~4 chars/token).
Summarization: calls the local Ollama router (src.ai.llm_provider) with a
small instruction; never raises — a summarization failure just keeps the
full history and lets the budget overflow warn without blocking trading.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


DEFAULT_MAX_TOKENS = int(os.getenv("ACT_AGENTIC_MAX_TOKENS", "80000"))
DEFAULT_KEEP_RECENT_ROUNDS = int(os.getenv("ACT_AGENTIC_KEEP_ROUNDS", "4"))
TOKEN_BUDGET_WARN_AT = 0.75   # warn when 75% consumed


def _approx_tokens(text: str) -> int:
    """Fallback token estimator — 1 token ≈ 4 chars for English.

    Used when tiktoken isn't installed. Good enough for budget-warning
    thresholds; not good enough for billing.
    """
    return max(1, len(text) // 4)


def _count_tokens(text: str) -> int:
    try:
        import tiktoken  # optional dep
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return _approx_tokens(text)


def count_message_tokens(messages: List[Dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            # tool-use / tool-result blocks — serialize JSON-ish before counting
            content = json.dumps(content, default=str)
        elif not isinstance(content, str):
            content = str(content)
        total += _count_tokens(content) + 4  # ~4 tokens per message envelope
    return total


@dataclass
class AgenticContext:
    """Per-cycle context state. One instance per (asset, cycle)."""
    asset: str
    max_tokens: int = DEFAULT_MAX_TOKENS
    keep_recent_rounds: int = DEFAULT_KEEP_RECENT_ROUNDS
    messages: List[Dict[str, Any]] = field(default_factory=list)
    token_count: int = 0

    # ── Seed ────────────────────────────────────────────────────────────

    def build_seed_context(
        self,
        regime: str = "UNKNOWN",
        quant_data: str = "",
        recent_critiques: Optional[List[Dict[str, Any]]] = None,
        similar_trades: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Compose the initial system + user messages for this cycle.

        Callers pass in pre-fetched quant_data / similar_trades / critiques;
        this module does not fetch them itself (keeps it test-friendly and
        avoids hard dependencies on MemoryVault being available at import).
        """
        system = self._system_prompt()
        user_blocks: List[str] = []

        if quant_data:
            user_blocks.append(f"## VERIFIED QUANT DATA ({self.asset})\n{quant_data}")

        user_blocks.append(f"## MARKET REGIME\n{regime}")

        if similar_trades:
            lines = ["## SIMILAR PAST SETUPS (top-k by embedding similarity)"]
            for i, t in enumerate(similar_trades[:5], 1):
                meta = t.get("metadata") or {}
                sim = t.get("similarity", 0.0)
                pnl = meta.get("pnl_pct", "?")
                reg = meta.get("regime") or meta.get("market_regime", "?")
                lines.append(f"{i}. similarity={sim:.2f} regime={reg} pnl_pct={pnl}")
            user_blocks.append("\n".join(lines))

        if recent_critiques:
            lines = ["## LAST SELF-CRITIQUES (for calibration)"]
            for c in recent_critiques[:5]:
                lines.append(
                    f"- matched={c.get('matched_thesis', '?')} "
                    f"miss={c.get('miss_reason','')[:60]} "
                    f"δconf={c.get('confidence_calibration_delta', 0):+.2f}"
                )
            user_blocks.append("\n".join(lines))

        user_blocks.append(
            "## YOUR TURN\n"
            "Gather any additional context you need via the available tools, "
            "then emit a TradePlan (or SKIP). Ground every number in the "
            "VERIFIED QUANT DATA block above or in tool results."
        )

        self.messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(user_blocks)},
        ]
        self.token_count = count_message_tokens(self.messages)
        return list(self.messages)

    def _system_prompt(self) -> str:
        """Import-lazy so this module loads even if prompt_constraints is absent."""
        try:
            from src.ai.prompt_constraints import SYSTEM_PROMPT_BASE
            from src.ai.authority_rules import AUTHORITY_SYSTEM_PROMPT
            return AUTHORITY_SYSTEM_PROMPT + "\n\n" + SYSTEM_PROMPT_BASE
        except Exception as e:
            logger.debug("agentic_context: system prompt import failed, using minimal fallback: %s", e)
            return (
                "You are ACT's trade-plan compiler. Ground every number in "
                "VERIFIED QUANT DATA. Never exceed position-size or stop-distance "
                "limits. When you have enough evidence, compile a TradePlan; "
                "otherwise SKIP."
            )

    # ── Multi-turn growth ───────────────────────────────────────────────

    def append(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Add a message (assistant tool-use, or tool-result user message)."""
        self.messages.append(message)
        self.token_count = count_message_tokens(self.messages)
        return list(self.messages)

    def token_budget_exceeded(self) -> bool:
        return self.token_count >= self.max_tokens

    def utilization_pct(self) -> float:
        if self.max_tokens <= 0:
            return 0.0
        return 100.0 * self.token_count / self.max_tokens

    def should_warn(self) -> bool:
        return self.utilization_pct() >= TOKEN_BUDGET_WARN_AT * 100

    # ── Summarization ───────────────────────────────────────────────────

    def summarize_older_rounds(self, summarizer=None) -> List[Dict[str, Any]]:
        """Compact old tool-use rounds into one summary message.

        Keeps the system + seed user messages and the last `keep_recent_rounds`
        messages verbatim. Older middle rounds become one summary.

        `summarizer` is a callable `List[Dict] -> str`; if None, we try the
        local LLM router, and if that fails fall back to a deterministic
        mechanical summary so we never hang trading on an unavailable model.
        """
        if len(self.messages) <= 2 + self.keep_recent_rounds:
            return list(self.messages)

        head = self.messages[:2]                                # system + seed
        recent = self.messages[-self.keep_recent_rounds:]       # keep verbatim
        middle = self.messages[2 : len(self.messages) - self.keep_recent_rounds]

        summary_text = self._do_summarize(middle, summarizer)
        summary_msg = {
            "role": "user",
            "content": f"## SUMMARY OF EARLIER TOOL CALLS\n{summary_text}",
        }
        self.messages = head + [summary_msg] + recent
        self.token_count = count_message_tokens(self.messages)
        return list(self.messages)

    def _do_summarize(self, middle: List[Dict[str, Any]], summarizer) -> str:
        if summarizer is not None:
            try:
                return str(summarizer(middle))
            except Exception as e:
                logger.debug("agentic_context: custom summarizer failed: %s", e)

        # Try the local LLM router.
        try:
            from src.ai.llm_provider import LLMRouter
            router = LLMRouter()
            router.add_from_env()
            transcript = "\n---\n".join(
                _stringify_message(m) for m in middle
            )
            prompt = (
                "Summarize the following tool-call transcript into <= 8 bullet "
                "points. Keep every numeric value verbatim. Drop reasoning filler.\n\n"
                f"{transcript}"
            )
            resp = router.complete(prompt, max_tokens=300)
            if resp and isinstance(resp, str):
                return resp.strip()
        except Exception as e:
            logger.debug("agentic_context: LLM summarizer unavailable: %s", e)

        # Mechanical fallback — at least keep the tool names + short digests.
        bullets: List[str] = []
        for m in middle:
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        bullets.append(f"- {role}: called {block.get('name','?')}")
                    elif isinstance(block, dict) and block.get("type") == "tool_result":
                        txt = json.dumps(block.get("content", ""), default=str)[:100]
                        bullets.append(f"- {role}: tool_result {txt}")
            else:
                txt = str(content)[:120]
                bullets.append(f"- {role}: {txt}")
        return "\n".join(bullets) or "(no prior rounds)"


def _stringify_message(m: Dict[str, Any]) -> str:
    content = m.get("content", "")
    if isinstance(content, (list, dict)):
        content = json.dumps(content, default=str)
    return f"[{m.get('role','?')}] {content}"
