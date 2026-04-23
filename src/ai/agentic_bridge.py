"""
End-to-end glue for the agentic trade loop.

What this module does:
  * Composes all C1–C3 modules + the existing LLMRouter into one callable
    entry point: `compile_agentic_plan(asset, regime, quant_data, ...)`
    returns a `LoopResult` with a validated `TradePlan` (or a skip-plan).
  * Provides a small CLI (`python -m src.ai.agentic_bridge --asset BTC
    --dry-run`) so the operator can exercise the loop manually before
    wiring it into the executor.

What this module does NOT do:
  * Place orders. That's the executor's job (C4d hook).
  * Own new configuration. All tunables come from env + existing
    `config.yaml` blocks (agentic_loop, emergency_mode).

Reuses (no re-implementation):
  * src/ai/agentic_context.py  — sliding context + summarization (M2)
  * src/ai/agentic_trade_loop.py — multi-turn ReAct driver (C3)
  * src/ai/trade_tools.py      — tool registry (C3)
  * src/ai/llm_provider.py     — provider-agnostic LLM dispatch
  * src/trading/trade_plan.py  — TradePlan validation (C1)
  * src/orchestration/warm_store.py — read last N self_critiques
  * src/ai/memory_vault.py     — cross-session RAG (if available)

Graceful degradation: if the LLM router is unavailable, the bridge
returns a SKIP plan with reason "no_llm" rather than raising. The
executor's default path is unaffected by bridge failures.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
from typing import Any, Callable, Dict, List, Optional

from src.ai.agentic_context import AgenticContext
from src.ai.agentic_trade_loop import AgenticTradeLoop, LoopResult
from src.ai.trade_tools import ToolRegistry, build_default_registry
from src.trading.trade_plan import TradePlan

logger = logging.getLogger(__name__)


# ── LLM-call closure builder ────────────────────────────────────────────


def build_llm_call(
    system_prompt_override: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Callable[[List[Dict[str, Any]]], str]:
    """Return a closure: `(messages) -> str` that dispatches the agentic
    loop's turns to an LLM.

    Routing (C5 dual-brain):
      * If `ai.dual_brain.enabled` is true in config, route to the
        ANALYST brain (Devstral 24B by default) — its structured-JSON
        and tool-use training fits the loop's multi-turn envelope
        protocol.
      * Otherwise, use the legacy flatten-and-dispatch path through
        the first available LLMRouter provider. Same behavior as
        before C5; existing tests / fallback unchanged.
    """
    # Prefer the dual-brain Analyst when enabled.
    try:
        from src.ai.dual_brain import build_analyst_llm_call, is_enabled as _dual_enabled
        if _dual_enabled(config):
            return build_analyst_llm_call(config)
    except Exception as e:
        logger.debug("agentic_bridge: dual_brain unavailable, using flat router: %s", e)
    def _call(messages: List[Dict[str, Any]]) -> str:
        if not messages:
            return ""

        # Separate system from the rest.
        system_parts: List[str] = []
        rest: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                content = json.dumps(content, default=str)
            content = str(content)
            if role == "system":
                system_parts.append(content)
            else:
                rest.append(f"[{role}]\n{content}")
        sys_prompt = system_prompt_override or "\n\n".join(system_parts)
        prompt = "\n\n".join(rest)

        try:
            from src.ai.llm_provider import LLMRouter
            router = LLMRouter()
            router.add_from_env()
            for _name, provider in list(router.providers.items()):
                try:
                    resp = provider.generate(prompt=prompt, system_prompt=sys_prompt)
                    text = resp.get("response") if isinstance(resp, dict) else str(resp)
                    if text:
                        return str(text)
                except Exception as e:
                    logger.debug("agentic_bridge provider %s failed: %s", _name, e)
                    continue
        except Exception as e:
            logger.debug("agentic_bridge LLMRouter unavailable: %s", e)
        return ""

    return _call


# ── Seed-context loaders ────────────────────────────────────────────────


def load_recent_critiques(limit: int = 5) -> List[Dict[str, Any]]:
    """Read the last N self_critique rows from warm_store.

    Graceful: returns [] on any error so a cold-start / missing db
    doesn't block the loop.
    """
    try:
        from src.orchestration.warm_store import get_store
        store = get_store()
        conn = sqlite3.connect(store.db_path, timeout=2.0)
        try:
            rows = conn.execute(
                "SELECT self_critique FROM decisions "
                "WHERE self_critique != '{}' AND self_critique IS NOT NULL "
                "ORDER BY ts_ns DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        finally:
            conn.close()
        out: List[Dict[str, Any]] = []
        for (raw,) in rows:
            try:
                out.append(json.loads(raw or "{}"))
            except Exception:
                continue
        return out
    except Exception as e:
        logger.debug("load_recent_critiques failed: %s", e)
        return []


def load_similar_trades(
    asset: str,
    regime: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Cross-session RAG over MemoryVault. Returns [] on any failure —
    MemoryVault initialization is non-trivial (loads embedding model)."""
    try:
        from src.ai.memory_vault import MemoryVault
        vault = MemoryVault()
        return vault.find_similar_trades(
            asset=asset,
            current_regime=regime,
            current_funding=0.0,                        # executor fills real values
            current_sentiment={"bullish": 0.5, "bearish": 0.5},
            proposed_signal=0,
            top_k=top_k,
        ) or []
    except Exception as e:
        logger.debug("load_similar_trades failed: %s", e)
        return []


# ── Main entry ──────────────────────────────────────────────────────────


def _fetch_scan_context(asset: str) -> str:
    """Pull the latest scanner-brain report for `asset` (via C7b brain_memory)
    and format it as a short block for the analyst's seed context. Returns
    empty string if nothing fresh is available."""
    try:
        from src.ai.brain_memory import get_scan_for_analyst, get_recent_analyst_traces
        report = get_scan_for_analyst(asset)
    except Exception:
        return ""
    lines: List[str] = []
    if report is not None:
        lines.append(
            f"## SCANNER REPORT ({int(report.age_s())}s old)\n"
            f"- opportunity_score: {report.opportunity_score:.0f}\n"
            f"- proposed_direction: {report.proposed_direction}\n"
            f"- signals: {', '.join(report.top_signals[:5]) or 'none'}\n"
            f"- rationale: {report.rationale[:200]}"
        )
    try:
        traces = get_recent_analyst_traces(asset, limit=3)
    except Exception:
        traces = []
    if traces:
        bullets = [
            f"- {t.direction}/{t.tier} size={t.size_pct}% verdict={t.verdict or '-'}"
            for t in traces
        ]
        lines.append("## RECENT ANALYST DECISIONS\n" + "\n".join(bullets))
    return "\n\n".join(lines)


def compile_agentic_plan(
    asset: str,
    *,
    regime: str = "UNKNOWN",
    quant_data: str = "",
    registry: Optional[ToolRegistry] = None,
    context: Optional[AgenticContext] = None,
    llm_call: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
    max_steps: Optional[int] = None,
    similar_trades: Optional[List[Dict[str, Any]]] = None,
    recent_critiques: Optional[List[Dict[str, Any]]] = None,
) -> LoopResult:
    """Compose the full loop and run it. Safe to call from the executor;
    never raises — a compile failure returns a SKIP plan with reason.

    Scanner/analyst bridge (C7b): if the brain_memory has a fresh scan
    report or recent analyst traces for this asset, they're appended to
    quant_data before the loop seeds — so the analyst sees what the
    scanner flagged and what past decisions looked like.
    """

    # Kill switch wins over everything.
    if os.environ.get("ACT_DISABLE_AGENTIC_LOOP", "0") == "1":
        return LoopResult(
            plan=TradePlan.skip(asset, thesis="disabled by ACT_DISABLE_AGENTIC_LOOP"),
            steps_taken=0,
            terminated_reason="disabled",
        )

    try:
        # Fill seed bits the caller didn't pre-fetch.
        if similar_trades is None:
            similar_trades = load_similar_trades(asset, regime)
        if recent_critiques is None:
            recent_critiques = load_recent_critiques()

        # C7b — inject scanner report + recent analyst traces into quant_data
        # so the analyst sees the other brain's output.
        brain_block = _fetch_scan_context(asset)
        if brain_block:
            quant_data = (quant_data + "\n\n" + brain_block) if quant_data else brain_block

        registry = registry or build_default_registry()
        context = context or AgenticContext(asset=asset)
        llm_call = llm_call or build_llm_call()
        steps = int(max_steps) if max_steps is not None else int(
            os.getenv("ACT_AGENTIC_MAX_STEPS", "8")
        )

        loop = AgenticTradeLoop(
            asset=asset,
            llm_call=llm_call,
            registry=registry,
            context=context,
            max_steps=steps,
        )
        loop.seed(
            regime=regime,
            quant_data=quant_data,
            recent_critiques=recent_critiques,
            similar_trades=similar_trades,
        )
        result = loop.run()

        # C7b — write the analyst's decision back into brain_memory so
        # the scanner's next tick sees it. Compact trace; never raises.
        try:
            import time as _t
            from src.ai.brain_memory import AnalystTrace, publish_analyst_trace
            publish_analyst_trace(AnalystTrace(
                asset=asset, ts=_t.time(),
                plan_id=getattr(result.plan, 'plan_id', '') or '',
                direction=str(result.plan.direction),
                tier=str(result.plan.entry_tier),
                size_pct=float(getattr(result.plan, 'size_pct', 0.0)),
                thesis=str(getattr(result.plan, 'thesis', ''))[:300],
                verdict=result.terminated_reason,
            ))
        except Exception as _e:
            logger.debug("brain_memory publish_analyst_trace failed: %s", _e)

        return result
    except Exception as e:
        logger.debug("compile_agentic_plan fatal: %s", e)
        return LoopResult(
            plan=TradePlan.skip(asset, thesis=f"bridge_error: {type(e).__name__}"),
            steps_taken=0,
            terminated_reason="parse_failures",
        )


# ── Feature-flag helper (used by the C4d executor hook) ─────────────────


def agentic_loop_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Env flag wins; falls back to config.yaml `agentic_loop.enabled`.

    `ACT_DISABLE_AGENTIC_LOOP=1` always wins (returns False) to give ops
    a one-env-var kill switch.
    """
    if os.environ.get("ACT_DISABLE_AGENTIC_LOOP", "0") == "1":
        return False
    env = os.environ.get("ACT_AGENTIC_LOOP", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    if isinstance(config, dict):
        return bool((config.get("agentic_loop") or {}).get("enabled", False))
    return False


# ── CLI (dry-run) ───────────────────────────────────────────────────────


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run the agentic trade loop for one asset and print the result.",
    )
    parser.add_argument("--asset", default="BTC", help="Asset symbol (BTC, ETH)")
    parser.add_argument("--regime", default="UNKNOWN", help="Current regime tag")
    parser.add_argument("--quant", default="", help="Verified quant-data block to seed with")
    parser.add_argument("--max-steps", type=int, default=6, help="Hard cap on LLM tool-use rounds")
    parser.add_argument(
        "--stub-llm", action="store_true",
        help="Skip the LLM; return a canned SKIP plan so the pipeline "
        "can be tested without any model available.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    llm_call = None
    if args.stub_llm:
        llm_call = lambda _msgs: json.dumps({"skip": "cli --stub-llm"})

    result = compile_agentic_plan(
        asset=args.asset.upper(),
        regime=args.regime,
        quant_data=args.quant,
        llm_call=llm_call,
        max_steps=args.max_steps,
    )
    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0 if result.terminated_reason in ("plan", "skip") else 1


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
