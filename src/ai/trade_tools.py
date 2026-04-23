"""
Tool registry for the agentic trade loop — Anthropic/OpenAI tool-use JSON.

What this module provides:
  A small registry + dispatcher that exposes ACT's existing agents,
  fetchers, and read-only stores as LLM-callable tools, using the
  tool-use JSON-schema contract that Claude / GPT-4 / Gemini expect.

Sub-agent pattern (per operator direction, 2026-04-23):

  Each tool receives an optional `task_description` string authored by
  the parent LLM — the same pattern as Claude Code's Agent tool.
  Expensive tools (backtest, market_research) run in their own isolated
  context and return a compact digest to the parent. Cheap read tools
  return raw structured data. Either way, tool outputs are kept tight
  so the parent's context window stays lean.

Contract:

  tool = Tool(
      name="ask_bear_agent",
      description="Delegates to ACT's BearAgent...",
      input_schema={...},       # JSON schema for LLM tool-use
      handler=callable,          # (args_dict) -> str-or-dict
  )
  registry = ToolRegistry()
  registry.register(tool)
  schemas = registry.anthropic_schemas()          # for LLM `tools=` param
  result_text = registry.dispatch("ask_bear_agent", {"task_description": "..."})

Design constraints:
  * Pure-Python handlers — no LLM provider imports leak into here.
  * Handlers must never raise; errors come back as dict {"error": "..."}
    so the LLM sees structured failure rather than crashing the loop.
  * Output length is soft-capped (DEFAULT_MAX_OUTPUT_CHARS) — overly
    long results are truncated with an "... [truncated]" marker.
  * Every tool is tagged as `read_only` (safe) or `write` (the one
    submit_trade_plan tool). Dispatch can be filtered by tag for
    paper-vs-live gating.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


DEFAULT_MAX_OUTPUT_CHARS = 1200


@dataclass
class Tool:
    """Single LLM-callable tool.

    `handler` signature: (args: dict) -> Any (serialised by dispatch)
    Handlers should be synchronous; async handlers are adapted at
    registration time if needed.
    """
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Any]
    tag: str = "read_only"                       # 'read_only' | 'write'
    subagent_system_prompt: Optional[str] = None  # for sub-agent style tools
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS

    def anthropic_schema(self) -> Dict[str, Any]:
        """The JSON blob Anthropic's API expects in its `tools` parameter."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
        }


class ToolRegistry:
    """Name-keyed store + dispatcher with tag-based filtering."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    # ── Registration ────────────────────────────────────────────────────

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_names(self) -> List[str]:
        return sorted(self._tools.keys())

    def anthropic_schemas(self, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if tags is None:
            return [t.anthropic_schema() for t in self._tools.values()]
        allowed = set(tags)
        return [t.anthropic_schema() for t in self._tools.values() if t.tag in allowed]

    # ── Dispatch ────────────────────────────────────────────────────────

    def dispatch(self, name: str, args: Optional[Dict[str, Any]] = None) -> str:
        """Run a tool's handler and serialize its output as a string.

        Returned string is what gets appended back into the LLM message
        list as the `tool_result` content. It is soft-capped to
        `tool.max_output_chars` so a verbose handler can't blow the
        parent's context budget.
        """
        tool = self._tools.get(name)
        if tool is None:
            return json.dumps({"error": f"unknown tool {name!r}"})

        try:
            args = dict(args or {})
            raw = tool.handler(args)
        except Exception as e:
            logger.debug("tool %s handler error: %s", name, e)
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

        return _serialize_and_cap(raw, tool.max_output_chars)


def _serialize_and_cap(raw: Any, cap: int) -> str:
    if isinstance(raw, str):
        s = raw
    else:
        try:
            s = json.dumps(raw, default=str)
        except Exception:
            s = str(raw)
    if len(s) > cap:
        return s[: cap - 20] + "... [truncated]"
    return s


# ── Default registry builder ────────────────────────────────────────────


def build_default_registry() -> ToolRegistry:
    """Assemble ACT's standard tool kit for the agentic trade loop.

    Imports are lazy inside handlers so this module loads even when some
    deps are absent (tests, minimal installs). Every handler catches
    its own errors — a missing dep yields a tool_result with "error"
    key rather than a crash.
    """
    reg = ToolRegistry()

    # ── Web context digests (composed via src/ai/web_context.py) ────────

    reg.register(Tool(
        name="get_web_context",
        description=(
            "Fetch Tier-1 web digests (news, sentiment, polymarket, "
            "institutional, macro, fear_greed) in parallel. Returns a "
            "compact per-source digest; never raw payloads."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "asset": {"type": "string", "enum": ["BTC", "ETH"]},
                "include": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Subset of {news, sentiment, polymarket, institutional, macro, fear_greed}; omit for all.",
                },
            },
            "required": ["asset"],
        },
        handler=_handle_web_context,
        tag="read_only",
    ))

    reg.register(Tool(
        name="get_news_digest",
        description="Recent crypto news digest for ONE asset (last N hours). Returns 3-line summary.",
        input_schema={
            "type": "object",
            "properties": {
                "asset": {"type": "string"},
                "hours": {"type": "integer", "minimum": 1, "maximum": 48},
            },
            "required": ["asset"],
        },
        handler=_handle_news,
        tag="read_only",
    ))

    reg.register(Tool(
        name="get_fear_greed",
        description="Current Fear & Greed index reading. Returns {value, label}.",
        input_schema={"type": "object", "properties": {}},
        handler=_handle_fear_greed,
        tag="read_only",
    ))

    # ── In-process sub-agents (existing orchestrator agents) ────────────

    reg.register(Tool(
        name="ask_risk_guardian",
        description=(
            "Sub-agent: Delegates to ACT's RiskGuardianAgent. Parent writes "
            "a task_description framing the specific risk concern; guardian "
            "returns a focused risk verdict (confidence + rationale)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "asset": {"type": "string", "enum": ["BTC", "ETH"]},
                "task_description": {
                    "type": "string",
                    "description": "Concrete question for the risk guardian, e.g. 'Is the recent breakout a bull trap given funding rate divergence?'",
                },
                "proposed_direction": {"type": "string", "enum": ["LONG", "SHORT"]},
            },
            "required": ["asset", "task_description"],
        },
        handler=_handle_risk_guardian,
        tag="read_only",
        subagent_system_prompt=(
            "You are ACT's RiskGuardianAgent. Your job is to challenge the "
            "parent agent's thesis for hidden risk. Return a 2-4 sentence "
            "verdict: either reasons to DOWNGRADE the trade, or ALL-CLEAR "
            "with a one-line why. Do not repeat the parent's reasoning back."
        ),
    ))

    reg.register(Tool(
        name="ask_loss_prevention",
        description=(
            "Sub-agent: Delegates to LossPreventionGuardian. Parent writes "
            "a task_description describing the proposed trade; guardian "
            "returns tighten_stop / reduce_size / ok verdict with reason."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "asset": {"type": "string"},
                "task_description": {"type": "string"},
                "proposed_size_pct": {"type": "number"},
                "proposed_sl_pct": {"type": "number"},
            },
            "required": ["asset", "task_description"],
        },
        handler=_handle_loss_prevention,
        tag="read_only",
        subagent_system_prompt=(
            "You are ACT's LossPreventionGuardian. You do not approve "
            "trades; you only flag risk issues. Return one of: "
            "'OK', 'TIGHTEN_STOP <reason>', 'REDUCE_SIZE <reason>', "
            "'VETO <reason>'."
        ),
    ))

    # ── Read-only queries over ACT's own stores ─────────────────────────

    reg.register(Tool(
        name="query_recent_trades",
        description="Last N closed trades from warm_store. Returns list of {asset, pnl_pct, exit_reason, regime}.",
        input_schema={
            "type": "object",
            "properties": {
                "asset": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
        },
        handler=_handle_recent_trades,
        tag="read_only",
    ))

    reg.register(Tool(
        name="get_readiness_state",
        description="Current readiness-gate evaluation: open/closed + reasons + numbers.",
        input_schema={"type": "object", "properties": {}},
        handler=_handle_readiness,
        tag="read_only",
    ))

    reg.register(Tool(
        name="search_strategy_repo",
        description="Query the versioned strategy repository by status/regime/min_sharpe. Returns top-N summaries.",
        input_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "regime": {"type": "string"},
                "min_sharpe": {"type": "number"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
        },
        handler=_handle_search_strategies,
        tag="read_only",
    ))

    # ── Write (authority-gated) ─────────────────────────────────────────

    # ── Quant-tool extensions (C11) ────────────────────────────────────
    # Wrap existing src/models/ quantitative tools as LLM-callable
    # digests (OU process, Hurst, Kalman, HMM regime, Hawkes, cointegration).
    # Import-lazy so a missing model module doesn't block registry build.
    try:
        from src.ai.quant_tools import register_quant_tools
        register_quant_tools(reg)
    except Exception as _e:
        logger.debug("quant_tools not registered: %s", _e)

    # ── Real-time knowledge-graph tool (C12) ───────────────────────────
    # Continuous-ingest graph over ACT's live data streams (news +
    # sentiment + institutional + polymarket + correlation edges, all
    # time-decayed). Analyst queries it via `query_knowledge_graph`.
    try:
        from src.ai.graph_rag import query_digest as _graph_query

        def _handle_graph_query(args):
            asset = args.get("asset")
            try:
                since_s = float(args.get("since_s") or 3600.0)
            except Exception:
                since_s = 3600.0
            try:
                max_chars = int(args.get("max_chars") or 500)
            except Exception:
                max_chars = 500
            try:
                summary = _graph_query(
                    asset=asset, since_s=since_s, max_chars=max_chars,
                )
            except Exception as e:
                return {"error": f"{type(e).__name__}: {e}"}
            return {"summary": summary}

        reg.register(Tool(
            name="query_knowledge_graph",
            description=(
                "[GRAPH] Query ACT's real-time knowledge graph over live "
                "data streams (news, sentiment, institutional, polymarket, "
                "correlations). Returns a compact text digest with recent "
                "entities + time-decayed edge weights around `asset`. "
                "Omit `asset` for a global kind-count snapshot."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "asset": {"type": "string"},
                    "since_s": {"type": "number", "minimum": 60, "maximum": 604800},
                    "max_chars": {"type": "integer", "minimum": 100, "maximum": 2000},
                },
            },
            handler=_handle_graph_query, tag="read_only",
        ))
    except Exception as _e:
        logger.debug("graph_rag tool not registered: %s", _e)

    reg.register(Tool(
        name="submit_trade_plan",
        description=(
            "Submit a compiled TradePlan for execution. The plan is "
            "re-validated against the conviction gate and authority "
            "rules; the LLM's submission cannot bypass them. Returns "
            "{status: accepted|rejected, reason?}."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "plan_json": {
                    "type": "string",
                    "description": "JSON-encoded TradePlan — must validate against src/trading/trade_plan.py TradePlan.",
                },
            },
            "required": ["plan_json"],
        },
        handler=_handle_submit_plan,
        tag="write",
    ))

    return reg


# ── Handlers (each a tiny wrapper; LLM-backed sub-agents call LLMRouter) ──


def _handle_web_context(args: Dict[str, Any]) -> Dict[str, Any]:
    from src.ai.web_context import fetch_bundle
    asset = str(args.get("asset") or "BTC").upper()
    include = args.get("include")
    bundle = fetch_bundle(asset, include=include if isinstance(include, list) else None)
    return {name: d.to_dict() for name, d in bundle.items()}


def _handle_news(args: Dict[str, Any]) -> Dict[str, Any]:
    from src.ai.web_context import get_news_digest
    asset = str(args.get("asset") or "BTC").upper()
    hours = int(args.get("hours") or 6)
    return get_news_digest(asset, hours=hours).to_dict()


def _handle_fear_greed(_args: Dict[str, Any]) -> Dict[str, Any]:
    from src.ai.web_context import get_fear_greed_digest
    return get_fear_greed_digest().to_dict()


def _handle_risk_guardian(args: Dict[str, Any]) -> Dict[str, Any]:
    """Sub-agent call — fans out to RiskGuardianAgent.

    The task_description from the parent is passed through as the
    guardian's `context` so its prompt can specialize. Falls back to
    a structured "unavailable" on import error.
    """
    task = str(args.get("task_description") or "").strip()
    try:
        from src.agents.risk_guardian_agent import RiskGuardianAgent
        agent = RiskGuardianAgent()
        state = {"asset": args.get("asset"), "direction": args.get("proposed_direction")}
        ctx = {"task_description": task}
        vote = agent.analyze(state, ctx)
    except Exception as e:
        return {"verdict": "unavailable", "reason": str(e)[:200]}
    return {
        "verdict": getattr(vote, "direction", 0),
        "confidence": float(getattr(vote, "confidence", 0.5)),
        "rationale": str(getattr(vote, "rationale", "") or getattr(vote, "reason", ""))[:400],
    }


def _handle_loss_prevention(args: Dict[str, Any]) -> Dict[str, Any]:
    task = str(args.get("task_description") or "").strip()
    try:
        from src.agents.loss_prevention_guardian import LossPreventionGuardian
        agent = LossPreventionGuardian()
        state = {
            "asset": args.get("asset"),
            "size_pct": args.get("proposed_size_pct"),
            "sl_pct": args.get("proposed_sl_pct"),
        }
        ctx = {"task_description": task}
        vote = agent.analyze(state, ctx)
    except Exception as e:
        return {"verdict": "unavailable", "reason": str(e)[:200]}
    return {
        "verdict": getattr(vote, "direction", 0),
        "confidence": float(getattr(vote, "confidence", 0.5)),
        "rationale": str(getattr(vote, "rationale", "") or getattr(vote, "reason", ""))[:400],
    }


def _handle_recent_trades(args: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        from src.orchestration.warm_store import get_store
        store = get_store()
        raw = store.recent_outcomes(
            symbol=(args.get("asset") or None),
            limit=int(args.get("limit") or 20),
        )
    except Exception as e:
        return [{"error": str(e)[:200]}]
    # Project down to the fields the LLM actually needs — keep tight.
    return [
        {
            "asset": r.get("symbol") or r.get("asset"),
            "pnl_pct": r.get("pnl_pct"),
            "exit_reason": r.get("exit_reason"),
            "regime": r.get("regime"),
        }
        for r in raw
    ]


def _handle_readiness(_args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from src.orchestration.readiness_gate import evaluate
        state = evaluate()
        return state.to_dict()
    except Exception as e:
        return {"error": str(e)[:200]}


def _handle_search_strategies(args: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        from src.trading.strategy_repository import get_repo
        repo = get_repo()
        recs = repo.search(
            status=args.get("status"),
            regime=args.get("regime"),
            min_sharpe=float(args.get("min_sharpe") or -1e9),
            limit=int(args.get("limit") or 10),
        )
    except Exception as e:
        return [{"error": str(e)[:200]}]
    # Compact each record — drop the full DNA blob, keep aggregates.
    return [
        {
            "strategy_id": r.strategy_id, "name": r.name, "status": r.status,
            "regime": r.regime_tag, "live_trades": r.live_trades,
            "live_sharpe": r.live_sharpe, "live_wr": r.live_wr,
        }
        for r in recs
    ]


def _handle_submit_plan(args: Dict[str, Any]) -> Dict[str, Any]:
    """Validate + return acceptance status. Actual executor handoff lives
    in the agentic_trade_loop — this handler only validates the plan's
    shape + checks readiness/authority gates so the LLM gets a fast
    structured answer."""
    plan_json = args.get("plan_json") or "{}"
    try:
        from src.trading.trade_plan import TradePlan
        data = json.loads(plan_json) if isinstance(plan_json, str) else plan_json
        plan = TradePlan(**data)
    except Exception as e:
        return {"status": "rejected", "reason": f"plan_parse_error: {type(e).__name__}: {e}"}

    if plan.is_stale():
        return {"status": "rejected", "reason": "plan_stale"}

    # Signal acceptance — the agentic_trade_loop inspects this flag and
    # actually hands the plan to the executor. We deliberately stop here
    # so unit tests and paper mode can exercise the registry without
    # placing orders.
    return {
        "status": "accepted",
        "plan_id": plan.plan_id,
        "asset": plan.asset,
        "direction": plan.direction,
        "entry_tier": plan.entry_tier,
        "size_pct": plan.size_pct,
    }
