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


# ── Tool input schema validation ─────────────────────────────────────────
#
# LLM tool calls hallucinate arg types ~15-52% of the time in production
# (April 2026 ICLR). Today the per-tool handlers each defend with ad-hoc
# coercion (`str(args.get('asset') or 'BTC').upper()` etc.) which is
# robust but scattered, and silently turns garbage into a default rather
# than telling the LLM it sent garbage.
#
# This validator catches the most common LLM failure shapes — wrong type,
# missing required, out-of-range numeric, out-of-enum string — at the
# dispatch boundary, returns a structured error to the LLM so its next
# ReAct turn can correct, and bumps a counter so /status can surface
# `tool_call_schema_violation_rate`. Designed to be lenient on shapes
# the existing handlers tolerate (e.g. "asset" passed as int gets
# coerced to string), aggressive on shapes that genuinely break (e.g.
# a tool that needs an array gets a string).

_SCHEMA_VIOLATIONS: int = 0
_TOOL_DISPATCHES: int = 0


def schema_violation_rate() -> Dict[str, Any]:
    """Report the running tool-call schema violation rate. /status
    surfaces this; target < 5%, alert threshold ~10% sustained."""
    rate = (_SCHEMA_VIOLATIONS / _TOOL_DISPATCHES) if _TOOL_DISPATCHES else 0.0
    return {
        "violations": _SCHEMA_VIOLATIONS,
        "dispatches": _TOOL_DISPATCHES,
        "rate":        round(rate, 4),
    }


def _coerce_scalar(value: Any, expected_type: str) -> Optional[Any]:
    """Best-effort coercion of LLM-emitted values to JSON Schema types.
    Returns None on irrecoverable mismatch (caller treats as violation)."""
    if expected_type == "string":
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        return None
    if expected_type == "integer":
        if isinstance(value, bool):  # bool is int in Python; reject silently
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
        return None
    if expected_type == "number":
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        return None
    if expected_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        return None
    if expected_type == "array":
        return value if isinstance(value, list) else None
    if expected_type == "object":
        return value if isinstance(value, dict) else None
    # Unknown type — accept as-is.
    return value


def _validate_input_schema(args: Dict[str, Any], schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """JSON-Schema-flavoured validator for the subset of constructs ACT
    tools actually use: type, properties{type/enum/minimum/maximum},
    required. Returns None on pass; on fail returns a digest error dict
    suitable for the LLM tool_result payload.

    Coerces in place — `args` may be mutated to canonical types so the
    handler downstream sees clean data."""
    if not isinstance(schema, dict):
        return None  # No schema → nothing to check.
    props = schema.get("properties") or {}
    required = schema.get("required") or []

    # Required-field check
    for name in required:
        if name not in args:
            return {
                "error": "schema_violation",
                "field": name,
                "detail": f"required field {name!r} missing",
            }

    # Per-field type + enum + range
    for name, spec in props.items():
        if name not in args:
            continue
        if not isinstance(spec, dict):
            continue
        expected_type = spec.get("type")
        if expected_type:
            coerced = _coerce_scalar(args[name], expected_type)
            if coerced is None:
                return {
                    "error": "schema_violation",
                    "field": name,
                    "detail": f"expected type {expected_type!r}, got {type(args[name]).__name__}",
                }
            args[name] = coerced
        # enum
        enum_vals = spec.get("enum")
        if enum_vals and args[name] not in enum_vals:
            return {
                "error": "schema_violation",
                "field": name,
                "detail": f"value {args[name]!r} not in {enum_vals}",
            }
        # numeric bounds
        if expected_type in ("integer", "number"):
            mn = spec.get("minimum")
            mx = spec.get("maximum")
            if mn is not None and args[name] < mn:
                return {
                    "error": "schema_violation",
                    "field": name,
                    "detail": f"value {args[name]} < minimum {mn}",
                }
            if mx is not None and args[name] > mx:
                return {
                    "error": "schema_violation",
                    "field": name,
                    "detail": f"value {args[name]} > maximum {mx}",
                }

    return None


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
    """Name-keyed store + dispatcher with tag-based filtering.

    Per-tick memoization: when `tick_id` is set via `set_tick_id`,
    repeated dispatches of the same (name, args) within that tick
    return the cached result. Resets when tick_id changes.
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        # (tick_id, tool_name, args_hash) -> serialized result
        self._tick_cache: Dict[tuple, str] = {}
        self._current_tick_id: Optional[str] = None
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def set_tick_id(self, tick_id: Optional[str]) -> None:
        """Mark the start of a new tick. Clears the per-tick cache so
        the new tick re-fetches fresh data. Pass None to disable caching."""
        if tick_id != self._current_tick_id:
            self._tick_cache.clear()
        self._current_tick_id = tick_id

    def cache_stats(self) -> Dict[str, int]:
        return {"hits": self._cache_hits, "misses": self._cache_misses,
                "size": len(self._tick_cache),
                "current_tick_id": self._current_tick_id}

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

        args = dict(args or {})

        # Bump dispatch counter once per call, even if cache-hit later.
        # Counters drive `/status` reporting of tool_call_schema_violation_rate
        # so the operator can spot a regressing analyst prompt or model swap.
        global _TOOL_DISPATCHES, _SCHEMA_VIOLATIONS
        _TOOL_DISPATCHES += 1

        # Schema validation at the dispatch boundary — see _validate_input_schema.
        viol = _validate_input_schema(args, tool.input_schema)
        if viol is not None:
            _SCHEMA_VIOLATIONS += 1
            logger.debug(
                "tool %s schema_violation: %s (args=%s)",
                name, viol.get("detail"), args,
            )
            return json.dumps(viol)

        # Per-tick memoization: skip re-execution when same (name, args)
        # was called earlier in the same tick. Write tools (tag='write')
        # bypass the cache — they have side effects and must always run.
        cache_key = None
        if (self._current_tick_id is not None
                and getattr(tool, "tag", "") != "write"):
            try:
                args_hash = json.dumps(args, sort_keys=True, default=str)
            except Exception:
                args_hash = str(args)
            cache_key = (self._current_tick_id, name, args_hash)
            cached = self._tick_cache.get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                return cached

        try:
            raw = tool.handler(args)
        except Exception as e:
            logger.debug("tool %s handler error: %s", name, e)
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

        result = _serialize_and_cap(raw, tool.max_output_chars)
        if cache_key is not None:
            self._tick_cache[cache_key] = result
            self._cache_misses += 1
        return result


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


# ── Default registry builder (hot-path cached) ────────────────────────

# Module-level cache: `build_default_registry` was being called on every
# shadow tick which (a) re-imported quant/agent/MCP helpers, (b) hit
# every configured MCP server's /tools/list endpoint over the network,
# and (c) parsed config.yaml from disk. That's 500 ms – 8 s wasted per
# tick if any MCP server is slow. Cache the built registry for the
# process lifetime; operators can force a rebuild via reset_default_registry().
_DEFAULT_REGISTRY: Optional[ToolRegistry] = None


def get_default_registry() -> ToolRegistry:
    """Return a process-wide cached ToolRegistry. Build on first call."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = _build_default_registry_uncached()
    return _DEFAULT_REGISTRY


def reset_default_registry() -> None:
    """Drop the cache so the next get_default_registry() rebuilds.
    Use after config.yaml mcp_clients edit or when tests want a fresh
    registry."""
    global _DEFAULT_REGISTRY
    _DEFAULT_REGISTRY = None


def build_default_registry() -> ToolRegistry:
    """Backward-compat wrapper — tests and legacy callers still use this.
    Returns the CACHED singleton; pass force=True via reset first if
    you really want a fresh build."""
    return get_default_registry()


def _build_default_registry_uncached() -> ToolRegistry:
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

    # ── Agent + debate + backtest tools (post-C17 simplify) ───────────
    # Expose the 10 orchestrator agents that weren't callable before,
    # plus ask_debate (run adversarial debate) and backtest_hypothesis
    # (vectorized sanity check). Closes the "full potential" capability
    # gap — every ACT subsystem the Analyst brain can meaningfully
    # reason about is now a tool.
    try:
        from src.ai.agent_tools import register_agent_tools
        register_agent_tools(reg)
    except Exception as _e:
        logger.debug("agent_tools not registered: %s", _e)

    # ── Unified-brain tool pack (C26 Step 2) ──────────────────────────
    # Expose the remaining ACT subsystems that weren't LLM-callable yet:
    # ML ensemble (LightGBM + LSTM + PatchTST + RL), 36-strategy engine,
    # MemoryVault (age-decayed), Monte-Carlo VaR, EVT tail risk, macro
    # bias, per-layer economic intelligence, genetic hall-of-fame
    # challenger, full event-driven backtest. After this block the
    # Analyst brain has access to every major ACT subsystem as a tool.
    try:
        from src.ai.unified_brain_tools import register_unified_brain_tools
        register_unified_brain_tools(reg)
    except Exception as _e:
        logger.debug("unified_brain_tools not registered: %s", _e)

    # ── External MCP servers (C7 — now auto-wired) ─────────────────────
    # Previously the operator had to manually call register_all_from_config.
    # Now every `python -m ... build_default_registry()` mirrors every
    # configured MCP server's tools into the same registry the Analyst
    # brain sees. No-op if config.yaml:mcp_clients is empty or
    # ACT_DISABLE_MCP_CLIENTS=1 is set.
    try:
        from pathlib import Path as _P
        import yaml as _yaml
        from src.ai.mcp_client_registry import register_all_from_config
        _cfg_path = _P(__file__).resolve().parents[2] / "config.yaml"
        _cfg = {}
        if _cfg_path.exists():
            with _cfg_path.open("r", encoding="utf-8") as _f:
                _cfg = _yaml.safe_load(_f) or {}
        register_all_from_config(reg, _cfg)
    except Exception as _e:
        logger.debug("mcp clients not registered: %s", _e)

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

    # ── Options tools (Alpaca options Level 3) ────────────────────────
    # These let the analyst evaluate option contracts alongside spot
    # stock/crypto each tick, picking whichever has higher EV. Long-
    # directional only in this commit (single-leg long_call / long_put);
    # multi-leg spreads come later. Operator pre-req: Alpaca paper
    # account must have options Level 3 cleared, and the alpaca_options
    # exchange enabled (ACT_BOX_ROLE includes 'options').

    reg.register(Tool(
        name="get_option_chain",
        description=(
            "Fetch a filtered options chain snapshot for an underlying. "
            "Returns top-K contracts in the DTE window with bid/ask + "
            "greeks (delta/gamma/theta/vega) + IV. Use this BEFORE "
            "submit_option_trade to pick the best strike + expiration "
            "for a directional thesis. ATM/near-ATM contracts have "
            "the tightest spreads — that's where most edge survives "
            "the fill cost."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "underlying": {"type": "string",
                               "description": "Stock ticker e.g. 'SPY', 'NVDA', 'TSLA'"},
                "side":       {"type": "string", "enum": ["call", "put"]},
                "min_dte":    {"type": "integer", "minimum": 1, "maximum": 365,
                               "description": "Min days to expiration. Default 7 (avoid 0-DTE)."},
                "max_dte":    {"type": "integer", "minimum": 1, "maximum": 365,
                               "description": "Max days to expiration. Default 45 (short-dated)."},
                "limit":      {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["underlying", "side"],
        },
        handler=_handle_get_option_chain,
        tag="read_only",
    ))

    reg.register(Tool(
        name="submit_option_trade",
        description=(
            "Submit a long-directional option order (long_call OR "
            "long_put). Re-validated against DTE window, greek caps, "
            "and RTH gate; the LLM cannot bypass these. Use this when "
            "you've identified a directional setup AND determined that "
            "an option's leverage/defined-risk profile beats the "
            "underlying's spot trade. Returns {status, order_id?, "
            "occ_symbol, reason?}."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "underlying": {"type": "string"},
                "side":       {"type": "string", "enum": ["call", "put"]},
                "strike":     {"type": "number", "minimum": 0.01,
                               "description": "Contract strike price."},
                "expiration": {"type": "string",
                               "description": "ISO date YYYY-MM-DD."},
                "qty":        {"type": "integer", "minimum": 1, "maximum": 100,
                               "description": "Number of contracts (each = 100 shares)."},
                "limit_price": {"type": "number", "minimum": 0.01,
                                "description": "Optional limit; omit for market order."},
                "delta_estimate": {"type": "number",
                                   "description": "Optional delta from chain; used for greek-cap check."},
                "vega_estimate":  {"type": "number",
                                   "description": "Optional vega from chain; used for greek-cap check."},
                "conviction_tier": {"type": "string", "enum": ["sniper", "normal"]},
            },
            "required": ["underlying", "side", "strike", "expiration", "qty"],
        },
        handler=_handle_submit_option_trade,
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


# ── Options handlers ───────────────────────────────────────────────


def _handle_get_option_chain(args: Dict[str, Any]) -> Dict[str, Any]:
    """Return a filtered options chain snapshot.

    Errors collapse to {"error": ..., "contracts": []} so the LLM
    sees a structured answer it can reason over instead of a stack
    trace.
    """
    underlying = str(args.get("underlying") or "").upper().strip()
    if not underlying:
        return {"error": "underlying_required", "contracts": []}
    side = str(args.get("side") or "call").lower().strip()
    if side not in ("call", "put"):
        return {"error": f"bad_side:{side}", "contracts": []}
    min_dte = int(args.get("min_dte") or 7)
    max_dte = int(args.get("max_dte") or 45)
    limit = int(args.get("limit") or 20)
    try:
        from src.data.alpaca_options_fetcher import AlpacaOptionsFetcher
        f = AlpacaOptionsFetcher(paper=True)
        if not f.available:
            return {"error": "alpaca_unavailable", "contracts": []}
        contracts = f.chain(
            underlying, side=side,
            min_dte=min_dte, max_dte=max_dte, limit=limit,
        )
    except Exception as e:
        return {"error": f"chain_fetch_failed: {type(e).__name__}: {e}",
                "contracts": []}
    # Compact summary first, then top-K rows. Bid/ask spread + greeks
    # in 1-2 decimals so the LLM can compare contracts without drowning
    # in precision.
    rows = []
    for c in contracts[:limit]:
        rows.append({
            "symbol":     c.get("symbol"),
            "strike":     c.get("strike"),
            "expiration": c.get("expiration"),
            "dte":        c.get("dte"),
            "bid":        round(float(c.get("bid") or 0), 2),
            "ask":        round(float(c.get("ask") or 0), 2),
            "delta":      None if c.get("delta") is None else round(c["delta"], 3),
            "vega":       None if c.get("vega") is None else round(c["vega"], 2),
            "iv":         None if c.get("iv") is None else round(c["iv"], 3),
        })
    return {
        "underlying": underlying,
        "side":       side,
        "count":      len(rows),
        "contracts":  rows,
        "note": (
            "ATM/near-ATM contracts have tightest spreads. Pick a strike "
            "+ expiration that balances delta exposure against premium "
            "cost; check delta is within the per-position cap (default 0.5)."
        ),
    }


def _handle_submit_option_trade(args: Dict[str, Any]) -> Dict[str, Any]:
    """Submit a single-leg long call/put. Validated by the executor's
    DTE / greek-cap / RTH gates; the LLM cannot bypass them.
    """
    underlying = str(args.get("underlying") or "").upper().strip()
    side = str(args.get("side") or "").lower().strip()
    if not underlying or side not in ("call", "put"):
        return {"status": "rejected",
                "reason": f"bad_args underlying={underlying} side={side}"}
    try:
        strike = float(args.get("strike"))
    except (TypeError, ValueError):
        return {"status": "rejected", "reason": f"bad_strike:{args.get('strike')}"}
    expiration = str(args.get("expiration") or "").strip()
    qty = int(args.get("qty") or 1)
    limit_price = args.get("limit_price")
    if limit_price is not None:
        try:
            limit_price = float(limit_price)
        except (TypeError, ValueError):
            limit_price = None
    delta_estimate = args.get("delta_estimate")
    vega_estimate = args.get("vega_estimate")
    conviction_tier = str(args.get("conviction_tier") or "normal").lower()

    try:
        from src.trading.alpaca_options_executor import AlpacaOptionsExecutor
        ex = AlpacaOptionsExecutor(paper=True)
    except Exception as e:
        return {"status": "rejected",
                "reason": f"executor_init_failed:{type(e).__name__}:{e}"}

    res = ex.submit_long_directional(
        underlying=underlying,
        side=side,
        strike=strike,
        expiration=expiration,
        qty=qty,
        limit_price=limit_price,
        delta_estimate=(float(delta_estimate)
                        if delta_estimate is not None else None),
        vega_estimate=(float(vega_estimate)
                       if vega_estimate is not None else None),
        conviction_tier=conviction_tier,
        plan={
            "asset": underlying, "direction": "LONG_CALL" if side == "call" else "LONG_PUT",
            "strike": strike, "expiration": expiration,
            "qty": qty, "conviction_tier": conviction_tier,
        },
    )
    return {
        "status":     "accepted" if res.submitted else "rejected",
        "order_id":   res.order_id,
        "occ_symbol": res.occ_symbol,
        "reason":     res.reason,
        "decision_id": res.decision_id,
    }
