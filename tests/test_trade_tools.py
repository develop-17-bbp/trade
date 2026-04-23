"""Tests for src/ai/trade_tools.py — LLM tool-use registry + dispatch."""
from __future__ import annotations

import json
from unittest import mock

import pytest

from src.ai.trade_tools import (
    DEFAULT_MAX_OUTPUT_CHARS,
    Tool,
    ToolRegistry,
    build_default_registry,
)


# ── Tool + registry basics ──────────────────────────────────────────────


def test_tool_anthropic_schema_shape():
    t = Tool(
        name="echo",
        description="echoes input",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        handler=lambda args: args,
    )
    s = t.anthropic_schema()
    assert s["name"] == "echo"
    assert s["description"] == "echoes input"
    assert s["input_schema"]["properties"]["x"]["type"] == "string"


def test_registry_register_and_get():
    reg = ToolRegistry()
    reg.register(Tool("a", "", {"type": "object"}, lambda x: "A"))
    assert reg.get("a") is not None
    assert reg.get("missing") is None
    assert reg.list_names() == ["a"]


def test_registry_duplicate_raises():
    reg = ToolRegistry()
    reg.register(Tool("a", "", {"type": "object"}, lambda x: None))
    with pytest.raises(ValueError):
        reg.register(Tool("a", "", {"type": "object"}, lambda x: None))


def test_registry_filter_by_tag():
    reg = ToolRegistry()
    reg.register(Tool("r1", "", {"type": "object"}, lambda x: None, tag="read_only"))
    reg.register(Tool("w1", "", {"type": "object"}, lambda x: None, tag="write"))
    read_names = [s["name"] for s in reg.anthropic_schemas(tags=["read_only"])]
    assert read_names == ["r1"]
    write_names = [s["name"] for s in reg.anthropic_schemas(tags=["write"])]
    assert write_names == ["w1"]
    both = [s["name"] for s in reg.anthropic_schemas()]
    assert set(both) == {"r1", "w1"}


# ── Dispatch ────────────────────────────────────────────────────────────


def test_dispatch_unknown_tool_returns_error_json():
    reg = ToolRegistry()
    out = reg.dispatch("nope")
    assert json.loads(out) == {"error": "unknown tool 'nope'"}


def test_dispatch_serializes_dict_result():
    reg = ToolRegistry()
    reg.register(Tool("echo", "", {"type": "object"}, lambda args: {"got": args.get("x")}))
    out = reg.dispatch("echo", {"x": 42})
    assert json.loads(out) == {"got": 42}


def test_dispatch_catches_handler_exception():
    reg = ToolRegistry()
    def _boom(_):
        raise RuntimeError("bad")
    reg.register(Tool("boom", "", {"type": "object"}, _boom))
    out = reg.dispatch("boom")
    assert "error" in json.loads(out)
    assert "RuntimeError" in json.loads(out)["error"]


def test_dispatch_truncates_long_output():
    reg = ToolRegistry()
    reg.register(Tool(
        "big", "", {"type": "object"},
        handler=lambda args: "x" * 5000,
        max_output_chars=200,
    ))
    out = reg.dispatch("big")
    assert len(out) <= 200
    assert out.endswith("[truncated]")


# ── Default registry content ────────────────────────────────────────────


def test_default_registry_has_expected_tools():
    reg = build_default_registry()
    names = set(reg.list_names())
    required = {
        "get_web_context", "get_news_digest", "get_fear_greed",
        "ask_risk_guardian", "ask_loss_prevention",
        "query_recent_trades", "get_readiness_state",
        "search_strategy_repo", "submit_trade_plan",
    }
    assert required <= names


def test_submit_trade_plan_rejects_bad_json():
    reg = build_default_registry()
    out = json.loads(reg.dispatch("submit_trade_plan", {"plan_json": "{not json"}))
    assert out["status"] == "rejected"
    assert "plan_parse_error" in out["reason"]


def test_submit_trade_plan_accepts_valid_plan():
    reg = build_default_registry()
    plan = {
        "asset": "BTC",
        "direction": "LONG",
        "entry_tier": "normal",
        "entry_price": 60000.0,
        "size_pct": 5.0,
        "sl_price": 58800.0,
    }
    out = json.loads(reg.dispatch("submit_trade_plan", {"plan_json": json.dumps(plan)}))
    assert out["status"] == "accepted"
    assert out["asset"] == "BTC"
    assert out["entry_tier"] == "normal"


def test_submit_trade_plan_rejects_stale():
    reg = build_default_registry()
    # Forge a valid plan but with valid_until in the past.
    from datetime import datetime, timedelta, timezone
    past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    plan = {
        "asset": "BTC", "direction": "LONG", "entry_tier": "normal",
        "entry_price": 60000.0, "size_pct": 5.0, "sl_price": 58800.0,
        "valid_until": past,
    }
    out = json.loads(reg.dispatch("submit_trade_plan", {"plan_json": json.dumps(plan)}))
    assert out["status"] == "rejected"
    assert out["reason"] == "plan_stale"


def test_only_one_write_tool():
    reg = build_default_registry()
    writes = [s["name"] for s in reg.anthropic_schemas(tags=["write"])]
    assert writes == ["submit_trade_plan"]


# ── Read handlers (graceful when deps absent) ───────────────────────────


def test_query_recent_trades_handles_store_error():
    reg = build_default_registry()
    with mock.patch("src.orchestration.warm_store.get_store", side_effect=RuntimeError("x")):
        out = json.loads(reg.dispatch("query_recent_trades", {"limit": 5}))
    assert isinstance(out, list)
    assert out[0].get("error")


def test_risk_guardian_returns_unavailable_if_errors():
    reg = build_default_registry()
    with mock.patch("src.agents.risk_guardian_agent.RiskGuardianAgent", side_effect=RuntimeError("no module")):
        out = json.loads(reg.dispatch("ask_risk_guardian", {
            "asset": "BTC",
            "task_description": "Is the breakout real?",
            "proposed_direction": "LONG",
        }))
    assert out["verdict"] == "unavailable"


def test_sub_agent_tool_has_system_prompt():
    reg = build_default_registry()
    tool = reg.get("ask_risk_guardian")
    assert tool is not None
    assert tool.subagent_system_prompt
    assert "RiskGuardianAgent" in tool.subagent_system_prompt


def test_default_max_output_chars_sane():
    assert DEFAULT_MAX_OUTPUT_CHARS >= 500
