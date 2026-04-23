"""Tests for src/ai/agentic_trade_loop.py — multi-turn ReAct driver."""
from __future__ import annotations

import json
import os

import pytest

from src.ai.agentic_trade_loop import (
    DEFAULT_MAX_STEPS,
    DISABLE_ENV,
    AgenticTradeLoop,
    LoopResult,
    _extract_json,
)
from src.ai.trade_tools import Tool, ToolRegistry
from src.trading.trade_plan import TradePlan


# ── JSON extraction ─────────────────────────────────────────────────────


def test_extract_json_raw():
    assert _extract_json('{"tool_call": {"name": "x", "args": {}}}') == {
        "tool_call": {"name": "x", "args": {}}
    }


def test_extract_json_fenced():
    text = 'prose\n```json\n{"skip": "flat"}\n```\nmore prose'
    assert _extract_json(text) == {"skip": "flat"}


def test_extract_json_embedded_in_prose():
    text = "I think the answer is {\"plan\": {\"asset\": \"BTC\"}} and that's it."
    out = _extract_json(text)
    assert out and out.get("plan", {}).get("asset") == "BTC"


def test_extract_json_returns_none_when_no_object():
    assert _extract_json("no json here") is None


# ── Loop flow control ───────────────────────────────────────────────────


def _mini_registry():
    reg = ToolRegistry()
    reg.register(Tool("echo", "echoes", {"type": "object"}, lambda args: {"echo": args}))
    return reg


def _valid_plan_payload():
    return {
        "plan": {
            "direction": "LONG",
            "entry_tier": "normal",
            "entry_price": 60000.0,
            "size_pct": 5.0,
            "sl_price": 58800.0,
        }
    }


def test_loop_disabled_by_env_returns_skip(monkeypatch):
    monkeypatch.setenv(DISABLE_ENV, "1")
    loop = AgenticTradeLoop(
        asset="BTC", llm_call=lambda msgs: "", registry=_mini_registry(),
    )
    result = loop.run()
    assert result.terminated_reason == "disabled"
    assert result.plan.direction == "SKIP"


def test_loop_emits_plan_on_first_turn():
    llm_calls = iter([json.dumps(_valid_plan_payload())])
    loop = AgenticTradeLoop(
        asset="BTC",
        llm_call=lambda msgs: next(llm_calls),
        registry=_mini_registry(),
    )
    loop.seed(quant_data="[PRICE=60000]")
    result = loop.run()
    assert result.terminated_reason == "plan"
    assert result.steps_taken == 1
    assert isinstance(result.plan, TradePlan)
    assert result.plan.direction == "LONG"


def test_loop_emits_skip():
    llm_calls = iter([json.dumps({"skip": "no setup"})])
    loop = AgenticTradeLoop(
        asset="ETH",
        llm_call=lambda msgs: next(llm_calls),
        registry=_mini_registry(),
    )
    loop.seed()
    result = loop.run()
    assert result.terminated_reason == "skip"
    assert result.plan.direction == "SKIP"
    assert "no setup" in result.plan.thesis


def test_loop_handles_tool_call_then_plan():
    replies = [
        json.dumps({"tool_call": {"name": "echo", "args": {"q": "price?"}}}),
        json.dumps(_valid_plan_payload()),
    ]
    it = iter(replies)
    loop = AgenticTradeLoop(
        asset="BTC",
        llm_call=lambda msgs: next(it),
        registry=_mini_registry(),
    )
    loop.seed()
    result = loop.run()
    assert result.terminated_reason == "plan"
    assert result.steps_taken == 2
    assert result.tool_calls[0]["name"] == "echo"


def test_loop_unknown_tool_returns_error_and_continues():
    replies = [
        json.dumps({"tool_call": {"name": "does_not_exist", "args": {}}}),
        json.dumps({"skip": "giving up"}),
    ]
    it = iter(replies)
    loop = AgenticTradeLoop(
        asset="BTC",
        llm_call=lambda msgs: next(it),
        registry=_mini_registry(),
    )
    loop.seed()
    result = loop.run()
    assert result.terminated_reason == "skip"
    assert "unknown tool" in result.tool_calls[0]["result_preview"]


def test_loop_parse_failures_bail_out():
    loop = AgenticTradeLoop(
        asset="BTC",
        llm_call=lambda msgs: "no JSON here at all",
        registry=_mini_registry(),
        max_steps=5,
    )
    loop.seed()
    result = loop.run()
    assert result.terminated_reason == "parse_failures"
    assert result.plan.direction == "SKIP"


def test_loop_invalid_plan_validation_retries_then_skips():
    replies = [
        # First plan is invalid (SL above entry for LONG).
        json.dumps({"plan": {
            "direction": "LONG", "entry_tier": "normal",
            "entry_price": 60000.0, "size_pct": 5.0, "sl_price": 61000.0,
        }}),
        json.dumps({"skip": "couldn't fix"}),
    ]
    it = iter(replies)
    loop = AgenticTradeLoop(
        asset="BTC",
        llm_call=lambda msgs: next(it),
        registry=_mini_registry(),
    )
    loop.seed()
    result = loop.run()
    assert result.terminated_reason == "skip"


def test_loop_respects_max_steps():
    # Every reply is a tool call → never terminates of its own accord.
    loop = AgenticTradeLoop(
        asset="BTC",
        llm_call=lambda msgs: json.dumps({"tool_call": {"name": "echo", "args": {}}}),
        registry=_mini_registry(),
        max_steps=3,
    )
    loop.seed()
    result = loop.run()
    assert result.terminated_reason == "max_steps"
    assert result.steps_taken == 3


def test_loop_llm_exception_returns_empty_string():
    # LLM raises → _call_llm swallows → JSON parse fails → parse_failures path.
    def boom(_):
        raise RuntimeError("network")
    loop = AgenticTradeLoop(asset="BTC", llm_call=boom, registry=_mini_registry())
    loop.seed()
    result = loop.run()
    assert result.terminated_reason == "parse_failures"


def test_default_max_steps_sane():
    assert DEFAULT_MAX_STEPS >= 4
