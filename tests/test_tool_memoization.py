"""Tests for per-tick tool-result memoization in ToolRegistry."""
from __future__ import annotations

import pytest


def test_memoization_disabled_when_no_tick_id():
    """When tick_id is None (default), every call hits the handler."""
    from src.ai.trade_tools import ToolRegistry, Tool

    call_count = {"n": 0}

    def handler(args):
        call_count["n"] += 1
        return {"value": call_count["n"]}

    r = ToolRegistry()
    r.register(Tool(name="counter", description="t",
                    input_schema={"type": "object", "properties": {}},
                    handler=handler))
    r.dispatch("counter", {})
    r.dispatch("counter", {})
    assert call_count["n"] == 2  # both fired


def test_memoization_caches_within_same_tick():
    from src.ai.trade_tools import ToolRegistry, Tool

    call_count = {"n": 0}

    def handler(args):
        call_count["n"] += 1
        return {"value": call_count["n"]}

    r = ToolRegistry()
    r.register(Tool(name="counter", description="t",
                    input_schema={"type": "object", "properties": {}},
                    handler=handler))
    r.set_tick_id("tick_1")
    a = r.dispatch("counter", {})
    b = r.dispatch("counter", {})
    assert a == b               # same result returned from cache
    assert call_count["n"] == 1  # handler ran exactly once


def test_memoization_different_args_separate_cache():
    from src.ai.trade_tools import ToolRegistry, Tool

    call_count = {"n": 0}

    def handler(args):
        call_count["n"] += 1
        return {"value": args.get("x", 0) * 2}

    r = ToolRegistry()
    r.register(Tool(name="doubler", description="t",
                    input_schema={"type": "object", "properties": {}},
                    handler=handler))
    r.set_tick_id("tick_1")
    r.dispatch("doubler", {"x": 3})
    r.dispatch("doubler", {"x": 5})
    r.dispatch("doubler", {"x": 3})  # cache hit
    assert call_count["n"] == 2  # x=3 once, x=5 once


def test_memoization_resets_on_tick_change():
    from src.ai.trade_tools import ToolRegistry, Tool

    call_count = {"n": 0}

    def handler(args):
        call_count["n"] += 1
        return {"value": call_count["n"]}

    r = ToolRegistry()
    r.register(Tool(name="counter", description="t",
                    input_schema={"type": "object", "properties": {}},
                    handler=handler))
    r.set_tick_id("tick_1")
    r.dispatch("counter", {})
    r.set_tick_id("tick_2")  # new tick → cache cleared
    r.dispatch("counter", {})
    assert call_count["n"] == 2


def test_write_tools_bypass_cache():
    """Tools tagged 'write' must always execute (side effects)."""
    from src.ai.trade_tools import ToolRegistry, Tool

    call_count = {"n": 0}

    def handler(args):
        call_count["n"] += 1
        return {"executed": call_count["n"]}

    r = ToolRegistry()
    r.register(Tool(name="writer", description="t",
                    input_schema={"type": "object", "properties": {}},
                    handler=handler, tag="write"))
    r.set_tick_id("tick_1")
    r.dispatch("writer", {})
    r.dispatch("writer", {})  # write tool bypasses cache
    assert call_count["n"] == 2


def test_cache_stats():
    from src.ai.trade_tools import ToolRegistry, Tool

    def handler(args):
        return {"ok": True}

    r = ToolRegistry()
    r.register(Tool(name="t", description="t",
                    input_schema={"type": "object", "properties": {}},
                    handler=handler))
    r.set_tick_id("tick_1")
    r.dispatch("t", {})
    r.dispatch("t", {})
    stats = r.cache_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["current_tick_id"] == "tick_1"


def test_existing_default_registry_unaffected():
    """Make sure the default registry (no tick_id set) still works
    end-to-end without caching."""
    from src.ai.trade_tools import build_default_registry
    r = build_default_registry()
    res1 = r.dispatch("query_session_bias", {})
    res2 = r.dispatch("query_session_bias", {})
    # Both calls succeed (no caching by default — tick_id is None)
    assert "session" in str(res1).lower()
    assert "session" in str(res2).lower()
