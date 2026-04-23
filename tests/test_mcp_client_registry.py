"""Tests for src/ai/mcp_client_registry.py — external MCP client mirroring."""
from __future__ import annotations

from unittest import mock

import pytest

from src.ai.mcp_client_registry import (
    DISABLE_ENV,
    MCPHTTPClient,
    _expand_env,
    _normalize_response,
    list_registered_mcp_tools,
    register_all_from_config,
    register_mcp_server,
)
from src.ai.trade_tools import ToolRegistry


# ── _expand_env ─────────────────────────────────────────────────────────


def test_expand_env_simple(monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "abc123")
    assert _expand_env("${MY_TOKEN}") == "abc123"
    assert _expand_env("Bearer ${MY_TOKEN}") == "Bearer abc123"


def test_expand_env_missing_becomes_empty(monkeypatch):
    monkeypatch.delenv("UNSET_VAR", raising=False)
    assert _expand_env("${UNSET_VAR}") == ""


def test_expand_env_dict_and_list(monkeypatch):
    monkeypatch.setenv("X", "val")
    assert _expand_env({"a": "${X}", "b": ["${X}", "static"]}) == {
        "a": "val", "b": ["val", "static"],
    }


def test_expand_env_non_string_passes_through():
    assert _expand_env(42) == 42
    assert _expand_env(None) is None


# ── _normalize_response ─────────────────────────────────────────────────


def test_normalize_fastmcp_content_list():
    resp = {"content": [
        {"type": "text", "text": "BTC price is 60000"},
        {"type": "text", "text": "sentiment bullish"},
    ]}
    out = _normalize_response(resp, "get_price")
    assert "summary" in out
    assert "BTC price" in out["summary"]
    assert "sentiment bullish" in out["summary"]
    assert out["tool"] == "get_price"


def test_normalize_handles_json_block():
    resp = {"content": [{"type": "json", "data": {"x": 1, "y": 2}}]}
    out = _normalize_response(resp, "tool")
    assert "1" in out["summary"] and "2" in out["summary"]


def test_normalize_plain_dict_fallback():
    out = _normalize_response({"foo": "bar"}, "tool")
    assert "foo" in out["summary"] and "bar" in out["summary"]


def test_normalize_non_dict_returns_summary():
    out = _normalize_response("raw string", "tool")
    assert out == {"summary": "raw string"}


# ── register_mcp_server ─────────────────────────────────────────────────


def _fake_client(tools):
    fake = mock.MagicMock()
    fake.list_tools.return_value = tools
    return lambda: fake


def test_register_mirrors_remote_tools():
    reg = ToolRegistry()
    remote_tools = [
        {"name": "get_price", "description": "price", "inputSchema": {"type": "object"}},
        {"name": "get_news",  "description": "news",  "inputSchema": {"type": "object"}},
    ]
    registered = register_mcp_server(
        reg, url="http://x", tag="cg",
        client_factory=_fake_client(remote_tools),
    )
    assert registered == ["cg_get_price", "cg_get_news"]
    assert reg.get("cg_get_price") is not None
    assert reg.get("cg_get_price").description.startswith("[MCP:cg]")


def test_register_empty_tools_list():
    reg = ToolRegistry()
    registered = register_mcp_server(
        reg, url="http://x", tag="empty",
        client_factory=_fake_client([]),
    )
    assert registered == []


def test_register_server_unreachable_logs_and_returns_empty():
    reg = ToolRegistry()
    def boom():
        raise RuntimeError("connection refused")
    registered = register_mcp_server(reg, url="http://nope", tag="dead",
                                     client_factory=boom)
    assert registered == []
    assert reg.list_names() == []


def test_registered_tool_forwards_to_client():
    fake = mock.MagicMock()
    fake.list_tools.return_value = [{"name": "ping", "description": "", "inputSchema": {"type": "object"}}]
    fake.call_tool.return_value = {"content": [{"type": "text", "text": "pong"}]}
    reg = ToolRegistry()
    register_mcp_server(reg, url="http://x", tag="t", client_factory=lambda: fake)
    out = reg.dispatch("t_ping", {"arg": 1})
    import json
    assert "pong" in json.loads(out)["summary"]
    fake.call_tool.assert_called_once_with("ping", {"arg": 1})


def test_registered_tool_handles_call_error():
    fake = mock.MagicMock()
    fake.list_tools.return_value = [{"name": "ping", "description": "", "inputSchema": {"type": "object"}}]
    fake.call_tool.side_effect = RuntimeError("server 500")
    reg = ToolRegistry()
    register_mcp_server(reg, url="http://x", tag="t", client_factory=lambda: fake)
    out = reg.dispatch("t_ping")
    import json
    parsed = json.loads(out)
    assert "error" in parsed
    assert "RuntimeError" in parsed["error"]


def test_register_skips_malformed_tool_entries():
    reg = ToolRegistry()
    bad = [
        "not a dict",
        {"no_name": True},
        {"name": "", "description": ""},
        {"name": "good", "description": "ok", "inputSchema": {"type": "object"}},
    ]
    registered = register_mcp_server(reg, url="http://x", tag="t",
                                     client_factory=_fake_client(bad))
    assert registered == ["t_good"]


# ── register_all_from_config ────────────────────────────────────────────


def test_bulk_register_skips_when_env_disabled(monkeypatch):
    monkeypatch.setenv(DISABLE_ENV, "1")
    reg = ToolRegistry()
    cfg = {"mcp_clients": [{"url": "http://x", "tag": "t"}]}
    out = register_all_from_config(reg, cfg)
    assert out == {}
    assert reg.list_names() == []


def test_bulk_register_skips_disabled_entries(monkeypatch):
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    reg = ToolRegistry()
    cfg = {"mcp_clients": [
        {"url": "http://x1", "tag": "t1", "enabled": False},
        {"url": "http://x2", "tag": "t2"},
    ]}
    with mock.patch("src.ai.mcp_client_registry.register_mcp_server",
                    return_value=["t2_ping"]) as reg_mock:
        out = register_all_from_config(reg, cfg)
    assert out == {"t2": ["t2_ping"]}
    reg_mock.assert_called_once()
    # The one call should have been for t2, not t1.
    args, kwargs = reg_mock.call_args
    assert kwargs.get("url") == "http://x2" or args[1] == "http://x2"


def test_bulk_register_none_config():
    reg = ToolRegistry()
    assert register_all_from_config(reg, None) == {}
    assert register_all_from_config(reg, {"mcp_clients": "not a list"}) == {}


def test_bulk_register_expands_env_in_headers(monkeypatch):
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    monkeypatch.setenv("MY_TOKEN", "sekret")
    reg = ToolRegistry()
    cfg = {"mcp_clients": [
        {"url": "http://x", "tag": "t", "headers": {"Authorization": "Bearer ${MY_TOKEN}"}},
    ]}
    captured = {}
    def _spy(registry, *, url, tag, headers, timeout_s):
        captured["headers"] = headers
        return []
    with mock.patch("src.ai.mcp_client_registry.register_mcp_server", side_effect=_spy):
        register_all_from_config(reg, cfg)
    assert captured["headers"]["Authorization"] == "Bearer sekret"


# ── Catalogue listing ──────────────────────────────────────────────────


def test_list_registered_mcp_tools_filters_by_tag():
    reg = ToolRegistry()
    register_mcp_server(reg, url="http://a", tag="cg",
                        client_factory=_fake_client([
                            {"name": "price", "description": "desc", "inputSchema": {"type": "object"}}]))
    # Also add a native tool that isn't MCP-origin.
    from src.ai.trade_tools import Tool
    reg.register(Tool("native", "native desc", {"type": "object"}, lambda args: {"ok": True}))
    names = [t["name"] for t in list_registered_mcp_tools(reg)]
    assert "cg_price" in names
    assert "native" not in names
