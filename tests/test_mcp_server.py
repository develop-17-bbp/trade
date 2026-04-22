"""Tests for src/mcp_server/tools.py — pure tool implementations.

Does not spin up the FastMCP HTTP server; tests the tool functions directly
so they're fast + hermetic. Covers: read-only tool return shapes, path-
traversal rejection, mutation gate default-off, secret redaction.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def test_mutations_disabled_by_default(monkeypatch):
    from src.mcp_server.tools import mutations_allowed

    monkeypatch.delenv("ACT_MCP_ALLOW_MUTATIONS", raising=False)
    assert mutations_allowed() is False

    monkeypatch.setenv("ACT_MCP_ALLOW_MUTATIONS", "0")
    assert mutations_allowed() is False


def test_mutations_enabled_via_env(monkeypatch):
    from src.mcp_server.tools import mutations_allowed

    for val in ("1", "true", "YES", "on"):
        monkeypatch.setenv("ACT_MCP_ALLOW_MUTATIONS", val)
        assert mutations_allowed() is True, f"should enable for {val!r}"


def test_restart_bot_refuses_when_gated(monkeypatch):
    from src.mcp_server.tools import restart_bot

    monkeypatch.delenv("ACT_MCP_ALLOW_MUTATIONS", raising=False)
    result = restart_bot()
    assert result["error"] == "mutations_disabled"
    assert "ACT_MCP_ALLOW_MUTATIONS" in result["message"]


def test_trigger_retrain_refuses_when_gated(monkeypatch):
    from src.mcp_server.tools import trigger_retrain

    monkeypatch.delenv("ACT_MCP_ALLOW_MUTATIONS", raising=False)
    assert trigger_retrain("BTC")["error"] == "mutations_disabled"


def test_trigger_retrain_rejects_unknown_asset(monkeypatch):
    from src.mcp_server.tools import trigger_retrain

    monkeypatch.setenv("ACT_MCP_ALLOW_MUTATIONS", "1")
    result = trigger_retrain("NOT_A_REAL_ASSET_XYZ")
    # unknown asset is caught before subprocess; no 10-min timeout risk
    assert "unknown_asset" in result.get("error", "")


def test_tail_log_rejects_path_traversal():
    from src.mcp_server.tools import tail_log

    assert tail_log("../secret.txt")["error"] == "invalid_log_name"
    assert tail_log("..\\secret.txt")["error"] == "invalid_log_name"
    assert tail_log(".hidden")["error"] == "invalid_log_name"


def test_tail_log_rejects_non_text_extensions():
    from src.mcp_server.tools import tail_log

    assert "only_text" in tail_log("state.pkl").get("error", "")
    assert "only_text" in tail_log("data.sqlite").get("error", "")


def test_tail_log_caps_line_count(tmp_path, monkeypatch):
    from src.mcp_server import tools

    # Create a log file with 100 lines in a fake logs/ dir
    logs = tmp_path / "logs"
    logs.mkdir()
    log = logs / "smoke.log"
    log.write_text("\n".join(f"line{i}" for i in range(100)), encoding="utf-8")

    monkeypatch.setattr(tools, "REPO_ROOT", tmp_path)
    result = tools.tail_log("smoke.log", lines=50)
    assert result["returned"] == 50
    assert result["total_lines"] == 100


def test_tail_log_missing_file_returns_error(tmp_path, monkeypatch):
    from src.mcp_server import tools

    monkeypatch.setattr(tools, "REPO_ROOT", tmp_path)
    (tmp_path / "logs").mkdir()
    result = tools.tail_log("nonexistent.log")
    assert "not_found" in result.get("error", "")


def test_list_logs_ignores_non_text(tmp_path, monkeypatch):
    from src.mcp_server import tools

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "a.log").write_text("x")
    (logs / "b.jsonl").write_text("x")
    (logs / "c.pkl").write_bytes(b"x")
    (logs / "d.parquet").write_bytes(b"x")

    monkeypatch.setattr(tools, "REPO_ROOT", tmp_path)
    r = tools.list_logs()
    names = {log["name"] for log in r["logs"]}
    assert "a.log" in names
    assert "b.jsonl" in names
    assert "c.pkl" not in names
    assert "d.parquet" not in names


def test_env_flags_redacts_secrets(monkeypatch):
    from src.mcp_server.tools import env_flags

    monkeypatch.setenv("DASHBOARD_API_KEY", "real-secret-never-expose")
    monkeypatch.setenv("ACT_MCP_TOKEN", "another-secret")
    monkeypatch.setenv("ACT_DISABLE_ML", "1")

    env = env_flags()["env"]
    assert env["ACT_DISABLE_ML"] == "1"
    assert env["DASHBOARD_API_KEY"] == "(set)"
    assert env["ACT_MCP_TOKEN"] == "(set)"
    # Ensure the actual value never leaks
    blob = json.dumps(env_flags())
    assert "real-secret-never-expose" not in blob
    assert "another-secret" not in blob


def test_env_flags_reports_unset_as_none(monkeypatch):
    from src.mcp_server.tools import env_flags

    monkeypatch.delenv("DASHBOARD_API_KEY", raising=False)
    monkeypatch.delenv("ACT_MCP_TOKEN", raising=False)
    env = env_flags()["env"]
    assert env["DASHBOARD_API_KEY"] is None
    assert env["ACT_MCP_TOKEN"] is None


def test_git_status_returns_expected_keys():
    """Runs against the actual repo — must at least parse."""
    from src.mcp_server.tools import git_status

    r = git_status()
    # Either succeeds fully or returns a clean error
    if "error" in r:
        pytest.skip(f"git not available in test env: {r['error']}")
    assert "head" in r and len(r["head"]) == 40
    assert "branch" in r
    assert "uncommitted_count" in r


def test_component_state_returns_expected_shape():
    from src.mcp_server.tools import component_state

    r = component_state()
    assert "components" in r
    assert isinstance(r["components"], list)
    assert len(r["components"]) >= 6   # at least the core ACT_* flags


def test_status_returns_dict_even_on_missing_files(tmp_path, monkeypatch):
    """Status should work even on a fresh box with no logs yet."""
    from src.mcp_server import tools

    monkeypatch.setattr(tools, "REPO_ROOT", tmp_path)
    (tmp_path / "logs").mkdir()
    r = tools.status()
    assert isinstance(r, dict)
    # paper_state + safe_entries may be absent; readiness_gate still runs


def test_mcp_server_builds_and_registers_all_tools():
    """Confirm the FastMCP app registers the expected tool names.
    Regression guard: if we rename or drop a tool, this fails fast."""
    import asyncio
    from src.mcp_server.act_mcp import build_server

    mcp = build_server()
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    expected = {
        "status", "evaluator_report", "component_state", "paper_state",
        "shadow_stats", "readiness_gate", "tail_log", "list_logs",
        "recent_trades", "git_status", "env_flags",
        "restart_bot", "trigger_retrain",
    }
    assert expected.issubset(names), f"missing tools: {expected - names}"
