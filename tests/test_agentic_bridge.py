"""Tests for src/ai/agentic_bridge.py — end-to-end glue + feature flag + CLI."""
from __future__ import annotations

import json
import os
from unittest import mock

import pytest

from src.ai.agentic_bridge import (
    agentic_loop_enabled,
    build_llm_call,
    compile_agentic_plan,
    load_recent_critiques,
    load_similar_trades,
)
from src.ai.trade_tools import Tool, ToolRegistry
from src.trading.trade_plan import TradePlan


# ── Feature flag resolution ─────────────────────────────────────────────


def test_flag_env_on_beats_config(monkeypatch):
    monkeypatch.setenv("ACT_AGENTIC_LOOP", "1")
    monkeypatch.delenv("ACT_DISABLE_AGENTIC_LOOP", raising=False)
    assert agentic_loop_enabled({"agentic_loop": {"enabled": False}}) is True


def test_flag_env_off_beats_config(monkeypatch):
    monkeypatch.setenv("ACT_AGENTIC_LOOP", "0")
    monkeypatch.delenv("ACT_DISABLE_AGENTIC_LOOP", raising=False)
    assert agentic_loop_enabled({"agentic_loop": {"enabled": True}}) is False


def test_flag_kill_switch_wins(monkeypatch):
    monkeypatch.setenv("ACT_AGENTIC_LOOP", "1")
    monkeypatch.setenv("ACT_DISABLE_AGENTIC_LOOP", "1")
    assert agentic_loop_enabled({"agentic_loop": {"enabled": True}}) is False


def test_flag_falls_through_to_config(monkeypatch):
    monkeypatch.delenv("ACT_AGENTIC_LOOP", raising=False)
    monkeypatch.delenv("ACT_DISABLE_AGENTIC_LOOP", raising=False)
    assert agentic_loop_enabled({"agentic_loop": {"enabled": True}}) is True
    assert agentic_loop_enabled({"agentic_loop": {"enabled": False}}) is False
    assert agentic_loop_enabled({}) is False
    assert agentic_loop_enabled(None) is False


# ── compile_agentic_plan: kill switch ───────────────────────────────────


def test_compile_disabled_returns_skip(monkeypatch):
    monkeypatch.setenv("ACT_DISABLE_AGENTIC_LOOP", "1")
    result = compile_agentic_plan(asset="BTC")
    assert result.terminated_reason == "disabled"
    assert result.plan.direction == "SKIP"


# ── compile_agentic_plan: full path with injected LLM ───────────────────


def test_compile_injected_llm_yields_plan(monkeypatch):
    monkeypatch.delenv("ACT_DISABLE_AGENTIC_LOOP", raising=False)
    plan_json = {
        "plan": {
            "direction": "LONG", "entry_tier": "normal",
            "entry_price": 60000.0, "size_pct": 5.0, "sl_price": 58800.0,
        }
    }

    def stub_llm(_messages):
        return json.dumps(plan_json)

    # Minimal registry — we don't want bridge side-effects pulling real
    # tools into this test.
    reg = ToolRegistry()
    reg.register(Tool("echo", "", {"type": "object"}, lambda args: {"ok": True}))

    result = compile_agentic_plan(
        asset="BTC",
        quant_data="[PRICE=60000]",
        registry=reg,
        llm_call=stub_llm,
        similar_trades=[],           # skip the MemoryVault load
        recent_critiques=[],         # skip the warm_store read
        max_steps=3,
    )
    assert result.terminated_reason == "plan"
    assert isinstance(result.plan, TradePlan)
    assert result.plan.asset == "BTC"
    assert result.plan.direction == "LONG"


def test_compile_injected_skip(monkeypatch):
    monkeypatch.delenv("ACT_DISABLE_AGENTIC_LOOP", raising=False)
    result = compile_agentic_plan(
        asset="ETH",
        llm_call=lambda _m: json.dumps({"skip": "low vol"}),
        similar_trades=[],
        recent_critiques=[],
        max_steps=3,
    )
    assert result.terminated_reason == "skip"
    assert result.plan.direction == "SKIP"
    assert "low vol" in result.plan.thesis


def test_compile_swallows_internal_error(monkeypatch):
    monkeypatch.delenv("ACT_DISABLE_AGENTIC_LOOP", raising=False)

    def boom(_m):
        raise RuntimeError("LLM down")

    result = compile_agentic_plan(
        asset="BTC",
        llm_call=boom,
        similar_trades=[],
        recent_critiques=[],
        max_steps=2,
    )
    # The loop's internal handling catches the exception per-turn; after
    # MAX_PARSE_FAILURES the loop bails with a skip plan.
    assert result.terminated_reason in ("parse_failures", "skip")
    assert result.plan.direction == "SKIP"


# ── build_llm_call ──────────────────────────────────────────────────────


def test_build_llm_call_empty_messages_returns_empty():
    call = build_llm_call()
    assert call([]) == ""


def test_build_llm_call_uses_router_when_available():
    fake_provider = mock.MagicMock()
    fake_provider.generate.return_value = {"response": "ok"}
    fake_router = mock.MagicMock()
    fake_router.providers = {"local": fake_provider}
    with mock.patch("src.ai.llm_provider.LLMRouter", return_value=fake_router):
        call = build_llm_call()
        out = call([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ])
    assert out == "ok"
    # System prompt should have been extracted; provider gets prompt + sys.
    _args, kwargs = fake_provider.generate.call_args
    assert kwargs["system_prompt"] == "sys"
    assert "[user]" in kwargs["prompt"]


def test_build_llm_call_returns_empty_on_router_failure():
    with mock.patch("src.ai.llm_provider.LLMRouter", side_effect=RuntimeError("no")):
        call = build_llm_call()
        assert call([{"role": "user", "content": "x"}]) == ""


# ── Seed loaders ────────────────────────────────────────────────────────


def test_load_recent_critiques_reads_warm_store(tmp_path, monkeypatch):
    from src.orchestration.warm_store import WarmStore
    import src.orchestration.warm_store as ws_mod

    db = tmp_path / "brg.sqlite"
    store = WarmStore(str(db))
    for i in range(3):
        store.write_decision({
            "decision_id": f"d-{i}", "symbol": "BTC",
            "self_critique": {"matched_thesis": i % 2 == 0, "idx": i},
        })
    store.flush()
    monkeypatch.setattr(ws_mod, "_store_singleton", store, raising=False)

    critiques = load_recent_critiques(limit=5)
    assert len(critiques) == 3
    assert all("matched_thesis" in c for c in critiques)


def test_load_recent_critiques_handles_missing_store(monkeypatch):
    with mock.patch("src.orchestration.warm_store.get_store", side_effect=RuntimeError("x")):
        assert load_recent_critiques() == []


def test_load_similar_trades_graceful_on_memoryvault_error():
    with mock.patch("src.ai.memory_vault.MemoryVault", side_effect=RuntimeError("nope")):
        assert load_similar_trades("BTC", "TRENDING") == []


# ── CLI ────────────────────────────────────────────────────────────────


def test_cli_stub_llm_exits_zero(capsys):
    import sys
    from src.ai import agentic_bridge as brg

    argv = ["agentic_bridge", "--asset", "BTC", "--stub-llm", "--max-steps", "2"]
    with mock.patch.object(sys, "argv", argv):
        rc = brg._cli()
    assert rc == 0
    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert out["plan"]["direction"] == "SKIP"
    assert out["terminated_reason"] == "skip"
