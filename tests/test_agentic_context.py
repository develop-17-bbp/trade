"""Tests for src/ai/agentic_context.py — multi-turn message window + token budget."""
from __future__ import annotations

import pytest

from src.ai.agentic_context import (
    DEFAULT_KEEP_RECENT_ROUNDS,
    DEFAULT_MAX_TOKENS,
    AgenticContext,
    count_message_tokens,
)


# ── Seed context ────────────────────────────────────────────────────────


def test_build_seed_context_minimal():
    ctx = AgenticContext(asset="BTC")
    msgs = ctx.build_seed_context()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "MARKET REGIME" in msgs[1]["content"]
    assert "YOUR TURN" in msgs[1]["content"]


def test_build_seed_context_with_quant_and_rag():
    ctx = AgenticContext(asset="ETH")
    msgs = ctx.build_seed_context(
        regime="TRENDING",
        quant_data="[PRICE=3000] [RSI=62] [EMA8=2950]",
        similar_trades=[
            {"similarity": 0.91, "metadata": {"regime": "TRENDING", "pnl_pct": 1.4}},
            {"similarity": 0.87, "metadata": {"regime": "TRENDING", "pnl_pct": -0.8}},
        ],
        recent_critiques=[
            {"matched_thesis": True, "miss_reason": "", "confidence_calibration_delta": 0.05},
            {"matched_thesis": False, "miss_reason": "exited too early", "confidence_calibration_delta": -0.10},
        ],
    )
    body = msgs[1]["content"]
    assert "TRENDING" in body
    assert "PRICE=3000" in body
    assert "similarity=0.91" in body
    assert "miss=exited too early" in body


def test_seed_context_updates_token_count():
    ctx = AgenticContext(asset="BTC")
    assert ctx.token_count == 0
    ctx.build_seed_context(quant_data="[PRICE=60000]" * 10)
    assert ctx.token_count > 0


# ── Append / utilization ────────────────────────────────────────────────


def test_append_grows_message_list():
    ctx = AgenticContext(asset="BTC")
    ctx.build_seed_context()
    before = len(ctx.messages)
    ctx.append({"role": "assistant", "content": "OK"})
    assert len(ctx.messages) == before + 1


def test_token_budget_warn_and_exceeded():
    ctx = AgenticContext(asset="BTC", max_tokens=100)
    ctx.build_seed_context(quant_data="x" * 2000)  # definitely over budget
    assert ctx.token_budget_exceeded() is True
    assert ctx.utilization_pct() > 75.0
    assert ctx.should_warn() is True


def test_utilization_under_budget():
    ctx = AgenticContext(asset="BTC", max_tokens=10_000)
    ctx.build_seed_context(quant_data="short")
    assert ctx.token_budget_exceeded() is False


# ── Summarization ───────────────────────────────────────────────────────


def test_summarize_noop_when_below_keep_threshold():
    ctx = AgenticContext(asset="BTC", keep_recent_rounds=4)
    ctx.build_seed_context()
    ctx.append({"role": "assistant", "content": "r1"})
    ctx.append({"role": "user", "content": "r2"})
    n_before = len(ctx.messages)
    ctx.summarize_older_rounds()
    assert len(ctx.messages) == n_before  # nothing to compact yet


def test_summarize_compacts_middle_with_mechanical_fallback():
    ctx = AgenticContext(asset="BTC", keep_recent_rounds=2)
    ctx.build_seed_context()
    # Seven middle rounds to be compacted.
    for i in range(7):
        ctx.append({"role": "assistant", "content": f"tool-use round {i}"})
    # Custom summarizer that always errors → forces fallback path.
    def broken(_middle):
        raise RuntimeError("boom")
    ctx.summarize_older_rounds(summarizer=broken)
    # Head (2) + summary (1) + last keep_recent_rounds (2) = 5 messages.
    assert len(ctx.messages) == 5
    assert "SUMMARY OF EARLIER" in ctx.messages[2]["content"]


def test_summarize_with_custom_summarizer():
    ctx = AgenticContext(asset="BTC", keep_recent_rounds=1)
    ctx.build_seed_context()
    for i in range(5):
        ctx.append({"role": "assistant", "content": f"round {i}"})
    ctx.summarize_older_rounds(summarizer=lambda middle: f"compacted {len(middle)} rounds")
    summary_msg = ctx.messages[2]
    assert "compacted" in summary_msg["content"]


# ── Token counter ───────────────────────────────────────────────────────


def test_count_message_tokens_list_content():
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [{"type": "tool_use", "name": "get_bars"}]},
    ]
    n = count_message_tokens(msgs)
    assert n > 0


def test_default_tunables_are_sane():
    assert DEFAULT_MAX_TOKENS >= 4000
    assert DEFAULT_KEEP_RECENT_ROUNDS >= 1
