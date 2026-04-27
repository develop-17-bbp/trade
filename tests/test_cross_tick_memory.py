"""Unit 7 — Cross-tick analyst memory (chain-of-reasoning).

The analyst should see its OWN last-5 decisions for an asset when
seeded for a new tick. This gives it continuity instead of starting
fresh every tick.
"""
from __future__ import annotations

import json
import time

import pytest

from src.ai.brain_memory import (
    AnalystTrace,
    BrainMemory,
    read_recent_analyst_traces,
)


# ── read_recent_analyst_traces helper ───────────────────────────────────


def test_read_recent_analyst_traces_default_limit_and_window(monkeypatch, tmp_path):
    """The new helper is a thin, safe wrapper that returns up to `limit`
    traces inside `max_age_s`. Defaults: limit=5, max_age_s=900."""
    from src.ai import brain_memory as bm

    fresh = BrainMemory(str(tmp_path / "u7.sqlite"))
    monkeypatch.setattr(bm, "_brain_singleton", fresh, raising=False)

    now = time.time()
    # Seed 7 traces — only the 5 most-recent within 15 min should come back.
    for i in range(7):
        fresh.write_analyst_trace(AnalystTrace(
            asset="BTC", ts=now - i * 30, plan_id=f"p-{i}",
            direction="LONG" if i % 2 == 0 else "SKIP",
            tier="normal", size_pct=1.0,
            thesis=f"thesis-{i}", verdict="plan" if i % 2 == 0 else "skip",
        ))
    # Plus one ancient trace beyond the 15-min window.
    fresh.write_analyst_trace(AnalystTrace(
        asset="BTC", ts=now - 7200, plan_id="ancient",
        direction="LONG", tier="normal", size_pct=1.0,
        thesis="ancient", verdict="plan",
    ))

    got = read_recent_analyst_traces("BTC")
    assert len(got) == 5
    # Most recent first.
    assert got[0].plan_id == "p-0"
    assert all(t.plan_id != "ancient" for t in got)


def test_read_recent_analyst_traces_swallows_errors(monkeypatch):
    from src.ai import brain_memory as bm

    class _Broken:
        def read_recent_traces(self, *_, **__):
            raise RuntimeError("boom")

    monkeypatch.setattr(bm, "_brain_singleton", _Broken(), raising=False)
    assert read_recent_analyst_traces("BTC") == []


# ── Trace block formatting ──────────────────────────────────────────────


def test_traces_block_format_matches_unit_spec(monkeypatch, tmp_path):
    """The trace block should emit lines like:
       [HH:MM] LONG, thesis=..., verdict=plan
       [HH:MM] SKIP, reason=...
    """
    from src.ai import brain_memory as bm
    from src.ai.context_builders import _traces_block, clear_cache

    fresh = BrainMemory(str(tmp_path / "u7-fmt.sqlite"))
    monkeypatch.setattr(bm, "_brain_singleton", fresh, raising=False)
    clear_cache()

    now = time.time()
    fresh.write_analyst_trace(AnalystTrace(
        asset="BTC", ts=now - 120, plan_id="p-1",
        direction="LONG", tier="normal", size_pct=5.0,
        thesis="tf_aligned + macro_neutral", verdict="plan",
    ))
    fresh.write_analyst_trace(AnalystTrace(
        asset="BTC", ts=now - 60, plan_id="p-2",
        direction="SKIP", tier="-", size_pct=0.0,
        thesis="parse_failure", verdict="skip",
    ))
    fresh.write_analyst_trace(AnalystTrace(
        asset="BTC", ts=now - 10, plan_id="p-3",
        direction="LONG", tier="normal", size_pct=5.0,
        thesis="trend_continuation", verdict="plan",
    ))

    block = _traces_block("BTC")
    assert block.startswith("Recent thoughts")
    assert "LONG" in block
    assert "SKIP" in block
    assert "thesis=" in block
    assert "verdict=plan" in block
    assert "reason=" in block
    # Cap budget — unit spec demands ≤300 chars.
    assert len(block) <= 300


def test_traces_block_empty_when_no_history(monkeypatch, tmp_path):
    from src.ai import brain_memory as bm
    from src.ai.context_builders import _traces_block, clear_cache

    fresh = BrainMemory(str(tmp_path / "u7-empty.sqlite"))
    monkeypatch.setattr(bm, "_brain_singleton", fresh, raising=False)
    clear_cache()

    assert _traces_block("BTC") == ""


# ── EvidenceDocument integration ────────────────────────────────────────


def test_evidence_document_includes_recent_traces(monkeypatch, tmp_path):
    from src.ai import brain_memory as bm
    from src.ai.context_builders import build_evidence_document, clear_cache

    fresh = BrainMemory(str(tmp_path / "u7-ed.sqlite"))
    monkeypatch.setattr(bm, "_brain_singleton", fresh, raising=False)
    clear_cache()

    now = time.time()
    fresh.write_analyst_trace(AnalystTrace(
        asset="ETH", ts=now - 30, plan_id="e-1",
        direction="LONG", tier="normal", size_pct=3.0,
        thesis="bullish breakout", verdict="plan",
    ))

    doc = build_evidence_document(
        "ETH",
        include_scanner=False, include_news=False,
        include_fear_greed=False, include_graph=False,
        include_body_controls=False, include_traces=True,
    )
    sec = doc.section("RECENT_ANALYST_TRACES")
    assert sec is not None
    assert "Recent thoughts" in sec.content
    assert "LONG" in sec.content


# ── compile_agentic_plan: traces appear in the LLM prompt ───────────────


def _stub_llm_capture(seen):
    def _call(messages):
        texts = [str(m.get("content")) for m in messages if m.get("role") == "user"]
        seen["user_text"] = "\n---\n".join(texts)
        # Return a valid skip envelope so the loop terminates immediately.
        return json.dumps({"skip": "test"})
    return _call


def test_compile_agentic_plan_injects_recent_traces(monkeypatch, tmp_path):
    from src.ai import brain_memory as bm
    from src.ai.context_builders import clear_cache
    import src.ai.agentic_bridge as brg

    fresh = BrainMemory(str(tmp_path / "u7-bridge.sqlite"))
    monkeypatch.setattr(bm, "_brain_singleton", fresh, raising=False)
    monkeypatch.delenv("ACT_DISABLE_AGENTIC_LOOP", raising=False)
    clear_cache()

    # Seed five analyst traces — the analyst should see them on its
    # next compile.
    now = time.time()
    seeded = [
        ("LONG", "tf_aligned + macro_neutral", "plan"),
        ("SKIP", "parse_failure", "skip"),
        ("LONG", "trend_continuation", "plan"),
        ("SHORT", "rejection_at_resistance", "plan"),
        ("LONG", "vwap_reclaim", "plan"),
    ]
    for i, (direction, thesis, verdict) in enumerate(seeded):
        fresh.write_analyst_trace(AnalystTrace(
            asset="BTC", ts=now - (300 - i * 30),
            plan_id=f"p-{i}", direction=direction, tier="normal",
            size_pct=5.0, thesis=thesis, verdict=verdict,
        ))

    seen: dict = {}
    result = brg.compile_agentic_plan(
        asset="BTC",
        quant_data="[PRICE=78200]",
        llm_call=_stub_llm_capture(seen),
        similar_trades=[],
        recent_critiques=[],
        max_steps=2,
    )
    assert result.terminated_reason == "skip"
    user_text = seen.get("user_text", "")
    assert "Recent thoughts" in user_text, (
        "analyst seed must include the Recent-thoughts trace block"
    )
    # At least one of the seeded thesis fragments should make it into
    # the trace block (truncation may drop the last lines past 300 chars,
    # so we don't assert all five — just continuity).
    assert any(
        frag in user_text for frag in (
            "tf_aligned", "trend_continuation", "vwap_reclaim",
            "rejection_at_resistance", "parse_failure",
        )
    )


def test_compile_agentic_plan_with_empty_trace_history_still_seeds(
    monkeypatch, tmp_path,
):
    """Edge case: brand-new asset / empty trace history. The analyst
    should still seed normally — no Recent-thoughts block, but other
    seed sections must still arrive."""
    from src.ai import brain_memory as bm
    from src.ai.context_builders import clear_cache
    import src.ai.agentic_bridge as brg

    fresh = BrainMemory(str(tmp_path / "u7-empty-bridge.sqlite"))
    monkeypatch.setattr(bm, "_brain_singleton", fresh, raising=False)
    monkeypatch.delenv("ACT_DISABLE_AGENTIC_LOOP", raising=False)
    clear_cache()

    seen: dict = {}
    result = brg.compile_agentic_plan(
        asset="SOL",
        quant_data="[PRICE=200]",
        llm_call=_stub_llm_capture(seen),
        similar_trades=[],
        recent_critiques=[],
        max_steps=2,
    )
    assert result.terminated_reason == "skip"
    user_text = seen.get("user_text", "")
    # Quant-data block is the load-bearing seed; it must be present.
    assert "VERIFIED QUANT DATA" in user_text
    # And no Recent-thoughts block since history is empty.
    assert "Recent thoughts" not in user_text
