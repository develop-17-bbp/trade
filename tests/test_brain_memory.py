"""Tests for src/ai/brain_memory.py — scanner↔analyst corpus callosum."""
from __future__ import annotations

import time

import pytest

from src.ai.brain_memory import (
    AnalystTrace,
    BrainMemory,
    ScanReport,
)


@pytest.fixture
def mem(tmp_path):
    m = BrainMemory(str(tmp_path / "bm.sqlite"), lru_size=8)
    yield m
    m.close()


# ── ScanReport write/read ───────────────────────────────────────────────


def test_write_then_read_latest_scan(mem):
    report = ScanReport(
        asset="BTC", ts=time.time(),
        opportunity_score=72.0,
        proposed_direction="LONG",
        top_signals=["ema_cross", "hh_hl"],
        rationale="trend intact",
        raw={"opportunity_score": 72.0, "proposed_direction": "LONG"},
    )
    mem.write_scan_report(report)
    got = mem.read_latest_scan("BTC")
    assert got is not None
    assert got.opportunity_score == pytest.approx(72.0)
    assert got.proposed_direction == "LONG"
    assert "ema_cross" in got.top_signals


def test_read_returns_none_when_stale(mem):
    old = ScanReport(
        asset="ETH", ts=time.time() - 10_000,
        opportunity_score=40.0, proposed_direction="FLAT",
        raw={"opportunity_score": 40.0, "proposed_direction": "FLAT"},
    )
    mem.write_scan_report(old)
    assert mem.read_latest_scan("ETH", max_age_s=60.0) is None
    assert mem.read_latest_scan("ETH", max_age_s=20_000.0) is not None


def test_read_latest_returns_most_recent(mem):
    asset = "BTC"
    for score, lag in [(30, 10), (60, 5), (45, 0)]:
        mem.write_scan_report(ScanReport(
            asset=asset, ts=time.time() - lag,
            opportunity_score=score, proposed_direction="LONG",
            raw={"opportunity_score": score, "proposed_direction": "LONG"},
        ))
    # Read should return the one with the newest ts (lag=0, score=45).
    r = mem.read_latest_scan(asset)
    assert r is not None
    assert r.opportunity_score == pytest.approx(45.0)


def test_read_unknown_asset_returns_none(mem):
    assert mem.read_latest_scan("SOL") is None


def test_lru_serves_hot_reads_without_hitting_db(mem):
    r = ScanReport(
        asset="BTC", ts=time.time(),
        opportunity_score=10.0, proposed_direction="FLAT",
        raw={"opportunity_score": 10.0, "proposed_direction": "FLAT"},
    )
    mem.write_scan_report(r)
    # Close the underlying connection to force a raise on any DB read.
    mem._conn.close()
    mem._conn = None
    # LRU should still serve the most recent write.
    cached = mem.read_latest_scan("BTC")
    assert cached is not None
    assert cached.opportunity_score == pytest.approx(10.0)


def test_asset_is_case_insensitive_for_reads(mem):
    mem.write_scan_report(ScanReport(
        asset="btc", ts=time.time(),
        opportunity_score=50.0, proposed_direction="LONG",
        raw={"opportunity_score": 50.0, "proposed_direction": "LONG"},
    ))
    assert mem.read_latest_scan("BTC") is not None


# ── AnalystTrace write/read ─────────────────────────────────────────────


def test_write_and_read_analyst_trace(mem):
    tr = AnalystTrace(
        asset="BTC", ts=time.time(),
        plan_id="p-1", direction="LONG", tier="normal",
        size_pct=5.0, thesis="trend+macro", verdict="plan",
    )
    mem.write_analyst_trace(tr)
    traces = mem.read_recent_traces("BTC", limit=5)
    assert len(traces) == 1
    assert traces[0].plan_id == "p-1"
    assert traces[0].direction == "LONG"
    assert traces[0].size_pct == pytest.approx(5.0)


def test_read_recent_traces_respects_limit_and_age(mem):
    now = time.time()
    for i in range(5):
        mem.write_analyst_trace(AnalystTrace(
            asset="BTC", ts=now - i * 60, plan_id=f"p-{i}",
            direction="LONG" if i % 2 == 0 else "SKIP",
            tier="normal", size_pct=1.0 * i,
        ))
    traces = mem.read_recent_traces("BTC", limit=3)
    assert len(traces) == 3
    # Most recent first.
    assert traces[0].plan_id == "p-0"
    # Old traces excluded by max_age_s.
    old = mem.read_recent_traces("BTC", limit=10, max_age_s=30.0)
    assert len(old) == 1


# ── Convenience helpers (singleton) ─────────────────────────────────────


def test_convenience_helpers_use_singleton(monkeypatch, tmp_path):
    from src.ai import brain_memory as bm
    fresh = BrainMemory(str(tmp_path / "c.sqlite"))
    monkeypatch.setattr(bm, "_brain_singleton", fresh, raising=False)

    bm.publish_scan(ScanReport(
        asset="BTC", ts=time.time(),
        opportunity_score=80.0, proposed_direction="LONG",
        raw={"opportunity_score": 80.0, "proposed_direction": "LONG"},
    ))
    assert bm.get_scan_for_analyst("BTC") is not None

    bm.publish_analyst_trace(AnalystTrace(
        asset="BTC", ts=time.time(), plan_id="p-h", direction="LONG",
    ))
    assert len(bm.get_recent_analyst_traces("BTC")) == 1


def test_convenience_helpers_swallow_errors(monkeypatch):
    from src.ai import brain_memory as bm

    class _Broken:
        def write_scan_report(self, *_):
            raise RuntimeError("x")
        def read_latest_scan(self, *_, **__):
            raise RuntimeError("x")
        def write_analyst_trace(self, *_):
            raise RuntimeError("x")
        def read_recent_traces(self, *_, **__):
            raise RuntimeError("x")

    monkeypatch.setattr(bm, "_brain_singleton", _Broken(), raising=False)

    # None of these should raise.
    bm.publish_scan(ScanReport(
        asset="BTC", ts=time.time(), opportunity_score=0, proposed_direction="FLAT",
    ))
    assert bm.get_scan_for_analyst("BTC") is None
    bm.publish_analyst_trace(AnalystTrace(asset="BTC", ts=0.0, plan_id="x", direction="SKIP"))
    assert bm.get_recent_analyst_traces("BTC") == []


# ── Integration with agentic_bridge ─────────────────────────────────────


def test_agentic_bridge_reads_scan_and_publishes_trace(monkeypatch, tmp_path):
    from src.ai import brain_memory as bm
    fresh = BrainMemory(str(tmp_path / "bridge.sqlite"))
    monkeypatch.setattr(bm, "_brain_singleton", fresh, raising=False)
    monkeypatch.delenv("ACT_DISABLE_AGENTIC_LOOP", raising=False)

    # Seed with a scanner report.
    bm.publish_scan(ScanReport(
        asset="BTC", ts=time.time(),
        opportunity_score=70.0, proposed_direction="LONG",
        top_signals=["breakout", "trend"], rationale="bullish",
        raw={"opportunity_score": 70.0, "proposed_direction": "LONG"},
    ))

    # Capture what the loop sees as quant_data.
    import src.ai.agentic_bridge as brg
    seen_prompts = {}

    def stub_llm(messages):
        # Capture ALL user messages (seed + response-format reminder).
        texts = [str(m.get("content")) for m in messages if m.get("role") == "user"]
        seen_prompts["user_text"] = "\n---\n".join(texts)
        # Return a valid skip envelope so the loop terminates cleanly.
        import json
        return json.dumps({"skip": "test"})

    result = brg.compile_agentic_plan(
        asset="BTC",
        quant_data="[PRICE=60000]",
        llm_call=stub_llm,
        similar_trades=[],
        recent_critiques=[],
        max_steps=2,
    )
    assert result.terminated_reason == "skip"
    # Analyst saw the scanner's report.
    assert "SCANNER REPORT" in seen_prompts.get("user_text", "")
    assert "breakout" in seen_prompts.get("user_text", "")

    # And the analyst's decision was written back for the next scanner tick.
    traces = bm.get_recent_analyst_traces("BTC")
    assert len(traces) == 1
    assert traces[0].direction == "SKIP"
    assert traces[0].verdict == "skip"
