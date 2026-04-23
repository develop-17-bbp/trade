"""Tests for src/ai/shadow_tick.py — unified per-asset tick orchestration."""
from __future__ import annotations

import json
from unittest import mock

import pytest

from src.ai.shadow_tick import (
    FRESH_SCAN_S,
    INGEST_EVERY_TICKS,
    PERSONA_REFRESH_EVERY_TICKS,
    _extract_first_json,
    _tick_counters,
    run_tick,
)


@pytest.fixture(autouse=True)
def _reset_counters():
    _tick_counters.clear()
    yield
    _tick_counters.clear()


# ── JSON extractor ─────────────────────────────────────────────────────


def test_extract_first_json_happy():
    out = _extract_first_json('{"opportunity_score": 72, "proposed_direction": "LONG"}')
    assert out and out.get("opportunity_score") == 72


def test_extract_first_json_embedded():
    text = "Here is my take: {\"opportunity_score\": 60} — hope that's enough"
    out = _extract_first_json(text)
    assert out and out.get("opportunity_score") == 60


def test_extract_first_json_none_on_bad_text():
    assert _extract_first_json("no json here") is None
    assert _extract_first_json("") is None


# ── run_tick orchestration ─────────────────────────────────────────────


@pytest.fixture
def mocks(monkeypatch):
    """Stub every external subsystem so run_tick is hermetic.

    Returns a dict of MagicMocks keyed by subsystem for assertion.
    """
    import sys

    # web_context
    web_module = mock.MagicMock()
    fake_bundle = {"news": mock.MagicMock(), "sentiment": mock.MagicMock()}
    web_module.fetch_bundle.return_value = fake_bundle
    web_module.bundle_to_prompt_block.return_value = "[BUNDLE]"
    monkeypatch.setitem(sys.modules, "src.ai.web_context", web_module)

    # brain_memory
    bm_module = mock.MagicMock()
    bm_module.get_scan_for_analyst.return_value = None     # no fresh scan
    bm_module.publish_scan = mock.MagicMock()
    # Pass through the real ScanReport class so run_tick can instantiate.
    from src.ai.brain_memory import ScanReport as _RealScanReport
    bm_module.ScanReport = _RealScanReport
    monkeypatch.setitem(sys.modules, "src.ai.brain_memory", bm_module)

    # dual_brain — scanner returns valid JSON envelope.
    db_module = mock.MagicMock()
    scan_resp = mock.MagicMock(
        ok=True, model="stub",
        text='{"opportunity_score": 68, "proposed_direction": "LONG", '
             '"top_signals": ["ema_cross"], "rationale": "test"}',
        fallback_used=False,
    )
    db_module.scan.return_value = scan_resp
    monkeypatch.setitem(sys.modules, "src.ai.dual_brain", db_module)

    # graph_rag ingest helpers (throttled — may or may not be called).
    gr_module = mock.MagicMock()
    gr_module.ingest_news.return_value = 2
    gr_module.ingest_sentiment.return_value = True
    gr_module.ingest_institutional.return_value = 3
    gr_module.ingest_polymarket.return_value = 1
    gr_module.ingest_correlation.return_value = True
    monkeypatch.setitem(sys.modules, "src.ai.graph_rag", gr_module)

    # agentic_bridge.compile_agentic_plan
    bridge_module = mock.MagicMock()
    from src.ai.agentic_trade_loop import LoopResult
    from src.trading.trade_plan import TradePlan
    fake_plan = TradePlan(
        asset="BTC", direction="LONG", entry_tier="normal",
        entry_price=60000, size_pct=5.0, sl_price=58800,
    )
    fake_loop = LoopResult(
        plan=fake_plan, steps_taken=2, tool_calls=[],
        terminated_reason="plan",
    )
    bridge_module.compile_agentic_plan.return_value = fake_loop
    monkeypatch.setitem(sys.modules, "src.ai.agentic_bridge", bridge_module)

    # persona manager
    pm_module = mock.MagicMock()
    pm_manager = mock.MagicMock()
    pm_manager.refresh.return_value = {"spawned": [], "dissolved": [], "kept": []}
    pm_module.get_manager.return_value = pm_manager
    monkeypatch.setitem(sys.modules, "src.agents.persona_from_graph", pm_module)

    # institutional + polymarket fetchers
    inst_module = mock.MagicMock()
    inst_instance = mock.MagicMock()
    inst_instance.get_all_institutional.return_value = {"stablecoin_flow": 1.5}
    inst_module.InstitutionalFetcher.return_value = inst_instance
    monkeypatch.setitem(sys.modules, "src.data.institutional_fetcher", inst_module)

    pm_fetch_module = mock.MagicMock()
    pm_fetcher = mock.MagicMock()
    pm_fetcher.fetch_crypto_markets.return_value = [
        {"market_id": "m1", "question": "Will BTC close above 70k?", "yes_price": 0.4},
    ]
    pm_fetch_module.PolymarketFetcher.return_value = pm_fetcher
    monkeypatch.setitem(sys.modules, "src.data.polymarket_fetcher", pm_fetch_module)

    return {
        "web": web_module, "bm": bm_module, "db": db_module,
        "gr": gr_module, "bridge": bridge_module, "pm": pm_module,
    }


def test_run_tick_returns_structured_summary(mocks):
    out = run_tick("BTC", quant_digest="[Q]")
    assert out["asset"] == "BTC"
    assert "plan" in out
    assert out["plan"]["direction"] == "LONG"
    assert out["plan"]["terminated_reason"] == "plan"


def test_scanner_publishes_when_no_fresh_scan(mocks):
    out = run_tick("BTC")
    assert out["scanner_published"] is True
    # publish_scan was called exactly once.
    assert mocks["bm"].publish_scan.call_count == 1
    # Scanner was asked.
    assert mocks["db"].scan.call_count == 1


def test_scanner_skipped_when_fresh_scan_exists(mocks):
    # Pretend brain_memory already has a recent scan.
    from src.ai.brain_memory import ScanReport
    import time as _t
    fresh = ScanReport(
        asset="BTC", ts=_t.time(),
        opportunity_score=55, proposed_direction="LONG",
    )
    mocks["bm"].get_scan_for_analyst.return_value = fresh

    out = run_tick("BTC")
    assert out["scanner_published"] is False
    assert mocks["db"].scan.call_count == 0


def test_compile_agentic_plan_always_called(mocks):
    out = run_tick("BTC")
    assert mocks["bridge"].compile_agentic_plan.call_count == 1
    assert out.get("_loop_result") is not None


def test_ingest_throttled_by_tick_count(mocks):
    """Ingest runs every INGEST_EVERY_TICKS calls."""
    for i in range(INGEST_EVERY_TICKS * 2):
        run_tick("BTC")
    # Should have ingested exactly 2 times over 2*INGEST_EVERY_TICKS ticks.
    # (ingest_news called on the ticks where counter % interval == 0)
    assert mocks["gr"].ingest_news.call_count == 2


def test_persona_refresh_throttled(mocks):
    for i in range(PERSONA_REFRESH_EVERY_TICKS * 2):
        run_tick("BTC")
    # Manager.refresh called exactly twice.
    assert mocks["pm"].get_manager.return_value.refresh.call_count == 2


def test_run_tick_tolerates_scanner_failure(mocks):
    # scan() returns ok=False → no publish, but the plan path still runs.
    mocks["db"].scan.return_value = mock.MagicMock(ok=False, text="", model="")
    out = run_tick("BTC")
    assert out["scanner_published"] is False
    assert out["plan"] is not None   # analyst still compiled


def test_run_tick_tolerates_web_bundle_failure(monkeypatch, mocks):
    mocks["web"].fetch_bundle.side_effect = RuntimeError("network down")
    out = run_tick("BTC")
    # No crash; plan still compiled.
    assert out["asset"] == "BTC"
    assert out["plan"] is not None


def test_run_tick_tolerates_compile_failure(mocks):
    mocks["bridge"].compile_agentic_plan.side_effect = RuntimeError("llm down")
    out = run_tick("BTC")
    assert out["plan"] is None


def test_tick_counter_is_per_asset(mocks):
    run_tick("BTC")
    run_tick("BTC")
    run_tick("ETH")
    assert _tick_counters["BTC"] == 2
    assert _tick_counters["ETH"] == 1


def test_scanner_parses_trailing_json_noise(mocks):
    mocks["db"].scan.return_value = mock.MagicMock(
        ok=True, model="m", fallback_used=False,
        text='Response: {"opportunity_score": 77, "proposed_direction": "SHORT"} - done',
    )
    mocks["bm"].get_scan_for_analyst.return_value = None
    out = run_tick("BTC")
    assert out["scanner_published"] is True
