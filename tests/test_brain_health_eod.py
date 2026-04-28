"""Tests for brain_health + eod_review."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _clear_caches():
    """Reset brain_health/eod_review module caches between tests so
    monkeypatched data sources are seen fresh."""
    from src.ai import brain_health as bh
    from src.ai import eod_review as eod
    bh._CACHE.clear()
    eod._YESTERDAY_CACHE = None
    yield
    bh._CACHE.clear()
    eod._YESTERDAY_CACHE = None


# ── Brain Health ────────────────────────────────────────────────────────


def test_thesis_quality_score_empty():
    from src.ai.brain_health import _score_thesis_quality
    assert _score_thesis_quality("") == 0.0
    assert _score_thesis_quality("short") == 0.0


def test_thesis_quality_score_rich_thesis():
    from src.ai.brain_health import _score_thesis_quality
    rich = (
        "macro risk_on (DXY -0.4%) + halving cycle markup phase + "
        "BTC dominance rotation_zone + CVD slope +1 + sniper PASS "
        "confluence 5/3 + DSR=0.42 on similar setups in this regime."
    )
    score = _score_thesis_quality(rich)
    assert score > 0.5


def test_thesis_quality_score_generic_thesis():
    from src.ai.brain_health import _score_thesis_quality
    generic = "looks like a good trade today price is going up"
    score = _score_thesis_quality(generic)
    assert score < 0.3


def test_brain_health_handles_no_decisions(monkeypatch):
    from src.ai import brain_health as bh
    monkeypatch.setattr(bh, "_read_recent_decisions", lambda **kw: [])
    snap = bh.compute_brain_health()
    assert snap.n_ticks_observed == 0
    assert "no_decisions" in snap.sample_warning


def test_brain_health_render_empty_when_too_few_ticks(monkeypatch):
    from src.ai import brain_health as bh
    monkeypatch.setattr(bh, "_read_recent_decisions", lambda **kw: [])
    assert bh.render_summary_for_tick() == ""


def test_brain_health_with_sample_decisions(monkeypatch):
    from src.ai import brain_health as bh
    sample = [
        {"ts_ns": 1, "direction": "LONG", "thesis": "macro DXY rising",
         "tool_calls": [{"name": "query_macro_overlay", "args": {}}],
         "steps_taken": 2, "final_action": "submit"},
        {"ts_ns": 2, "direction": "SKIP", "thesis": "macro headwind",
         "tool_calls": [{"name": "query_macro_overlay", "args": {}},
                         {"name": "query_btc_dominance", "args": {}}],
         "steps_taken": 3, "final_action": "skip"},
    ] * 12  # 24 sample decisions
    monkeypatch.setattr(bh, "_read_recent_decisions", lambda **kw: sample)
    snap = bh.compute_brain_health()
    assert snap.n_ticks_observed == 24
    assert snap.n_distinct_tools_used == 2
    assert 0.0 <= snap.tool_variety_score <= 1.0


def test_brain_health_tool_registered():
    from src.ai.trade_tools import build_default_registry
    r = build_default_registry()
    assert "query_brain_health" in set(r.list_names())


# ── EOD Review ────────────────────────────────────────────────────────


def test_eod_review_handles_few_trades(monkeypatch):
    from src.ai import eod_review as eod
    monkeypatch.setattr(eod, "_read_today_trades", lambda d: [])
    review = eod.compute_eod_review("2026-04-29")
    assert review.n_trades_closed == 0
    assert "only" in review.skipped_reason or "need" in review.skipped_reason


def test_eod_review_with_sample_trades(monkeypatch):
    from src.ai import eod_review as eod
    sample = [
        {"ts_ns": 1, "asset": "BTC", "direction": "LONG", "tier": "normal",
         "thesis": "macro risk_on bias 0.6", "pnl_pct": 1.5,
         "regime": "TREND_UP", "miss_reason": "", "lessons": "good"},
        {"ts_ns": 2, "asset": "BTC", "direction": "LONG", "tier": "sniper",
         "thesis": "sniper conf 6/3", "pnl_pct": -1.5,
         "regime": "CHOP", "miss_reason": "macro_flip", "lessons": "skip in chop"},
        {"ts_ns": 3, "asset": "ETH", "direction": "LONG", "tier": "normal",
         "thesis": "BTC.D falling", "pnl_pct": 2.0,
         "regime": "TREND_UP", "miss_reason": "", "lessons": "great"},
    ]
    monkeypatch.setattr(eod, "_read_today_trades", lambda d: sample)
    review = eod.compute_eod_review("2026-04-29")
    assert review.n_trades_closed == 3
    assert review.win_rate == 2 / 3
    assert len(review.best_trades) == 3
    assert len(review.worst_trades) == 3
    # Best should be the +2.0% trade
    assert review.best_trades[0]["pnl_pct"] == 2.0
    # Worst should be -1.5%
    assert review.worst_trades[0]["pnl_pct"] == -1.5


def test_eod_review_writes_markdown(monkeypatch, tmp_path):
    from src.ai import eod_review as eod
    sample = [
        {"ts_ns": 1, "asset": "BTC", "direction": "LONG", "tier": "normal",
         "thesis": "test", "pnl_pct": 1.5, "regime": "TREND_UP",
         "miss_reason": "", "lessons": ""},
    ] * 5
    monkeypatch.setattr(eod, "_read_today_trades", lambda d: sample)
    monkeypatch.setattr(eod, "EOD_DIR", str(tmp_path))
    review = eod.compute_eod_review("2026-04-29")
    path = eod.write_eod_review(review)
    assert path
    assert "EOD Review" in Path(path).read_text(encoding="utf-8")


def test_eod_tool_registered():
    from src.ai.trade_tools import build_default_registry
    r = build_default_registry()
    assert "run_eod_review" in set(r.list_names())


def test_format_for_brain_renders_brain_health():
    from src.ai import tick_state as ts
    ts.update("BTC", brain_health_summary=(
        "BRAIN_HEALTH: skip_rate=60% thesis_quality=70% tool_variety=25%"
    ))
    out = ts.format_for_brain("BTC")
    assert "BRAIN_HEALTH" in out


def test_format_for_brain_renders_eod_summary():
    from src.ai import tick_state as ts
    ts.update("BTC", eod_review_yesterday=(
        "YESTERDAY_REVIEW: 5 trades WR=60% +0.42% calibration=neutral"
    ))
    out = ts.format_for_brain("BTC")
    assert "YESTERDAY_REVIEW" in out
