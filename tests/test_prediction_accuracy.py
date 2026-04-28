"""Tests for prediction_accuracy — self-critique feedback to brain."""
from __future__ import annotations

import pytest


def test_extract_bias_score_from_thesis():
    from src.ai.prediction_accuracy import _extract_bias_score
    assert _extract_bias_score("LONG bias=+0.62 confluence 5/3") == 0.62
    assert _extract_bias_score("score: -0.4 weak setup") == -0.4
    assert _extract_bias_score("long_bias_score = 0.85") == 0.85
    assert _extract_bias_score("no number here") is None
    assert _extract_bias_score("") is None


def test_bucket_bias_score():
    from src.ai.prediction_accuracy import _bucket_bias
    assert _bucket_bias(0.7) == "strong_long"
    assert _bucket_bias(0.3) == "mild_long"
    assert _bucket_bias(0.0) == "neutral"
    assert _bucket_bias(-0.3) == "mild_short"
    assert _bucket_bias(-0.7) == "strong_short"
    assert _bucket_bias(None) == "no_bias_score"


def test_compute_accuracy_handles_no_data(monkeypatch):
    from src.ai import prediction_accuracy as pa
    monkeypatch.setattr(pa, "_read_closed_decisions", lambda **kw: [])
    snap = pa.compute_accuracy()
    assert snap.n_closed_trades == 0
    assert snap.overall_win_rate == 0.0
    assert "no_closed_trades" in snap.sample_warning


def test_aggregate_buckets_basic():
    from src.ai.prediction_accuracy import _aggregate_buckets
    rows = [
        {"realized_pnl_pct": 1.5},
        {"realized_pnl_pct": -0.5},
        {"realized_pnl_pct": 2.0},
        {"realized_pnl_pct": 1.0},
    ]
    buckets = _aggregate_buckets(rows, lambda r: "all")
    assert len(buckets) == 1
    b = buckets[0]
    assert b.n == 4
    assert b.wins == 3
    assert b.win_rate == 0.75


def test_calibration_label_well_calibrated():
    from src.ai.prediction_accuracy import _calibration_label, AccuracyBucket
    by_bias = [
        AccuracyBucket(label="strong_long", n=10, wins=7, win_rate=0.7),
    ]
    assert _calibration_label(by_bias) == "well_calibrated"


def test_calibration_label_over_confident():
    from src.ai.prediction_accuracy import _calibration_label, AccuracyBucket
    by_bias = [
        AccuracyBucket(label="strong_long", n=10, wins=4, win_rate=0.40),
    ]
    assert _calibration_label(by_bias) == "over_confident"


def test_render_summary_empty_when_no_trades(monkeypatch):
    from src.ai import prediction_accuracy as pa
    monkeypatch.setattr(pa, "_read_closed_decisions", lambda **kw: [])
    assert pa.render_summary_for_tick() == ""


def test_render_summary_format(monkeypatch):
    from src.ai import prediction_accuracy as pa
    sample_rows = [
        {"realized_pnl_pct": 1.5, "direction": "LONG", "tier": "normal",
         "regime": "TREND_UP", "bias_score": 0.6, "miss_reason": ""},
        {"realized_pnl_pct": -1.0, "direction": "LONG", "tier": "normal",
         "regime": "TREND_UP", "bias_score": 0.3, "miss_reason": "macro_flip"},
        {"realized_pnl_pct": 2.0, "direction": "LONG", "tier": "sniper",
         "regime": "TREND_UP", "bias_score": 0.7, "miss_reason": ""},
    ]
    monkeypatch.setattr(pa, "_read_closed_decisions", lambda **kw: sample_rows)
    summary = pa.render_summary_for_tick()
    assert "PREDICTION_ACCURACY" in summary
    assert "n=3" in summary
    assert "WR=" in summary
    assert "calibration=" in summary


def test_query_prediction_accuracy_tool_registered():
    from src.ai.trade_tools import build_default_registry
    r = build_default_registry()
    assert "query_prediction_accuracy" in set(r.list_names())


def test_format_for_brain_renders_accuracy_line():
    from src.ai import tick_state as ts
    ts.update("BTC", prediction_accuracy_summary=(
        "PREDICTION_ACCURACY: n=20 WR=55% avg_pnl=+0.30% calibration=neutral"
    ))
    out = ts.format_for_brain("BTC")
    assert "PREDICTION_ACCURACY" in out
