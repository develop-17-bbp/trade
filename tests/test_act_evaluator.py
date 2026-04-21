"""Tests for src/evaluation/act_evaluator.py — locks the contract that the
CLI + Streamlit dashboard depend on."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest


def _write_paper_journal(path: str, events: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _entry(trade_id: str, asset: str, direction: str, price: float, score: int = 5,
           llm_conf: float = 0.75, ml_conf: float = 0.0, spread: float = 1.0,
           size: float = 2.0) -> dict:
    return {
        "event": "ENTRY", "timestamp": f"2026-04-22T00:00:00+00:00",
        "asset": asset, "direction": direction,
        "fill_price": price, "rh_bid": price, "rh_ask": price, "rh_mid": price,
        "spread_pct": spread, "quantity": 0.01,
        "score": score, "sl": price * 0.99, "tp": 0,
        "ml_confidence": ml_conf, "llm_confidence": llm_conf,
        "size_pct": size, "reasoning": "test", "equity": 10000.0,
    }


def _exit(asset: str, direction: str, entry_price: float, exit_price: float,
          pnl_pct: float, reason: str = "SL L2 hit") -> dict:
    pnl_usd = (exit_price - entry_price) * 0.01 if direction == "LONG" else (entry_price - exit_price) * 0.01
    return {
        "event": "EXIT", "timestamp": f"2026-04-22T00:30:00+00:00",
        "asset": asset, "direction": direction,
        "entry_price": entry_price, "exit_price": exit_price,
        "rh_bid": exit_price, "rh_ask": exit_price,
        "pnl_pct": pnl_pct, "pnl_usd": pnl_usd,
        "reason": reason, "equity": 10000.0, "bars_held": 15, "entry_spread_pct": 0.0,
    }


def test_load_paper_trades_matches_fifo(tmp_path):
    from src.evaluation.act_evaluator import load_paper_trades

    p = str(tmp_path / "paper.jsonl")
    events = [
        _entry("T1", "BTC", "LONG", 100.0),
        _exit("BTC", "LONG", 100.0, 101.0, 1.0),
        _entry("T2", "ETH", "LONG", 2000.0),
        _exit("ETH", "LONG", 2000.0, 1980.0, -1.0),
    ]
    _write_paper_journal(p, events)
    trades = load_paper_trades(p)
    assert len(trades) == 2
    assert trades[0]["asset"] == "BTC" and trades[0]["win"] == 1
    assert trades[1]["asset"] == "ETH" and trades[1]["win"] == 0


def test_bucket_attribution_aggregates_correctly(tmp_path):
    from src.evaluation.act_evaluator import load_paper_trades, bucket_attribution, SCORE_BUCKETS

    p = str(tmp_path / "paper.jsonl")
    events = [
        _entry("T1", "BTC", "LONG", 100.0, score=2),  # low score, loser
        _exit("BTC", "LONG", 100.0, 99.0, -1.0),
        _entry("T2", "BTC", "LONG", 100.0, score=8),  # high score, winner
        _exit("BTC", "LONG", 100.0, 102.0, 2.0),
        _entry("T3", "BTC", "LONG", 100.0, score=8),  # high score, another winner
        _exit("BTC", "LONG", 100.0, 101.0, 1.0),
    ]
    _write_paper_journal(p, events)
    trades = load_paper_trades(p)
    rows = bucket_attribution(trades, "score", SCORE_BUCKETS)

    low_bucket = [r for r in rows if r["lo"] == -10][0]   # [-10, 3)
    high_bucket = [r for r in rows if r["lo"] == 7][0]    # [7, 10)
    assert low_bucket["n"] == 1
    assert low_bucket["wr"] == 0.0
    assert high_bucket["n"] == 2
    assert high_bucket["wr"] == 1.0


def test_rolling_sharpe_needs_enough_samples(tmp_path):
    from src.evaluation.act_evaluator import load_paper_trades, rolling_sharpe_series

    p = str(tmp_path / "paper.jsonl")
    events = []
    for i in range(5):
        events.append(_entry(f"T{i}", "BTC", "LONG", 100.0))
        events.append(_exit("BTC", "LONG", 100.0, 101.0, 1.0))
    _write_paper_journal(p, events)
    trades = load_paper_trades(p)
    rs = rolling_sharpe_series(trades, window=30)
    assert rs == [], "should be empty when below window size"


def test_rolling_sharpe_on_35_trades(tmp_path):
    from src.evaluation.act_evaluator import load_paper_trades, rolling_sharpe_series

    p = str(tmp_path / "paper.jsonl")
    events = []
    for i in range(35):
        events.append(_entry(f"T{i}", "BTC", "LONG", 100.0))
        # alternating +2%/-1% = +EV
        pnl = 2.0 if i % 2 == 0 else -1.0
        events.append(_exit("BTC", "LONG", 100.0, 100.0 + pnl, pnl))
    _write_paper_journal(p, events)
    trades = load_paper_trades(p)
    rs = rolling_sharpe_series(trades, window=30)
    assert len(rs) == 6   # 35 - 30 + 1
    for r in rs:
        assert r["n"] == 30
        # Mean is ~0.5, std ~1.5 → Sharpe ~0.33; must be positive on +EV stream
        assert r["sharpe"] > 0


def test_component_state_reflects_env(monkeypatch):
    """Kill switches (ACT_DISABLE_ML) display the FEATURE state — not the raw env.
    ACT_DISABLE_ML=1 means kill switch on -> feature OFF."""
    from src.evaluation.act_evaluator import load_component_state

    monkeypatch.setenv("ACT_SAFE_ENTRIES", "1")
    monkeypatch.setenv("ACT_DISABLE_ML", "1")         # kill switch ON
    monkeypatch.setenv("ACT_LGBM_DEVICE", "gpu")
    monkeypatch.delenv("ACT_META_SHADOW_MODE", raising=False)

    cs = load_component_state()
    by_name = {c["env"]: c for c in cs["components"]}
    assert by_name["ACT_SAFE_ENTRIES"]["is_on"] is True
    # ACT_DISABLE_ML=1 inverts: feature (ML Gate) is OFF when var=1
    assert by_name["ACT_DISABLE_ML"]["is_on"] is False
    assert by_name["ACT_LGBM_DEVICE"]["is_on"] is True
    assert by_name["ACT_META_SHADOW_MODE"]["is_on"] is False


def test_ml_gate_on_when_kill_switch_unset(monkeypatch):
    """ACT_DISABLE_ML unset -> kill switch OFF -> ML Gate feature ON."""
    from src.evaluation.act_evaluator import load_component_state

    monkeypatch.delenv("ACT_DISABLE_ML", raising=False)
    cs = load_component_state()
    by_name = {c["env"]: c for c in cs["components"]}
    assert by_name["ACT_DISABLE_ML"]["is_on"] is True


def test_ml_gate_on_when_kill_switch_zero(monkeypatch):
    """ACT_DISABLE_ML=0 -> kill switch OFF -> ML Gate feature ON."""
    from src.evaluation.act_evaluator import load_component_state

    monkeypatch.setenv("ACT_DISABLE_ML", "0")
    cs = load_component_state()
    by_name = {c["env"]: c for c in cs["components"]}
    assert by_name["ACT_DISABLE_ML"]["is_on"] is True


def test_recommendations_fire_on_losing_bucket(tmp_path):
    """Score bucket with WR=0 across 10+ trades must trigger a recommendation."""
    from src.evaluation.act_evaluator import build_report

    p = str(tmp_path / "paper.jsonl")
    events = []
    for i in range(12):
        events.append(_entry(f"T{i}", "BTC", "LONG", 100.0, score=2))  # all low-score
        events.append(_exit("BTC", "LONG", 100.0, 99.0, -1.0))  # all losers
    _write_paper_journal(p, events)

    report = build_report(
        paper_journal_path=p,
        shadow_log_path=str(tmp_path / "nonexistent_shadow.jsonl"),
        safe_state_path=str(tmp_path / "nonexistent_safe.json"),
        retrain_history_path=str(tmp_path / "nonexistent_retrain.json"),
    )
    recs = report["recommendations"]
    reasons = " | ".join(r["reason"] for r in recs)
    assert "score" in reasons.lower() or "wr" in reasons.lower()
    assert any(r["severity"] in ("high", "medium") for r in recs)


def test_recommendations_quiet_on_insufficient_data(tmp_path):
    """Empty journal should produce a single info-level "keep soaking" message."""
    from src.evaluation.act_evaluator import build_report

    empty = str(tmp_path / "empty.jsonl")
    Path(empty).write_text("", encoding="utf-8")

    report = build_report(
        paper_journal_path=empty,
        shadow_log_path=str(tmp_path / "nonexistent_shadow.jsonl"),
        safe_state_path=str(tmp_path / "nonexistent_safe.json"),
        retrain_history_path=str(tmp_path / "nonexistent_retrain.json"),
    )
    recs = report["recommendations"]
    assert len(recs) == 1
    assert recs[0]["severity"] == "info"
    assert "not enough" in recs[0]["reason"].lower() or "trades" in recs[0]["reason"].lower()


def test_build_report_contract_keys_stable(tmp_path):
    """CLI + Streamlit depend on these exact top-level keys — lock them."""
    from src.evaluation.act_evaluator import build_report

    empty = str(tmp_path / "empty.jsonl")
    Path(empty).write_text("", encoding="utf-8")

    report = build_report(
        paper_journal_path=empty,
        shadow_log_path=str(tmp_path / "nonexistent_shadow.jsonl"),
        safe_state_path=str(tmp_path / "nonexistent_safe.json"),
        retrain_history_path=str(tmp_path / "nonexistent_retrain.json"),
    )
    expected = {
        "components", "trades", "totals", "attribution",
        "rolling_sharpe_30", "shadow", "safe_entries",
        "retrain", "recommendations",
    }
    assert expected.issubset(report.keys()), f"missing: {expected - set(report.keys())}"


def test_exit_reason_family_covers_common_cases():
    from src.evaluation.act_evaluator import exit_reason_family

    assert exit_reason_family("SL L2 hit (candle close $74,530.20)") == "SL ratchet"
    assert exit_reason_family("Hard stop -1.8%") == "SL hard stop"
    assert exit_reason_family("ROI table (ROI: profit 0.59%)") == "ROI target"
    assert exit_reason_family("Time exit (720min)") == "Time exit"
    assert exit_reason_family("EMA new down line") == "EMA flip"
    assert exit_reason_family("something weird") == "Other"
