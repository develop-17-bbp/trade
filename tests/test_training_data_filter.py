"""Tests for src/ai/training_data_filter.py — quality filter over warm_store."""
from __future__ import annotations

import json
import sqlite3
import time

import pytest

from src.ai.training_data_filter import (
    DEFAULT_MIN_PNL_ABS_PCT,
    ExperienceSample,
    FilterStats,
    format_analyst_sft_example,
    format_dpo_pairs,
    format_scanner_sft_example,
    load_experience_samples,
)


def _make_db(tmp_path):
    db = tmp_path / "warm.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute("""
        CREATE TABLE decisions (
            decision_id TEXT PRIMARY KEY,
            trace_id TEXT, symbol TEXT, ts_ns INTEGER, direction INTEGER,
            confidence REAL, consensus TEXT, veto INTEGER, raw_signal INTEGER,
            final_action TEXT, authority_violations TEXT, payload_json TEXT,
            component_signals TEXT DEFAULT '{}',
            plan_json TEXT DEFAULT '{}',
            self_critique TEXT DEFAULT '{}'
        )
    """)
    conn.execute("""
        CREATE TABLE outcomes (
            outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id TEXT, symbol TEXT, direction TEXT,
            entry_price REAL, exit_price REAL, pnl_pct REAL, pnl_usd REAL,
            duration_s REAL, exit_reason TEXT, regime TEXT,
            entry_ts REAL, exit_ts REAL, payload_json TEXT
        )
    """)
    conn.commit()
    return db, conn


def _insert(conn, decision_id, symbol, pnl_pct, *, matched=True, plan=None, ts_ns=None,
            shadow=False, component_signals=None, direction="LONG"):
    ts_ns = ts_ns or time.time_ns()
    exit_ts = ts_ns / 1_000_000_000
    conn.execute(
        "INSERT INTO decisions (decision_id, symbol, ts_ns, direction, final_action, "
        "plan_json, self_critique, component_signals) VALUES (?,?,?,?,?,?,?,?)",
        (decision_id, symbol, ts_ns, 1 if direction == "LONG" else -1, direction,
         json.dumps(plan or {"direction": direction, "size_pct": 5.0}),
         json.dumps({"matched_thesis": matched, "miss_reason": "" if matched else "off"}),
         json.dumps(component_signals or {})),
    )
    conn.execute(
        "INSERT INTO outcomes (decision_id, symbol, direction, pnl_pct, exit_ts, "
        "payload_json) VALUES (?,?,?,?,?,?)",
        (decision_id, symbol, direction, pnl_pct, exit_ts,
         json.dumps({"pnl_pct": pnl_pct, "exit_reason": "tp" if pnl_pct > 0 else "sl"})),
    )
    conn.commit()


# ── Happy path ──────────────────────────────────────────────────────────


def test_load_empty_db_returns_empty(tmp_path, monkeypatch):
    db, conn = _make_db(tmp_path)
    conn.close()
    monkeypatch.setenv("ACT_WARM_DB_PATH", str(db))
    samples, stats = load_experience_samples()
    assert samples == []
    assert stats.total == 0


def test_load_filters_out_shadow_rows(tmp_path, monkeypatch):
    db, conn = _make_db(tmp_path)
    _insert(conn, "shadow-1", "BTC", 1.5)
    _insert(conn, "real-1", "BTC", 1.5)
    conn.close()
    monkeypatch.setenv("ACT_WARM_DB_PATH", str(db))
    samples, _ = load_experience_samples()
    ids = {s.decision_id for s in samples}
    assert "real-1" in ids
    assert "shadow-1" not in ids


def test_load_rejects_low_pnl_noise(tmp_path, monkeypatch):
    db, conn = _make_db(tmp_path)
    _insert(conn, "tiny-1", "BTC", 0.05)     # below min_pnl_abs_pct=0.3
    _insert(conn, "good-1", "BTC", 2.0)
    conn.close()
    monkeypatch.setenv("ACT_WARM_DB_PATH", str(db))
    samples, stats = load_experience_samples()
    ids = {s.decision_id for s in samples}
    assert "good-1" in ids
    assert "tiny-1" not in ids
    assert stats.below_min_pnl == 1


def test_load_excludes_thesis_unmatched_positive(tmp_path, monkeypatch):
    db, conn = _make_db(tmp_path)
    # Lucky win: PnL positive but thesis didn't match.
    _insert(conn, "lucky-1", "BTC", 2.0, matched=False)
    _insert(conn, "earned-1", "BTC", 2.0, matched=True)
    conn.close()
    monkeypatch.setenv("ACT_WARM_DB_PATH", str(db))
    samples, _ = load_experience_samples(require_matched_thesis=True)
    ids = {s.decision_id for s in samples}
    # lucky-1 has positive pnl but matched=False and our include_negatives
    # path only keeps unmatched-thesis negatives (pnl < 0); a positive pnl
    # lucky win is filtered out entirely.
    assert "earned-1" in ids
    assert "lucky-1" not in ids


def test_load_keeps_matched_negatives_as_dpo_fuel(tmp_path, monkeypatch):
    db, conn = _make_db(tmp_path)
    _insert(conn, "honest-loss-1", "BTC", -1.5, matched=False)  # thesis miss + loss
    _insert(conn, "good-1", "BTC", 2.0, matched=True)
    conn.close()
    monkeypatch.setenv("ACT_WARM_DB_PATH", str(db))
    samples, stats = load_experience_samples(include_negatives=True)
    ids = {s.decision_id for s in samples}
    assert "honest-loss-1" in ids
    labels = {s.decision_id: s.label for s in samples}
    assert labels["honest-loss-1"] == "negative"
    assert labels["good-1"] == "positive"


def test_load_include_negatives_false(tmp_path, monkeypatch):
    db, conn = _make_db(tmp_path)
    _insert(conn, "loss-1", "BTC", -1.0, matched=False)
    _insert(conn, "win-1", "BTC", 1.0, matched=True)
    conn.close()
    monkeypatch.setenv("ACT_WARM_DB_PATH", str(db))
    samples, _ = load_experience_samples(include_negatives=False)
    ids = {s.decision_id for s in samples}
    assert ids == {"win-1"}


def test_load_respects_age_cutoff(tmp_path, monkeypatch):
    db, conn = _make_db(tmp_path)
    ancient_ts_ns = int((time.time() - 200 * 86400) * 1_000_000_000)
    _insert(conn, "ancient", "BTC", 1.5, ts_ns=ancient_ts_ns)
    _insert(conn, "recent", "BTC", 1.5)
    conn.close()
    monkeypatch.setenv("ACT_WARM_DB_PATH", str(db))
    samples, _ = load_experience_samples(max_age_days=30)
    ids = {s.decision_id for s in samples}
    assert "recent" in ids
    assert "ancient" not in ids


def test_load_filters_by_asset(tmp_path, monkeypatch):
    db, conn = _make_db(tmp_path)
    _insert(conn, "btc-1", "BTC", 1.5)
    _insert(conn, "eth-1", "ETH", 1.5)
    conn.close()
    monkeypatch.setenv("ACT_WARM_DB_PATH", str(db))
    samples, _ = load_experience_samples(asset="BTC")
    assert {s.asset for s in samples} == {"BTC"}


# ── ExperienceSample helpers ────────────────────────────────────────────


def test_sample_to_dict_roundtrip(tmp_path, monkeypatch):
    db, conn = _make_db(tmp_path)
    _insert(conn, "r-1", "BTC", 1.2)
    conn.close()
    monkeypatch.setenv("ACT_WARM_DB_PATH", str(db))
    samples, _ = load_experience_samples()
    d = samples[0].to_dict()
    assert d["decision_id"] == "r-1"
    assert d["asset"] == "BTC"
    assert d["label"] == "positive"
    assert d["matched_thesis"] is True


# ── Formatters ──────────────────────────────────────────────────────────


def _sample(asset="BTC", label="positive", pnl=1.2, direction="LONG", scanner_tag=...):
    # Sentinel so explicitly-passed empty dict is preserved (empty dict is falsy).
    if scanner_tag is ...:
        scanner_tag = {"top_signals": ["ema_cross"]}
    return ExperienceSample(
        decision_id=f"d-{asset}",
        asset=asset, ts=time.time(), pnl_pct=pnl, direction=direction,
        matched_thesis=True, label=label,
        plan={"direction": direction, "size_pct": 5.0, "entry_price": 60000},
        outcome={"pnl_pct": pnl, "exit_reason": "tp"},
        self_critique={"matched_thesis": True},
        scanner_tag=scanner_tag,
    )


def test_format_analyst_sft_example_has_prompt_and_completion():
    s = _sample()
    out = format_analyst_sft_example(s)
    assert "prompt" in out and "completion" in out
    assert "BTC" in out["prompt"]
    # Completion should parse back to a plan dict with direction.
    parsed = json.loads(out["completion"])
    assert parsed.get("direction") == "LONG"


def test_format_scanner_sft_example_positive_only():
    pos = _sample(label="positive")
    neg = _sample(label="negative")
    assert format_scanner_sft_example(pos) is not None
    assert format_scanner_sft_example(neg) is None


def test_format_scanner_sft_drops_when_no_scanner_tag():
    s = _sample(scanner_tag={})
    assert format_scanner_sft_example(s) is None


def test_format_dpo_pairs_matches_pos_to_nearest_neg():
    now = time.time()
    pos = ExperienceSample(
        decision_id="p", asset="BTC", ts=now, pnl_pct=1.5,
        direction="LONG", matched_thesis=True, label="positive",
        plan={"direction": "LONG"}, outcome={}, self_critique={},
    )
    neg_close = ExperienceSample(
        decision_id="n1", asset="BTC", ts=now - 60, pnl_pct=-1.0,
        direction="LONG", matched_thesis=False, label="negative",
        plan={"direction": "LONG"}, outcome={}, self_critique={},
    )
    neg_far = ExperienceSample(
        decision_id="n2", asset="BTC", ts=now - 5000, pnl_pct=-0.8,
        direction="LONG", matched_thesis=False, label="negative",
        plan={"direction": "LONG"}, outcome={}, self_critique={},
    )
    pairs = format_dpo_pairs([pos, neg_close, neg_far])
    assert len(pairs) == 1
    # Nearest-in-time negative wins.
    rejected = json.loads(pairs[0]["rejected"])
    assert rejected == {"direction": "LONG"}
    assert pairs[0]["chosen_pnl_pct"] == 1.5


def test_format_dpo_pairs_skips_when_no_negatives():
    pos = _sample(label="positive")
    assert format_dpo_pairs([pos]) == []


# ── Defaults ───────────────────────────────────────────────────────────


def test_default_min_pnl_sane():
    assert DEFAULT_MIN_PNL_ABS_PCT >= 0.1
