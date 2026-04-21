"""Tests for src/ml/shadow_log.py and src/scripts/shadow_retrain.py.

Shadow mode lets the meta model predict on every rule-signaled entry and log
(features, prob, decision) without vetoing anything. Outcomes are joined by
trade_id at exit, and shadow_retrain uses the joined dataset to replace the
meta model with one trained on live ground truth.
"""
from __future__ import annotations

import json
import os

import pytest


def test_shadow_enabled_env_parsing(monkeypatch):
    from src.ml.shadow_log import is_enabled

    monkeypatch.delenv("ACT_DISABLE_ML", raising=False)

    for val in ("1", "true", "YES", "on", "True"):
        monkeypatch.setenv("ACT_META_SHADOW_MODE", val)
        assert is_enabled() is True, f"enabled expected for ACT_META_SHADOW_MODE={val}"

    for val in ("0", "false", "", "off"):
        monkeypatch.setenv("ACT_META_SHADOW_MODE", val)
        assert is_enabled() is False


def test_shadow_disabled_if_ml_killed(monkeypatch):
    """ACT_DISABLE_ML=1 trumps ACT_META_SHADOW_MODE=1 — no shadow logs when ML is off."""
    from src.ml.shadow_log import is_enabled

    monkeypatch.setenv("ACT_META_SHADOW_MODE", "1")
    monkeypatch.setenv("ACT_DISABLE_ML", "1")
    assert is_enabled() is False


def test_log_predict_no_op_when_disabled(tmp_path, monkeypatch):
    """log_predict must not touch the filesystem when shadow mode is off."""
    from src.ml import shadow_log

    monkeypatch.delenv("ACT_META_SHADOW_MODE", raising=False)
    p = str(tmp_path / "meta_shadow.jsonl")
    shadow_log.log_predict(
        trade_id="T1", asset="BTC", direction="LONG", entry_price=100.0,
        entry_score=5, meta_prob_raw=0.3, meta_prob_cal=0.35,
        take_threshold=0.4, features=[0.1] * 10, path=p,
    )
    assert not os.path.exists(p), "log_predict must no-op when shadow mode off"


def test_log_predict_writes_when_enabled(tmp_path, monkeypatch):
    from src.ml import shadow_log

    monkeypatch.setenv("ACT_META_SHADOW_MODE", "1")
    monkeypatch.delenv("ACT_DISABLE_ML", raising=False)
    p = str(tmp_path / "meta_shadow.jsonl")

    shadow_log.log_predict(
        trade_id="T1", asset="BTC", direction="LONG", entry_price=100.0,
        entry_score=5, meta_prob_raw=0.3, meta_prob_cal=0.35,
        take_threshold=0.4, features=[0.1] * 50, path=p,
    )
    assert os.path.exists(p)
    with open(p, "r", encoding="utf-8") as f:
        rec = json.loads(f.readline())
    assert rec["event"] == "shadow_predict"
    assert rec["trade_id"] == "T1"
    assert rec["would_veto"] is True  # prob 0.35 < threshold 0.4
    assert rec["meta_decision"] == "SKIP"
    assert len(rec["features"]) == 50


def test_join_predict_outcome_matches_by_trade_id(tmp_path, monkeypatch):
    from src.ml import shadow_log

    monkeypatch.setenv("ACT_META_SHADOW_MODE", "1")
    monkeypatch.delenv("ACT_DISABLE_ML", raising=False)
    p = str(tmp_path / "meta_shadow.jsonl")

    # Two trades: T1 has both predict + outcome; T2 is still open (predict only)
    shadow_log.log_predict(
        trade_id="T1", asset="BTC", direction="LONG", entry_price=100.0,
        entry_score=5, meta_prob_raw=0.6, meta_prob_cal=0.55,
        take_threshold=0.4, features=[1.0] * 10, path=p,
    )
    shadow_log.log_outcome(
        trade_id="T1", pnl_pct=1.2, pnl_usd=12.0, exit_price=101.2,
        bars_held=15, exit_reason="EMA exit", path=p,
    )
    shadow_log.log_predict(
        trade_id="T2", asset="ETH", direction="LONG", entry_price=2000.0,
        entry_score=4, meta_prob_raw=0.3, meta_prob_cal=0.32,
        take_threshold=0.4, features=[0.5] * 10, path=p,
    )

    joined = shadow_log.join_predict_outcome(path=p)
    assert len(joined) == 1, f"only T1 should join; got {len(joined)}"
    r = joined[0]
    assert r["trade_id"] == "T1"
    assert r["asset"] == "BTC"
    assert r["win"] == 1
    assert r["pnl_pct"] == pytest.approx(1.2)
    assert r["would_veto"] is False


def test_shadow_stats_basic(tmp_path, monkeypatch):
    from src.ml import shadow_log

    # Synthetic joined records (no file I/O)
    joined = [
        {"trade_id": "T1", "asset": "BTC", "would_veto": True, "win": 0, "pnl_pct": -1.5, "features": []},
        {"trade_id": "T2", "asset": "BTC", "would_veto": True, "win": 1, "pnl_pct": 2.0, "features": []},
        {"trade_id": "T3", "asset": "BTC", "would_veto": False, "win": 1, "pnl_pct": 1.0, "features": []},
        {"trade_id": "T4", "asset": "BTC", "would_veto": False, "win": 0, "pnl_pct": -0.5, "features": []},
    ]
    s = shadow_log.shadow_stats(joined)
    assert s["n"] == 4
    assert s["actual_wr"] == 0.5  # 2 wins / 4
    assert s["meta_veto_count"] == 2
    assert s["meta_take_count"] == 2
    # Of 2 veto suggestions: 1 was correct (T1 lost). Precision = 0.5
    assert s["veto_precision_loss"] == 0.5
    # Of 2 take suggestions: 1 was correct (T3 won). Precision = 0.5
    assert s["take_precision_win"] == 0.5
    # If vetoed pnl = pnl of trades where would_veto=False = T3 + T4 = 1.0 - 0.5 = 0.5
    assert s["if_vetoed_pnl_pct"] == pytest.approx(0.5)


def test_shadow_retrain_refuses_small_dataset(tmp_path, monkeypatch):
    """shadow_retrain must bail cleanly when the joined dataset is below the threshold."""
    from src.scripts.shadow_retrain import retrain

    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    # Empty shadow log
    shadow_log_path = tmp_path / "logs" / "meta_shadow.jsonl"
    shadow_log_path.write_text("", encoding="utf-8")

    result = retrain(asset="BTC", min_joined=100, models_dir=str(tmp_path / "models"), force=False)
    assert result["promoted"] is False
    assert result["reason"] == "insufficient_data"
    assert result["n"] == 0


def test_shadow_retrain_cli_main_exists():
    from src.scripts import shadow_retrain
    assert hasattr(shadow_retrain, "main")
    assert callable(shadow_retrain.main)
