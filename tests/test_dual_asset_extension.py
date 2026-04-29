"""End-to-end tests for the dual-asset extension (Phase B-D).

Covers:
    * asset_class classifier (CRYPTO / STOCK / POLYMARKET routing)
    * market_hours NYSE schedule + holiday + early-close + leveraged blackout
    * stocks_conviction tier logic + IEX-skew bump
    * authority_rules_stocks RTH-only + leveraged-ETF rules + ETB-shorts
    * warm_store schema + asset_class column + idempotent migration
    * warm_store_sync round-trip + INSERT OR IGNORE dedup
    * quant_tools _validate_quant bounds (Hurst/OU/Hawkes)
    * trade_tools _validate_input_schema dispatch-boundary

Tests are pure unit / smoke — no live external deps. Designed to run
in CI on either the 4060 or 5090 as a regression guard for the
dual-asset wiring.
"""
from __future__ import annotations

import datetime as _dt
import sys
import tempfile
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── asset_class ─────────────────────────────────────────────────────


def test_asset_class_crypto():
    from src.models.asset_class import classify, AssetClass
    assert classify("BTCUSDT").asset_class is AssetClass.CRYPTO
    assert classify("BTC").symbol == "BTC"
    assert classify("eth").asset_class is AssetClass.CRYPTO


def test_asset_class_index_etfs_open_15pct():
    from src.models.asset_class import classify, AssetClass
    for sym in ("SPY", "QQQ"):
        m = classify(sym)
        assert m.asset_class is AssetClass.STOCK
        assert m.is_index_etf
        assert not m.is_leveraged_etf
        assert m.intraday_pct_max() == 15.0
        assert m.overnight_pct_max() == 5.0


def test_asset_class_leveraged_etf_caps():
    from src.models.asset_class import classify, is_leveraged_etf
    for sym in ("TQQQ", "SOXL", "UPRO", "SQQQ"):
        m = classify(sym)
        assert m.is_leveraged_etf, sym
        assert m.intraday_pct_max() == 5.0
        assert m.overnight_pct_max() == 0.0
    assert is_leveraged_etf("TQQQ") is True
    assert is_leveraged_etf("SPY") is False


def test_asset_class_polymarket_prefix():
    from src.models.asset_class import classify, AssetClass
    m = classify("polymarket-fed-cut-2026")
    assert m.asset_class is AssetClass.POLYMARKET


def test_asset_class_unknown_without_hint():
    from src.models.asset_class import classify, AssetClass
    assert classify("AAPL").asset_class is AssetClass.UNKNOWN
    assert classify("AAPL", venue_hint="alpaca").asset_class is AssetClass.STOCK


# ── market_hours ─────────────────────────────────────────────────────


def test_market_hours_holiday_full_closure():
    from src.utils.market_hours import session_for_date_et
    sess = session_for_date_et(_dt.date(2026, 7, 3))
    assert not sess.is_trading_day  # Independence Day observed


def test_market_hours_early_close_2026_11_27():
    from src.utils.market_hours import session_for_date_et
    sess = session_for_date_et(_dt.date(2026, 11, 27))
    assert sess.is_trading_day
    assert sess.is_early_close
    assert sess.close_et == _dt.time(13, 0)


def test_market_hours_weekend():
    from src.utils.market_hours import session_for_date_et
    sat = session_for_date_et(_dt.date(2026, 5, 2))
    sun = session_for_date_et(_dt.date(2026, 5, 3))
    assert not sat.is_trading_day
    assert not sun.is_trading_day


def test_market_hours_pre_close_blackouts():
    from src.utils.market_hours import is_pre_close_blackout, is_pre_close_leveraged_blackout
    near_close = _dt.datetime(2026, 4, 30, 19, 56, tzinfo=_dt.timezone.utc)  # 15:56 ET
    assert is_pre_close_blackout(near_close, minutes=5.0) is True
    assert is_pre_close_leveraged_blackout(near_close, minutes=30.0) is True
    well_before = _dt.datetime(2026, 4, 30, 14, 0, tzinfo=_dt.timezone.utc)  # 10:00 ET
    assert is_pre_close_blackout(well_before, minutes=5.0) is False
    assert is_pre_close_leveraged_blackout(well_before, minutes=30.0) is False


# ── stocks_conviction ───────────────────────────────────────────────


def test_stocks_conviction_iex_threshold_above_sip():
    """IEX feed (default) should require a wider min-move than SIP."""
    import os
    from src.trading.stocks_conviction import _effective_min_move, SIP_MIN_EXPECTED_MOVE_PCT

    os.environ["ACT_ALPACA_DATA_FEED"] = "iex"
    iex_min = _effective_min_move()
    os.environ["ACT_ALPACA_DATA_FEED"] = "sip"
    sip_min = _effective_min_move()
    os.environ["ACT_ALPACA_DATA_FEED"] = "iex"  # restore

    assert iex_min > sip_min
    assert sip_min == SIP_MIN_EXPECTED_MOVE_PCT


def test_stocks_conviction_rejects_low_move():
    from src.trading.stocks_conviction import evaluate
    r = evaluate(symbol="SPY", direction="LONG", expected_move_pct=0.05,
                 tf_5m_direction="UP", tf_15m_direction="UP", tf_1h_direction="UP",
                 multi_strategy_counts={"long": 5, "short": 0})
    assert r.tier == "reject"
    assert any("expected_move" in reason for reason in r.reasons)


def test_stocks_conviction_sniper_qqq():
    from src.trading.stocks_conviction import evaluate
    r = evaluate(symbol="QQQ", direction="LONG", expected_move_pct=0.40,
                 tf_5m_direction="UP", tf_15m_direction="UP", tf_1h_direction="UP",
                 multi_strategy_counts={"long": 6, "short": 0})
    assert r.tier == "sniper"
    assert r.intraday_pct_cap == 15.0
    assert r.size_multiplier == 3.0


def test_stocks_conviction_tqqq_caps_tighter():
    from src.trading.stocks_conviction import evaluate
    r = evaluate(symbol="TQQQ", direction="LONG", expected_move_pct=0.40,
                 tf_5m_direction="UP", tf_15m_direction="UP", tf_1h_direction="UP",
                 multi_strategy_counts={"long": 6, "short": 0})
    assert r.intraday_pct_cap == 5.0
    assert r.overnight_pct_cap == 0.0


def test_stocks_conviction_rejects_non_stock():
    from src.trading.stocks_conviction import evaluate
    r = evaluate(symbol="BTC", direction="LONG", expected_move_pct=2.0,
                 multi_strategy_counts={"long": 5, "short": 0})
    assert r.tier == "reject"
    assert any("not_a_stock" in reason for reason in r.reasons)


# ── authority_rules_stocks ──────────────────────────────────────────


def test_authority_rules_leveraged_overnight_refused():
    from src.ai.authority_rules_stocks import evaluate
    v = evaluate(symbol="TQQQ", side="buy", qty=1.0, is_overnight=True, intent="open")
    rules = [x.rule for x in v]
    assert "stocks.leveraged_etf_no_overnight" in rules


def test_authority_rules_fractional_short_refused():
    from src.ai.authority_rules_stocks import evaluate
    v = evaluate(symbol="SPY", side="sell", qty=0.5, fractional=True, intent="open")
    rules = [x.rule for x in v]
    assert "stocks.fractional_short_unsupported" in rules


def test_authority_rules_short_requires_etb():
    from src.ai.authority_rules_stocks import evaluate
    v = evaluate(symbol="ZZZNOTREAL", side="sell", qty=1.0, intent="open")
    rules = [x.rule for x in v]
    assert "stocks.short_requires_etb" in rules


# ── warm_store schema migration + writes ────────────────────────────


def test_warm_store_migration_adds_asset_class(tmp_path):
    from src.orchestration.warm_store import WarmStore
    import sqlite3
    ws = WarmStore(db_path=str(tmp_path / "test.sqlite"))
    conn = sqlite3.connect(str(tmp_path / "test.sqlite"))
    cols = [r[1] for r in conn.execute("PRAGMA table_info(decisions)").fetchall()]
    assert "asset_class" in cols
    assert "venue" in cols
    cols = [r[1] for r in conn.execute("PRAGMA table_info(outcomes)").fetchall()]
    assert "asset_class" in cols
    assert "venue" in cols


def test_warm_store_writes_asset_class_default(tmp_path):
    from src.orchestration.warm_store import WarmStore
    import sqlite3
    ws = WarmStore(db_path=str(tmp_path / "test.sqlite"))
    ws.write_decision({"decision_id": "t1", "symbol": "BTC", "ts_ns": time.time_ns(),
                       "direction": 1, "final_action": "BUY"})
    ws.write_decision({"decision_id": "t2", "symbol": "SPY", "ts_ns": time.time_ns(),
                       "direction": 1, "final_action": "BUY",
                       "asset_class": "STOCK", "venue": "alpaca"})
    ws.flush()
    rows = sqlite3.connect(str(tmp_path / "test.sqlite")).execute(
        "SELECT decision_id, asset_class, venue FROM decisions ORDER BY ts_ns").fetchall()
    by_id = {r[0]: r for r in rows}
    assert by_id["t1"][1] == "CRYPTO"
    assert by_id["t1"][2] == "robinhood"
    assert by_id["t2"][1] == "STOCK"
    assert by_id["t2"][2] == "alpaca"


# ── warm_store_sync round-trip + dedup ──────────────────────────────


def test_warm_store_sync_idempotent_ingest(tmp_path):
    from src.orchestration.warm_store import WarmStore
    from scripts.warm_store_sync import (
        _new_decision_rows, _new_outcome_rows, _write_delta_parquet, _ingest_delta,
    )
    src = WarmStore(db_path=str(tmp_path / "src.sqlite"))
    dst = WarmStore(db_path=str(tmp_path / "dst.sqlite"))

    src.write_decision({"decision_id": "rt-SPY", "symbol": "SPY",
                         "ts_ns": time.time_ns(), "direction": 1, "final_action": "BUY",
                         "asset_class": "STOCK", "venue": "alpaca"})
    src.flush()

    decisions = _new_decision_rows(str(tmp_path / "src.sqlite"), since_ts_ns=0)
    outcomes  = _new_outcome_rows(str(tmp_path / "src.sqlite"), since_exit_ts=0.0)
    bundle = _write_delta_parquet(tmp_path / "outbox", decisions, outcomes)
    assert bundle is not None

    n_d, n_o = _ingest_delta(bundle, str(tmp_path / "dst.sqlite"))
    assert n_d == 1 and n_o == 0
    n_d2, _ = _ingest_delta(bundle, str(tmp_path / "dst.sqlite"))
    assert n_d2 == 0  # idempotent


# ── quant_tools bounds-check ────────────────────────────────────────


def test_quant_tools_rejects_nan_hurst():
    import math
    from src.ai.quant_tools import _validate_quant
    e = _validate_quant("hurst", {"hurst": math.nan}, {"hurst": ("range", 0.0, 1.0)})
    assert e is not None and "non-finite" in e["error"]


def test_quant_tools_rejects_inf_half_life():
    import math
    from src.ai.quant_tools import _validate_quant
    e = _validate_quant("ou", {"half_life": math.inf}, {"half_life": ("positive",)})
    assert e is not None and "non-finite" in e["error"]


def test_quant_tools_rejects_negative_hawkes_alpha():
    from src.ai.quant_tools import _validate_quant
    e = _validate_quant("hawkes", {"alpha": -1.0}, {"alpha": ("positive",)})
    assert e is not None and "not positive" in e["error"]


def test_quant_tools_accepts_valid_hmm_confidence_at_boundary():
    from src.ai.quant_tools import _validate_quant
    assert _validate_quant("hmm", {"confidence": 1.0}, {"confidence": ("range", 0.0, 1.0 + 1e-6)}) is None


# ── trade_tools schema validation ────────────────────────────────────


def test_schema_validation_required_field():
    from src.ai.trade_tools import _validate_input_schema
    schema = {"properties": {"asset": {"type": "string"}}, "required": ["asset"]}
    err = _validate_input_schema({}, schema)
    assert err is not None and err.get("field") == "asset"


def test_schema_validation_coerces_int_to_string():
    from src.ai.trade_tools import _validate_input_schema
    args = {"asset": 999}
    err = _validate_input_schema(args, {"properties": {"asset": {"type": "string"}}})
    assert err is None  # coerces silently
    assert args["asset"] == "999"


def test_schema_validation_enum_violation():
    from src.ai.trade_tools import _validate_input_schema
    err = _validate_input_schema(
        {"tf": "17m"},
        {"properties": {"tf": {"type": "string", "enum": ["1h", "4h", "1d"]}}},
    )
    assert err is not None and "not in" in err["detail"]


def test_schema_validation_minimum_max():
    from src.ai.trade_tools import _validate_input_schema
    err = _validate_input_schema(
        {"bars": 9999},
        {"properties": {"bars": {"type": "integer", "minimum": 50, "maximum": 1000}}},
    )
    assert err is not None and "9999" in err["detail"]


# ── alpaca_executor reject paths (no creds smoke) ───────────────────


def test_alpaca_executor_rejects_non_stock():
    import os
    os.environ.pop("ACT_DISABLE_AGENTIC_LOOP", None)
    from src.trading.alpaca_executor import AlpacaExecutor
    e = AlpacaExecutor(paper=True)
    r = e.submit_order("BTC", "buy", 1.0)
    assert r.submitted is False
    assert "not_a_stock" in r.reason


def test_alpaca_executor_kill_switch():
    import os
    os.environ["ACT_DISABLE_AGENTIC_LOOP"] = "1"
    try:
        from src.trading.alpaca_executor import AlpacaExecutor
        e = AlpacaExecutor(paper=True)
        r = e.submit_order("SPY", "buy", 1.0)
        assert r.submitted is False
        assert r.reason == "kill_switch"
    finally:
        os.environ.pop("ACT_DISABLE_AGENTIC_LOOP", None)
