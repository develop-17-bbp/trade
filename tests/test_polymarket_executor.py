"""Tests for PolymarketExecutor — shadow-by-default, gated live path."""
from __future__ import annotations

import json
import sqlite3
from unittest import mock

import pytest

from src.exchanges.polymarket_executor import (
    DISABLE_ENV,
    LIVE_ENV,
    PolymarketExecutor,
    PolymarketOrderResult,
)


@pytest.fixture
def warm_store_tmp(tmp_path, monkeypatch):
    """Redirect warm_store to a tmp sqlite for each test."""
    from src.orchestration.warm_store import WarmStore
    import src.orchestration.warm_store as ws_mod
    db = tmp_path / "w.sqlite"
    store = WarmStore(str(db))
    monkeypatch.setattr(ws_mod, "_store_singleton", store, raising=False)
    return store


# ── Mode resolution ────────────────────────────────────────────────────


def test_defaults_to_shadow(monkeypatch):
    monkeypatch.delenv(LIVE_ENV, raising=False)
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    ex = PolymarketExecutor(config={"polymarket": {"enabled": False}})
    assert ex.mode() == "shadow"


def test_live_env_alone_not_enough(monkeypatch):
    monkeypatch.setenv(LIVE_ENV, "1")
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    # enabled=False → still shadow.
    ex = PolymarketExecutor(config={"polymarket": {"enabled": False}})
    assert ex.mode() == "shadow"


def test_disable_env_forces_shadow(monkeypatch):
    monkeypatch.setenv(LIVE_ENV, "1")
    monkeypatch.setenv(DISABLE_ENV, "1")
    ex = PolymarketExecutor(config={"polymarket": {"enabled": True}})
    assert ex.mode() == "shadow"


# ── Input validation ───────────────────────────────────────────────────


def test_invalid_side_returns_failure():
    ex = PolymarketExecutor()
    r = ex.place_order(market_id="m1", side="MAYBE", shares=10, price=0.4)
    assert r.ok is False
    assert "invalid side" in (r.reason or "").lower()


def test_invalid_shares_rejected():
    ex = PolymarketExecutor()
    r = ex.place_order(market_id="m1", side="YES", shares=0, price=0.4)
    assert r.ok is False


def test_invalid_price_rejected():
    ex = PolymarketExecutor()
    r = ex.place_order(market_id="m1", side="YES", shares=5, price=1.5)
    assert r.ok is False


# ── Shadow-mode happy path ─────────────────────────────────────────────


def test_shadow_mode_logs_to_warm_store(warm_store_tmp, monkeypatch):
    monkeypatch.delenv(LIVE_ENV, raising=False)
    ex = PolymarketExecutor()
    r = ex.place_order(
        market_id="btc-up-1h", side="YES", shares=50, price=0.42,
        plan_digest={"test": True},
    )
    assert r.ok is True
    assert r.mode == "shadow"
    assert r.decision_id is not None
    assert r.order_id is None

    warm_store_tmp.flush()
    conn = sqlite3.connect(warm_store_tmp.db_path)
    row = conn.execute(
        "SELECT final_action, component_signals FROM decisions "
        "WHERE decision_id=?", (r.decision_id,),
    ).fetchone()
    conn.close()
    assert row is not None
    assert row[0] == "PM_SHADOW"
    cs = json.loads(row[1])
    assert cs["source"] == "polymarket_executor"
    assert cs["mode"] == "shadow"
    assert cs["side"] == "YES"
    assert cs["shares"] == 50


def test_shadow_mode_tolerates_warm_store_failure(monkeypatch):
    ex = PolymarketExecutor()
    with mock.patch("src.orchestration.warm_store.get_store", side_effect=RuntimeError("db")):
        r = ex.place_order(market_id="m", side="YES", shares=5, price=0.5)
    # Returns ok=True (shadow placeholder succeeds even if log fails).
    assert r.ok is True
    assert r.mode == "shadow"


# ── Live-mode fallback when client unavailable ─────────────────────────


def test_live_requested_but_client_missing_degrades_to_shadow(monkeypatch, warm_store_tmp):
    monkeypatch.setenv(LIVE_ENV, "1")
    # Fake a "live-ready" state by patching the mode-check directly.
    cfg = {"polymarket": {"enabled": True}}
    ex = PolymarketExecutor(config=cfg)
    ex._live_ready = True       # force the live branch for this test

    # With no py_clob_client installed, _live_client raises → degrade.
    r = ex.place_order(market_id="m", side="YES", shares=5, price=0.4)
    assert r.mode == "shadow"
    assert r.ok is False
    assert "live client unavailable" in (r.reason or "").lower()


def test_live_mode_places_order_when_client_available(monkeypatch, warm_store_tmp):
    monkeypatch.setenv(LIVE_ENV, "1")
    ex = PolymarketExecutor(config={"polymarket": {"enabled": True}})
    ex._live_ready = True   # force

    fake_client = mock.MagicMock()
    fake_client.place_order.return_value = {"order_id": "ord-1", "ok": True}
    monkeypatch.setattr(ex, "_live_client", lambda: fake_client)

    r = ex.place_order(market_id="m", side="YES", shares=5, price=0.4)
    assert r.ok is True
    assert r.mode == "live"
    assert r.order_id == "ord-1"
    fake_client.place_order.assert_called_once()


def test_live_mode_handles_client_exception(monkeypatch, warm_store_tmp):
    monkeypatch.setenv(LIVE_ENV, "1")
    ex = PolymarketExecutor(config={"polymarket": {"enabled": True}})
    ex._live_ready = True

    fake_client = mock.MagicMock()
    fake_client.place_order.side_effect = RuntimeError("api rate limit")
    monkeypatch.setattr(ex, "_live_client", lambda: fake_client)

    r = ex.place_order(market_id="m", side="YES", shares=5, price=0.4)
    assert r.ok is False
    assert r.mode == "live"
    assert "api rate limit" in (r.reason or "").lower()


# ── Result shape ───────────────────────────────────────────────────────


def test_result_to_dict_shape():
    r = PolymarketOrderResult(
        ok=True, mode="shadow", market_id="x", side="YES",
        shares=10, price=0.5, decision_id="abc",
    )
    d = r.to_dict()
    for k in ("ok", "mode", "market_id", "side", "shares", "price",
              "order_id", "decision_id", "reason"):
        assert k in d
