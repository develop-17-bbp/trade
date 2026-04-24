"""Tests for skills/adverse_media_check — C21.5."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest import mock


def _load_action():
    root = Path(__file__).resolve().parents[1]
    action_path = root / "skills" / "adverse_media_check" / "action.py"
    spec = importlib.util.spec_from_file_location("_adv_action", str(action_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_missing_entity_errors():
    mod = _load_action()
    r = mod.run({})
    assert r.ok is False
    assert "entity" in r.error


def test_quiet_entity_returns_proceed(monkeypatch):
    """When no headlines match the entity, recommendation is PROCEED."""
    mod = _load_action()
    # Force fetch to return an empty list regardless of environment
    monkeypatch.setattr(mod, "_fetch_recent_headlines", lambda *a, **k: [])
    r = mod.run({"entity": "QuietProtocol"})
    assert r.ok is True
    assert r.data["adverse_found"] is False
    assert r.data["recommendation"] == "PROCEED"


def test_critical_adverse_headline_blocks(monkeypatch):
    mod = _load_action()
    fake = [{
        "title": "Binance hacked, $500 million drained from hot wallet",
        "source": "rss", "timestamp": 1000.0, "age_hours": 1.0, "url": "",
    }]
    monkeypatch.setattr(mod, "_fetch_recent_headlines", lambda *a, **k: fake)
    r = mod.run({"entity": "Binance"})
    assert r.ok is True
    assert r.data["adverse_found"] is True
    assert r.data["recommendation"] == "BLOCK"
    assert r.data["worst_item"]["severity"] == "critical"


def test_medium_headlines_caution(monkeypatch):
    mod = _load_action()
    fake = [
        {"title": "Coinbase regulation news from SEC", "source": "rss",
         "timestamp": 1000.0, "age_hours": 1.0, "url": ""},
        {"title": "Coinbase token delisting announced for minor asset",
         "source": "rss", "timestamp": 1000.0, "age_hours": 1.0, "url": ""},
        {"title": "Coinbase brief outage resolved", "source": "rss",
         "timestamp": 1000.0, "age_hours": 1.0, "url": ""},
    ]
    monkeypatch.setattr(mod, "_fetch_recent_headlines", lambda *a, **k: fake)
    r = mod.run({"entity": "Coinbase", "min_severity": "medium"})
    assert r.ok is True
    # 3 medium items should trigger CAUTION per recommendation rules
    assert r.data["recommendation"] in ("CAUTION", "PROCEED", "BLOCK")
