"""Tests for skills/paper_soak_loose — C22."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


def _load_action():
    root = Path(__file__).resolve().parents[1]
    action_path = root / "skills" / "paper_soak_loose" / "action.py"
    spec = importlib.util.spec_from_file_location("_psl_action", str(action_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_missing_enable_arg_errors():
    mod = _load_action()
    r = mod.run({})
    assert r.ok is False
    assert "enable" in r.error.lower()


def test_enable_writes_overlay(tmp_path, monkeypatch):
    mod = _load_action()
    overlay_file = tmp_path / "paper_soak_loose.json"
    monkeypatch.setattr(mod, "OVERLAY_FILE", overlay_file)
    monkeypatch.delenv("ACT_REAL_CAPITAL_ENABLED", raising=False)
    r = mod.run({"enable": "true"})
    assert r.ok is True
    assert overlay_file.exists()
    payload = json.loads(overlay_file.read_text(encoding="utf-8"))
    assert payload["sniper"]["min_score"] == 4
    assert payload["requires_paper_mode"] is True


def test_disable_removes_overlay(tmp_path, monkeypatch):
    mod = _load_action()
    overlay_file = tmp_path / "paper_soak_loose.json"
    monkeypatch.setattr(mod, "OVERLAY_FILE", overlay_file)
    monkeypatch.delenv("ACT_REAL_CAPITAL_ENABLED", raising=False)
    # Enable then disable
    mod.run({"enable": True})
    assert overlay_file.exists()
    r = mod.run({"enable": False})
    assert r.ok is True
    assert not overlay_file.exists()


def test_refuses_to_enable_under_real_capital(tmp_path, monkeypatch):
    mod = _load_action()
    overlay_file = tmp_path / "paper_soak_loose.json"
    monkeypatch.setattr(mod, "OVERLAY_FILE", overlay_file)
    monkeypatch.setenv("ACT_REAL_CAPITAL_ENABLED", "1")
    r = mod.run({"enable": True})
    assert r.ok is False
    assert "real" in r.error.lower() or "ACT_REAL_CAPITAL" in r.error
    assert not overlay_file.exists()


def test_custom_thresholds_respected(tmp_path, monkeypatch):
    mod = _load_action()
    overlay_file = tmp_path / "paper_soak_loose.json"
    monkeypatch.setattr(mod, "OVERLAY_FILE", overlay_file)
    monkeypatch.delenv("ACT_REAL_CAPITAL_ENABLED", raising=False)
    r = mod.run({"enable": True, "min_score": 3, "min_move_pct": 1.5})
    assert r.ok is True
    payload = json.loads(overlay_file.read_text(encoding="utf-8"))
    assert payload["sniper"]["min_score"] == 3
    assert payload["sniper"]["min_expected_move_pct"] == 1.5


def test_get_overlay_returns_none_when_disabled(tmp_path, monkeypatch):
    mod = _load_action()
    overlay_file = tmp_path / "paper_soak_loose.json"
    monkeypatch.setattr(mod, "OVERLAY_FILE", overlay_file)
    monkeypatch.delenv("ACT_REAL_CAPITAL_ENABLED", raising=False)
    assert mod.get_paper_soak_overlay() is None


def test_get_overlay_safe_under_real_capital(tmp_path, monkeypatch):
    """Even if the overlay file exists, real-capital mode must NOT return it."""
    mod = _load_action()
    overlay_file = tmp_path / "paper_soak_loose.json"
    monkeypatch.setattr(mod, "OVERLAY_FILE", overlay_file)
    # Write an overlay manually
    overlay_file.write_text(json.dumps({
        "sniper": {"min_score": 3}, "requires_paper_mode": True,
    }), encoding="utf-8")
    monkeypatch.setenv("ACT_REAL_CAPITAL_ENABLED", "1")
    assert mod.get_paper_soak_overlay() is None


def test_overlay_includes_bypass_macro_crisis_default(tmp_path, monkeypatch):
    mod = _load_action()
    overlay_file = tmp_path / "paper_soak_loose.json"
    monkeypatch.setattr(mod, "OVERLAY_FILE", overlay_file)
    monkeypatch.delenv("ACT_REAL_CAPITAL_ENABLED", raising=False)
    r = mod.run({"enable": True})
    assert r.ok is True
    payload = json.loads(overlay_file.read_text(encoding="utf-8"))
    assert payload["conviction"]["bypass_macro_crisis"] is True


def test_overlay_bypass_macro_crisis_disabled_when_requested(tmp_path, monkeypatch):
    mod = _load_action()
    overlay_file = tmp_path / "paper_soak_loose.json"
    monkeypatch.setattr(mod, "OVERLAY_FILE", overlay_file)
    monkeypatch.delenv("ACT_REAL_CAPITAL_ENABLED", raising=False)
    r = mod.run({"enable": True, "bypass_macro_crisis": False})
    assert r.ok is True
    payload = json.loads(overlay_file.read_text(encoding="utf-8"))
    assert payload["conviction"]["bypass_macro_crisis"] is False
