"""Tests for C26 Step 5 — autonomous pursuit loop + update_overlay."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _load_psl():
    root = Path(__file__).resolve().parents[1]
    p = root / "skills" / "paper_soak_loose" / "action.py"
    spec = importlib.util.spec_from_file_location("_psl_ut", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── update_overlay ──────────────────────────────────────────────────────


def test_update_overlay_no_file_returns_none(tmp_path, monkeypatch):
    mod = _load_psl()
    monkeypatch.setattr(mod, "OVERLAY_FILE", tmp_path / "nope.json")
    monkeypatch.delenv("ACT_REAL_CAPITAL_ENABLED", raising=False)
    assert mod.update_overlay({"sniper": {"min_score": -1}}) is None


def test_update_overlay_loosens_within_floor(tmp_path, monkeypatch):
    mod = _load_psl()
    overlay_file = tmp_path / "overlay.json"
    overlay_file.write_text(json.dumps({
        "sniper": {"min_score": 4, "min_expected_move_pct": 2.0, "min_confluence": 3},
        "cost_gate": {"min_margin_pct": 0.3},
        "conviction": {"min_normal_strategies_agreeing": 2},
        "requires_paper_mode": True,
    }), encoding="utf-8")
    monkeypatch.setattr(mod, "OVERLAY_FILE", overlay_file)
    monkeypatch.delenv("ACT_REAL_CAPITAL_ENABLED", raising=False)

    updated = mod.update_overlay({
        "sniper": {"min_score": -1, "min_expected_move_pct": -0.3},
        "cost_gate": {"min_margin_pct": -0.1},
    })
    assert updated is not None
    assert updated["sniper"]["min_score"] == 3      # 4 - 1
    assert updated["sniper"]["min_expected_move_pct"] == pytest.approx(1.7)
    assert updated["cost_gate"]["min_margin_pct"] == pytest.approx(0.2)


def test_update_overlay_respects_floor(tmp_path, monkeypatch):
    mod = _load_psl()
    overlay_file = tmp_path / "overlay.json"
    overlay_file.write_text(json.dumps({
        "sniper": {"min_score": 2, "min_expected_move_pct": 1.0, "min_confluence": 2},
        "cost_gate": {"min_margin_pct": 0.1},
        "conviction": {"min_normal_strategies_agreeing": 1},
        "requires_paper_mode": True,
    }), encoding="utf-8")
    monkeypatch.setattr(mod, "OVERLAY_FILE", overlay_file)
    monkeypatch.delenv("ACT_REAL_CAPITAL_ENABLED", raising=False)

    updated = mod.update_overlay({
        "sniper": {"min_score": -5, "min_expected_move_pct": -5.0},
        "cost_gate": {"min_margin_pct": -5.0},
    })
    # Already at floor — stays there
    assert updated["sniper"]["min_score"] == 2
    assert updated["sniper"]["min_expected_move_pct"] == 1.0
    assert updated["cost_gate"]["min_margin_pct"] == 0.1


def test_update_overlay_refuses_under_real_capital(tmp_path, monkeypatch):
    mod = _load_psl()
    overlay_file = tmp_path / "overlay.json"
    overlay_file.write_text(json.dumps({
        "sniper": {"min_score": 4}, "requires_paper_mode": True,
    }), encoding="utf-8")
    monkeypatch.setattr(mod, "OVERLAY_FILE", overlay_file)
    monkeypatch.setenv("ACT_REAL_CAPITAL_ENABLED", "1")
    assert mod.update_overlay({"sniper": {"min_score": -1}}) is None


def test_update_overlay_records_adjustments_history(tmp_path, monkeypatch):
    mod = _load_psl()
    overlay_file = tmp_path / "overlay.json"
    overlay_file.write_text(json.dumps({
        "sniper": {"min_score": 4, "min_expected_move_pct": 2.0, "min_confluence": 3},
        "cost_gate": {"min_margin_pct": 0.3},
        "requires_paper_mode": True,
    }), encoding="utf-8")
    monkeypatch.setattr(mod, "OVERLAY_FILE", overlay_file)
    monkeypatch.delenv("ACT_REAL_CAPITAL_ENABLED", raising=False)

    updated = mod.update_overlay({
        "sniper": {"min_score": -1},
        "reason": "pursuit:no_submits_4h",
    })
    assert "adjustments" in updated
    assert len(updated["adjustments"]) == 1
    assert updated["adjustments"][0]["reason"] == "pursuit:no_submits_4h"


# ── _pursuit_step ───────────────────────────────────────────────────────


def _pursuit_loop(tmp_path, monkeypatch):
    """Build an AutonomousLoop with pursuit helpers patched."""
    import src.scripts.autonomous_loop as al
    monkeypatch.setattr(al, "PROJECT_ROOT", tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "logs").mkdir(exist_ok=True)
    return al.AutonomousLoop(dry_run=True)


def test_pursuit_noop_when_overlay_inactive(tmp_path, monkeypatch):
    loop = _pursuit_loop(tmp_path, monkeypatch)
    # No overlay → action = noop
    import skills.paper_soak_loose.action as psl
    monkeypatch.setattr(psl, "OVERLAY_FILE", tmp_path / "missing.json")
    result = loop._pursuit_step({})
    assert result["action"] == "noop"
    assert "not active" in result["reason"]


def test_pursuit_relaxes_when_no_submits(tmp_path, monkeypatch):
    loop = _pursuit_loop(tmp_path, monkeypatch)

    import skills.paper_soak_loose.action as psl
    overlay_file = tmp_path / "overlay.json"
    overlay_file.write_text(json.dumps({
        "sniper": {"min_score": 4, "min_expected_move_pct": 2.0, "min_confluence": 3},
        "cost_gate": {"min_margin_pct": 0.3},
        "conviction": {"min_normal_strategies_agreeing": 2},
        "requires_paper_mode": True,
    }), encoding="utf-8")
    monkeypatch.setattr(psl, "OVERLAY_FILE", overlay_file)
    monkeypatch.delenv("ACT_REAL_CAPITAL_ENABLED", raising=False)

    # No recent trades + rolling_pnl at 0
    result = loop._pursuit_step({"recent_trades": [], "rolling_pnl": 0.0})
    # dry_run=True → just reports the intent, doesn't write
    assert result["action"] == "relax_one_step"
    assert "no submits" in result["reason"]


def test_pursuit_tightens_when_losing_streak(tmp_path, monkeypatch):
    loop = _pursuit_loop(tmp_path, monkeypatch)

    import skills.paper_soak_loose.action as psl
    overlay_file = tmp_path / "overlay.json"
    overlay_file.write_text(json.dumps({
        "sniper": {"min_score": 3, "min_expected_move_pct": 1.7, "min_confluence": 3},
        "cost_gate": {"min_margin_pct": 0.2},
        "requires_paper_mode": True,
    }), encoding="utf-8")
    monkeypatch.setattr(psl, "OVERLAY_FILE", overlay_file)
    monkeypatch.delenv("ACT_REAL_CAPITAL_ENABLED", raising=False)

    # 4 recent trades + negative PnL
    fake_trades = [{"pnl_pct": -0.5}, {"pnl_pct": -0.3}, {"pnl_pct": -0.8}, {"pnl_pct": 0.1}]
    result = loop._pursuit_step({
        "recent_trades": fake_trades,
        "rolling_pnl": -2.5,
    })
    assert result["action"] == "tighten_one_step"
    assert "losing" in result["reason"]


def test_pursuit_holds_when_at_target(tmp_path, monkeypatch):
    loop = _pursuit_loop(tmp_path, monkeypatch)

    import skills.paper_soak_loose.action as psl
    overlay_file = tmp_path / "overlay.json"
    overlay_file.write_text(json.dumps({
        "sniper": {"min_score": 4, "min_expected_move_pct": 2.0, "min_confluence": 3},
        "cost_gate": {"min_margin_pct": 0.3},
        "requires_paper_mode": True,
    }), encoding="utf-8")
    monkeypatch.setattr(psl, "OVERLAY_FILE", overlay_file)
    monkeypatch.delenv("ACT_REAL_CAPITAL_ENABLED", raising=False)

    # approx_daily_pct = rolling_pnl/7 = 7.5/7 = 1.07% >= 1.0% target
    result = loop._pursuit_step({
        "recent_trades": [{"pnl_pct": 1.0}] * 10,
        "rolling_pnl": 7.5,
    })
    assert result["action"] == "hold"
    assert "target" in result["reason"]
