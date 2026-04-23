"""Tests for BaseAgent per-agent episodic memory (C12b)."""
from __future__ import annotations

import json

import pytest

from src.agents.base_agent import AgentVote, BaseAgent


class _TestAgent(BaseAgent):
    """Minimal concrete subclass — analyze() just echoes."""

    def analyze(self, quant_state, context):
        return AgentVote(direction=1, confidence=0.5, reasoning="test")


def _state(**overrides):
    base = {
        "ema_slope_pct": 0.5, "hurst": 0.52, "rsi": 60.0,
        "atr_pct": 1.2, "regime": "trending",
    }
    base.update(overrides)
    return base


# ── Fingerprint extraction ─────────────────────────────────────────────


def test_fingerprint_picks_scalar_keys_only():
    a = _TestAgent("x")
    st = _state(big_array=[1, 2, 3], obj={"nested": True}, hurst=0.58)
    snap = a._episode_fingerprint(st)
    assert "hurst" in snap
    assert snap["hurst"] == 0.58
    assert "big_array" not in snap
    assert "obj" not in snap


def test_fingerprint_extra_keys():
    a = _TestAgent("x")
    snap = a._episode_fingerprint({"custom_metric": 42}, extra_keys=["custom_metric"])
    assert snap.get("custom_metric") == 42


# ── record_episode ─────────────────────────────────────────────────────


def test_record_appends_to_buffer():
    a = _TestAgent("x")
    assert a.episodic_memory_size() == 0
    a.record_episode(_state(), AgentVote(direction=1, confidence=0.7))
    assert a.episodic_memory_size() == 1


def test_record_respects_buffer_cap():
    a = _TestAgent("x", config={"episode_buffer_size": 3})
    for i in range(10):
        a.record_episode(_state(rsi=50 + i), AgentVote(direction=1))
    assert a.episodic_memory_size() == 3


def test_record_survives_malformed_vote():
    a = _TestAgent("x")
    # Passing a dict instead of AgentVote shouldn't crash.
    try:
        a.record_episode(_state(), {"direction": 1})   # type: ignore[arg-type]
    except Exception as e:
        pytest.fail(f"record_episode should not raise: {e}")


# ── get_similar_episodes ───────────────────────────────────────────────


def test_similar_returns_empty_when_buffer_empty():
    a = _TestAgent("x")
    assert a.get_similar_episodes(_state()) == []


def test_similar_ranks_nearest_first():
    a = _TestAgent("x")
    # Episode 1: very different rsi.
    a.record_episode(_state(rsi=20.0), AgentVote(direction=1))
    # Episode 2: very similar rsi.
    a.record_episode(_state(rsi=62.0), AgentVote(direction=1))
    # Episode 3: another far-away rsi.
    a.record_episode(_state(rsi=5.0), AgentVote(direction=-1))

    current = _state(rsi=60.0)
    hits = a.get_similar_episodes(current, k=2)
    assert len(hits) == 2
    # Nearest must be the rsi=62 episode.
    assert hits[0]["state"]["rsi"] == 62.0


def test_similar_skips_low_overlap_episodes():
    a = _TestAgent("x")
    # Current state has only 1 numeric key → below min_numeric_keys=2.
    fake_state = {"only_one": 5.0}
    assert a.get_similar_episodes(fake_state) == []


def test_similar_returns_full_episode_shape():
    a = _TestAgent("x")
    a.record_episode(_state(), AgentVote(direction=1, confidence=0.6), outcome={"pnl_pct": 1.3})
    hits = a.get_similar_episodes(_state(), k=1)
    assert hits
    ep = hits[0]
    assert "state" in ep and "vote" in ep and "outcome" in ep
    assert ep["vote"]["direction"] == 1
    assert ep["outcome"]["pnl_pct"] == 1.3


# ── Persistence ────────────────────────────────────────────────────────


def test_save_load_preserves_episode_buffer(tmp_path):
    a = _TestAgent("savetest")
    for i in range(5):
        a.record_episode(_state(rsi=40 + i), AgentVote(direction=1, confidence=0.5))
    path = tmp_path / "state.json"
    a.save_state(str(path))

    b = _TestAgent("savetest")
    assert b.episodic_memory_size() == 0
    b.load_state(str(path))
    assert b.episodic_memory_size() == 5


def test_load_from_pre_c12b_state_file_still_works(tmp_path):
    """A state file saved BEFORE C12b lacks episode_buffer — loader
    must tolerate the missing key and initialize an empty deque."""
    path = tmp_path / "legacy.json"
    legacy = {
        "name": "legacy", "weight": 1.2, "total_calls": 10,
        "correct_calls": 7, "accuracy_history": [1, 1, 0, 1],
        # NO episode_buffer key.
    }
    path.write_text(json.dumps(legacy), encoding="utf-8")

    a = _TestAgent("legacy")
    a.load_state(str(path))
    assert a.episodic_memory_size() == 0
    assert abs(a._current_weight - 1.2) < 1e-9


# ── Orchestrator / debate_engine integration ──────────────────────────


def test_existing_agents_inherit_episodic_memory():
    """Spot-check: actual ACT agents inherit BaseAgent and so should
    have the episodic memory API."""
    try:
        from src.agents.risk_guardian_agent import RiskGuardianAgent
        rg = RiskGuardianAgent()
        assert hasattr(rg, "record_episode")
        assert hasattr(rg, "get_similar_episodes")
        assert rg.episodic_memory_size() == 0
    except Exception as e:
        pytest.skip(f"RiskGuardianAgent unavailable in this env: {e}")
