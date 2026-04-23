"""Tests for src/agents/persona_from_graph.py — transient persona agents."""
from __future__ import annotations

import json
import time
from unittest import mock

import pytest

from src.agents.persona_from_graph import (
    DISABLE_ENV,
    PersonaDescriptor,
    PersonaManager,
    TransientPersonaAgent,
    get_active_personas,
    get_manager,
)
from src.agents.base_agent import AgentVote


@pytest.fixture
def fresh_manager(monkeypatch):
    import src.agents.persona_from_graph as pfg
    mgr = PersonaManager(max_concurrent=3, min_spawn_heat=0.5,
                          dissolve_heat=0.2, max_age_s=3600)
    monkeypatch.setattr(pfg, "_manager_singleton", mgr, raising=False)
    yield mgr


@pytest.fixture
def fresh_graph(tmp_path, monkeypatch):
    """Tmp KnowledgeGraph so tests don't pollute the shared store."""
    from src.ai.graph_rag import KnowledgeGraph
    import src.ai.graph_rag as gr
    g = KnowledgeGraph(str(tmp_path / "kg.sqlite"))
    monkeypatch.setattr(gr, "_graph_singleton", g, raising=False)
    yield g
    g.close()


def _seed_hot_cluster(graph, asset: str = "BTC",
                      anchors=(("SEC approves ETF", 5.0), ("whale moves", 3.0))):
    """Helper — seed the graph with edges weighty enough to look 'hot'."""
    asset_eid = graph.upsert_entity(kind="asset", name=asset)
    for name, weight in anchors:
        nid = graph.upsert_entity(kind="news", name=name)
        graph.add_edge(
            src_id=asset_eid, dst_id=nid, relation="mentions",
            kind="news", weight=weight,
        )
    return asset_eid


# ── TransientPersonaAgent ──────────────────────────────────────────────


def test_persona_inherits_base_agent_api():
    desc = PersonaDescriptor(
        persona_id="persona:test", name="Tester", theme="unit test",
        anchor_entity_id="x", cluster_heat=1.0,
    )
    agent = TransientPersonaAgent(desc)
    # Bayesian accuracy + episodic memory inherited from BaseAgent.
    assert hasattr(agent, "record_episode")
    assert hasattr(agent, "get_similar_episodes")
    assert agent.get_weight() == 1.0


def test_persona_analyze_returns_vote_even_when_llm_unavailable(monkeypatch):
    """If LLM import fails, analyze() still returns a neutral vote."""
    desc = PersonaDescriptor(
        persona_id="persona:test", name="NoLLM", theme="t",
        anchor_entity_id="x", cluster_heat=1.0,
    )
    agent = TransientPersonaAgent(desc)

    import sys
    fake_module = mock.MagicMock(analyze=mock.MagicMock(side_effect=RuntimeError("x")))
    monkeypatch.setitem(sys.modules, "src.ai.dual_brain", fake_module)
    vote = agent.analyze({"asset": "BTC"}, {"asset": "BTC"})
    assert isinstance(vote, AgentVote)
    assert vote.direction == 0
    assert "LLM unavailable" in vote.reasoning


def test_persona_analyze_parses_llm_json(monkeypatch):
    desc = PersonaDescriptor(
        persona_id="persona:bullish", name="Bull", theme="hot-streak",
        anchor_entity_id="x", cluster_heat=2.0,
    )
    agent = TransientPersonaAgent(desc)

    fake_resp = mock.MagicMock(
        ok=True,
        text='Sure. {"direction": 1, "confidence": 0.72, "rationale": "tight stop"}',
    )
    import sys
    fake_module = mock.MagicMock(analyze=mock.MagicMock(return_value=fake_resp))
    monkeypatch.setitem(sys.modules, "src.ai.dual_brain", fake_module)

    vote = agent.analyze({"asset": "BTC", "rsi": 60}, {"asset": "BTC"})
    assert vote.direction == 1
    assert vote.confidence == pytest.approx(0.72)
    assert "tight stop" in vote.reasoning
    assert vote.metadata.get("persona") == "persona:bullish"


def test_persona_analyze_garbage_llm_falls_back(monkeypatch):
    desc = PersonaDescriptor(
        persona_id="persona:x", name="X", theme="t",
        anchor_entity_id="x", cluster_heat=1.0,
    )
    agent = TransientPersonaAgent(desc)
    fake_resp = mock.MagicMock(ok=True, text="no json here at all")
    import sys
    fake_module = mock.MagicMock(analyze=mock.MagicMock(return_value=fake_resp))
    monkeypatch.setitem(sys.modules, "src.ai.dual_brain", fake_module)

    vote = agent.analyze({}, {})
    # Falls back to neutral via the except path.
    assert vote.direction == 0


# ── PersonaManager ─────────────────────────────────────────────────────


def test_manager_empty_at_start(fresh_manager):
    assert fresh_manager.active() == []


def test_manager_disabled_by_env(monkeypatch, fresh_manager, fresh_graph):
    monkeypatch.setenv(DISABLE_ENV, "1")
    report = fresh_manager.refresh(asset="BTC")
    assert report.get("disabled") is True
    assert fresh_manager.active() == []


def test_manager_spawns_for_hot_clusters(fresh_manager, fresh_graph):
    _seed_hot_cluster(fresh_graph, "BTC")
    report = fresh_manager.refresh(asset="BTC")
    # Two anchors above heat threshold → two personas (capped by max_concurrent=3).
    assert len(report["spawned"]) >= 1
    assert len(fresh_manager.active()) >= 1
    # Heat values reported.
    for s in report["spawned"]:
        assert s["heat"] >= fresh_manager.min_spawn_heat


def test_manager_respects_max_concurrent(fresh_manager, fresh_graph):
    # Seed MORE clusters than the cap (3).
    anchors = [(f"anchor-{i}", 3.0) for i in range(10)]
    _seed_hot_cluster(fresh_graph, "BTC", anchors=anchors)
    fresh_manager.refresh(asset="BTC")
    assert len(fresh_manager.active()) <= fresh_manager.max_concurrent


def test_manager_skips_cold_clusters(fresh_manager, fresh_graph):
    # All heats below the spawn threshold (0.5).
    anchors = [("weak-1", 0.1), ("weak-2", 0.2)]
    _seed_hot_cluster(fresh_graph, "BTC", anchors=anchors)
    report = fresh_manager.refresh(asset="BTC")
    assert report["spawned"] == []


def test_manager_dissolves_cold_personas(fresh_manager, fresh_graph):
    # First pass: heavy anchors spawn personas.
    _seed_hot_cluster(fresh_graph, "BTC",
                       anchors=[("heavy", 5.0)])
    fresh_manager.refresh(asset="BTC")
    assert len(fresh_manager.active()) == 1

    # Stub top_connected to return zero heat → persona should dissolve.
    with mock.patch.object(fresh_graph, "top_connected", return_value=[]):
        report = fresh_manager.refresh(asset="BTC")
    assert report["dissolved"]
    assert fresh_manager.active() == []


def test_manager_dissolves_by_age(fresh_manager, fresh_graph):
    _seed_hot_cluster(fresh_graph, "BTC", anchors=[("anchor", 5.0)])
    fresh_manager.refresh(asset="BTC")
    # Age the persona beyond max_age_s.
    for a in fresh_manager._active.values():
        a.descriptor.spawned_at = time.time() - fresh_manager.max_age_s - 100
    report = fresh_manager.refresh(asset="BTC")
    assert any(d["reason"] == "age" for d in report["dissolved"])


def test_manager_refresh_tolerates_graph_error(fresh_manager, monkeypatch):
    import src.ai.graph_rag as gr

    def _boom():
        raise RuntimeError("db down")
    monkeypatch.setattr(gr, "get_graph", _boom)
    report = fresh_manager.refresh(asset="BTC")
    # Report still returns cleanly.
    assert isinstance(report, dict)
    assert report["spawned"] == []


# ── Singleton + integration ────────────────────────────────────────────


def test_get_manager_singleton():
    a = get_manager()
    b = get_manager()
    assert a is b


def test_get_active_personas_returns_list(fresh_manager, fresh_graph):
    _seed_hot_cluster(fresh_graph, "BTC", anchors=[("anchor", 5.0)])
    fresh_manager.refresh(asset="BTC")
    personas = get_active_personas()
    assert isinstance(personas, list)
    assert all(isinstance(p, TransientPersonaAgent) for p in personas)


def test_get_active_personas_never_raises():
    """Even if the manager internals blow up, the orchestrator-facing
    shortcut must return a list (never raise)."""
    # Can't easily force get_manager() to raise, but we can verify the
    # public contract by calling repeatedly.
    for _ in range(3):
        out = get_active_personas()
        assert isinstance(out, list)
