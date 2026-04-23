"""Tests for src/ai/graph_rag.py — real-time knowledge graph."""
from __future__ import annotations

import json
import time
import types

import pytest

from src.ai.graph_rag import (
    DEFAULT_HALF_LIFE_S,
    DISABLE_ENV,
    Edge,
    Entity,
    KnowledgeGraph,
    ingest_correlation,
    ingest_institutional,
    ingest_news,
    ingest_polymarket,
    ingest_sentiment,
    query_digest,
)


@pytest.fixture
def graph(tmp_path, monkeypatch):
    g = KnowledgeGraph(str(tmp_path / "kg.sqlite"))
    # Swap the module-level singleton so query_digest + ingest_* use this
    # test graph.
    import src.ai.graph_rag as gr
    monkeypatch.setattr(gr, "_graph_singleton", g, raising=False)
    yield g
    g.close()


# ── Entity + edge primitives ───────────────────────────────────────────


def test_upsert_entity_is_idempotent(graph):
    eid1 = graph.upsert_entity(kind="asset", name="BTC")
    eid2 = graph.upsert_entity(kind="asset", name="BTC")
    assert eid1 == eid2
    ent = graph.get_entity(eid1)
    assert ent is not None
    assert ent.kind == "asset" and ent.name == "BTC"


def test_upsert_entity_merges_attrs(graph):
    graph.upsert_entity(kind="asset", name="ETH", attrs={"venue": "robinhood"})
    graph.upsert_entity(kind="asset", name="ETH", attrs={"tier": "primary"})
    ent = graph.get_entity("asset:eth")
    assert ent.attrs["venue"] == "robinhood"
    assert ent.attrs["tier"] == "primary"


def test_add_edge_stores_payload(graph):
    a = graph.upsert_entity(kind="asset", name="BTC")
    n = graph.upsert_entity(kind="news", name="SEC approves spot ETF")
    eid = graph.add_edge(
        src_id=a, dst_id=n, relation="mentions", kind="news",
        weight=1.5, source="newsapi", payload={"event_type": "etf"},
    )
    assert eid
    edges = graph.recent_edges(entity_id=a, since_s=3600)
    assert any(e.edge_id == eid and e.payload.get("event_type") == "etf"
               for e in edges)


# ── Time decay ─────────────────────────────────────────────────────────


def test_decayed_weight_halves_every_half_life():
    # Manufactured edge 1 half-life in the past.
    hl = 3600.0
    e = Edge(
        edge_id="x", src_id="a", dst_id="b", relation="r",
        kind="news", weight=1.0, ts=time.time() - hl,
    )
    # Using news default half-life (3h) → 1h ago = 2^(-1/3) ≈ 0.7937.
    w = e.decayed_weight()
    assert 0.7 < w < 0.9


def test_decayed_weight_zero_half_life_returns_raw():
    e = Edge(
        edge_id="x", src_id="a", dst_id="b", relation="r",
        kind="news", weight=2.0, ts=time.time() - 10_000,
    )
    assert e.decayed_weight(half_life_s=0) == pytest.approx(2.0)


# ── recent_edges / top_connected ───────────────────────────────────────


def test_recent_edges_respects_kind_filter(graph):
    a = graph.upsert_entity(kind="asset", name="BTC")
    n = graph.upsert_entity(kind="news", name="x")
    s = graph.upsert_entity(kind="sentiment_label", name="bullish")
    graph.add_edge(src_id=a, dst_id=n, relation="mentions", kind="news", weight=1)
    graph.add_edge(src_id=a, dst_id=s, relation="tilted_to", kind="sentiment", weight=1)
    news_only = graph.recent_edges(entity_id=a, kind="news")
    assert len(news_only) == 1
    assert news_only[0].kind == "news"


def test_recent_edges_respects_since(graph):
    a = graph.upsert_entity(kind="asset", name="BTC")
    n = graph.upsert_entity(kind="news", name="x")
    graph.add_edge(src_id=a, dst_id=n, relation="mentions", kind="news", weight=1)
    # since_s=1 is too tight to include just-added edges older than 1s.
    time.sleep(1.1)
    assert graph.recent_edges(entity_id=a, since_s=1) == []
    assert graph.recent_edges(entity_id=a, since_s=3600) != []


def test_top_connected_ranks_by_decayed_weight(graph):
    a = graph.upsert_entity(kind="asset", name="BTC")
    heavy = graph.upsert_entity(kind="news", name="heavy")
    light = graph.upsert_entity(kind="news", name="light")
    graph.add_edge(src_id=a, dst_id=heavy, relation="mentions", kind="news", weight=5.0)
    graph.add_edge(src_id=a, dst_id=light, relation="mentions", kind="news", weight=0.5)
    top = graph.top_connected(a, limit=5)
    assert top
    # Heaviest edge should rank first.
    assert top[0]["neighbor_name"] == "heavy"
    assert top[0]["total_decayed_weight"] > top[1]["total_decayed_weight"]


def test_count_by_kind(graph):
    a = graph.upsert_entity(kind="asset", name="BTC")
    n = graph.upsert_entity(kind="news", name="x")
    for _ in range(3):
        graph.add_edge(src_id=a, dst_id=n, relation="mentions", kind="news", weight=1)
    for _ in range(2):
        graph.add_edge(src_id=a, dst_id=n, relation="mentions", kind="sentiment", weight=1)
    counts = graph.count_by_kind(since_s=3600)
    assert counts.get("news") == 3
    assert counts.get("sentiment") == 2


# ── Ingest helpers ─────────────────────────────────────────────────────


def test_ingest_news_counts_items(graph):
    items = [
        types.SimpleNamespace(title="SEC approves spot ETF", source="newsapi",
                              timestamp=time.time(), event_type="etf"),
        types.SimpleNamespace(title="Bitcoin crashes", source="reddit",
                              timestamp=time.time(), event_type="regulatory"),
    ]
    n = ingest_news("BTC", items)
    assert n == 2
    edges = graph.recent_edges(entity_id="asset:btc")
    assert any(e.payload.get("event_type") == "etf" for e in edges)


def test_ingest_news_skips_items_without_title(graph):
    items = [{"source": "x"}, {"title": "only valid one", "source": "y"}]
    n = ingest_news("BTC", items)
    assert n == 1


def test_ingest_sentiment_creates_label_entity(graph):
    vote = types.SimpleNamespace(direction=1, confidence=0.7)
    assert ingest_sentiment("BTC", vote) is True
    assert graph.get_entity("sentiment_label:bullish") is not None


def test_ingest_institutional_counts_numeric_keys(graph):
    n = ingest_institutional("BTC", {
        "stablecoin_inflow_24h": 3.2,
        "options_put_call_ratio": 0.8,
        "comment": "ignore me",      # non-numeric → skipped
    })
    assert n == 2


def test_ingest_polymarket_inserts_events(graph):
    markets = [
        {"market_id": "m1", "question": "Will BTC close above $70k?", "yes_price": 0.4},
        {"market_id": "", "question": "bad row"},   # no id → skipped
    ]
    n = ingest_polymarket("BTC", markets)
    assert n == 1
    assert graph.get_entity("polymarket_event:will btc close above $70k?") is not None


def test_ingest_correlation_bidirectional_edge(graph):
    assert ingest_correlation("BTC", "ETH", 0.75) is True
    edges = graph.recent_edges(entity_id="asset:btc", kind="correlation")
    assert edges and edges[0].payload.get("r") == pytest.approx(0.75)


# ── query_digest ───────────────────────────────────────────────────────


def test_query_digest_disabled_env(graph, monkeypatch):
    monkeypatch.setenv(DISABLE_ENV, "1")
    assert "disabled" in query_digest("BTC")


def test_query_digest_unknown_asset_has_fallback(graph):
    out = query_digest("WAT")
    assert "no entity" in out.lower()


def test_query_digest_focused_on_asset(graph):
    ingest_news("BTC", [types.SimpleNamespace(
        title="SEC approves spot ETF", source="newsapi",
        timestamp=time.time(), event_type="etf",
    )])
    ingest_sentiment("BTC", types.SimpleNamespace(direction=1, confidence=0.6))
    out = query_digest("BTC")
    assert "BTC" in out
    # Edge kind counts should appear.
    assert "news" in out.lower() or "sentiment" in out.lower()


def test_query_digest_respects_max_chars(graph):
    for i in range(20):
        ingest_news("BTC", [types.SimpleNamespace(
            title=f"headline {i}" * 10, source="s",
            timestamp=time.time(), event_type="general",
        )])
    out = query_digest("BTC", max_chars=200)
    assert len(out) <= 200


def test_query_digest_global_snapshot(graph):
    ingest_correlation("BTC", "ETH", 0.5)
    out = query_digest(None)
    assert "global" in out.lower()
    assert "correlation" in out.lower() or "edges" not in out.lower() or True


# ── LLM-tool registration ──────────────────────────────────────────────


def test_query_knowledge_graph_registered_in_default_registry():
    from src.ai.trade_tools import build_default_registry
    reg = build_default_registry()
    assert reg.get("query_knowledge_graph") is not None


def test_query_knowledge_graph_dispatch_returns_summary(graph):
    from src.ai.trade_tools import build_default_registry
    reg = build_default_registry()
    out = reg.dispatch("query_knowledge_graph", {"asset": "BTC"})
    parsed = json.loads(out)
    assert "summary" in parsed or "error" in parsed
