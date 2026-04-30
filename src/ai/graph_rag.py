"""
Real-time knowledge graph — C12 (operator-corrected paradigm).

A lightweight knowledge graph over ACT's LIVE data streams, NOT static
documents. The graph is continuously updated by background ingest ticks
that pull from the existing fetchers (news, sentiment, on-chain,
institutional, macro, polymarket, orderbook) and emit
entity + relationship records with time-decaying weights.

Design principles:
  * No new data sources — composes existing fetchers.
  * Time-aware edges: weight = w0 × exp(-dt / tau); old edges naturally
    fade, recent confirmations strengthen. Tau is per-edge-type.
  * SQLite schema (same pattern as warm_store / strategy_repository /
    brain_memory) — single file, WAL mode, no new deps.
  * LLM-callable as a tool: `query_knowledge_graph(question)` returns
    a compact digest (same ≤500-char discipline as web_context.py).
  * Web-fetch fallback for context the graph doesn't have: on-demand
    via the existing web_context tools (news search, web search) with
    short TTL. NOT primary; graph is primary.
  * Kill switch: `ACT_DISABLE_GRAPH_RAG=1` disables both ingest + query.

This module ships the storage + query layer. Ingest hooks live here as
thin wrappers that pull one fetcher and insert edges; a separate
scheduler job (wired in C12 part 2) calls them on cadence.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


DEFAULT_DB = os.getenv(
    "ACT_GRAPH_DB_PATH",
    str(Path(__file__).resolve().parents[2] / "data" / "knowledge_graph.sqlite"),
)
DISABLE_ENV = "ACT_DISABLE_GRAPH_RAG"

# Default half-life per edge kind (seconds). Fresh signals matter more.
DEFAULT_HALF_LIFE_S = {
    "news":          3 * 3600,         # news decays across 3 hours
    "sentiment":     1 * 3600,
    "funding":       30 * 60,          # funding rate window (CRYPTO)
    "on_chain":      2 * 3600,         # whale/exchange flows (CRYPTO)
    "polymarket":    1 * 3600,         # prediction markets (CRYPTO)
    "macro":         24 * 3600,        # macro persists longer
    "correlation":   6 * 3600,
    "orderbook":     5 * 60,           # order-book signal fast-decays
    "event":         4 * 3600,
    # Equity-specific edge kinds
    "earnings":      12 * 3600,        # earnings dates persist multi-day
    "options_skew":  1 * 3600,         # IV skew / PCR — equity flow signal
    "_default":      60 * 60,
}


_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS entities (
        entity_id    TEXT PRIMARY KEY,
        kind         TEXT NOT NULL,       -- asset | event | institution | macro | whale | market
        name         TEXT NOT NULL,
        attrs        TEXT DEFAULT '{}',
        first_seen   REAL NOT NULL,
        last_seen    REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_entities_kind_name ON entities(kind, name)",
    """
    CREATE TABLE IF NOT EXISTS edges (
        edge_id      TEXT PRIMARY KEY,
        src_id       TEXT NOT NULL,
        dst_id       TEXT NOT NULL,
        relation     TEXT NOT NULL,       -- mentions | correlates | precedes | flows_into | ...
        kind         TEXT NOT NULL,       -- news | sentiment | funding | on_chain | ...
        weight       REAL NOT NULL,       -- last-observed raw weight
        ts           REAL NOT NULL,       -- when this edge was last observed
        source       TEXT,
        payload      TEXT DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id, ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id, ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation, ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_edges_kind ON edges(kind, ts DESC)",
]


# ── Data types ──────────────────────────────────────────────────────────


@dataclass
class Entity:
    entity_id: str
    kind: str
    name: str
    attrs: Dict[str, Any] = field(default_factory=dict)
    first_seen: float = 0.0
    last_seen: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id, "kind": self.kind, "name": self.name,
            "attrs": dict(self.attrs),
            "first_seen": self.first_seen, "last_seen": self.last_seen,
        }


@dataclass
class Edge:
    edge_id: str
    src_id: str
    dst_id: str
    relation: str
    kind: str
    weight: float           # raw observed weight at time `ts`
    ts: float
    source: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    def decayed_weight(self, now: Optional[float] = None,
                       half_life_s: Optional[float] = None) -> float:
        """w × 2^(-dt/HL). HL = half-life in seconds."""
        now = now if now is not None else time.time()
        hl = (half_life_s if half_life_s is not None
              else DEFAULT_HALF_LIFE_S.get(self.kind, DEFAULT_HALF_LIFE_S["_default"]))
        if hl <= 0:
            return float(self.weight)
        dt = max(0.0, now - float(self.ts))
        return float(self.weight) * math.pow(2.0, -dt / hl)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id, "src": self.src_id, "dst": self.dst_id,
            "relation": self.relation, "kind": self.kind,
            "weight": round(self.weight, 4),
            "decayed_weight": round(self.decayed_weight(), 4),
            "ts": self.ts, "source": self.source,
            "payload": dict(self.payload),
        }


# ── Core store ──────────────────────────────────────────────────────────


class KnowledgeGraph:
    """SQLite-backed real-time graph. Thread-safe."""

    def __init__(self, db_path: str = DEFAULT_DB):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_schema()

    def _conn_get(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(self.db_path, timeout=5.0,
                                   isolation_level=None, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._conn = conn
        return self._conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._conn_get()
            for stmt in _SCHEMA:
                conn.execute(stmt)
            conn.commit()

    # ── Entity upsert ───────────────────────────────────────────────────

    def upsert_entity(
        self, *, kind: str, name: str, attrs: Optional[Dict[str, Any]] = None,
        entity_id: Optional[str] = None,
    ) -> str:
        """Create if missing, update last_seen/attrs if exists. Returns id.

        Entity id is deterministic per (kind, name) unless caller pins it,
        so repeated ingestion de-duplicates cleanly.
        """
        eid = entity_id or f"{kind}:{name.lower()}"
        now = time.time()
        attrs_json = json.dumps(attrs or {}, default=str)
        with self._lock:
            conn = self._conn_get()
            existing = conn.execute(
                "SELECT first_seen, attrs FROM entities WHERE entity_id=?", (eid,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO entities (entity_id, kind, name, attrs, "
                    "first_seen, last_seen) VALUES (?,?,?,?,?,?)",
                    (eid, kind, name, attrs_json, now, now),
                )
            else:
                # Merge attrs on update.
                try:
                    prev_attrs = json.loads(existing[1] or "{}")
                except Exception:
                    prev_attrs = {}
                prev_attrs.update(attrs or {})
                conn.execute(
                    "UPDATE entities SET attrs=?, last_seen=? WHERE entity_id=?",
                    (json.dumps(prev_attrs, default=str), now, eid),
                )
            conn.commit()
        return eid

    # ── Edge upsert ─────────────────────────────────────────────────────

    def add_edge(
        self, *, src_id: str, dst_id: str, relation: str, kind: str,
        weight: float = 1.0, source: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Append an edge. Edges are append-only so time decay is
        meaningful — we never overwrite history."""
        eid = uuid.uuid4().hex
        now = time.time()
        with self._lock:
            conn = self._conn_get()
            conn.execute(
                "INSERT INTO edges (edge_id, src_id, dst_id, relation, kind, "
                "weight, ts, source, payload) VALUES (?,?,?,?,?,?,?,?,?)",
                (eid, src_id, dst_id, relation, kind, float(weight), now,
                 source, json.dumps(payload or {}, default=str)),
            )
            conn.commit()
        return eid

    def add_edges_bulk(self, rows: List[Dict[str, Any]]) -> List[str]:
        """Batch-insert N edges in one transaction (N inserts, 1 commit).

        Each row is a dict with add_edge-shaped keys. Malformed rows are
        skipped with a debug log rather than aborting the batch.
        Used by ingest_news etc. so a 10-item news batch is 1 commit
        instead of 10.
        """
        if not rows:
            return []
        now = time.time()
        prepared: List[tuple] = []
        ids: List[str] = []
        for r in rows:
            try:
                eid = uuid.uuid4().hex
                prepared.append((
                    eid,
                    r["src_id"], r["dst_id"],
                    r["relation"], r["kind"],
                    float(r.get("weight", 1.0)),
                    now, r.get("source", "") or "",
                    json.dumps(r.get("payload") or {}, default=str),
                ))
                ids.append(eid)
            except Exception as e:
                logger.debug("add_edges_bulk: skipping malformed row: %s", e)
        if not prepared:
            return []
        with self._lock:
            conn = self._conn_get()
            conn.executemany(
                "INSERT INTO edges (edge_id, src_id, dst_id, relation, kind, "
                "weight, ts, source, payload) VALUES (?,?,?,?,?,?,?,?,?)",
                prepared,
            )
            conn.commit()
        return ids

    # ── Queries ─────────────────────────────────────────────────────────

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        row = self._conn_get().execute(
            "SELECT entity_id, kind, name, attrs, first_seen, last_seen "
            "FROM entities WHERE entity_id=?", (entity_id,),
        ).fetchone()
        if not row:
            return None
        try:
            attrs = json.loads(row[3] or "{}")
        except Exception:
            attrs = {}
        return Entity(entity_id=row[0], kind=row[1], name=row[2], attrs=attrs,
                      first_seen=float(row[4] or 0.0),
                      last_seen=float(row[5] or 0.0))

    def recent_edges(
        self, *, entity_id: Optional[str] = None, kind: Optional[str] = None,
        since_s: Optional[float] = None, limit: int = 100,
    ) -> List[Edge]:
        """Return edges matching filters, newest first."""
        cutoff = time.time() - (since_s if since_s is not None else 86400.0)
        q = ("SELECT edge_id, src_id, dst_id, relation, kind, weight, ts, "
             "source, payload FROM edges WHERE ts >= ?")
        params: List[Any] = [cutoff]
        if entity_id:
            q += " AND (src_id=? OR dst_id=?)"
            params.extend([entity_id, entity_id])
        if kind:
            q += " AND kind=?"
            params.append(kind)
        q += " ORDER BY ts DESC LIMIT ?"
        params.append(int(limit))
        rows = self._conn_get().execute(q, tuple(params)).fetchall()
        out: List[Edge] = []
        for r in rows:
            try:
                pl = json.loads(r[8] or "{}")
            except Exception:
                pl = {}
            out.append(Edge(
                edge_id=r[0], src_id=r[1], dst_id=r[2],
                relation=r[3], kind=r[4], weight=float(r[5] or 0.0),
                ts=float(r[6] or 0.0), source=r[7] or "", payload=pl,
            ))
        return out

    def top_connected(
        self, entity_id: str, *, limit: int = 5, since_s: float = 86400.0,
    ) -> List[Dict[str, Any]]:
        """Neighbors of `entity_id` ranked by total decayed weight."""
        edges = self.recent_edges(entity_id=entity_id, since_s=since_s,
                                  limit=1000)
        if not edges:
            return []
        now = time.time()
        agg: Dict[str, float] = {}
        rels: Dict[str, List[str]] = {}
        for e in edges:
            other = e.dst_id if e.src_id == entity_id else e.src_id
            w = e.decayed_weight(now)
            agg[other] = agg.get(other, 0.0) + w
            rels.setdefault(other, []).append(e.relation)
        ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(limit))]
        out: List[Dict[str, Any]] = []
        for other, total in ranked:
            ent = self.get_entity(other)
            out.append({
                "neighbor_id": other,
                "neighbor_name": ent.name if ent else other,
                "neighbor_kind": ent.kind if ent else "?",
                "total_decayed_weight": round(total, 3),
                "relations": list(dict.fromkeys(rels.get(other, [])))[:5],
            })
        return out

    def count_by_kind(self, since_s: float = 86400.0) -> Dict[str, int]:
        cutoff = time.time() - since_s
        rows = self._conn_get().execute(
            "SELECT kind, COUNT(*) FROM edges WHERE ts >= ? GROUP BY kind",
            (cutoff,),
        ).fetchall()
        return {k: int(n) for k, n in rows}

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None


# ── Singleton ──────────────────────────────────────────────────────────

_graph_singleton: Optional[KnowledgeGraph] = None
_graph_lock = threading.Lock()


def get_graph() -> KnowledgeGraph:
    global _graph_singleton
    with _graph_lock:
        if _graph_singleton is None:
            _graph_singleton = KnowledgeGraph()
        return _graph_singleton


# ── Query layer — LLM-facing compact digest ────────────────────────────


def query_digest(
    asset: Optional[str] = None,
    *,
    since_s: float = 3600.0,
    max_chars: int = 500,
) -> str:
    """Format the graph's current state for the LLM as a ≤ max_chars digest.

    If `asset` is given, focuses on that entity's neighborhood. Otherwise
    returns a global kind-count snapshot. Always returns SOMETHING — on
    any error falls back to a one-line "unavailable" string.
    """
    if os.environ.get(DISABLE_ENV, "0") == "1":
        return "[graph disabled by ACT_DISABLE_GRAPH_RAG]"
    try:
        g = get_graph()
    except Exception as e:
        return f"[graph unavailable: {type(e).__name__}]"

    lines: List[str] = []
    if asset:
        eid = f"asset:{asset.lower()}"
        ent = g.get_entity(eid)
        if ent is None:
            return f"[graph] no entity for {asset}"
        top = g.top_connected(eid, limit=5, since_s=since_s)
        counts = g.count_by_kind(since_s=since_s)
        lines.append(f"[graph/{asset.upper()} last {int(since_s/60)}m]")
        if counts:
            lines.append(
                "edges: " + ", ".join(f"{k}={v}" for k, v in
                                      sorted(counts.items(), key=lambda kv: -kv[1])[:5])
            )
        for n in top:
            lines.append(
                f"• {n['neighbor_name']} ({n['neighbor_kind']}) w={n['total_decayed_weight']} "
                f"[{','.join(n['relations'][:3])}]"
            )
    else:
        counts = g.count_by_kind(since_s=since_s)
        lines.append(f"[graph global last {int(since_s/60)}m]")
        if counts:
            lines.append(
                ", ".join(f"{k}={v}" for k, v in
                          sorted(counts.items(), key=lambda kv: -kv[1])[:8])
            )
        else:
            lines.append("no edges")
    out = "\n".join(lines)
    return out if len(out) <= max_chars else out[: max_chars - 3] + "..."


# ── Ingest helpers — thin wrappers over existing fetchers ──────────────


def ingest_news(asset: str, news_items: List[Dict[str, Any]]) -> int:
    """Convert a batch of news items into entity+edge inserts.

    `news_items` is whatever news_fetcher.fetch_all() returns (each item
    has `title`, `source`, `timestamp`, `event_type`, optional `tickers`).
    Returns number of edges added.
    """
    if not news_items:
        return 0
    try:
        g = get_graph()
    except Exception:
        return 0
    asset_eid = g.upsert_entity(kind="asset", name=asset.upper())
    # Build all edges first, then insert in one commit via add_edges_bulk.
    rows: List[Dict[str, Any]] = []
    for item in news_items:
        title = getattr(item, "title", None) or (item.get("title") if isinstance(item, dict) else None)
        if not title:
            continue
        src = getattr(item, "source", None) or (item.get("source") if isinstance(item, dict) else "") or "news"
        event_type = (getattr(item, "event_type", None)
                      or (item.get("event_type") if isinstance(item, dict) else "general"))
        news_eid = g.upsert_entity(
            kind="news", name=title[:200],
            attrs={"source": src, "event_type": event_type},
        )
        impact_weight = {
            "regulatory": 1.5, "etf": 1.3, "hack": 1.4, "macro": 1.0,
            "exchange": 0.9, "adoption": 0.8, "general": 0.6,
        }.get(event_type, 0.6)
        rows.append({
            "src_id": asset_eid, "dst_id": news_eid,
            "relation": "mentions", "kind": "news",
            "weight": impact_weight, "source": src,
            "payload": {"event_type": event_type},
        })
    inserted = g.add_edges_bulk(rows)
    return len(inserted)


def ingest_sentiment(asset: str, sentiment_vote) -> bool:
    """One sentiment observation → one edge. `sentiment_vote` is an
    AgentVote or dict with direction/confidence fields."""
    try:
        g = get_graph()
        asset_eid = g.upsert_entity(kind="asset", name=asset.upper())
        direction = int(getattr(sentiment_vote, "direction",
                                (sentiment_vote.get("direction", 0)
                                 if isinstance(sentiment_vote, dict) else 0)))
        conf = float(getattr(sentiment_vote, "confidence",
                             (sentiment_vote.get("confidence", 0.5)
                              if isinstance(sentiment_vote, dict) else 0.5)))
        label = "bullish" if direction > 0 else ("bearish" if direction < 0 else "neutral")
        sent_eid = g.upsert_entity(kind="sentiment_label", name=label)
        g.add_edge(
            src_id=asset_eid, dst_id=sent_eid,
            relation="tilted_to", kind="sentiment",
            weight=conf, source="sentiment_decoder_agent",
            payload={"direction": direction},
        )
        return True
    except Exception as e:
        logger.debug("ingest_sentiment failed: %s", e)
        return False


def ingest_institutional(asset: str, institutional_data: Dict[str, float]) -> int:
    """Institutional signals (L/S ratio, options, stablecoin flows)
    → edges. Each numeric key becomes a flow-style edge with magnitude."""
    if not isinstance(institutional_data, dict):
        return 0
    try:
        g = get_graph()
        asset_eid = g.upsert_entity(kind="asset", name=asset.upper())
    except Exception:
        return 0
    rows: List[Dict[str, Any]] = []
    for key, val in institutional_data.items():
        if not isinstance(val, (int, float)):
            continue
        signal_eid = g.upsert_entity(kind="institutional_signal", name=str(key))
        weight = min(5.0, abs(float(val)))
        rel = "positive_flow" if val >= 0 else "negative_flow"
        rows.append({
            "src_id": asset_eid, "dst_id": signal_eid,
            "relation": rel,
            "kind": "on_chain" if "flow" in key.lower() else "institutional",
            "weight": weight, "source": "institutional_fetcher",
            "payload": {"raw_value": val},
        })
    return len(g.add_edges_bulk(rows))


def ingest_polymarket(asset: str, markets: List[Dict[str, Any]]) -> int:
    """Polymarket markets → event entities + asset→event edges."""
    if not markets:
        return 0
    try:
        g = get_graph()
        asset_eid = g.upsert_entity(kind="asset", name=asset.upper())
    except Exception:
        return 0
    rows: List[Dict[str, Any]] = []
    for m in markets:
        mid = str(m.get("market_id") or m.get("id") or "")
        question = str(m.get("question") or "")
        if not mid or not question:
            continue
        yes_price = float(m.get("yes_price") or 0.5)
        event_eid = g.upsert_entity(
            kind="polymarket_event", name=question[:200],
            attrs={"market_id": mid, "yes_price": yes_price},
        )
        rows.append({
            "src_id": asset_eid, "dst_id": event_eid,
            "relation": "referenced_by", "kind": "polymarket",
            "weight": yes_price, "source": "polymarket",
            "payload": {"market_id": mid, "yes_price": yes_price},
        })
    return len(g.add_edges_bulk(rows))


def ingest_earnings(asset: str, days_to_earnings: float,
                    confirmed: bool = True) -> bool:
    """Equity-specific: an upcoming earnings print → asset→earnings_event
    edge. Weight is inversely proportional to days-to-earnings so the
    closer the print, the heavier the signal. ETFs and tickers without
    earnings are no-ops.

    Edge kind: 'earnings' (12h half-life — earnings is a multi-day
    proximity signal, not a fast-fade fact).
    """
    try:
        days = float(days_to_earnings)
    except Exception:
        return False
    if not (days >= 0 and days < 365):     # +inf/None/garbage → skip
        return False
    try:
        g = get_graph()
        asset_eid = g.upsert_entity(kind="asset", name=asset.upper())
        # Bucket the proximity so the entity name is stable across ticks
        # (entities are deduped on (kind, name); we don't want a fresh
        # entity every minute as days_to_earnings drifts).
        if days <= 1.0:
            bucket = "0-1d"
        elif days <= 3.0:
            bucket = "1-3d"
        elif days <= 7.0:
            bucket = "3-7d"
        elif days <= 14.0:
            bucket = "7-14d"
        else:
            bucket = ">14d"
        ev_eid = g.upsert_entity(
            kind="earnings_event",
            name=f"{asset.upper()}_earnings_{bucket}",
            attrs={"days_to_earnings": days, "confirmed": confirmed},
        )
        # Weight: 0-1d→1.0, 1-3d→0.8, 3-7d→0.5, 7-14d→0.3, >14d→0.1
        if days <= 1.0:
            w = 1.0
        elif days <= 3.0:
            w = 0.8
        elif days <= 7.0:
            w = 0.5
        elif days <= 14.0:
            w = 0.3
        else:
            w = 0.1
        g.add_edge(
            src_id=asset_eid, dst_id=ev_eid,
            relation="prints_in", kind="earnings",
            weight=w, source="earnings_calendar",
            payload={"days_to_earnings": days, "confirmed": confirmed},
        )
        return True
    except Exception as e:
        logger.debug("ingest_earnings failed: %s", e)
        return False


def ingest_options_skew(asset: str, *,
                         put_call_ratio: Optional[float] = None,
                         iv_skew: Optional[float] = None,
                         vix: Optional[float] = None) -> int:
    """Equity-specific: options-market structure → edges to a regime-
    indicator entity. PCR > 1.0 = bearish positioning; iv_skew < 0
    = put-side stress; high VIX = systemic fear.

    Returns number of edges written (0 if no fields supplied).
    """
    g = get_graph()
    try:
        asset_eid = g.upsert_entity(kind="asset", name=asset.upper())
    except Exception:
        return 0
    rows: List[Dict[str, Any]] = []
    if put_call_ratio is not None:
        try:
            pcr = float(put_call_ratio)
            if pcr >= 0:
                indicator_eid = g.upsert_entity(
                    kind="options_indicator", name=f"{asset.upper()}_pcr",
                )
                rel = "elevated_puts" if pcr > 1.0 else "elevated_calls"
                rows.append({
                    "src_id": asset_eid, "dst_id": indicator_eid,
                    "relation": rel, "kind": "options_skew",
                    "weight": min(2.0, abs(pcr - 1.0) * 2.0),
                    "source": "equity_risk_pulse",
                    "payload": {"put_call_ratio": pcr},
                })
        except Exception:
            pass
    if iv_skew is not None:
        try:
            sk = float(iv_skew)
            indicator_eid = g.upsert_entity(
                kind="options_indicator", name=f"{asset.upper()}_iv_skew",
            )
            rel = "put_skew_high" if sk < 0 else "call_skew_high"
            rows.append({
                "src_id": asset_eid, "dst_id": indicator_eid,
                "relation": rel, "kind": "options_skew",
                "weight": min(2.0, abs(sk)),
                "source": "equity_risk_pulse",
                "payload": {"iv_skew": sk},
            })
        except Exception:
            pass
    if vix is not None:
        try:
            v = float(vix)
            if v > 0:
                indicator_eid = g.upsert_entity(
                    kind="macro_indicator", name="VIX",
                )
                # VIX > 25 is the "elevated" line; > 40 is "panic".
                rel = "vix_panic" if v > 40 else ("vix_elevated" if v > 25 else "vix_calm")
                rows.append({
                    "src_id": asset_eid, "dst_id": indicator_eid,
                    "relation": rel, "kind": "macro",
                    "weight": min(2.0, v / 25.0),
                    "source": "equity_risk_pulse",
                    "payload": {"vix": v},
                })
        except Exception:
            pass
    if not rows:
        return 0
    return len(g.add_edges_bulk(rows))


def ingest_correlation(asset_a: str, asset_b: str, correlation: float,
                       window_s: float = 86400.0) -> bool:
    """Bidirectional correlation edge between two assets."""
    try:
        g = get_graph()
        a_eid = g.upsert_entity(kind="asset", name=asset_a.upper())
        b_eid = g.upsert_entity(kind="asset", name=asset_b.upper())
        w = min(2.0, abs(float(correlation)) * 2.0)
        rel = "correlates_with" if correlation >= 0 else "inversely_correlates_with"
        g.add_edge(
            src_id=a_eid, dst_id=b_eid,
            relation=rel, kind="correlation",
            weight=w, source=f"computed_{int(window_s/3600)}h",
            payload={"r": correlation, "window_s": window_s},
        )
        return True
    except Exception as e:
        logger.debug("ingest_correlation failed: %s", e)
        return False
