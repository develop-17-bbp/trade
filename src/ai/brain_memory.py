"""
Brain memory — the corpus callosum between scanner and analyst.

Two-way context store so the right hemisphere (Qwen 32B scanner) and
left hemisphere (Devstral 24B analyst) can see each other's output:

  Scanner → Analyst
    Every tick, the scanner writes its assessment here:
      {opportunity_score, top_signals, proposed_direction, rationale}
    When the agentic trade loop fires the analyst, the loop reads
    the latest scan for that asset and injects it into the analyst's
    seed context — the analyst sees what the scanner flagged instead
    of starting blind.

  Analyst → Scanner
    After the analyst compiles a TradePlan (or skips), it writes a
    compact trace here:
      {plan_id, direction, tier, size_pct, thesis, verdict}
    The scanner's NEXT run sees what the analyst actually did, which
    lets it learn patterns like "when my opportunity_score was X, the
    analyst skipped — adjust future scoring downward for that setup."

Design constraints:
  * Same SQLite-WAL warm_store pattern used elsewhere (uses the existing
    warm_store for durability so state survives restarts).
  * In-memory LRU on top of the DB for hot-path reads (~microsecond
    latency, unlike the SQLite round-trip).
  * Staleness — readers specify max_age_s so they don't act on stale
    data. Default: 180s (one swing-profile tick). Scanner reports older
    than that are ignored by the analyst.
  * Zero new deps; never raises (errors logged at DEBUG + swallowed).
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


DEFAULT_DB = os.getenv(
    "ACT_BRAIN_MEMORY_PATH",
    str(Path(__file__).resolve().parents[2] / "data" / "brain_memory.sqlite"),
)
DEFAULT_MAX_AGE_S = float(os.getenv("ACT_BRAIN_MEM_MAX_AGE_S", "180"))
DEFAULT_LRU_SIZE = int(os.getenv("ACT_BRAIN_MEM_LRU_SIZE", "64"))


_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS scan_reports (
        scan_id    TEXT PRIMARY KEY,
        asset      TEXT NOT NULL,
        ts         REAL NOT NULL,
        payload    TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_scan_asset_ts ON scan_reports(asset, ts DESC)",
    """
    CREATE TABLE IF NOT EXISTS analyst_traces (
        trace_id   TEXT PRIMARY KEY,
        asset      TEXT NOT NULL,
        ts         REAL NOT NULL,
        plan_id    TEXT,
        direction  TEXT,
        payload    TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_trace_asset_ts ON analyst_traces(asset, ts DESC)",
]


@dataclass
class ScanReport:
    """One scanner-brain assessment, per asset per tick."""
    asset: str
    ts: float
    opportunity_score: float              # 0-100
    proposed_direction: str               # 'LONG' | 'SHORT' | 'FLAT'
    top_signals: List[str] = field(default_factory=list)
    rationale: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def age_s(self, now: Optional[float] = None) -> float:
        return (now or time.time()) - self.ts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset, "ts": self.ts,
            "opportunity_score": round(float(self.opportunity_score), 1),
            "proposed_direction": self.proposed_direction,
            "top_signals": list(self.top_signals)[:10],
            "rationale": self.rationale[:400],
        }


@dataclass
class AnalystTrace:
    """One analyst-brain decision trace per compiled plan."""
    asset: str
    ts: float
    plan_id: str
    direction: str               # 'LONG' | 'SHORT' | 'FLAT' | 'SKIP'
    tier: str = ""
    size_pct: float = 0.0
    thesis: str = ""
    verdict: str = ""            # short verdict label for future scanner tuning
    raw: Dict[str, Any] = field(default_factory=dict)

    def age_s(self, now: Optional[float] = None) -> float:
        return (now or time.time()) - self.ts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset, "ts": self.ts, "plan_id": self.plan_id,
            "direction": self.direction, "tier": self.tier,
            "size_pct": round(self.size_pct, 2),
            "thesis": self.thesis[:300],
            "verdict": self.verdict[:120],
        }


# ── Core ────────────────────────────────────────────────────────────────


class BrainMemory:
    """SQLite-backed shared store + in-process LRU for hot reads."""

    def __init__(self, db_path: str = DEFAULT_DB, lru_size: int = DEFAULT_LRU_SIZE):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._scan_lru: "OrderedDict[str, ScanReport]" = OrderedDict()
        self._trace_lru: "OrderedDict[str, AnalystTrace]" = OrderedDict()
        self._lru_size = int(max(4, lru_size))
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._conn = conn
        return self._conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._get_conn()
            for stmt in _SCHEMA:
                conn.execute(stmt)
            conn.commit()

    def _lru_put(self, lru: OrderedDict, key: str, value: Any) -> None:
        lru[key] = value
        lru.move_to_end(key)
        while len(lru) > self._lru_size:
            lru.popitem(last=False)

    # ── Scanner side ────────────────────────────────────────────────────

    def write_scan_report(self, report: ScanReport) -> None:
        payload = json.dumps({
            "top_signals": report.top_signals,
            "rationale": report.rationale,
            "raw": report.raw,
        }, default=str)
        scan_id = f"{report.asset}-{int(report.ts * 1000)}"
        try:
            with self._lock:
                self._get_conn().execute(
                    "INSERT OR REPLACE INTO scan_reports "
                    "(scan_id, asset, ts, payload) VALUES (?, ?, ?, ?)",
                    (scan_id, report.asset.upper(), float(report.ts), payload),
                )
                self._get_conn().commit()
                self._lru_put(self._scan_lru, report.asset.upper(), report)
        except Exception as e:
            logger.debug("write_scan_report failed: %s", e)

    def read_latest_scan(
        self, asset: str, max_age_s: float = DEFAULT_MAX_AGE_S,
    ) -> Optional[ScanReport]:
        asset = asset.upper()
        now = time.time()

        # LRU first.
        hit = self._scan_lru.get(asset)
        if hit and hit.age_s(now) <= max_age_s:
            return hit

        try:
            with self._lock:
                row = self._get_conn().execute(
                    "SELECT ts, payload FROM scan_reports "
                    "WHERE asset=? ORDER BY ts DESC LIMIT 1",
                    (asset,),
                ).fetchone()
        except Exception as e:
            logger.debug("read_latest_scan failed: %s", e)
            return None
        if not row:
            return None
        ts, payload_raw = row
        if now - float(ts) > max_age_s:
            return None

        try:
            payload = json.loads(payload_raw or "{}")
        except Exception:
            payload = {}
        # Rehydrate the main fields from the raw dict; the core
        # opportunity_score / proposed_direction come from raw so we can
        # return a complete ScanReport even after restart.
        raw = payload.get("raw") or {}
        report = ScanReport(
            asset=asset, ts=float(ts),
            opportunity_score=float(raw.get("opportunity_score") or 0.0),
            proposed_direction=str(raw.get("proposed_direction") or "FLAT"),
            top_signals=list(payload.get("top_signals") or []),
            rationale=str(payload.get("rationale") or ""),
            raw=raw,
        )
        self._lru_put(self._scan_lru, asset, report)
        return report

    # ── Analyst side ────────────────────────────────────────────────────

    def write_analyst_trace(self, trace: AnalystTrace) -> None:
        payload = json.dumps({
            "tier": trace.tier, "size_pct": trace.size_pct,
            "thesis": trace.thesis, "verdict": trace.verdict,
            "raw": trace.raw,
        }, default=str)
        trace_id = trace.plan_id or f"{trace.asset}-{int(trace.ts * 1000)}"
        try:
            with self._lock:
                self._get_conn().execute(
                    "INSERT OR REPLACE INTO analyst_traces "
                    "(trace_id, asset, ts, plan_id, direction, payload) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (trace_id, trace.asset.upper(), float(trace.ts),
                     trace.plan_id, trace.direction, payload),
                )
                self._get_conn().commit()
                self._lru_put(self._trace_lru, trace.asset.upper(), trace)
        except Exception as e:
            logger.debug("write_analyst_trace failed: %s", e)

    def read_recent_traces(
        self, asset: str, limit: int = 3, max_age_s: float = 3600.0,
    ) -> List[AnalystTrace]:
        asset = asset.upper()
        now = time.time()
        try:
            with self._lock:
                rows = self._get_conn().execute(
                    "SELECT ts, plan_id, direction, payload FROM analyst_traces "
                    "WHERE asset=? AND ts >= ? ORDER BY ts DESC LIMIT ?",
                    (asset, now - max_age_s, int(limit)),
                ).fetchall()
        except Exception as e:
            logger.debug("read_recent_traces failed: %s", e)
            return []
        out: List[AnalystTrace] = []
        for ts, plan_id, direction, payload_raw in rows:
            try:
                payload = json.loads(payload_raw or "{}")
            except Exception:
                payload = {}
            out.append(AnalystTrace(
                asset=asset, ts=float(ts),
                plan_id=str(plan_id or ""), direction=str(direction or ""),
                tier=str(payload.get("tier") or ""),
                size_pct=float(payload.get("size_pct") or 0.0),
                thesis=str(payload.get("thesis") or ""),
                verdict=str(payload.get("verdict") or ""),
                raw=payload.get("raw") or {},
            ))
        return out

    # ── Housekeeping ────────────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None


# ── Singleton ───────────────────────────────────────────────────────────

_brain_singleton: Optional[BrainMemory] = None
_brain_lock = threading.Lock()


def get_brain_memory() -> BrainMemory:
    global _brain_singleton
    with _brain_lock:
        if _brain_singleton is None:
            _brain_singleton = BrainMemory()
        return _brain_singleton


# ── Convenience helpers used by the agentic loop ────────────────────────


def publish_scan(report: ScanReport) -> None:
    """Called by the scanner brain after each tick. Safe / no-op on error."""
    try:
        get_brain_memory().write_scan_report(report)
    except Exception as e:
        logger.debug("publish_scan failed: %s", e)


def get_scan_for_analyst(asset: str, max_age_s: float = DEFAULT_MAX_AGE_S) -> Optional[ScanReport]:
    """Called by the analyst (or the agentic bridge) before compiling."""
    try:
        return get_brain_memory().read_latest_scan(asset, max_age_s=max_age_s)
    except Exception as e:
        logger.debug("get_scan_for_analyst failed: %s", e)
        return None


def publish_analyst_trace(trace: AnalystTrace) -> None:
    try:
        get_brain_memory().write_analyst_trace(trace)
    except Exception as e:
        logger.debug("publish_analyst_trace failed: %s", e)


def get_recent_analyst_traces(asset: str, limit: int = 3) -> List[AnalystTrace]:
    try:
        return get_brain_memory().read_recent_traces(asset, limit=limit)
    except Exception as e:
        logger.debug("get_recent_analyst_traces failed: %s", e)
        return []
