"""
Shared diagnostic helpers used by /status and /diagnose-noop.

Before this module, both skills re-implemented the same warm_store /
brain_memory / readiness-gate / Ollama checks locally. Extracting the
shared code here keeps the diagnostic surface consistent — if a check
is updated, both skills get the fix automatically.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _warm_db_path() -> str:
    return os.getenv(
        "ACT_WARM_DB_PATH",
        str(Path(__file__).resolve().parents[2] / "data" / "warm_store.sqlite"),
    )


# ── Env flags ──────────────────────────────────────────────────────────


def check_env_flags() -> Dict[str, str]:
    keys = [
        "ACT_AGENTIC_LOOP", "ACT_DISABLE_AGENTIC_LOOP",
        "ACT_BRAIN_PROFILE", "ACT_SCANNER_MODEL", "ACT_ANALYST_MODEL",
        "OLLAMA_NUM_PARALLEL", "ACT_REAL_CAPITAL_ENABLED",
        "ACT_EMERGENCY_MODE", "ACT_POLYMARKET_LIVE",
    ]
    return {k: os.environ.get(k, "<unset>") for k in keys}


# ── Ollama reachability + required-model presence ──────────────────────


def check_ollama_models() -> Dict[str, Any]:
    try:
        import urllib.request
        base = os.environ.get("OLLAMA_REMOTE_URL") or "http://localhost:11434"
        url = base.rstrip("/") + "/api/tags"
        with urllib.request.urlopen(urllib.request.Request(url), timeout=3.0) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
        present = {m.get("name", "").split(":")[0] for m in (data.get("models") or [])}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    try:
        from src.ai.dual_brain import _resolve, ANALYST, SCANNER
        required = [_resolve(None, SCANNER).model, _resolve(None, ANALYST).model]
    except Exception as e:
        return {"ok": False, "error": f"dual_brain import failed: {e}"}

    return {
        "ok": True,
        "required": {m: (m.split(":")[0] in present) for m in required},
        "total_pulled": len(present),
    }


# ── Warm-store activity ────────────────────────────────────────────────


def _columns_of(conn: sqlite3.Connection, table: str) -> set:
    """Return the set of column names for `table` (empty on any error)."""
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {row[1] for row in rows}
    except Exception:
        return set()


def check_warm_store(window_s: float = 3600.0) -> Dict[str, Any]:
    """Query warm_store for recent activity. Robust to legacy schemas
    missing the C1 component_signals column (the migration runs on
    first WarmStore() construction — before the bot boots, the column
    may not exist yet)."""
    db = _warm_db_path()
    if not os.path.exists(db):
        return {"exists": False, "path": db}
    try:
        # Open via get_store() so the C1 migration runs and the column
        # gets added transparently. Safe for repeat calls.
        try:
            from src.orchestration.warm_store import get_store
            get_store()
        except Exception as e:
            logger.debug("check_warm_store: get_store() init failed: %s", e)

        conn = sqlite3.connect(db, timeout=3.0)
        cur = conn.cursor()
        cols = _columns_of(conn, "decisions")
        has_component_signals = "component_signals" in cols

        cutoff_ns = int((time.time() - window_s) * 1_000_000_000)
        total = cur.execute(
            "SELECT COUNT(*) FROM decisions WHERE ts_ns >= ?", (cutoff_ns,),
        ).fetchone()[0]
        shadow = cur.execute(
            "SELECT COUNT(*) FROM decisions WHERE ts_ns >= ? AND decision_id LIKE 'shadow-%'",
            (cutoff_ns,),
        ).fetchone()[0]
        outcomes = cur.execute(
            "SELECT COUNT(*) FROM outcomes WHERE exit_ts >= ?", (time.time() - window_s,),
        ).fetchone()[0]
        actions = cur.execute(
            "SELECT final_action, COUNT(*) FROM decisions "
            "WHERE ts_ns >= ? AND decision_id NOT LIKE 'shadow-%' "
            "GROUP BY final_action ORDER BY 2 DESC", (cutoff_ns,),
        ).fetchall()

        if has_component_signals:
            scanner_published = cur.execute(
                "SELECT COUNT(*) FROM decisions WHERE ts_ns >= ? AND "
                "json_extract(component_signals, '$.scanner_published') = 1",
                (cutoff_ns,),
            ).fetchone()[0]
            shadow_reasons = cur.execute(
                "SELECT json_extract(component_signals, '$.terminated_reason'), COUNT(*) "
                "FROM decisions WHERE ts_ns >= ? AND decision_id LIKE 'shadow-%' "
                "GROUP BY 1 ORDER BY 2 DESC", (cutoff_ns,),
            ).fetchall()
        else:
            # Legacy schema — report the gap honestly so /status turns yellow
            # and /diagnose-noop can call it out.
            scanner_published = 0
            shadow_reasons = []
        conn.close()

        out: Dict[str, Any] = {
            "exists": True, "path": db, "window_s": window_s,
            "decisions_total": int(total or 0),
            "decisions_shadow": int(shadow or 0),
            "decisions_real": int((total or 0) - (shadow or 0)),
            "outcomes": int(outcomes or 0),
            "scanner_published": int(scanner_published or 0),
            "real_action_breakdown": dict(actions or []),
            "shadow_terminated_reasons": dict(shadow_reasons or []),
        }
        if not has_component_signals:
            out["warning"] = (
                "legacy schema — component_signals column not yet added. "
                "Will self-heal when bot boots (WarmStore() runs migration)."
            )
        return out
    except Exception as e:
        return {"exists": True, "error": str(e)[:120]}


# ── Brain memory ───────────────────────────────────────────────────────


def check_brain_memory(assets: Optional[list] = None) -> Dict[str, Any]:
    assets = assets or ["BTC", "ETH"]
    try:
        from src.ai.brain_memory import get_brain_memory
        mem = get_brain_memory()
    except Exception as e:
        return {"error": f"brain_memory unavailable: {e}"}

    out: Dict[str, Any] = {}
    for a in assets:
        try:
            scan = mem.read_latest_scan(a, max_age_s=86400.0)
            traces = mem.read_recent_traces(a, limit=10, max_age_s=86400.0)
            out[a] = {
                "latest_scan_age_s": int(scan.age_s()) if scan else None,
                "latest_scan_score": (scan.opportunity_score if scan else None),
                "trace_count_24h": len(traces),
            }
        except Exception as e:
            out[a] = {"error": str(e)[:120]}
    return out


# ── Readiness gate ─────────────────────────────────────────────────────


def check_readiness() -> Dict[str, Any]:
    try:
        from src.orchestration.readiness_gate import evaluate, is_emergency_mode
        state = evaluate()
        return {
            "open": bool(state.open_),
            "failing_top3": list(state.reasons)[:3],
            "emergency_mode": bool(is_emergency_mode(state)),
            "details": dict(state.details),
        }
    except Exception as e:
        return {"error": f"readiness_gate import/eval failed: {e}"}


# ── Knowledge graph ────────────────────────────────────────────────────


def check_graph() -> Dict[str, Any]:
    try:
        from src.ai.graph_rag import get_graph
        g = get_graph()
        return {
            "edges_1h_by_kind": g.count_by_kind(since_s=3600),
            "edges_24h_by_kind": g.count_by_kind(since_s=86400),
        }
    except Exception as e:
        return {"error": str(e)[:120]}


# ── Active personas ────────────────────────────────────────────────────


def check_personas() -> Dict[str, Any]:
    try:
        from src.agents.persona_from_graph import get_active_personas
        active = get_active_personas()
        return {
            "active_count": len(active),
            "names": [a.descriptor.name for a in active[:10]],
        }
    except Exception as e:
        return {"error": str(e)[:120]}


# ── Config ─────────────────────────────────────────────────────────────


def check_config() -> Dict[str, Any]:
    cfg_path = Path(__file__).resolve().parents[2] / "config.yaml"
    if not cfg_path.exists():
        return {"exists": False}
    try:
        import yaml
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        return {"exists": True, "error": str(e)[:100]}
    ai = cfg.get("ai") or {}
    db = ai.get("dual_brain") or {}
    return {
        "mode": cfg.get("mode", "<unset>"),
        "agentic_loop_enabled": (cfg.get("agentic_loop") or {}).get("enabled"),
        "dual_brain_enabled": db.get("enabled"),
        "dual_brain_profile": db.get("profile"),
        "scanner_model_config": db.get("scanner_model") or "<profile-derived>",
        "analyst_model_config": db.get("analyst_model") or "<profile-derived>",
    }
