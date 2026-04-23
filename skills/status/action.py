"""Skill: /status — one-shot system verification after pull + START_ALL."""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List

from src.skills.registry import SkillResult


def _env() -> Dict[str, str]:
    keys = [
        "ACT_AGENTIC_LOOP", "ACT_DISABLE_AGENTIC_LOOP",
        "ACT_BRAIN_PROFILE", "ACT_SCANNER_MODEL", "ACT_ANALYST_MODEL",
        "OLLAMA_NUM_PARALLEL", "ACT_REAL_CAPITAL_ENABLED",
        "ACT_EMERGENCY_MODE", "ACT_POLYMARKET_LIVE",
    ]
    return {k: os.environ.get(k, "<unset>") for k in keys}


def _ollama() -> Dict[str, Any]:
    try:
        import urllib.request
        import json
        base = os.environ.get("OLLAMA_REMOTE_URL") or "http://localhost:11434"
        url = base.rstrip("/") + "/api/tags"
        with urllib.request.urlopen(urllib.request.Request(url), timeout=3.0) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
        names = {m.get("name", "").split(":")[0] for m in (data.get("models") or [])}
        from src.ai.dual_brain import _resolve, ANALYST, SCANNER
        req = {
            _resolve(None, SCANNER).model: _resolve(None, SCANNER).model.split(":")[0] in names,
            _resolve(None, ANALYST).model: _resolve(None, ANALYST).model.split(":")[0] in names,
        }
        return {"ok": True, "required": req, "total_pulled": len(names)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _warm_store() -> Dict[str, Any]:
    db = os.getenv(
        "ACT_WARM_DB_PATH",
        str(Path(__file__).resolve().parents[3] / "data" / "warm_store.sqlite"),
    )
    if not os.path.exists(db):
        return {"exists": False}
    try:
        conn = sqlite3.connect(db, timeout=3.0)
        cur = conn.cursor()
        cutoff_ns = int((time.time() - 3600) * 1_000_000_000)
        total_1h = cur.execute(
            "SELECT COUNT(*) FROM decisions WHERE ts_ns >= ?", (cutoff_ns,),
        ).fetchone()[0]
        shadow_1h = cur.execute(
            "SELECT COUNT(*) FROM decisions WHERE ts_ns >= ? AND decision_id LIKE 'shadow-%'",
            (cutoff_ns,),
        ).fetchone()[0]
        # C17 tick telemetry — are scanner publishes + graph ingests happening?
        scanner_published_count = cur.execute(
            "SELECT COUNT(*) FROM decisions WHERE ts_ns >= ? AND "
            "json_extract(component_signals, '$.scanner_published') = 1",
            (cutoff_ns,),
        ).fetchone()[0]
        conn.close()
        return {
            "exists": True, "path": db,
            "decisions_1h_total": int(total_1h or 0),
            "decisions_1h_shadow": int(shadow_1h or 0),
            "scanner_published_1h": int(scanner_published_count or 0),
        }
    except Exception as e:
        return {"exists": True, "error": str(e)[:120]}


def _brain_memory() -> Dict[str, Any]:
    try:
        from src.ai.brain_memory import get_brain_memory
        mem = get_brain_memory()
        out: Dict[str, Any] = {}
        for asset in ("BTC", "ETH"):
            scan = mem.read_latest_scan(asset, max_age_s=86400.0)
            traces = mem.read_recent_traces(asset, limit=10, max_age_s=86400.0)
            out[asset] = {
                "latest_scan_age_s": int(scan.age_s()) if scan else None,
                "latest_scan_score": (scan.opportunity_score if scan else None),
                "trace_count_24h": len(traces),
            }
        return out
    except Exception as e:
        return {"error": str(e)[:120]}


def _graph() -> Dict[str, Any]:
    try:
        from src.ai.graph_rag import get_graph
        g = get_graph()
        return {
            "edges_1h_by_kind": g.count_by_kind(since_s=3600),
            "edges_24h_by_kind": g.count_by_kind(since_s=86400),
        }
    except Exception as e:
        return {"error": str(e)[:120]}


def _personas() -> Dict[str, Any]:
    try:
        from src.agents.persona_from_graph import get_active_personas
        active = get_active_personas()
        return {
            "active_count": len(active),
            "names": [a.descriptor.name for a in active[:10]],
        }
    except Exception as e:
        return {"error": str(e)[:120]}


def _readiness() -> Dict[str, Any]:
    try:
        from src.orchestration.readiness_gate import evaluate, is_emergency_mode
        state = evaluate()
        return {
            "open": bool(state.open_),
            "failing_top3": list(state.reasons)[:3],
            "emergency_mode": bool(is_emergency_mode(state)),
            "trades": state.details.get("trades"),
            "soak_days": state.details.get("soak_days"),
            "rolling_sharpe": state.details.get("rolling_sharpe"),
        }
    except Exception as e:
        return {"error": str(e)[:120]}


def run(args: Dict[str, Any]) -> SkillResult:
    data: Dict[str, Any] = {
        "env": _env(),
        "ollama": _ollama(),
        "warm_store": _warm_store(),
        "brain_memory": _brain_memory(),
        "graph": _graph(),
        "personas": _personas(),
        "readiness": _readiness(),
    }

    # Traffic-light summary — green / yellow / red per subsystem.
    def _light(sub: Dict[str, Any], checks: List) -> str:
        for check_fn, status in checks:
            if check_fn(sub):
                return status
        return "yellow"

    lights: Dict[str, str] = {}
    lights["env"] = _light(data["env"], [
        (lambda e: e.get("ACT_DISABLE_AGENTIC_LOOP") == "1", "red"),
        (lambda e: e.get("ACT_AGENTIC_LOOP") in ("1", "true"), "green"),
    ])
    lights["ollama"] = "green" if data["ollama"].get("ok") and all(
        data["ollama"].get("required", {}).values()
    ) else ("yellow" if data["ollama"].get("ok") else "red")
    ws = data["warm_store"]
    lights["warm_store"] = "green" if ws.get("exists") and ws.get("decisions_1h_shadow", 0) > 0 \
        else ("yellow" if ws.get("exists") else "red")
    bm = data["brain_memory"]
    btc = (bm or {}).get("BTC") or {}
    lights["brain_memory"] = "green" if btc.get("latest_scan_age_s") is not None \
        and btc["latest_scan_age_s"] < 600 else "yellow"
    graph_counts = (data.get("graph") or {}).get("edges_1h_by_kind") or {}
    lights["graph"] = "green" if sum(graph_counts.values()) > 0 else "yellow"
    lights["personas"] = "green" if (data.get("personas") or {}).get("active_count", 0) > 0 else "yellow"
    lights["readiness"] = "green" if (data.get("readiness") or {}).get("open") else "yellow"

    lines = ["ACT /status — subsystem traffic lights"]
    for k, v in lights.items():
        icon = {"green": "✓", "yellow": "…", "red": "✗"}.get(v, "?")
        lines.append(f"  [{icon}] {k:<12} = {v}")
    lines.append("")
    lines.append(f"  ACT_AGENTIC_LOOP     = {data['env'].get('ACT_AGENTIC_LOOP')}")
    lines.append(f"  Brain profile        = {data['env'].get('ACT_BRAIN_PROFILE')}")
    lines.append(f"  Scanner / Analyst    = {data['env'].get('ACT_SCANNER_MODEL')} / {data['env'].get('ACT_ANALYST_MODEL')}")
    if ws.get("exists"):
        lines.append(
            f"  warm_store 1h        = total={ws.get('decisions_1h_total', 0)}, "
            f"shadow={ws.get('decisions_1h_shadow', 0)}, "
            f"scanner_pub={ws.get('scanner_published_1h', 0)}"
        )
    if data.get("readiness", {}).get("trades") is not None:
        r = data["readiness"]
        lines.append(
            f"  Readiness            = open={r.get('open')} "
            f"trades={r.get('trades')} soak_days={r.get('soak_days')} "
            f"sharpe={r.get('rolling_sharpe')}"
        )
    if graph_counts:
        edge_summary = ", ".join(f"{k}={v}" for k, v in
                                 sorted(graph_counts.items(), key=lambda kv: -kv[1])[:5])
        lines.append(f"  Graph 1h             = {edge_summary}")
    p = data.get("personas") or {}
    if "active_count" in p:
        lines.append(f"  Active personas      = {p.get('active_count', 0)}")

    data["lights"] = lights
    overall_ok = lights["warm_store"] != "red" and lights["ollama"] != "red"
    return SkillResult(ok=overall_ok, message="\n".join(lines), data=data)
