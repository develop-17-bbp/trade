"""
Skill: /diagnose-noop — why hasn't the bot placed any profitable trades?

Audits the full stack in priority order — each check returns a short
status string. The most-likely-cause line at the end is a rules-based
summary of the findings.

No network calls besides Ollama /api/tags (local). Safe, read-only.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.skills.registry import SkillResult


# ── Individual checks ──────────────────────────────────────────────────


def _check_env_flags() -> Dict[str, Any]:
    """Inspect the env flags that gate whether the agentic loop runs."""
    return {
        "ACT_AGENTIC_LOOP": os.environ.get("ACT_AGENTIC_LOOP", "<unset>"),
        "ACT_DISABLE_AGENTIC_LOOP": os.environ.get("ACT_DISABLE_AGENTIC_LOOP", "<unset>"),
        "ACT_BRAIN_PROFILE": os.environ.get("ACT_BRAIN_PROFILE", "<unset>"),
        "ACT_SCANNER_MODEL": os.environ.get("ACT_SCANNER_MODEL", "<unset>"),
        "ACT_ANALYST_MODEL": os.environ.get("ACT_ANALYST_MODEL", "<unset>"),
        "ACT_REAL_CAPITAL_ENABLED": os.environ.get("ACT_REAL_CAPITAL_ENABLED", "<unset>"),
        "ACT_EMERGENCY_MODE": os.environ.get("ACT_EMERGENCY_MODE", "<unset>"),
        "ACT_POLYMARKET_ENABLED": os.environ.get("ACT_POLYMARKET_ENABLED", "<unset>"),
    }


def _check_ollama_models() -> Dict[str, Any]:
    """Ping local Ollama and report which of the required models are pulled."""
    required: List[str] = []
    try:
        from src.ai.dual_brain import _resolve, SCANNER, ANALYST
        required = [_resolve(None, SCANNER).model, _resolve(None, ANALYST).model]
    except Exception as e:
        return {"ok": False, "error": f"dual_brain import failed: {e}"}

    try:
        import urllib.request
        base = os.environ.get("OLLAMA_REMOTE_URL") or "http://localhost:11434"
        url = base.rstrip("/") + "/api/tags"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=3.0) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
        present = {m.get("name", "").split(":")[0] for m in (data.get("models") or [])}
        # Match by prefix — tags like `deepseek-r1:7b` vs stored `deepseek-r1`.
        required_map = {}
        for m in required:
            head = m.split(":")[0]
            required_map[m] = head in present
        return {"ok": True, "required": required_map, "pulled_count": len(present)}
    except Exception as e:
        return {"ok": False, "error": f"ollama unreachable: {e}"}


def _check_warm_store_counts() -> Dict[str, Any]:
    """Count decisions in the last 24h: total, shadow, real, outcomes."""
    db_path = os.getenv(
        "ACT_WARM_DB_PATH",
        str(Path(__file__).resolve().parents[3] / "data" / "warm_store.sqlite"),
    )
    if not os.path.exists(db_path):
        return {"exists": False, "path": db_path}
    try:
        conn = sqlite3.connect(db_path, timeout=3.0)
        cur = conn.cursor()
        cutoff_ns = int((time.time() - 86400) * 1_000_000_000)
        total = cur.execute("SELECT COUNT(*) FROM decisions WHERE ts_ns >= ?", (cutoff_ns,)).fetchone()[0]
        shadow = cur.execute(
            "SELECT COUNT(*) FROM decisions WHERE ts_ns >= ? AND decision_id LIKE 'shadow-%'",
            (cutoff_ns,),
        ).fetchone()[0]
        real = total - shadow
        outcomes_24h = cur.execute(
            "SELECT COUNT(*) FROM outcomes WHERE exit_ts >= ?", (time.time() - 86400,),
        ).fetchone()[0]
        # Breakdown of actions on real (non-shadow) rows — what was the executor doing?
        actions = cur.execute(
            "SELECT final_action, COUNT(*) FROM decisions "
            "WHERE ts_ns >= ? AND decision_id NOT LIKE 'shadow-%' "
            "GROUP BY final_action ORDER BY 2 DESC", (cutoff_ns,),
        ).fetchall()
        # Breakdown of terminated_reason on shadow plans
        shadow_reasons = cur.execute(
            "SELECT json_extract(component_signals, '$.terminated_reason'), COUNT(*) "
            "FROM decisions WHERE ts_ns >= ? AND decision_id LIKE 'shadow-%' "
            "GROUP BY 1 ORDER BY 2 DESC", (cutoff_ns,),
        ).fetchall()
        conn.close()
        return {
            "exists": True,
            "decisions_24h_total": int(total or 0),
            "decisions_24h_shadow": int(shadow or 0),
            "decisions_24h_real": int(real or 0),
            "outcomes_24h": int(outcomes_24h or 0),
            "real_action_breakdown": dict(actions or []),
            "shadow_terminated_reasons": dict(shadow_reasons or []),
        }
    except Exception as e:
        return {"exists": True, "error": f"warm_store query failed: {e}"}


def _check_readiness() -> Dict[str, Any]:
    try:
        from src.orchestration.readiness_gate import evaluate, is_emergency_mode
        state = evaluate()
        return {
            "open": bool(state.open_),
            "failing": list(state.reasons),
            "details": dict(state.details),
            "emergency_mode": bool(is_emergency_mode(state)),
        }
    except Exception as e:
        return {"error": f"readiness_gate import/eval failed: {e}"}


def _check_brain_memory() -> Dict[str, Any]:
    try:
        from src.ai.brain_memory import get_brain_memory
        mem = get_brain_memory()
    except Exception as e:
        return {"error": f"brain_memory unavailable: {e}"}

    out: Dict[str, Any] = {}
    for asset in ("BTC", "ETH"):
        try:
            scan = mem.read_latest_scan(asset, max_age_s=86400.0)
            traces = mem.read_recent_traces(asset, limit=5, max_age_s=86400.0)
            out[asset] = {
                "latest_scan_age_s": (int(scan.age_s()) if scan else None),
                "latest_scan_score": (scan.opportunity_score if scan else None),
                "recent_trace_count_24h": len(traces),
            }
        except Exception as e:
            out[asset] = {"error": str(e)[:100]}
    return out


def _check_config() -> Dict[str, Any]:
    """Inspect key config switches."""
    cfg_path = Path(__file__).resolve().parents[3] / "config.yaml"
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


# ── Rules-based diagnosis ──────────────────────────────────────────────


def _diagnose(data: Dict[str, Any]) -> str:
    """Return the most likely cause in one line. Priority ordered."""
    env = data.get("env") or {}
    ollama = data.get("ollama") or {}
    ws = data.get("warm_store") or {}
    gate = data.get("readiness") or {}
    mem = data.get("brain_memory") or {}

    # 0. Kill switch engaged
    if env.get("ACT_DISABLE_AGENTIC_LOOP") == "1":
        return "ACT_DISABLE_AGENTIC_LOOP=1 — agentic loop is HALTED by env kill switch"

    # 1. Agentic loop never turned on
    if env.get("ACT_AGENTIC_LOOP") not in ("1", "true", "yes", "on"):
        if not (data.get("config") or {}).get("agentic_loop_enabled"):
            return "ACT_AGENTIC_LOOP flag not set + config.agentic_loop.enabled=false — shadow loop isn't firing at all"

    # 2. Ollama models missing
    if ollama.get("ok") is True:
        missing = [m for m, ok in (ollama.get("required") or {}).items() if not ok]
        if missing:
            return f"Ollama models missing: {', '.join(missing)} — analyst/scanner calls return empty, loop skips every tick"
    elif ollama.get("ok") is False:
        return f"Ollama unreachable — {ollama.get('error', '')}"

    # 3. warm_store missing
    if ws.get("exists") is False:
        return "warm_store.sqlite does not exist — bot has never run (first start pending?)"

    # 4. Shadow rows exist but zero REAL decisions → gate is holding back
    if ws.get("decisions_24h_shadow", 0) > 0 and ws.get("decisions_24h_real", 0) == 0:
        gate_open = bool(gate.get("open"))
        if not gate_open:
            reasons = ", ".join((gate.get("failing") or [])[:3])
            return (
                f"Readiness gate CLOSED ({reasons or 'see details'}) — "
                f"bot is in paper soak; {ws.get('decisions_24h_shadow')} shadow plans compiled "
                f"in 24h but zero real trades fire until gate opens"
            )
        else:
            return (
                "Readiness gate open but zero real decisions — likely conviction gate "
                "is rejecting all shadow plans before they reach the executor"
            )

    # 5. No shadow rows at all → loop not firing (flag issue despite #1 check)
    if ws.get("decisions_24h_total", 0) == 0:
        return "Zero warm_store decisions in last 24h — shadow hook is not firing; verify bot restarted after git pull"

    # 6. Plenty of real decisions, no outcomes → stuck in open positions
    if ws.get("decisions_24h_real", 0) > 0 and ws.get("outcomes_24h", 0) == 0:
        return "Real decisions present but zero closed outcomes — positions may be stuck open; check failed_close_assets"

    # 7. Scanner quiet
    btc_scan_age = (mem.get("BTC") or {}).get("latest_scan_age_s")
    eth_scan_age = (mem.get("ETH") or {}).get("latest_scan_age_s")
    if btc_scan_age is None and eth_scan_age is None:
        return "brain_memory has no scanner reports — scanner brain isn't publishing; check dual_brain enabled + scanner model reachable"

    # 8. Shadow dominated by skip/max_steps
    st = ws.get("shadow_terminated_reasons") or {}
    total_shadow = sum(v for v in st.values() if isinstance(v, int))
    if total_shadow > 0:
        skip_ratio = (st.get("skip", 0) + st.get("max_steps", 0) + st.get("parse_failures", 0)) / total_shadow
        if skip_ratio > 0.9:
            return (
                f"{int(skip_ratio*100)}% of shadow runs terminated in skip/max_steps/parse_failures — "
                "analyst isn't compiling plans successfully; check LLM output format + temperature"
            )

    return "No obvious blocker. If still no profitable trades, suspect conviction-tier thresholds or market regime (low vol)."


# ── Skill entry point ──────────────────────────────────────────────────


def run(args: Dict[str, Any]) -> SkillResult:
    data = {
        "env": _check_env_flags(),
        "ollama": _check_ollama_models(),
        "warm_store": _check_warm_store_counts(),
        "readiness": _check_readiness(),
        "brain_memory": _check_brain_memory(),
        "config": _check_config(),
    }
    diagnosis = _diagnose(data)

    # Compact human-readable summary.
    ws = data["warm_store"]
    gate = data["readiness"]
    lines = [
        f"Diagnosis: {diagnosis}",
        "",
        f"  ACT_AGENTIC_LOOP        = {data['env'].get('ACT_AGENTIC_LOOP')}",
        f"  ACT_DISABLE_AGENTIC_LOOP= {data['env'].get('ACT_DISABLE_AGENTIC_LOOP')}",
        f"  Brain profile           = {data['env'].get('ACT_BRAIN_PROFILE') or data['config'].get('dual_brain_profile') or 'default'}",
    ]
    if isinstance(data["ollama"], dict) and data["ollama"].get("ok"):
        for m, ok in (data["ollama"].get("required") or {}).items():
            lines.append(f"  Ollama has {m:<24} = {'yes' if ok else 'NO'}")
    else:
        lines.append(f"  Ollama                  = unreachable ({(data['ollama'] or {}).get('error','')})")
    if ws.get("exists"):
        lines.append(f"  warm_store decisions/24h= {ws.get('decisions_24h_total', 0)} "
                     f"(shadow={ws.get('decisions_24h_shadow', 0)}, real={ws.get('decisions_24h_real', 0)})")
        lines.append(f"  warm_store outcomes/24h = {ws.get('outcomes_24h', 0)}")
    lines.append(f"  Readiness gate open     = {gate.get('open')}")
    if gate.get("failing"):
        lines.append(f"  Gate failing            = {', '.join(gate['failing'][:3])}")

    return SkillResult(ok=True, message="\n".join(lines), data=data)
