"""Skill: /status — one-shot system verification after pull + START_ALL."""
from __future__ import annotations

from typing import Any, Dict, List

from src.skills.diagnostics import (
    check_brain_memory,
    check_config,
    check_env_flags,
    check_graph,
    check_ollama_models,
    check_personas,
    check_polymarket,
    check_readiness,
    check_warm_store,
)
from src.skills.registry import SkillResult


def _lights(data: Dict[str, Any]) -> Dict[str, str]:
    env = data["env"]
    ollama = data["ollama"]
    ws = data["warm_store"]
    bm = data["brain_memory"]
    graph = data["graph"]
    personas = data["personas"]
    readiness = data["readiness"]

    lights: Dict[str, str] = {}
    if env.get("ACT_DISABLE_AGENTIC_LOOP") == "1":
        lights["env"] = "red"
    elif env.get("ACT_AGENTIC_LOOP") in ("1", "true", "yes", "on"):
        lights["env"] = "green"
    else:
        lights["env"] = "yellow"

    if ollama.get("ok") and all(ollama.get("required", {}).values()):
        lights["ollama"] = "green"
    elif ollama.get("ok"):
        lights["ollama"] = "yellow"
    else:
        lights["ollama"] = "red"

    if ws.get("exists") and ws.get("decisions_shadow", 0) > 0:
        lights["warm_store"] = "green"
    elif ws.get("exists"):
        lights["warm_store"] = "yellow"
    else:
        lights["warm_store"] = "red"

    btc_age = (bm or {}).get("BTC", {}).get("latest_scan_age_s")
    lights["brain_memory"] = "green" if btc_age is not None and btc_age < 600 else "yellow"

    edge_total = sum((graph.get("edges_1h_by_kind") or {}).values())
    lights["graph"] = "green" if edge_total > 0 else "yellow"
    lights["personas"] = "green" if (personas or {}).get("active_count", 0) > 0 else "yellow"
    lights["readiness"] = "green" if (readiness or {}).get("open") else "yellow"

    pm = data.get("polymarket") or {}
    if pm.get("executor_mode") == "live":
        lights["polymarket"] = "green"
    elif pm.get("enabled_in_config") or pm.get("recent_shadow_orders", 0) > 0:
        lights["polymarket"] = "yellow"   # shadow-active
    else:
        lights["polymarket"] = "yellow"
    return lights


def run(args: Dict[str, Any]) -> SkillResult:
    data: Dict[str, Any] = {
        "env": check_env_flags(),
        "ollama": check_ollama_models(),
        "warm_store": check_warm_store(window_s=3600),
        "brain_memory": check_brain_memory(),
        "graph": check_graph(),
        "personas": check_personas(),
        "readiness": check_readiness(),
        "polymarket": check_polymarket(),
        "config": check_config(),
    }
    lights = _lights(data)
    data["lights"] = lights

    lines: List[str] = ["ACT /status — subsystem traffic lights"]
    for k, v in lights.items():
        icon = {"green": "✓", "yellow": "…", "red": "✗"}.get(v, "?")
        lines.append(f"  [{icon}] {k:<12} = {v}")
    lines.append("")
    env = data["env"]
    lines.append(f"  ACT_AGENTIC_LOOP     = {env.get('ACT_AGENTIC_LOOP')}")
    lines.append(f"  Brain profile        = {env.get('ACT_BRAIN_PROFILE')}")
    lines.append(f"  Scanner / Analyst    = {env.get('ACT_SCANNER_MODEL')} / {env.get('ACT_ANALYST_MODEL')}")

    # Helpful hint: when all agentic env flags are <unset>, the operator
    # is likely running /status from a terminal that didn't source
    # START_ALL.ps1. setx persistence fixes this for future terminals.
    if env.get("ACT_AGENTIC_LOOP") == "<unset>" and \
       env.get("ACT_BRAIN_PROFILE") == "<unset>":
        lines.append("")
        lines.append("  HINT: env flags unset — either:")
        lines.append("    (a) open a fresh terminal after running START_ALL.ps1 ")
        lines.append("        (setx persists; current windows need a restart), or")
        lines.append("    (b) set manually:  setx ACT_AGENTIC_LOOP 1")
        lines.append("                       setx ACT_BRAIN_PROFILE dense_r1")

    ws = data["warm_store"]
    if ws.get("exists"):
        lines.append(
            f"  warm_store 1h        = total={ws.get('decisions_total', 0)}, "
            f"shadow={ws.get('decisions_shadow', 0)}, "
            f"scanner_pub={ws.get('scanner_published', 0)}"
        )
    r = data["readiness"]
    if "details" in r and r["details"].get("trades") is not None:
        det = r["details"]
        lines.append(
            f"  Readiness            = open={r.get('open')} "
            f"trades={det.get('trades')} soak_days={det.get('soak_days')} "
            f"sharpe={det.get('rolling_sharpe')}"
        )
    counts = (data.get("graph") or {}).get("edges_1h_by_kind") or {}
    if counts:
        line = ", ".join(f"{k}={v}" for k, v in
                         sorted(counts.items(), key=lambda kv: -kv[1])[:5])
        lines.append(f"  Graph 1h             = {line}")
    p = data.get("personas") or {}
    if "active_count" in p:
        lines.append(f"  Active personas      = {p.get('active_count', 0)}")

    overall_ok = lights["warm_store"] != "red" and lights["ollama"] != "red"
    return SkillResult(ok=overall_ok, message="\n".join(lines), data=data)
