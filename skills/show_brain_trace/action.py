"""Skill: /show-brain-trace — show what the LLMs are actually seeing.

Diagnostic skill for "is the model receiving context and analyzing?"

Pulls the last N scanner reports from brain_memory and the last N
analyst traces from warm_store, plus the last N shadow decisions
with their full plan_json + self_critique. Output is designed to
make it obvious whether:

  * the scanner is producing real rationale (vs empty / "unavailable")
  * the analyst is consuming the scanner's report (via corpus callosum)
  * the analyst's output JSON matches the expected envelope
  * parse failures are affecting a subset or all decisions

Never mutates state. Output is scrubbed through output_scrubber so
accidental secrets in traces don't leak to the operator's screen.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.skills.registry import SkillResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _warm_store_path() -> Path:
    return Path(
        os.getenv("ACT_WARM_DB_PATH")
        or str(PROJECT_ROOT / "data" / "warm_store.sqlite")
    )


def _scrub(text: Any) -> str:
    """Best-effort PII/secret scrubbing before surfacing to operator."""
    try:
        from src.ai.output_scrubber import scrub
        return scrub(str(text or "")).text
    except Exception:
        return str(text or "")


def _read_recent_decisions(asset: str, limit: int) -> List[Dict[str, Any]]:
    db = _warm_store_path()
    if not db.exists():
        return []
    try:
        conn = sqlite3.connect(str(db), timeout=5.0)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(decisions)").fetchall()}
    except Exception:
        return []

    select_cols = ["decision_id", "ts_ns"]
    for c in ("symbol", "direction", "final_action", "plan_json",
              "self_critique", "component_signals"):
        if c in cols:
            select_cols.append(c)

    asset_u = asset.upper()
    if "symbol" in cols:
        sql = (f"SELECT {', '.join(select_cols)} FROM decisions "
               "WHERE symbol = ? ORDER BY ts_ns DESC LIMIT ?")
        params = (asset_u, limit)
    else:
        sql = (f"SELECT {', '.join(select_cols)} FROM decisions "
               "ORDER BY ts_ns DESC LIMIT ?")
        params = (limit,)
    try:
        rows = conn.execute(sql, params).fetchall()
        conn.close()
    except Exception:
        return []
    return [dict(zip(select_cols, r)) for r in rows]


def _read_scanner_reports(asset: str, limit: int) -> List[Dict[str, Any]]:
    try:
        from src.ai.brain_memory import get_brain_memory
        mem = get_brain_memory()
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    try:
        latest = mem.read_latest_scan(asset.upper(), max_age_s=86400.0)
        if latest:
            out.append({
                "ts": getattr(latest, "ts", 0.0),
                "age_s": getattr(latest, "age_s", lambda: 0)(),
                "opportunity_score": getattr(latest, "opportunity_score", 0.0),
                "proposed_direction": getattr(latest, "proposed_direction", ""),
                "top_signals": list(getattr(latest, "top_signals", []) or []),
                "rationale": _scrub(getattr(latest, "rationale", ""))[:300],
                "model_metadata": getattr(latest, "model_metadata", {}) or {},
            })
    except Exception:
        pass
    try:
        traces = mem.read_recent_traces(asset.upper(), limit=limit, max_age_s=86400.0) or []
        for t in traces:
            out.append({
                "kind": "analyst_trace",
                "direction": getattr(t, "direction", ""),
                "tier": getattr(t, "tier", ""),
                "size_pct": getattr(t, "size_pct", 0),
                "verdict": getattr(t, "verdict", ""),
                "ts": getattr(t, "ts", 0),
            })
    except Exception:
        pass
    return out


def _summarize_plan_json(pj: str) -> Dict[str, Any]:
    if not pj or pj == "null":
        return {"empty": True}
    try:
        obj = json.loads(pj)
    except Exception:
        return {"parse_error": True, "raw_preview": _scrub(pj)[:200]}
    if not isinstance(obj, dict):
        return {"non_dict": True, "raw_preview": _scrub(str(obj))[:200]}
    # Extract the fields operators care about
    return {
        "tier": obj.get("entry_tier") or obj.get("tier"),
        "direction": obj.get("direction"),
        "size_pct": obj.get("size_pct"),
        "thesis": _scrub(obj.get("thesis", ""))[:300],
        "supporting_evidence_count": len(obj.get("supporting_evidence") or []),
        "has_sl": obj.get("sl_price") is not None,
        "has_tp": bool(obj.get("tp_levels")),
    }


def run(args: Optional[Dict[str, Any]] = None) -> SkillResult:
    args = dict(args or {})
    asset = str(args.get("asset", "BTC")).upper()
    try:
        limit = int(args.get("limit", 5))
    except (TypeError, ValueError):
        limit = 5
    limit = max(1, min(20, limit))

    decisions = _read_recent_decisions(asset, limit)
    scanner = _read_scanner_reports(asset, limit)

    # Classify decisions by outcome
    by_outcome: Dict[str, int] = {}
    parse_failures = 0
    empty_plans = 0
    real_plans = 0
    sample_plans: List[Dict[str, Any]] = []
    for d in decisions:
        did = str(d.get("decision_id") or "")
        pj_summary = _summarize_plan_json(d.get("plan_json") or "")
        if pj_summary.get("parse_error"):
            parse_failures += 1
        elif pj_summary.get("empty"):
            empty_plans += 1
        else:
            real_plans += 1
        by_outcome.setdefault("total", 0)
        by_outcome["total"] += 1
        tier = pj_summary.get("tier") or "unknown"
        by_outcome.setdefault(str(tier), 0)
        by_outcome[str(tier)] += 1
        sample_plans.append({
            "decision_id": did,
            "ts_iso": datetime.fromtimestamp(
                int(d.get("ts_ns", 0)) / 1e9, tz=timezone.utc
            ).isoformat() if d.get("ts_ns") else None,
            "plan_summary": pj_summary,
            "has_self_critique": bool(
                d.get("self_critique") and d.get("self_critique") not in ("{}", "null")
            ),
        })

    scanner_latest = scanner[0] if scanner else None
    scanner_has_rationale = bool(scanner_latest
                                  and str(scanner_latest.get("rationale", "")).strip())
    scanner_score = scanner_latest.get("opportunity_score") if scanner_latest else None

    # Verdict message
    if not decisions and not scanner:
        msg = (f"{asset}: no recent activity. The bot may not have run a "
               "tick yet, or warm_store/brain_memory paths are misaligned.")
        recommendation = "RESTART_AND_WAIT"
    elif parse_failures == len(decisions) and len(decisions) > 0:
        msg = (f"{asset}: ALL {parse_failures} recent decisions failed JSON parse. "
               "LLM is returning empty or malformed output — check Ollama VRAM, "
               "profile selection, and parse_failure preview in bot log.")
        recommendation = "CHECK_LLM_PROVIDER"
    elif real_plans > 0:
        msg = (f"{asset}: {real_plans}/{len(decisions)} decisions have "
               f"valid plan_json (parse={parse_failures}, empty={empty_plans}). "
               f"Scanner: {'has rationale' if scanner_has_rationale else 'EMPTY rationale'}.")
        recommendation = "HEALTHY" if scanner_has_rationale else "SCANNER_EMPTY"
    else:
        msg = (f"{asset}: {len(decisions)} decisions, 0 with valid plan_json. "
               f"Scanner: {'has rationale' if scanner_has_rationale else 'EMPTY rationale'}.")
        recommendation = "SCANNER_OR_PARSER_BROKEN"

    return SkillResult(
        ok=True,
        message=msg,
        data={
            "asset": asset,
            "recommendation": recommendation,
            "summary": {
                "decisions_checked": len(decisions),
                "real_plans": real_plans,
                "empty_plans": empty_plans,
                "parse_failures": parse_failures,
                "scanner_has_rationale": scanner_has_rationale,
                "scanner_score": scanner_score,
                "by_outcome": by_outcome,
            },
            "latest_scanner_report": scanner_latest,
            "decision_samples": sample_plans,
        },
    )
