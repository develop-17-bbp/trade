"""
Tool implementations for the ACT MCP server.

Each function is pure-ish: reads from disk / subprocess and returns a dict.
They NEVER raise to the caller — every failure becomes `{"error": "..."}` so
the Claude Code session sees a clean response rather than a transport error.

The functions are kept free of FastMCP decorators so they're importable +
testable without the full MCP runtime.
"""
from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────
# Audit log — one JSONL line per tool invocation
# ─────────────────────────────────────────────────────────────────────

_AUDIT_PATH = REPO_ROOT / "logs" / "mcp_audit.jsonl"
_audit_lock = Lock()


def _audit(tool_name: str, duration_ms: int, success: bool, error: Optional[str] = None) -> None:
    """Append one line to logs/mcp_audit.jsonl. Never raises."""
    try:
        _AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tool": tool_name,
            "duration_ms": duration_ms,
            "success": success,
        }
        if error:
            rec["error"] = str(error)[:200]
        with _audit_lock:
            with open(_AUDIT_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    except Exception:
        pass  # audit failures cannot break tool calls


def audited(fn: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    """Decorator that appends an audit-log entry per call."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        err = None
        try:
            result = fn(*args, **kwargs)
            success = not (isinstance(result, dict) and "error" in result)
            if not success:
                err = result.get("error")
            return result
        except Exception as e:
            err = str(e)[:200]
            success = False
            raise
        finally:
            _audit(fn.__name__, int((time.time() - t0) * 1000), success, err)
    return wrapper


# ─────────────────────────────────────────────────────────────────────
# Safety — mutation gate
# ─────────────────────────────────────────────────────────────────────

def mutations_allowed() -> bool:
    """True if mutating tools can run. Default: False."""
    return (os.environ.get("ACT_MCP_ALLOW_MUTATIONS") or "").strip().lower() in (
        "1", "true", "yes", "on"
    )


# ─────────────────────────────────────────────────────────────────────
# Read-only tools
# ─────────────────────────────────────────────────────────────────────

@audited
def status() -> Dict[str, Any]:
    """One-shot system status — bot uptime, equity, positions, readiness gate.

    Returns a single dict the remote caller can poll every ~30s to see
    overall health without making 5 separate calls.
    """
    out: Dict[str, Any] = {}
    # Paper state
    try:
        p = REPO_ROOT / "logs" / "robinhood_paper_state.json"
        if p.exists():
            out["paper"] = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        out["paper_error"] = str(e)
    # Readiness gate
    try:
        from src.orchestration.readiness_gate import evaluate
        out["readiness_gate"] = evaluate().to_dict()
    except Exception as e:
        out["gate_error"] = str(e)
    # Safe-entries summary
    try:
        from src.trading.safe_entries import SafeEntryState, default_state_path
        st = SafeEntryState.load(default_state_path(str(REPO_ROOT / "logs")))
        out["safe_entries"] = {
            "combined_rolling_sharpe_30": round(st.combined_rolling_sharpe(n=30), 4),
            "assets": {
                a: {
                    "consecutive_losses": s.consecutive_losses,
                    "n_trades": len(s.trade_pnl_pcts),
                    "rolling_sharpe_30": round(st.rolling_sharpe(a, n=30), 4),
                }
                for a, s in st.assets.items()
            },
        }
    except Exception as e:
        out["safe_entries_error"] = str(e)
    return out


@audited
def evaluator_report() -> Dict[str, Any]:
    """Full JSON evaluation report (same data as `python scripts/evaluate_act.py --json`)."""
    try:
        from src.evaluation.act_evaluator import build_report
        r = build_report()
        # Trim the heavy trades array for transport
        r["trades_count"] = len(r.get("trades") or [])
        r.pop("trades", None)
        return r
    except Exception as e:
        return {"error": f"evaluator_failed: {e}"}


@audited
def component_state() -> Dict[str, Any]:
    """Current ON/OFF state of every toggleable component + exact setx cmd to flip each."""
    try:
        from src.evaluation.act_evaluator import load_component_state
        return load_component_state()
    except Exception as e:
        return {"error": f"component_state_failed: {e}"}


@audited
def paper_state() -> Dict[str, Any]:
    """Paper-trading equity + stats + open positions."""
    try:
        p = REPO_ROOT / "logs" / "robinhood_paper_state.json"
        if not p.exists():
            return {"available": False}
        return {"available": True, **json.loads(p.read_text(encoding="utf-8"))}
    except Exception as e:
        return {"error": str(e)}


@audited
def shadow_stats() -> Dict[str, Any]:
    """Meta-model shadow-log stats (predictions + outcomes)."""
    try:
        from src.ml.shadow_log import read_all, join_predict_outcome, shadow_stats as _ss
        recs = read_all(str(REPO_ROOT / "logs" / "meta_shadow.jsonl"))
        joined = join_predict_outcome(recs)
        out = {
            "total_records": len(recs),
            "joined_trades": len(joined),
            "per_asset": {},
            "combined": _ss(joined),
        }
        for asset in {r.get("asset") for r in joined if r.get("asset")}:
            out["per_asset"][asset] = _ss([r for r in joined if r.get("asset") == asset])
        return out
    except Exception as e:
        return {"error": str(e)}


@audited
def readiness_gate() -> Dict[str, Any]:
    """Readiness-gate evaluation — is the bot allowed to place real-capital orders?"""
    try:
        from src.orchestration.readiness_gate import evaluate
        return evaluate().to_dict()
    except Exception as e:
        return {"error": str(e)}


@audited
def tail_log(log_name: str = "autonomous_loop.log", lines: int = 100) -> Dict[str, Any]:
    """Return the last `lines` lines of a log file under logs/.

    Only allows filenames (no path traversal). Caps `lines` at 5000.
    """
    # Harden path — reject anything with separators or dotfiles
    if "/" in log_name or "\\" in log_name or log_name.startswith("."):
        return {"error": "invalid_log_name"}
    if not log_name.endswith((".log", ".jsonl", ".json", ".txt")):
        return {"error": "only_text_logs_allowed"}
    lines = max(1, min(int(lines), 5000))

    p = REPO_ROOT / "logs" / log_name
    if not p.exists():
        return {"error": f"not_found:{log_name}", "path": str(p)}
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        tail = all_lines[-lines:]
        return {"name": log_name, "total_lines": len(all_lines),
                "returned": len(tail), "content": "".join(tail)}
    except Exception as e:
        return {"error": str(e)}


@audited
def list_logs() -> Dict[str, Any]:
    """Available log files under logs/ with sizes."""
    logs_dir = REPO_ROOT / "logs"
    if not logs_dir.exists():
        return {"logs": [], "error": "logs_dir_missing"}
    out = []
    for p in sorted(logs_dir.iterdir()):
        try:
            if p.is_file() and p.suffix in (".log", ".jsonl", ".json", ".txt"):
                out.append({
                    "name": p.name,
                    "size_bytes": p.stat().st_size,
                    "mtime": int(p.stat().st_mtime),
                })
        except Exception:
            continue
    out.sort(key=lambda x: -x["mtime"])
    return {"logs": out}


@audited
def recent_trades(limit: int = 20) -> Dict[str, Any]:
    """Last N completed trades from robinhood_paper.jsonl as structured records."""
    try:
        from src.evaluation.act_evaluator import load_paper_trades
        trades = load_paper_trades(str(REPO_ROOT / "logs" / "robinhood_paper.jsonl"))
        limit = max(1, min(int(limit), 500))
        return {"total": len(trades), "returned": min(limit, len(trades)),
                "trades": trades[-limit:]}
    except Exception as e:
        return {"error": str(e)}


@audited
def git_status() -> Dict[str, Any]:
    """Current git state — HEAD, branch, uncommitted files."""
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
        status_lines = subprocess.check_output(
            ["git", "status", "--short"], cwd=str(REPO_ROOT), text=True
        ).strip().splitlines()
        last_msg = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%s"], cwd=str(REPO_ROOT), text=True
        ).strip()
        return {
            "head": head, "branch": branch,
            "last_commit_subject": last_msg,
            "uncommitted_count": len(status_lines),
            "uncommitted": status_lines[:50],  # cap
        }
    except Exception as e:
        return {"error": str(e)}


@audited
def audit_log(limit: int = 50) -> Dict[str, Any]:
    """Return the last `limit` entries from logs/mcp_audit.jsonl.

    Lets the operator see what MCP tools Claude has called recently,
    when, how long each took, and whether they succeeded — from inside
    Claude Code without leaving the session.
    """
    limit = max(1, min(int(limit), 500))
    if not _AUDIT_PATH.exists():
        return {"total": 0, "entries": []}
    try:
        with open(_AUDIT_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        recs = []
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                continue
        return {"total": len(lines), "returned": len(recs), "entries": recs}
    except Exception as e:
        return {"error": str(e)}


@audited
def env_flags() -> Dict[str, Any]:
    """Current env flag state for all ACT_* variables the bot reads."""
    keys = [
        "ACT_DISABLE_ML", "ACT_SAFE_ENTRIES", "ACT_META_SHADOW_MODE",
        "ACT_LGBM_DEVICE", "ACT_ROBINHOOD_HARDEN", "ACT_GATE_MIN_SHARPE",
        "ACT_GATE_SHARPE_WINDOW", "ACT_REAL_CAPITAL_ENABLED",
        "ACT_METRICS_ENABLED", "ACT_TRACING_ENABLED",
        "TRADE_API_DEV_MODE", "DASHBOARD_API_KEY",
        "ACT_MCP_ALLOW_MUTATIONS", "ACT_MCP_TOKEN",
    ]
    return {
        "env": {k: (os.environ.get(k) if k != "DASHBOARD_API_KEY" and k != "ACT_MCP_TOKEN"
                    else ("(set)" if os.environ.get(k) else None))
                for k in keys},
        "note": "DASHBOARD_API_KEY and ACT_MCP_TOKEN values redacted",
    }


# ─────────────────────────────────────────────────────────────────────
# Mutating tools (gated by ACT_MCP_ALLOW_MUTATIONS=1)
# ─────────────────────────────────────────────────────────────────────

@audited
def restart_bot() -> Dict[str, Any]:
    """Run STOP_ALL.ps1 then START_ALL.ps1. Requires ACT_MCP_ALLOW_MUTATIONS=1."""
    if not mutations_allowed():
        return {
            "error": "mutations_disabled",
            "message": "Set ACT_MCP_ALLOW_MUTATIONS=1 on the GPU box and restart the MCP server to enable",
        }
    try:
        stop = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(REPO_ROOT / "STOP_ALL.ps1")],
            capture_output=True, text=True, timeout=60,
        )
        start = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(REPO_ROOT / "START_ALL.ps1")],
            capture_output=True, text=True, timeout=120,
        )
        return {
            "stop_rc": stop.returncode,
            "start_rc": start.returncode,
            "stop_tail": stop.stdout.splitlines()[-10:],
            "start_tail": start.stdout.splitlines()[-10:],
        }
    except Exception as e:
        return {"error": str(e)}


@audited
def trigger_retrain(asset: str = "BTC") -> Dict[str, Any]:
    """Fire train_all_models.py for the specified asset. Requires ACT_MCP_ALLOW_MUTATIONS=1."""
    if not mutations_allowed():
        return {"error": "mutations_disabled"}
    asset_up = str(asset).upper().strip()
    if asset_up not in ("BTC", "ETH", "AAVE", "SOL", "BNB", "ADA", "DOGE", "XRP", "AVAX"):
        return {"error": f"unknown_asset:{asset}"}
    try:
        res = subprocess.run(
            [sys.executable, "-m", "src.scripts.train_all_models",
             "--asset", asset_up, "--bars", "20000"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=600,
        )
        return {
            "rc": res.returncode,
            "tail": res.stdout.splitlines()[-30:],
            "stderr_tail": res.stderr.splitlines()[-10:],
        }
    except subprocess.TimeoutExpired:
        return {"error": "retrain_timeout_10min"}
    except Exception as e:
        return {"error": str(e)}
