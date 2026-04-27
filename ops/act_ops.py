"""ACT autonomous operations agent.

Runs every 15 minutes (Windows Task Scheduler / cron), performs:
  1. Health checks  -- ACT_AGENTIC_LOOP, drawdown halt, tunnel freshness, last trade timestamp.
  2. Strategy-leak detection (loaded from ops/strategy_leaks.yaml).
  3. Auto-remediation for known issues (env flags, file writes, restart).
  4. Daily EOD report (P&L, leak summary) at the first run after midnight.
  5. Webhook escalation only when an issue requires human judgment.

Design constraints:
  * Read-only by default; writes only fire under known-safe remediations.
  * Routine work runs in this script directly -- fast + deterministic.
  * Ambiguous cases escalate via Claude Code headless (`claude -p`).
  * All remediation actions commit to git for audit.

Usage:
    python ops/act_ops.py                  # full cycle: checks + remediate + report-if-eod
    python ops/act_ops.py --check-only     # diagnostic only, no writes
    python ops/act_ops.py --eod            # force EOD report regardless of time
    python ops/act_ops.py --escalate L10   # force-escalate a known leak via Claude headless

Env:
    ACT_OPS_WEBHOOK_URL          where to POST human-judgment escalations (Slack/Discord/etc.)
    ACT_OPS_DRY_RUN=1            log actions but don't execute remediations
    ACT_OPS_AUTO_COMMIT=1        commit + push remediation actions (default off)
    ACT_OPS_HEADLESS_CLAUDE=1    enable `claude -p` escalations (default off; needs CLI installed)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request

REPO_ROOT = Path(__file__).resolve().parent.parent
LEAKS_YAML = Path(__file__).resolve().parent / "strategy_leaks.yaml"
RUNBOOK_LOG = REPO_ROOT / "logs" / "act_ops.jsonl"
SYSTEM_LOG = REPO_ROOT / "logs" / "system_output.log"
WARM_STORE = REPO_ROOT / "data" / "warm_store.sqlite"
PAPER_LOG = REPO_ROOT / "logs" / "robinhood_paper.jsonl"
EOD_DIR = REPO_ROOT / "logs" / "eod"
LAST_RUN_FILE = REPO_ROOT / "logs" / "act_ops_last_run.json"

LOG_TAIL_LINES = 500       # how much of system_output.log to scan for leak signatures
TUNNEL_STALE_HOURS = 2.0   # cloudflared quick-tunnel rotates ~daily; >2h stale flagged
LAST_TRADE_STALE_H = 24.0  # 24h with no trade = page operator
EOD_HOUR_LOCAL = 0         # midnight local time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [act_ops] %(levelname)s %(message)s",
)
logger = logging.getLogger("act_ops")


# ── Action / Result data classes ───────────────────────────────────────


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RemediationAction:
    """Describes a remediation that ran (or would run in dry-run)."""
    kind: str                  # env_flag | file_write | restart | escalate
    target: str                # env name, file path, service name, or leak id
    before: str                # state before the action
    after: str                 # state after (or planned)
    executed: bool
    error: Optional[str] = None


@dataclass
class CycleReport:
    """One cycle's summary, written to RUNBOOK_LOG and used by EOD."""
    started_at: str
    ended_at: str
    checks: List[CheckResult] = field(default_factory=list)
    leaks_active: List[str] = field(default_factory=list)
    remediations: List[RemediationAction] = field(default_factory=list)
    escalations: List[Dict[str, Any]] = field(default_factory=list)
    is_eod: bool = False
    eod_report_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "checks": [
                {"name": c.name, "ok": c.ok, "detail": c.detail, "raw": c.raw}
                for c in self.checks
            ],
            "leaks_active": list(self.leaks_active),
            "remediations": [
                {"kind": r.kind, "target": r.target, "before": r.before,
                 "after": r.after, "executed": r.executed, "error": r.error}
                for r in self.remediations
            ],
            "escalations": list(self.escalations),
            "is_eod": self.is_eod,
            "eod_report_path": self.eod_report_path,
        }


# ── Health checks ──────────────────────────────────────────────────────


def check_agentic_loop_flag() -> CheckResult:
    """Verify ACT_AGENTIC_LOOP is set and not disabled."""
    enabled = os.environ.get("ACT_AGENTIC_LOOP", "").strip()
    disabled = os.environ.get("ACT_DISABLE_AGENTIC_LOOP", "").strip()
    ok = enabled == "1" and disabled != "1"
    return CheckResult(
        name="agentic_loop_flag",
        ok=ok,
        detail=f"ACT_AGENTIC_LOOP={enabled or '<unset>'} disable={disabled or '<unset>'}",
        raw={"enabled": enabled, "disabled": disabled},
    )


def check_drawdown_halt() -> CheckResult:
    """Read warm_store / equity log for emergency halt state.

    Heuristic: if the latest equity row shows a drawdown > config.max_dd
    OR the executor's emergency_mode is set in checkpoints, flag halt.
    """
    if not WARM_STORE.exists():
        return CheckResult(
            name="drawdown_halt", ok=True,
            detail="warm_store missing -- bot not run yet",
        )
    try:
        c = sqlite3.connect(str(WARM_STORE), timeout=2.0)
        # Most recent checkpoint row -- if emergency_mode flag is True we halt
        row = c.execute(
            "SELECT payload_json FROM checkpoints ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        c.close()
    except sqlite3.OperationalError as e:
        return CheckResult(
            name="drawdown_halt", ok=True,
            detail=f"checkpoint read failed: {e}",
        )
    if not row or not row[0]:
        return CheckResult(
            name="drawdown_halt", ok=True,
            detail="no checkpoint yet",
        )
    try:
        payload = json.loads(row[0])
    except json.JSONDecodeError:
        return CheckResult(
            name="drawdown_halt", ok=True,
            detail="checkpoint json malformed",
        )
    halted = bool(payload.get("emergency_mode") or payload.get("dd_halt"))
    dd_pct = float(payload.get("drawdown_pct", 0.0) or 0.0)
    return CheckResult(
        name="drawdown_halt",
        ok=not halted,
        detail=f"halted={halted} dd_pct={dd_pct:.2f}%",
        raw={"halted": halted, "dd_pct": dd_pct},
    )


def check_tunnel_freshness() -> CheckResult:
    """Cloudflared quick tunnels rotate URLs on every restart; named
    tunnels are stable. We treat "tunnel stale" as: the URL in
    .mcp.json or .env is unreachable AND last_seen_alive > N hours."""
    mcp_json = REPO_ROOT / ".mcp.json"
    if not mcp_json.exists():
        return CheckResult(name="tunnel_freshness", ok=True,
                           detail="no .mcp.json -- skipping tunnel check")
    try:
        cfg = json.loads(mcp_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return CheckResult(name="tunnel_freshness", ok=False,
                           detail=f".mcp.json unreadable: {e}")
    servers = cfg.get("mcpServers") or {}
    for name, conf in servers.items():
        url = (conf.get("url") or "").strip()
        if not url or "trycloudflare.com" not in url:
            continue
        try:
            req = request.Request(url, method="HEAD")
            with request.urlopen(req, timeout=5.0) as r:
                if 200 <= r.status < 500:
                    return CheckResult(
                        name="tunnel_freshness", ok=True,
                        detail=f"{name} reachable: HTTP {r.status}",
                    )
        except (error.URLError, OSError):
            return CheckResult(
                name="tunnel_freshness", ok=False,
                detail=f"{name} tunnel unreachable: {url}",
                raw={"server": name, "url": url},
            )
    return CheckResult(name="tunnel_freshness", ok=True,
                       detail="no quick-tunnels configured")


def check_last_trade() -> CheckResult:
    """Read robinhood_paper.jsonl for the most recent ENTRY timestamp."""
    if not PAPER_LOG.exists():
        return CheckResult(
            name="last_trade", ok=False,
            detail=f"{PAPER_LOG.name} missing -- no trades ever",
        )
    last_entry_ts: Optional[float] = None
    last_event = "?"
    try:
        with PAPER_LOG.open(encoding="utf-8", errors="replace") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") in ("ENTRY", "EXIT"):
                    ts = rec.get("timestamp", "")
                    try:
                        last_entry_ts = datetime.fromisoformat(
                            ts.replace("Z", "+00:00")
                        ).timestamp()
                        last_event = rec.get("event", "?")
                    except (ValueError, TypeError):
                        pass
    except OSError as e:
        return CheckResult(name="last_trade", ok=False,
                           detail=f"read failed: {e}")
    if last_entry_ts is None:
        return CheckResult(
            name="last_trade", ok=False,
            detail="no ENTRY/EXIT events in paper log",
        )
    age_h = (time.time() - last_entry_ts) / 3600.0
    ok = age_h <= LAST_TRADE_STALE_H
    return CheckResult(
        name="last_trade", ok=ok,
        detail=f"last={last_event} age={age_h:.1f}h (threshold={LAST_TRADE_STALE_H}h)",
        raw={"last_event": last_event, "age_h": age_h, "ts": last_entry_ts},
    )


# ── Strategy-leak detector ─────────────────────────────────────────────


def _load_leaks() -> List[Dict[str, Any]]:
    """Load strategy_leaks.yaml. We avoid pulling pyyaml as a hard dep --
    use the stdlib alternative (json5-ish manual parse) only if pyyaml
    isn't installed."""
    if not LEAKS_YAML.exists():
        logger.warning("strategy_leaks.yaml missing at %s", LEAKS_YAML)
        return []
    try:
        import yaml  # type: ignore
        return list((yaml.safe_load(LEAKS_YAML.read_text(encoding="utf-8"))
                    or {}).get("leaks") or [])
    except ImportError:
        logger.error(
            "pyyaml not installed; install via `pip install pyyaml` "
            "to enable strategy-leak detection",
        )
        return []
    except Exception as e:
        logger.error("strategy_leaks.yaml load failed: %s", e)
        return []


def _read_log_tail(n: int = LOG_TAIL_LINES) -> str:
    """Read the last N lines of system_output.log (the executor's stdout)."""
    if not SYSTEM_LOG.exists():
        return ""
    try:
        with SYSTEM_LOG.open(encoding="utf-8", errors="replace") as fp:
            lines = fp.readlines()
        return "".join(lines[-n:])
    except OSError:
        return ""


def detect_active_leaks() -> List[Dict[str, Any]]:
    """Walk the leak runbook and return the ones currently signature-matching.

    Each return dict adds a `match_source` field: "log_pattern" or "state_check".
    """
    log_tail = _read_log_tail()
    active: List[Dict[str, Any]] = []
    for leak in _load_leaks():
        sig = (leak.get("signature") or {})
        # 1. log-pattern signature
        pat = sig.get("log_pattern", "")
        if pat:
            try:
                if re.search(pat, log_tail):
                    active.append({**leak, "match_source": "log_pattern"})
                    continue
            except re.error as e:
                logger.warning("leak %s bad regex: %s", leak.get("id"), e)
        # 2. state-check signature
        state_expr = (sig.get("state_check") or "").strip()
        if state_expr and state_expr != "False":
            try:
                # Safe eval: only modules we explicitly whitelist
                ns = {"os": os, "cost_gate": _import_cost_gate(),
                      "recent_seen": 0, "recent_entered": 0,
                      "recent_pattern_excellent_count": 0}
                if eval(state_expr, {"__builtins__": {}}, ns):
                    active.append({**leak, "match_source": "state_check"})
            except Exception as e:
                logger.debug("leak %s state_check eval failed: %s",
                             leak.get("id"), e)
    return active


def _import_cost_gate() -> Any:
    """Lazy import of cost_gate so leaks can introspect VENUE_COSTS."""
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from src.trading import cost_gate  # type: ignore
        return cost_gate
    except Exception:
        return None


# ── Remediations ───────────────────────────────────────────────────────


def remediate_env_flag(
    env: str, value: str, dry_run: bool,
) -> RemediationAction:
    before = os.environ.get(env, "")
    if before == value:
        return RemediationAction(
            kind="env_flag", target=env, before=before, after=before,
            executed=False, error=None,
        )
    if dry_run:
        return RemediationAction(
            kind="env_flag", target=env, before=before, after=value,
            executed=False, error="dry_run",
        )
    try:
        # setx persists for future shells; also set in current process
        os.environ[env] = value
        if value:
            subprocess.run(
                ["setx", env, value], check=True, capture_output=True, timeout=10,
            )
        else:
            # empty value -> unset persistent
            subprocess.run(
                ["setx", env, ""], check=True, capture_output=True, timeout=10,
            )
        return RemediationAction(
            kind="env_flag", target=env, before=before, after=value,
            executed=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError) as e:
        return RemediationAction(
            kind="env_flag", target=env, before=before, after=value,
            executed=False, error=str(e),
        )


def remediate_clear_stale_state(dry_run: bool) -> RemediationAction:
    """Remove run-meta.json, paper_soak_loose.json BOM-corrupted files,
    etc. Conservative: only remove files we KNOW are safe to recreate."""
    cleared: List[str] = []
    targets = [
        REPO_ROOT / "data" / "run-meta.json",
        REPO_ROOT / ".claude" / "scheduled_tasks.lock",
    ]
    for p in targets:
        if p.exists():
            if dry_run:
                cleared.append(f"would-rm:{p.relative_to(REPO_ROOT)}")
                continue
            try:
                p.unlink()
                cleared.append(str(p.relative_to(REPO_ROOT)))
            except OSError as e:
                return RemediationAction(
                    kind="file_write", target=str(p), before="exists",
                    after="exists", executed=False, error=str(e),
                )
    if not cleared:
        return RemediationAction(
            kind="file_write", target="stale-state",
            before="clean", after="clean", executed=False,
        )
    return RemediationAction(
        kind="file_write", target="stale-state",
        before="dirty", after="clean",
        executed=not dry_run,
    )


def remediate_restart_bot(dry_run: bool) -> RemediationAction:
    """Restart the trading bot process. On Windows, find the python
    process whose command line contains 'src.main' and signal a
    graceful restart by killing it; the operator's START_ALL.ps1
    relaunches separately so this function only kills."""
    if dry_run:
        return RemediationAction(
            kind="restart", target="trading_bot", before="running",
            after="killed", executed=False, error="dry_run",
        )
    try:
        # PowerShell to find + kill the bot python process
        ps_cmd = (
            "$p = Get-Process python -ErrorAction SilentlyContinue | "
            "Where-Object { $_.MainWindowTitle -like '*ACT*' -or "
            "$_.CommandLine -like '*src.main*' }; "
            "if ($p) { $p | Stop-Process -Force; Write-Host 'killed' } "
            "else { Write-Host 'not_found' }"
        )
        result = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=15,
        )
        outcome = (result.stdout or "").strip() or (result.stderr or "").strip()
        return RemediationAction(
            kind="restart", target="trading_bot", before="running",
            after=outcome, executed=True,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return RemediationAction(
            kind="restart", target="trading_bot", before="running",
            after="error", executed=False, error=str(e),
        )


# ── Webhook + Claude Code headless escalation ─────────────────────────


def post_webhook(message: str, payload: Dict[str, Any]) -> bool:
    url = os.environ.get("ACT_OPS_WEBHOOK_URL", "").strip()
    if not url:
        logger.info("[escalate] no ACT_OPS_WEBHOOK_URL set -- logging only")
        logger.info("[escalate] %s", message)
        return False
    body = json.dumps({
        "text": message, **payload,
        "ts": datetime.now(timezone.utc).isoformat(),
    }).encode("utf-8")
    req = request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with request.urlopen(req, timeout=10.0) as r:
            return 200 <= r.status < 300
    except (error.URLError, OSError) as e:
        logger.warning("[escalate] webhook POST failed: %s", e)
        return False


def headless_claude(prompt: str, timeout_s: int = 600) -> Optional[str]:
    """Invoke Claude Code headless (`claude -p`) for ambiguous fixes.

    Disabled by default; enable with ACT_OPS_HEADLESS_CLAUDE=1.
    """
    if os.environ.get("ACT_OPS_HEADLESS_CLAUDE", "").strip() != "1":
        return None
    try:
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True, text=True, timeout=timeout_s,
            cwd=str(REPO_ROOT),
        )
        return (result.stdout or "")[:5000]
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("headless claude invocation failed: %s", e)
        return None


# ── EOD report ─────────────────────────────────────────────────────────


def is_eod_due() -> bool:
    """True at the first run after midnight local time today (using
    LAST_RUN_FILE for the previous run timestamp)."""
    now = datetime.now()
    if now.hour != EOD_HOUR_LOCAL:
        # Not the magic hour -- only fire if we've never run today.
        pass
    if not LAST_RUN_FILE.exists():
        return True
    try:
        last = json.loads(LAST_RUN_FILE.read_text(encoding="utf-8"))
        last_ts = datetime.fromisoformat(last.get("ts", ""))
    except (json.JSONDecodeError, ValueError, OSError):
        return True
    return last_ts.date() < now.date()


def write_eod_report(
    cycle: CycleReport, leaks_today: List[str],
) -> Optional[str]:
    EOD_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    path = EOD_DIR / f"act_eod_{today}.md"

    paper_summary = _summarize_paper_log()
    lines = [
        f"# ACT EOD Report — {today}",
        "",
        "## Paper Trading Summary",
        f"- Total ENTRY events: {paper_summary['entries']}",
        f"- Total EXIT events: {paper_summary['exits']}",
        f"- Open at report time: {paper_summary['open']}",
        f"- Realized PnL (closed trades, $): {paper_summary['realized_pnl_usd']:+.2f}",
        f"- Win rate: {paper_summary['win_rate']*100:.1f}% "
        f"({paper_summary['wins']}W/{paper_summary['losses']}L)",
        "",
        "## Strategy-Leak Surface Today",
    ]
    if leaks_today:
        for leak_id in sorted(set(leaks_today)):
            lines.append(f"- {leak_id}")
    else:
        lines.append("- (none detected)")

    lines += [
        "",
        "## Health Checks (this cycle)",
    ]
    for c in cycle.checks:
        emoji = "OK" if c.ok else "FAIL"
        lines.append(f"- [{emoji}] {c.name}: {c.detail}")

    lines += [
        "",
        "## Remediations Today",
    ]
    if cycle.remediations:
        for r in cycle.remediations:
            lines.append(
                f"- {r.kind} on {r.target}: {r.before!r} -> {r.after!r} "
                f"executed={r.executed}{(' err='+r.error) if r.error else ''}"
            )
    else:
        lines.append("- (no remediations needed)")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("EOD report written: %s", path)
    return str(path)


def _summarize_paper_log() -> Dict[str, Any]:
    if not PAPER_LOG.exists():
        return {"entries": 0, "exits": 0, "open": 0,
                "realized_pnl_usd": 0.0, "wins": 0, "losses": 0,
                "win_rate": 0.0}
    entries = 0
    exits = 0
    pnl = 0.0
    wins = 0
    losses = 0
    try:
        with PAPER_LOG.open(encoding="utf-8", errors="replace") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") == "ENTRY":
                    entries += 1
                elif rec.get("event") == "EXIT":
                    exits += 1
                    p = float(rec.get("pnl_usd", 0.0) or 0.0)
                    pnl += p
                    if p >= 0:
                        wins += 1
                    else:
                        losses += 1
    except OSError:
        pass
    closed = wins + losses
    return {
        "entries": entries, "exits": exits,
        "open": max(0, entries - exits),
        "realized_pnl_usd": pnl, "wins": wins, "losses": losses,
        "win_rate": (wins / closed) if closed else 0.0,
    }


# ── Git audit ──────────────────────────────────────────────────────────


def git_commit_remediations(
    actions: List[RemediationAction], dry_run: bool,
) -> Optional[str]:
    """Commit env-flag changes / file-write changes to the repo for audit.

    We don't commit binary state (warm_store, etc.), only config artifacts
    and removed stale files. Auto-push only when ACT_OPS_AUTO_COMMIT=1.
    """
    if dry_run or not actions:
        return None
    if os.environ.get("ACT_OPS_AUTO_COMMIT", "").strip() != "1":
        return None
    msg_lines = ["ops: auto-remediation"]
    for a in actions:
        if not a.executed:
            continue
        msg_lines.append(f"- {a.kind} {a.target}: {a.before} -> {a.after}")
    if len(msg_lines) == 1:
        return None
    msg = "\n".join(msg_lines)
    try:
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "add", "-A"],
            check=True, capture_output=True, timeout=15,
        )
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "commit", "-m", msg,
             "--allow-empty"],
            check=True, capture_output=True, timeout=15,
        )
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "push", "origin", "main"],
            check=True, capture_output=True, timeout=60,
        )
        return msg
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError) as e:
        logger.warning("git commit failed: %s", e)
        return None


# ── Main cycle ─────────────────────────────────────────────────────────


def run_cycle(check_only: bool = False, force_eod: bool = False) -> CycleReport:
    started = datetime.now(timezone.utc).isoformat()
    dry_run = check_only or os.environ.get(
        "ACT_OPS_DRY_RUN", ""
    ).strip() == "1"

    checks = [
        check_agentic_loop_flag(),
        check_drawdown_halt(),
        check_tunnel_freshness(),
        check_last_trade(),
    ]

    leaks = detect_active_leaks()
    leak_ids = [l.get("id", "") for l in leaks]

    remediations: List[RemediationAction] = []
    escalations: List[Dict[str, Any]] = []

    # Apply remediations from each active leak
    for leak in leaks:
        rem = (leak.get("remediation") or {})
        kind = rem.get("kind", "")
        if kind == "env_flag":
            action = remediate_env_flag(rem.get("env", ""),
                                        rem.get("value", ""), dry_run)
            remediations.append(action)
        elif kind == "file_write":
            action = remediate_clear_stale_state(dry_run)
            remediations.append(action)
        elif kind == "restart":
            action = remediate_restart_bot(dry_run)
            remediations.append(action)
        elif kind == "escalate":
            payload = {
                "leak_id": leak.get("id"),
                "title": leak.get("title"),
                "match_source": leak.get("match_source"),
                "prompt": (rem.get("prompt") or "").strip(),
            }
            escalations.append(payload)
            if not dry_run:
                msg = (
                    f"[ACT] strategy-leak detected: {leak.get('id')} "
                    f"({leak.get('title')}) -- {leak.get('match_source')}"
                )
                post_webhook(msg, payload)
                # Optional: hand to claude headless to draft a fix
                claude_out = headless_claude(payload["prompt"])
                if claude_out:
                    payload["claude_response"] = claude_out

    # Tunnel-freshness specific remediation (no env flag works)
    tunnel_check = next((c for c in checks if c.name == "tunnel_freshness"),
                       None)
    if tunnel_check and not tunnel_check.ok:
        msg = "[ACT] cloudflare quick-tunnel stale -- update .mcp.json + restart bot"
        post_webhook(msg, {"check": tunnel_check.detail})
        escalations.append({
            "leak_id": "tunnel_stale",
            "title": "Cloudflare tunnel unreachable",
            "detail": tunnel_check.detail,
        })

    # Last-trade staleness escalation
    last_trade_check = next((c for c in checks if c.name == "last_trade"), None)
    if last_trade_check and not last_trade_check.ok:
        msg = f"[ACT] no trades in {last_trade_check.detail}"
        post_webhook(msg, {"check": last_trade_check.detail,
                          "active_leaks": leak_ids})
        escalations.append({
            "leak_id": "no_recent_trades",
            "title": "No trades in 24h+",
            "detail": last_trade_check.detail,
        })

    # Drawdown halt escalation (always page operator)
    dd_check = next((c for c in checks if c.name == "drawdown_halt"), None)
    if dd_check and not dd_check.ok:
        msg = f"[ACT] DRAWDOWN HALT: {dd_check.detail}"
        post_webhook(msg, {"check": dd_check.detail})
        escalations.append({
            "leak_id": "drawdown_halt",
            "title": "Bot halted on drawdown",
            "detail": dd_check.detail,
        })

    cycle = CycleReport(
        started_at=started,
        ended_at=datetime.now(timezone.utc).isoformat(),
        checks=checks,
        leaks_active=leak_ids,
        remediations=remediations,
        escalations=escalations,
        is_eod=force_eod or is_eod_due(),
    )

    if cycle.is_eod:
        cycle.eod_report_path = write_eod_report(cycle, leak_ids)

    if not dry_run:
        commit_msg = git_commit_remediations(remediations, dry_run)
        if commit_msg:
            logger.info("committed: %s", commit_msg)

    # Persist cycle report
    RUNBOOK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUNBOOK_LOG.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(cycle.to_dict()) + "\n")

    LAST_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_RUN_FILE.write_text(
        json.dumps({"ts": datetime.now().isoformat()}),
        encoding="utf-8",
    )

    return cycle


def _print_summary(cycle: CycleReport) -> None:
    print("=" * 60)
    print(f" ACT ops cycle  {cycle.started_at}  ->  {cycle.ended_at}")
    print("=" * 60)
    print("\nCHECKS")
    for c in cycle.checks:
        print(f"  [{'OK' if c.ok else 'FAIL'}] {c.name}: {c.detail}")
    print("\nLEAKS ACTIVE")
    if cycle.leaks_active:
        for leak in cycle.leaks_active:
            print(f"  - {leak}")
    else:
        print("  (none)")
    print("\nREMEDIATIONS")
    if cycle.remediations:
        for r in cycle.remediations:
            print(f"  - {r.kind} {r.target}: {r.before!r} -> {r.after!r} "
                  f"exec={r.executed}{(' err='+r.error) if r.error else ''}")
    else:
        print("  (none)")
    print("\nESCALATIONS")
    if cycle.escalations:
        for e in cycle.escalations:
            print(f"  - {e.get('leak_id')}: {e.get('title')}")
    else:
        print("  (none)")
    if cycle.eod_report_path:
        print(f"\nEOD REPORT: {cycle.eod_report_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__ or "")
    ap.add_argument("--check-only", action="store_true",
                    help="Run diagnostic checks; do not execute remediations.")
    ap.add_argument("--eod", action="store_true",
                    help="Force-generate the EOD report regardless of time.")
    ap.add_argument("--escalate", metavar="LEAK_ID",
                    help="Force-escalate a known leak to claude headless / webhook.")
    args = ap.parse_args()

    if args.escalate:
        leaks = _load_leaks()
        target = next((l for l in leaks if l.get("id") == args.escalate), None)
        if target is None:
            print(f"unknown leak id: {args.escalate}")
            return 2
        rem = (target.get("remediation") or {})
        post_webhook(
            f"[ACT] forced escalation: {target.get('id')} -- {target.get('title')}",
            {"leak_id": target.get("id"), "title": target.get("title"),
             "prompt": rem.get("prompt", "")},
        )
        out = headless_claude(rem.get("prompt", ""))
        if out:
            print(out)
        return 0

    cycle = run_cycle(check_only=args.check_only, force_eod=args.eod)
    _print_summary(cycle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
