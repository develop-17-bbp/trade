"""
ACT MCP server — exposes runtime artifacts to Claude Code sessions over HTTP.

Transport: Streamable-HTTP (MCP 2024-11 spec). Routed via Cloudflare tunnel
to `mcp.<domain>` and gated by Cloudflare Access. Optional secondary auth via
X-MCP-Token header (belt-and-suspenders).

Run on the GPU box:
    python -m src.mcp_server.act_mcp
    python -m src.mcp_server.act_mcp --port 9100 --host 127.0.0.1

Tools exposed:
    READ-ONLY (always on):
      status                 - one-shot health snapshot
      evaluator_report       - full evaluation JSON
      component_state        - ON/OFF states + toggle commands
      paper_state            - paper equity + trade stats
      shadow_stats           - meta-model shadow log stats
      readiness_gate         - gate evaluation
      tail_log               - last N lines of a named log
      list_logs              - available log files
      recent_trades          - last N completed trades
      git_status             - HEAD, branch, uncommitted files
      env_flags              - ACT_* env state (values redacted for secrets)

    MUTATING (gated by ACT_MCP_ALLOW_MUTATIONS=1):
      restart_bot            - STOP_ALL + START_ALL
      trigger_retrain        - train_all_models for one asset

Auth:
  1. Cloudflare Access (primary) — set up in the CF Zero Trust dashboard.
  2. X-MCP-Token header (secondary) — checked against ACT_MCP_TOKEN env var
     if set. If ACT_MCP_TOKEN is unset, token check is skipped (CF Access
     is the only layer).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

from src.mcp_server import tools as _tools


logger = logging.getLogger("mcp.act")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def build_server():
    """Build and return the FastMCP app. Imported lazily so tests can import
    this module without pulling the whole mcp runtime."""
    from mcp.server.fastmcp import FastMCP
    try:
        # Available in mcp >= 1.x — configures DNS-rebinding Host validation.
        from mcp.server.transport_security import TransportSecuritySettings
    except Exception:
        TransportSecuritySettings = None

    # MCP's DNS-rebinding protection validates the Host header against an
    # allowlist. Default list is empty or too restrictive, so requests from
    # a Cloudflare tunnel (Host: *.trycloudflare.com) or after cloudflared's
    # --http-host-header localhost rewrite get rejected with 421 Misdirected
    # Request. We widen the list because auth is already provided by:
    #   1. Cloudflare Access (or Zero Trust policy) at the edge
    #   2. X-MCP-Token header checked server-side
    # DNS rebinding is a browser-only attack; we're only serving non-browser
    # MCP clients via header-authenticated HTTPS.
    allowed_hosts_env = (os.environ.get("ACT_MCP_ALLOWED_HOSTS") or "").strip()
    if allowed_hosts_env:
        allowed_hosts = [h.strip() for h in allowed_hosts_env.split(",") if h.strip()]
    else:
        allowed_hosts = [
            "*",                        # trust header auth; see note above
            "127.0.0.1:*", "localhost:*",
            "*.trycloudflare.com",      # quick tunnels
        ]
    allowed_origins = [
        "*",                            # same rationale — MCP clients are not browsers
    ]

    if TransportSecuritySettings is not None:
        tx_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=False,  # auth is via X-MCP-Token + CF Access
            allowed_hosts=allowed_hosts,
            allowed_origins=allowed_origins,
        )
        mcp = FastMCP(
            "ACT-GPU",
            instructions=(
                "ACT trading-system inspection server. Read-only by default. "
                "Use `status` for a quick health check; `evaluator_report` for full "
                "component attribution. Logs available via `list_logs` + `tail_log`."
            ),
            transport_security=tx_security,
        )
    else:
        mcp = FastMCP(
            "ACT-GPU",
            instructions=(
                "ACT trading-system inspection server. Read-only by default."
            ),
        )

    # ── Read-only tools ─────────────────────────────────────────────
    @mcp.tool()
    def status() -> dict:
        """One-shot bot health: paper equity, readiness-gate state, rolling Sharpe per asset."""
        return _tools.status()

    @mcp.tool()
    def evaluator_report() -> dict:
        """Full evaluation report — component state, attribution tables, recommendations.

        Same data as `python scripts/evaluate_act.py --json`. Omits the raw
        trades array (use `recent_trades` to pull those).
        """
        return _tools.evaluator_report()

    @mcp.tool()
    def component_state() -> dict:
        """ON/OFF state of every toggleable ACT component with exact setx cmd to flip each."""
        return _tools.component_state()

    @mcp.tool()
    def paper_state() -> dict:
        """Current paper-trading equity, peak, stats, open positions."""
        return _tools.paper_state()

    @mcp.tool()
    def shadow_stats() -> dict:
        """Meta-model shadow log stats — prediction count, joined-trade count, per-asset winrate."""
        return _tools.shadow_stats()

    @mcp.tool()
    def readiness_gate() -> dict:
        """Readiness-gate evaluation — is the bot cleared for real capital?"""
        return _tools.readiness_gate()

    @mcp.tool()
    def tail_log(log_name: str = "autonomous_loop.log", lines: int = 100) -> dict:
        """Return last `lines` lines of a log file under logs/. Max 5000 lines."""
        return _tools.tail_log(log_name=log_name, lines=lines)

    @mcp.tool()
    def list_logs() -> dict:
        """Available log files under logs/ with sizes + modification times."""
        return _tools.list_logs()

    @mcp.tool()
    def recent_trades(limit: int = 20) -> dict:
        """Last N completed paper trades, joined from ENTRY+EXIT events."""
        return _tools.recent_trades(limit=limit)

    @mcp.tool()
    def git_status() -> dict:
        """HEAD, branch, last commit subject, uncommitted-file count + list."""
        return _tools.git_status()

    @mcp.tool()
    def env_flags() -> dict:
        """Current ACT_* env variable state. Secret values (tokens, API keys) redacted."""
        return _tools.env_flags()

    @mcp.tool()
    def audit_log(limit: int = 50) -> dict:
        """Last N MCP tool invocations — timestamp, tool, duration_ms, success.
        Reads from logs/mcp_audit.jsonl which is appended on every tool call."""
        return _tools.audit_log(limit=limit)

    # ── Mutating tools (gated) ──────────────────────────────────────
    @mcp.tool()
    def restart_bot() -> dict:
        """STOP_ALL + START_ALL. GATED by ACT_MCP_ALLOW_MUTATIONS=1."""
        return _tools.restart_bot()

    @mcp.tool()
    def trigger_retrain(asset: str = "BTC") -> dict:
        """Force train_all_models for one asset. GATED by ACT_MCP_ALLOW_MUTATIONS=1."""
        return _tools.trigger_retrain(asset=asset)

    return mcp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("ACT_MCP_PORT", "9100")))
    ap.add_argument("--host", default=os.environ.get("ACT_MCP_HOST", "127.0.0.1"))
    args = ap.parse_args()

    mcp = build_server()
    logger.info(f"[MCP] Starting ACT MCP server on http://{args.host}:{args.port}")
    logger.info(f"[MCP] Mutations allowed: {_tools.mutations_allowed()}")
    logger.info(f"[MCP] Auth token required: {bool(os.environ.get('ACT_MCP_TOKEN'))}")
    # FastMCP's HTTP transport. Route via Cloudflare tunnel to mcp.<domain>.
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.run(transport="streamable-http")
    return 0


if __name__ == "__main__":
    sys.exit(main())
