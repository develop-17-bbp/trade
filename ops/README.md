# ACT Ops Agent — autonomous trading-bot operations

Self-healing layer that runs every 15 minutes, performs the routine
"is the bot trading?" checks, auto-remediates known issues from the
strategy-leak runbook, and pages the operator only when human
judgment is required.

## What it checks

| Check | What pages on | Auto-fix? |
|---|---|---|
| `ACT_AGENTIC_LOOP` flag | not 1 OR disable=1 | sets to 1 via `setx` |
| Drawdown halt | warm_store checkpoint shows `emergency_mode: true` | escalates (operator decides) |
| Tunnel freshness | `.mcp.json` URL HTTP HEAD fails | escalates (URL rotation needs operator) |
| Last trade timestamp | >24h since last ENTRY/EXIT | escalates with active-leak list |

## Strategy-leak runbook (`strategy_leaks.yaml`)

11 known leaks we've fixed before, each with:
- `signature.log_pattern` — regex over last 500 lines of `system_output.log`
- `signature.state_check` — python expression over current env / cost_gate state
- `remediation.kind` — one of `env_flag` / `file_write` / `restart` / `escalate`

Every cycle the agent walks the runbook, detects active leaks, applies
deterministic remediations, and escalates ambiguous ones via webhook
(+ optional `claude -p` headless to draft a one-time fix).

## Install

### Cron schedule (Windows Task Scheduler — every 15 min)

```cmd
schtasks /Create ^
  /TN "ACT Ops Agent" ^
  /TR "powershell -ExecutionPolicy Bypass -File C:\Users\admin\trade\ops\run_ops.ps1" ^
  /SC MINUTE /MO 15 ^
  /RU "%USERNAME%" ^
  /F
```

### Cron schedule (Linux)

```cron
*/15 * * * * cd /path/to/trade && python -m ops.act_ops >> logs/act_ops_cron.log 2>&1
```

### Webhook (optional — recommended)

Set `ACT_OPS_WEBHOOK_URL` to a Slack/Discord/Pagerduty incoming webhook so
escalations actually reach you. Without it, escalations log to
`logs/act_ops.jsonl` only.

```cmd
setx ACT_OPS_WEBHOOK_URL "https://hooks.slack.com/services/..."
```

### Headless Claude (optional)

If `claude` CLI is on PATH and `ACT_OPS_HEADLESS_CLAUDE=1`, the agent
will invoke `claude -p "<leak prompt>"` for ambiguous leaks and
include the response in the escalation payload. Useful for "the bot
isn't trading and I don't know why" cases.

```cmd
setx ACT_OPS_HEADLESS_CLAUDE 1
```

### Auto-commit remediations

To have the agent push every successful remediation to git for audit:

```cmd
setx ACT_OPS_AUTO_COMMIT 1
```

Otherwise it logs to `logs/act_ops.jsonl` only (recommended for first
days; flip to 1 once you trust it).

## Manual usage

```cmd
REM diagnostic check (no writes)
python -m ops.act_ops --check-only

REM force EOD report regardless of time
python -m ops.act_ops --eod

REM force-escalate a specific leak (opens claude -p prompt + webhook)
python -m ops.act_ops --escalate L10
```

## What it produces

| Output | Path |
|---|---|
| Per-cycle structured audit | `logs/act_ops.jsonl` |
| Cron run output | `logs/act_ops_cron.log` |
| Daily EOD reports | `logs/eod/act_eod_YYYY-MM-DD.md` |

Each EOD report contains:
- Paper-trading P&L (realized $, win rate, open positions)
- Strategy-leak surface for the day (which leak ids fired)
- All health-check results from the most recent cycle
- All remediations attempted today

## Operator quick reference

| Symptom | Run | Likely action |
|---|---|---|
| "bot isn't trading" | `--check-only` | look at `last_trade` + active leaks |
| "leak X is firing repeatedly" | `--escalate <ID>` | dispatch claude headless to draft fix |
| "what happened today?" | `--eod` | generate today's report |
| "stale tunnel" | manual | regen cloudflared, update `.mcp.json` |

## Why this design

The /insights report from this user's prior sessions flagged that
diagnose -> ship-fix -> "but did it actually run on the deploy box?"
loops were the dominant friction. This agent closes that loop
without requiring an operator-side script per fix:

* **Routine work runs in pure Python** — fast, deterministic, audited.
* **Ambiguous work escalates to Claude Code headless** — the same
  judgment the operator was getting via interactive sessions, but
  unattended.
* **Webhook-only paging** — the operator gets pinged ONLY when the
  agent can't fix it deterministically.
* **Strategy-leak runbook is data, not code** — adding a new leak
  is a YAML edit, not a code change.
