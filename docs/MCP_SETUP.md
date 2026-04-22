# ACT MCP server — Claude Code remote inspection

Connects any Claude Code session to the ACT bot running on the GPU box.
Once set up, Claude can directly call `evaluator_report()`, `tail_log()`,
`shadow_stats()` etc. instead of asking you to paste terminal output.

## Architecture

```
  Claude Code (your laptop)
        │
        │  HTTPS + Cloudflare Access
        ▼
  mcp.<yourdomain>
        │
        │  (Cloudflare Tunnel)
        ▼
  GPU box: 127.0.0.1:9100
        │
        │  MCP streamable-HTTP transport
        ▼
  src/mcp_server/act_mcp.py
        │
        ├── evaluator_report ── reads logs/ + models/
        ├── tail_log         ── reads logs/*.log
        ├── shadow_stats     ── reads logs/meta_shadow.jsonl
        ├── readiness_gate   ── runs evaluate()
        ├── status           ── one-shot health
        └── restart_bot      ── gated behind ACT_MCP_ALLOW_MUTATIONS=1
```

## One-time setup on the GPU box

### 1. Install the MCP SDK

```powershell
pip install mcp
```

### 2. Generate and persist a server token

```powershell
# Generate a 32-byte random token
$token = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 32 | ForEach-Object {[char]$_})
Write-Host "Token: $token"        # save this somewhere safe

setx ACT_MCP_TOKEN $token
```

Close the terminal, open a new one. The server will now require this token
in the `X-MCP-Token` header for every request (secondary to Cloudflare Access).

### 3. Start the MCP server

```powershell
cd C:\Users\admin\trade
powershell -ExecutionPolicy Bypass -File .\scripts\start_mcp.ps1
```

Leave this window open. First startup prints:

```
[MCP] Starting ACT MCP server on http://127.0.0.1:9100
[MCP] Mutations allowed: False
[MCP] Auth token required: True
```

### 4. Configure Cloudflare tunnel ingress

In `infra/cloudflared/config.yml` (the generated one, not the template),
ensure the `mcp.<yourdomain>` ingress rule is present. `setup_tunnel.ps1`
adds it automatically on fresh setups; for existing tunnels add manually:

```yaml
ingress:
  - hostname: mcp.yourdomain.com
    service: http://localhost:9100
    originRequest:
      noTLSVerify: true
```

Then DNS-route:
```powershell
cloudflared tunnel route dns act-gpu mcp.yourdomain.com
```

Restart the tunnel service:
```powershell
Restart-Service cloudflared
```

### 5. Zero Trust Access policy

Browser → https://one.dash.cloudflare.com/ → Access → Applications → Add → Self-hosted

- **Application domain**: `mcp.yourdomain.com`
- **Session duration**: 24 hours
- **Policy**: Allow, Include = Emails in { your email }

### 6. Create a service token (for Claude Code non-interactive auth)

Access → Service Auth → Service Tokens → Create
- Name: "Claude Code MCP"
- Duration: forever (or rotate annually)

Save the `CF-Access-Client-Id` and `CF-Access-Client-Secret`.

Go back to your Access application for `mcp.yourdomain.com` and add a second
policy: **Bypass** (no identity check), **Include** = Service Token =
"Claude Code MCP". This lets Claude Code authenticate programmatically
without the email-OTP flow.

## Claude Code MCP config

On the machine running Claude Code, edit `~/.claude/mcp.json` (create if missing):

```json
{
  "mcpServers": {
    "act-gpu": {
      "transport": "http",
      "url": "https://mcp.yourdomain.com/mcp",
      "headers": {
        "CF-Access-Client-Id": "<service-token-client-id>",
        "CF-Access-Client-Secret": "<service-token-secret>",
        "X-MCP-Token": "<the-token-from-step-2>"
      }
    }
  }
}
```

Restart Claude Code. New sessions will discover 13 tools under the `act-gpu`
namespace (e.g. `mcp__act-gpu__status`, `mcp__act-gpu__evaluator_report`).

## Verify end-to-end

From the GPU box itself, quick smoke:
```powershell
curl http://127.0.0.1:9100/health
```

From your laptop through the tunnel:
```bash
curl -H "CF-Access-Client-Id: ..." \
     -H "CF-Access-Client-Secret: ..." \
     -H "X-MCP-Token: ..." \
     https://mcp.yourdomain.com/mcp
```

Inside Claude Code, ask:
> Call `status` on the act-gpu MCP server.

You should get JSON back immediately without typing any command on the GPU box.

## Tools reference

### Read-only (always available)

| Tool | Returns |
|---|---|
| `status` | paper equity + stats + readiness gate + safe-entries rolling Sharpe — one-shot health |
| `evaluator_report` | Full evaluation JSON (component state, bucket attribution, recommendations) |
| `component_state` | ON/OFF of every toggleable subsystem + exact setx cmd to flip each |
| `paper_state` | Paper-trading state JSON |
| `shadow_stats` | Meta-model shadow log stats |
| `readiness_gate` | Gate evaluation (open/closed + reasons + numbers) |
| `tail_log(log_name, lines)` | Last N lines of a named log under logs/ |
| `list_logs` | Available log files with sizes |
| `recent_trades(limit)` | Last N completed trades joined from ENTRY + EXIT events |
| `git_status` | HEAD, branch, last commit, uncommitted files |
| `env_flags` | ACT_* env state, secrets redacted |

### Mutating (gated by `ACT_MCP_ALLOW_MUTATIONS=1`)

| Tool | Effect |
|---|---|
| `restart_bot` | Runs STOP_ALL.ps1 then START_ALL.ps1 |
| `trigger_retrain(asset)` | Runs train_all_models.py --asset <X> --bars 20000 |

## Security model

Three layers stacked:

1. **Cloudflare Access** — all traffic to `mcp.<domain>` must pass a CF Access
   policy. Browser visitors get email-OTP; Claude Code uses a service token.
2. **X-MCP-Token header** — if `ACT_MCP_TOKEN` is set on the GPU box, the
   MCP server rejects any request whose header doesn't match. Belt-and-
   suspenders for the case where CF Access is somehow bypassed.
3. **Mutation gate** — `ACT_MCP_ALLOW_MUTATIONS=1` is required before any
   tool that can change bot state will run. Off by default. The server's
   startup log prints the current state of this flag so the operator can
   verify at boot.

What's deliberately NOT exposed:
- Arbitrary shell exec
- File write outside `logs/`
- Env-var setters (keep `setx` manual so every change is reviewable)
- Order placement / live-trading actions

## Rollback

Stop the MCP server by closing its PowerShell window. The tunnel entry stays
configured; a restart of the server brings it back with zero operator
intervention needed.

To permanently disable the tunnel route:
```powershell
cloudflared tunnel route dns act-gpu mcp.yourdomain.com --delete
# + remove the mcp.* ingress block from infra/cloudflared/config.yml
Restart-Service cloudflared
```
