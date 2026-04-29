# Two-Box Parallel Fine-Tuning Setup

ACT trains two brains via QLoRA: **analyst** (`qwen3-coder:30b`, ~24 GB
VRAM) and **scanner** (`qwen2.5-coder:7b`, ~7 GB VRAM). With one GPU box
this happens sequentially in a single nightly window. With two GPU
boxes — RTX 5090 (Kirkland, runs the live bot) and RTX 4060 (India,
runs scanner training in parallel) — we get ~10× more learning cycles
per week.

This doc covers the one-time setup. Run-time orchestration is in:

- `scripts/finetune_scanner_4060.py` — 4060 trainer loop
- `scripts/start_scanner_finetune_4060.ps1` — 4060 launcher
- `scripts/nightly_analyst_finetune.py` — 5090 nightly analyst (auto-started by START_ALL Process 9)
- `scripts/scanner_adapter_watcher.py` — 5090 watcher (auto-started by START_ALL Process 10)

---

## Architecture (cross-continent)

```
┌────────────────────────────┐                    ┌────────────────────────────┐
│  RTX 4060  (India)         │                    │  RTX 5090  (Kirkland)      │
│  Workstation               │                    │  Entity4 — runs live bot   │
│  ──────────────────        │                    │  ──────────────────         │
│                            │                    │                             │
│  every 3h:                 │  scp warm_store    │  Process 9 (nightly):       │
│  finetune_scanner_4060.py  │ ◄─────────────────  │  03:00 UTC analyst QLoRA   │
│   1. pull warm_store       │                    │  → champion gate            │
│   2. LoRA train scanner    │  scp adapter.tar   │  → hot-swap on promote      │
│   3. tar + scp adapter     │ ──────────────────► │                             │
│   4. ssh: drop .ready mark │                    │  Process 10 (watcher):      │
│                            │  ssh New-Item      │  60s poll for .ready        │
│  Output: LoRA delta only   │ ──────────────────► │   → merge LoRA → GGUF      │
│  (~150 MB / cycle)         │                    │   → ollama create           │
│                            │                    │   → champion gate           │
│                            │                    │   → hot-swap on promote     │
└────────────────────────────┘                    └────────────────────────────┘
                                                          │
                                                  cloudflared tunnel (existing)
                                                          │
                                                  SSH ingress (new — see §1)
```

Both boxes have the same git repo. SSH transport rides the Cloudflared
tunnel that the 5090 already runs (Process 7), with a new SSH ingress.
Operator can swap to Tailscale or direct port-forward — only the
SSH-config alias on the 4060 changes.

---

## 1. RTX 5090 (Kirkland) — one-time setup

Run all of these from a Parsec session into the 5090 (or directly at
the keyboard).

### 1.1 Enable Windows OpenSSH Server

```powershell
# Run as Administrator
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Set-Service -Name sshd -StartupType Automatic
Start-Service sshd

# Lock down inbound to LAN-only — Cloudflared brokers public access.
New-NetFirewallRule -Name "OpenSSH-Server-In-LAN" -DisplayName "OpenSSH Server (LAN)" `
    -Direction Inbound -Protocol TCP -LocalPort 22 -Action Allow `
    -RemoteAddress LocalSubnet
```

Test locally: `ssh admin@localhost echo ok` (should print `ok`).

### 1.2 Add SSH ingress to cloudflared

Edit `~/.cloudflared/config.yml` (or `infra\cloudflared\config.yml` if
that's where the named tunnel config lives):

```yaml
tunnel: <existing tunnel UUID>
credentials-file: <existing path>

ingress:
  # Existing FastAPI ingress — keep
  - hostname: act-trade.<your-domain>
    service: http://localhost:8080

  # NEW — SSH ingress
  - hostname: ssh.act-trade.<your-domain>
    service: ssh://localhost:22

  - service: http_status:404
```

Apply:

```powershell
# If running cloudflared as a Windows service:
Restart-Service cloudflared

# If running via START_ALL Process 7:
.\STOP_ALL.ps1
.\START_ALL.ps1
```

Add a CNAME record in your Cloudflare DNS dashboard:
`ssh.act-trade.<your-domain>` → `<tunnel-UUID>.cfargotunnel.com`.

### 1.3 Add the 4060's SSH key (after generating it in §2.1)

Once the 4060 emits its public key, paste it (single line) into:

```
C:\ProgramData\ssh\administrators_authorized_keys
```

Restrict the key to safe operations (recommended — even if the 4060 key
leaks, attacker can only run scp + a tiny marker-touch command):

```
# Single line; replace AAAAC3... with the 4060's public key.
command="powershell.exe -NoProfile -File C:\Users\admin\trade\scripts\ssh_4060_wrapper.ps1",no-port-forwarding,no-X11-forwarding,no-agent-forwarding ssh-ed25519 AAAAC3...4060-india-finetune
```

(For the simplest non-restricted setup, drop the `command=` clause and
the key gets full SSH access — fine if you trust the 4060.)

Permissions on this file matter — Windows requires Administrators-only:

```powershell
icacls C:\ProgramData\ssh\administrators_authorized_keys /inheritance:r
icacls C:\ProgramData\ssh\administrators_authorized_keys /grant "Administrators:F" /grant "SYSTEM:F"
```

### 1.4 Install Unsloth (Python QLoRA dependency)

Inside your project venv:

```powershell
pip install unsloth
pip install --no-deps xformers trl peft accelerate bitsandbytes
```

Verify:

```powershell
python -c "from src.ai.unsloth_backend import _require_unsloth; _require_unsloth(); print('unsloth ok')"
```

### 1.5 Pull and run

```powershell
cd C:\Users\admin\trade
git pull
.\STOP_ALL.ps1
.\START_ALL.ps1
```

START_ALL now launches 10 processes. Process 9 = nightly analyst,
Process 10 = scanner adapter watcher. Both honor `ACT_DISABLE_FINETUNE=1`
as a global kill switch.

---

## 2. RTX 4060 (India) — one-time setup

### 2.1 Generate SSH key + cloudflared client

```powershell
# Generate ed25519 key (no passphrase for unattended use)
ssh-keygen -t ed25519 -f $HOME\.ssh\id_ed25519_act -C "4060-india-finetune" -N '""'

# Print the public key to paste into the 5090's authorized_keys
Get-Content $HOME\.ssh\id_ed25519_act.pub
```

Install cloudflared:

```powershell
# Use winget or download from cloudflare.com/products/tunnel
winget install --id Cloudflare.cloudflared
```

Authenticate (opens a browser to your Cloudflare account):

```powershell
cloudflared access login ssh.act-trade.<your-domain>
```

### 2.2 SSH config

Edit `$HOME\.ssh\config` (create if missing):

```
Host act5090
  HostName ssh.act-trade.<your-domain>
  User admin
  IdentityFile ~/.ssh/id_ed25519_act
  ProxyCommand cloudflared access ssh --hostname %h
  ServerAliveInterval 60
  ServerAliveCountMax 3
```

Test:

```powershell
ssh act5090 echo ok
```

Should print `ok`. If it hangs, cloudflared isn't authenticated; rerun
the `cloudflared access login` step. If it prompts for a password,
your public key isn't in the 5090's `administrators_authorized_keys`.

### 2.3 Persistent env vars

```powershell
setx ACT_5090_SSH_HOST "act5090"
setx ACT_5090_TRADE_DIR "C:/Users/admin/trade"
# Optional overrides:
# setx ACT_SCANNER_FINETUNE_INTERVAL_H "3"
# setx ACT_SCANNER_BASE_MODEL "qwen2.5-coder:7b"
# setx ACT_SCANNER_MIN_NEW_SAMPLES "30"
# setx ACT_SCANNER_MIN_VRAM_MB "7000"
```

Open a fresh PowerShell window so the `setx` values are visible to
subsequent commands.

### 2.4 Install Python deps

```powershell
git clone https://github.com/develop-17-bbp/trade.git C:\Users\CHIST\ACT\trade
cd C:\Users\CHIST\ACT\trade

# (or if you already cloned into an existing dir:)
# git checkout main

pip install -r requirements.txt
pip install unsloth
pip install --no-deps xformers trl peft accelerate bitsandbytes
```

Verify Unsloth + GPU:

```powershell
python -c "import unsloth, torch; print('unsloth', unsloth.__version__); print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 2.5 Smoke test

```powershell
# Stub backend, no GPU work — validates SSH path + corpus pull
.\scripts\start_scanner_finetune_4060.ps1 -DryRun -Once

# Real cycle, single iteration — needs Unsloth + 7+ GB free VRAM
.\scripts\start_scanner_finetune_4060.ps1 -Once
```

If the smoke test succeeds, run unattended:

```powershell
.\scripts\start_scanner_finetune_4060.ps1
```

This runs forever, training every `ACT_SCANNER_FINETUNE_INTERVAL_H`
hours (default 3). Logs go to `logs/fine_tune/scanner_4060.log` and
per-cycle JSON to `logs/fine_tune/<date>_scanner_4060.json`.

---

## 3. Verification (full end-to-end)

Sequence:

1. **4060** runs `start_scanner_finetune_4060.ps1 -Once`.
2. 4060 pulls `warm_store.sqlite` over SSH.
3. 4060 trains LoRA, exports to `models/unsloth_adapters/scanner-act-<ts>/`.
4. 4060 tars + scp the dir to the 5090's `models/unsloth_adapters/`.
5. 4060 sshes the 5090 and creates `scanner-act-<ts>.ready` marker.
6. **5090** Process 10 (watcher) sees the marker within 60s.
7. 5090 merges LoRA → Q4_K_M GGUF → `ollama create scanner-act-<ts>`.
8. 5090 runs `champion_gate.evaluate_gate('scanner', incumbent, scanner-act-<ts>, val)`.
9. If `promote=True` → `_hot_swap` sets `ACT_SCANNER_MODEL` env; bot reads next tick.
10. Marker deleted; rejected adapter renamed `<tag>-rejected`.

Watch progress on the 5090:

```powershell
Get-Content -Wait .\logs\fine_tune\watcher_5090.log
Get-Content -Wait .\logs\fine_tune\analyst_5090.log
```

Audit trail (per-cycle JSON):

```powershell
Get-ChildItem .\logs\fine_tune\*.json | Sort LastWriteTime | Select -Last 5
```

---

## 4. Kill switches + safety

| Switch | Effect |
|---|---|
| `setx ACT_DISABLE_FINETUNE 1` | Both boxes skip cycles silently. |
| 4060: stop the launcher | Loop exits cleanly; lock file removed. |
| 5090: STOP_ALL | Processes 9 + 10 die with the rest. |
| `setx ACT_SCANNER_MODEL <previous-tag>` | Force scanner to a previous incumbent. |
| `setx ACT_ANALYST_MODEL <previous-tag>` | Force analyst to a previous incumbent. |

Champion gate still has the final say — even if a cycle fires, an
adapter only swaps in when `direction_agreement` improves ≥ 2% AND no
metric regresses > 5%. Rejected adapters are renamed `<tag>-rejected`
and kept on disk for audit; they never ship.

---

## 5. Troubleshooting

**`ssh act5090` hangs forever**
Cloudflared not authenticated. Run `cloudflared access login ssh.act-trade.<your-domain>` again.

**`ssh act5090` prompts for password**
The 4060's public key isn't in the 5090's
`C:\ProgramData\ssh\administrators_authorized_keys`. Re-paste, fix
file permissions per §1.3.

**`pip install unsloth` fails on Windows**
Unsloth strongly prefers Linux. On Windows: try `pip install
unsloth-zoo` (a community Windows-friendly fork), or set up WSL2 with
CUDA passthrough. Alternative: skip Unsloth entirely and run
`StubBackend` (set `ACT_FINETUNE_BACKEND=stub` env) which trains nothing
but exercises the rest of the pipeline.

**4060 cycle aborts with "free VRAM N MB < threshold"**
Other apps eating VRAM. Close Chrome, Discord, OBS. Or lower the
threshold: `setx ACT_SCANNER_MIN_VRAM_MB 6000`.

**5090 watcher logs "merge_or_register_failed"**
Unsloth couldn't load the base model. Verify `ollama list` shows
`qwen2.5-coder:7b` and `pip show unsloth` returns a version. The
adapter dir is renamed `-failed` so re-poll won't loop on it.

**Champion gate keeps rejecting**
Either the adapter is genuinely worse OR the validation set is too
small. Check `logs/fine_tune/<date>_scanner_<tag>.json` →
`audit.gate.metrics_delta`. If `direction_agreement` improvement is
< 2%, the rejection is correct. Increase training samples by lowering
`ACT_SCANNER_MIN_NEW_SAMPLES` or wait for more trade data.

---

## 6. What this gives you

CLAUDE.md §13 Phase 2 forecast: "+0.1 to +0.3 Sharpe, translating to
~0.18-0.22%/day average."

With this two-box deploy:
- **Scanner cycles ~ 8× / day** (every 3h on 4060)
- **Analyst cycles ~ 1× / day** (nightly on 5090)

That's ~10× more learning iterations per week than today's manual
single-box flow. Champion gate is the only path to behavior change —
no rule was relaxed, no safety guard removed.
