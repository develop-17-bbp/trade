# Three-peer ACT mesh — setup runbook

This document covers the one-time bring-up of the 3-peer ACT topology:

```
+--------------------+      Tailscale mesh      +--------------------+
| Acer (operator)    | <----------------------> | RTX 5090 (Kirkland)|
| convo user         |                          | crypto live (RH)   |
| code + monitoring  |                          | analyst nightly QLoRA|
+--------------------+                          | scanner_adapter    |
                                                |   watcher          |
                                                +--------------------+
                                                          ^
                                                          | scp + ssh setx
                                                          v
                                                +--------------------+
                                                | RTX 4060 (India)   |
                                                | US stocks live     |
                                                |   (Alpaca paper)   |
                                                | finetune router    |
                                                |   (cross-class)    |
                                                | warm_store_sync    |
                                                +--------------------+
```

## 0. Prerequisites

- Tailscale account (free tier covers 100 devices). Auth via Google/GitHub/Microsoft.
- Both bots running Windows 11 with Admin access.
- Python 3.14 already installed on each box (project pin).
- Code already cloned (this repo) on all three boxes.

## 1. Tailscale install (each box)

**Important:** Tailscale's native SSH server (`--ssh`) is **not supported on Windows**.
Use Tailscale only for the mesh transport; run Windows OpenSSH Server on each Windows
box for the actual SSH daemon. (Confirmed Tailscale 1.96.3 returns
`500 Internal Server Error: The Tailscale SSH server is not supported on windows`.)

Run as Administrator (PowerShell, NOT cmd.exe — `#` lines are comments):

```powershell
# 1. Install Tailscale
winget install --id Tailscale.Tailscale

# 2. Bring up the mesh (no --ssh on Windows)
tailscale up --hostname=<acer | act5090 | act4060> --unattended --accept-routes --accept-dns
# Browser opens; sign in with the same account on all three boxes.

# 3. Verify
tailscale status
```

`--unattended` keeps the daemon up across reboots.
`--accept-routes` lets each box reach private subnets advertised by peers.
`--accept-dns` enables MagicDNS so peers resolve by short name (act5090, act4060).

Tag the bots in the Tailscale admin console:
- `act5090` → tag `tag:act-bot`
- `act4060` → tag `tag:act-bot`
- `acer` → leave untagged (operator)

## 1.5 Windows OpenSSH Server (on each bot — 5090 + 4060)

The Acer doesn't need this; it's the SSH client. The two bots need an SSH daemon.

```powershell
# As Administrator
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Set-Service -Name sshd -StartupType Automatic
Start-Service sshd

# Confirm port 22 is listening
Get-NetTCPConnection -LocalPort 22 -State Listen
```

The Tailscale firewall + ACL controls who can reach port 22, so no extra firewall
config is needed beyond the default Windows rule that OpenSSH installs.

### Generate keys + exchange

On the **Acer** (operator):
```powershell
ssh-keygen -t ed25519 -f $HOME\.ssh\id_ed25519_act -C "acer-operator" -N '""'
Get-Content $HOME\.ssh\id_ed25519_act.pub
```

On the **5090**:
```powershell
ssh-keygen -t ed25519 -f $HOME\.ssh\id_ed25519_act -C "act5090" -N '""'
Get-Content $HOME\.ssh\id_ed25519_act.pub
```

On the **4060**:
```powershell
ssh-keygen -t ed25519 -f $HOME\.ssh\id_ed25519_act -C "act4060" -N '""'
Get-Content $HOME\.ssh\id_ed25519_act.pub
```

Paste each box's public key into the *other* boxes' `C:\ProgramData\ssh\administrators_authorized_keys`:
- 5090's `administrators_authorized_keys` gets: Acer's pubkey + 4060's pubkey
- 4060's `administrators_authorized_keys` gets: Acer's pubkey + 5090's pubkey

Lock file ACL on each bot:
```powershell
icacls C:\ProgramData\ssh\administrators_authorized_keys /inheritance:r
icacls C:\ProgramData\ssh\administrators_authorized_keys /grant "Administrators:F" /grant "SYSTEM:F"
```

`~/.ssh/config` on each box (so `ssh act5090` resolves the right key + Tailscale name):
```
Host act5090
  HostName act5090
  User admin
  IdentityFile ~/.ssh/id_ed25519_act

Host act4060
  HostName act4060
  User admin
  IdentityFile ~/.ssh/id_ed25519_act
```

(The bare hostname works because Tailscale MagicDNS resolves it. Alternatively use
`act5090.<your-tailnet>.ts.net`.)

## 2. ACL policy (Tailscale admin console -> Access Controls)

Since we're using Windows OpenSSH (not Tailscale-native SSH), the `ssh` block doesn't
apply — use a regular `acls` block to allow tag-to-tag traffic on port 22:

```jsonc
{
  "tagOwners": { "tag:act-bot": ["autogroup:admin"] },
  "acls": [
    // Operator (untagged Acer) + bots can SSH the bots on port 22
    { "action": "accept", "src": ["autogroup:admin", "tag:act-bot"], "dst": ["tag:act-bot:22"] }
  ]
}
```

That's it — auth is handled by Windows OpenSSH key files; Tailscale only controls
network-layer reachability.

## 3. Smoke tests

```powershell
# From Acer
ssh act5090 echo ok
ssh act4060 echo ok
# From 4060
ssh act5090 echo ok
# From 5090
ssh act4060 echo ok
```

All four should print `ok`. If any hangs, the daemon hasn't propagated tag/ACL yet —
wait ~30s and retry.

## 4. Per-box configuration

### 4a. RTX 5090 (Kirkland) — keeps doing crypto

Add the new ENV var so the watcher can ssh-back to the 4060 for hot-swaps:

```powershell
setx ACT_4060_SSH_HOST "act4060"
```

Pull, restart:

```powershell
cd C:\Users\admin\trade
git pull
.\STOP_ALL.ps1
.\START_ALL.ps1
```

### 4b. RTX 4060 (India) — runs US stocks live + cross-class finetune

Required env (set once, persists across reboots):

```powershell
# Alpaca paper credentials (sign up at https://alpaca.markets, paper account is free)
setx APCA_API_KEY_ID    "<your-paper-key>"
setx APCA_API_SECRET_KEY "<your-paper-secret>"

# Box role + Tailscale peer aliases
setx ACT_BOX_ROLE       "stocks"
setx ACT_5090_SSH_HOST  "act5090"
setx ACT_5090_TRADE_DIR "C:/Users/admin/trade"
setx ACT_4060_SSH_HOST  "act4060"

# Optional: data feed (defaults to free IEX; pay $99/mo for SIP)
setx ACT_ALPACA_DATA_FEED "iex"

# Optional: finetune priority — stocks (default), crypto, alternate
setx ACT_FINETUNE_PRIORITY "stocks"

# Real-capital flags (DO NOT set until paper soak completes)
# setx ACT_STOCKS_REAL_CAPITAL_ENABLED "1"
# setx ACT_REAL_CAPITAL_ENABLED        "1"
```

Open a fresh PowerShell so setx values are visible.

```powershell
cd C:\Users\CHIST\ACT\trade
git pull
pip install -r requirements.txt
.\START_ALL_4060.ps1
```

5 background processes launch:
- ACT - Stocks Live (4060) — the trading bot, ACT_BOX_ROLE=stocks
- ACT - Finetune Router (4060) — schedules crypto/stocks training around RTH
- ACT - warm_store sync (4060) — parquet deltas to/from 5090 every 60s
- ACT - Silence watchdog (4060) — alerts if no decisions in 30 min

## 5. Verification

### Mesh
```powershell
ssh act5090 echo ok       # from Acer
ssh act4060 echo ok       # from Acer
```

### warm_store schema migrated
```powershell
sqlite3 data/warm_store.sqlite "SELECT name FROM pragma_table_info('decisions') WHERE name='asset_class';"
# expect: asset_class
```

### Stocks fetcher live
```powershell
python -c "from src.data.alpaca_fetcher import AlpacaFetcher; f=AlpacaFetcher(); print(f.health()); print(len(f.fetch_ohlcv('SPY', '1Min', 5)))"
```

### Finetune router decision cycle (dry run)
```powershell
python scripts/finetune_router_4060.py --once --dry-run --verbose
```

### warm_store_sync round trip
```powershell
# On 4060
python -m scripts.warm_store_sync --once --verbose
# On 5090
sqlite3 data/warm_store.sqlite "SELECT COUNT(*) FROM decisions WHERE asset_class='STOCK'"
# expect: > 0 once stocks bot has produced rows
```

### Silence watchdog
```powershell
python scripts/silence_watchdog.py --once --threshold-s 60 --verbose
# expect: CRITICAL alert fired (smoke test)
```

## 6. Operator basket

The 4060 trades exactly 4 symbols selected by the operator on 2026-04-29:

| Symbol | Class    | Role                                         |
|--------|----------|----------------------------------------------|
| SPY    | 1x ETF   | Best ETF overall, tightest spreads           |
| QQQ    | 1x ETF   | Best for higher returns (Nasdaq-100)         |
| TQQQ   | 3x ETF   | Aggressive AI thematic, no overnight allowed |
| SOXL   | 3x ETF   | Aggressive AI thematic, no overnight allowed |

Hard rules baked into `src/ai/authority_rules_stocks.py` + `src/trading/stocks_conviction.py`:
- TQQQ/SOXL: 5% max intraday position, 0% overnight (forced flat-by-EOD).
- TQQQ/SOXL refused inside 30-min pre-close blackout.
- All symbols refused inside 5-min pre-close blackout.
- RTH only (9:30-16:00 ET, weekdays, NYSE holidays excluded; Nov 27 + Dec 24 close at 13:00 ET).
- Daily margin exposure cap 200% (post-2026-04-14 PDT replacement).

## 7. Data feed honesty (paper soak limitation)

Alpaca free tier serves IEX market data only — ~2-8% of consolidated US equity volume.
Bid/ask the bot sees in paper soak does NOT match what live SIP would show.

While `ACT_ALPACA_DATA_FEED=iex` (the default), `stocks_conviction.py` bumps the
sniper-tier minimum-expected-move from 0.15% (SIP) to 0.25% (IEX + skew adder)
to compensate for the optimistic fill model.

Before deploying real capital: subscribe to Alpaca AlgoTrader Plus ($99/mo) for
SIP feed and flip `setx ACT_ALPACA_DATA_FEED sip`. Or use Polygon.io ($29/mo,
1-month delay on the free tier) for survivorship-bias-free historical bars when
training; live forward-tick stays on Alpaca.

## 8. Kill switches

| Switch                                | Effect                                           |
|---------------------------------------|--------------------------------------------------|
| `setx ACT_DISABLE_AGENTIC_LOOP 1`     | both boxes — stop submitting trades              |
| `setx ACT_DISABLE_FINETUNE 1`         | both boxes — finetune router exits next cycle    |
| `STOP_ALL_4060.ps1`                   | 4060 — kill all 5 child processes                |
| `STOP_ALL.ps1`                        | 5090 — kill all background processes             |

Silence watchdog will NOT alert when `ACT_DISABLE_AGENTIC_LOOP=1` is set — silence
is the intended state in that mode.

## 9. Troubleshooting

**`ssh act5090` hangs forever**
Tailscale daemon hasn't authenticated. Run `tailscale status` to confirm peer
visibility; re-run `tailscale up --hostname=...` if needed.

**warm_store_sync says `skipped: no_peer`**
`ACT_5090_SSH_HOST` is unset on the local box (or the env var didn't propagate
into the cmd.exe child process). Open a fresh PowerShell after `setx` and retry.

**`AlpacaExecutor.health()` returns `available: false / no_credentials`**
APCA env vars not visible to the child process. Verify with `[System.Environment]::GetEnvironmentVariable("APCA_API_KEY_ID","User")`.
After setx you must restart the cmd.exe / PowerShell host.

**Stocks bot generates zero decisions**
`/diagnose-noop` on the 4060 will report the cause. Most common during
paper soak:
- IEX min-expected-move floor too aggressive for current vol regime
  (lower temporarily via `ACT_STOCKS_MIN_MOVE_SIP=0.10` to widen the gate).
- Pre-close blackout fired (5-min window before NYSE close).
- TQQQ/SOXL got refused on overnight intent (this is by design — daily-reset decay).

**Watcher promotes a stocks adapter but the 4060 doesn't pick it up**
Cross-box hot-swap requires `ACT_4060_SSH_HOST` set on the 5090. After the
watcher logs `PROMOTED stocks scanner adapter ... on the 4060 (cross-box ssh)`,
verify on the 4060: `[System.Environment]::GetEnvironmentVariable("ACT_STOCKS_SCANNER_MODEL","User")`.
The agentic loop reads it on the next tick — restart the trading bot to force-pick-up.
