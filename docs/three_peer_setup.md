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

Run as Administrator:

```powershell
winget install --id Tailscale.Tailscale
tailscale up --hostname=<acer | act5090 | act4060> --ssh --unattended --accept-routes
```

`--ssh` enables Tailscale-native SSH so we don't need to run Windows OpenSSH Server.
`--unattended` keeps the daemon up across reboots.
`--accept-routes` lets each box reach private subnets advertised by peers.

Tag the bots in the Tailscale admin console:
- `act5090` → tag `tag:act-bot`
- `act4060` → tag `tag:act-bot`
- `acer` → leave untagged (operator)

## 2. ACL policy (Tailscale admin console -> Access Controls)

Paste:

```jsonc
{
  "tagOwners": { "tag:act-bot": ["autogroup:admin"] },
  "ssh": [
    // Operator (untagged) -> both bots, only as 'admin'
    { "action": "accept", "src": ["autogroup:admin"], "dst": ["tag:act-bot"], "users": ["admin"] },
    // Bot <-> bot (5090 <-> 4060) for warm_store sync + adapter shipping
    { "action": "accept", "src": ["tag:act-bot"], "dst": ["tag:act-bot"], "users": ["admin"] }
  ]
}
```

Important: do NOT use `autogroup:nonroot` in `users` when `dst` is a custom tag — that
opens ssh access as ANY non-root account on the destination. Lock to explicit `admin`.

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
