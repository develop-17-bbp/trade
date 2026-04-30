# ===============================================================
#  ACT 4060 (India) - US stocks live + cross-class finetune
#  Runs as a peer of the 5090 over Tailscale.
#  Run: powershell -ExecutionPolicy Bypass -File START_ALL_4060.ps1
# ===============================================================
# Operator basket: SPY / QQQ (1x ETFs) + TQQQ / SOXL (3x leveraged).
# The 4060 trades stocks during NYSE RTH (~6.5h/day) and uses the
# remaining ~17.5h/day to alternate crypto-scanner and stocks-scanner
# QLoRA finetune cycles (router decides which based on market hours).
#
# Required env (set via setx, see docs/three_peer_setup.md):
#   APCA_API_KEY_ID            Alpaca paper key
#   APCA_API_SECRET_KEY        Alpaca paper secret
#   ACT_BOX_ROLE=stocks        flips config.yaml exchanges[name=alpaca].enabled=true
#   ACT_5090_SSH_HOST=act5090  Tailscale alias for the 5090 (warm_store sync target)
#   ACT_5090_TRADE_DIR         remote repo root e.g. C:/Users/admin/trade
#   ACT_4060_SSH_HOST=act4060  this box's tailnet name (5090 ssh-back for hot-swap)
# Optional:
#   ACT_ALPACA_DATA_FEED       'iex' (default, free) or 'sip' (paid plan)
#   ACT_FINETUNE_PRIORITY      'stocks' (default) | 'crypto' | 'alternate'
#   ACT_DISABLE_AGENTIC_LOOP=1 kill switch for live trading
#   ACT_DISABLE_FINETUNE=1     kill switch for finetune router
#
# Auto-set defaults (this script populates them if unset):
#   OLLAMA_REMOTE_URL    = http://act5090:11434     analyst served on 5090
#   OLLAMA_REMOTE_MODEL  = qwen3-coder:30b          which model lives remote
#   ACT_BRAIN_PROFILE    = moe_agentic              7b local + 30b remote
# The 4060 has 8GB VRAM and CANNOT host the 30b analyst locally. Pointing
# it at the 5090 over Tailscale is the only way the moe_agentic profile
# fits. The 7b scanner stays local (fits in 8GB). Operator can override
# any of these by setting the env var manually before launching this
# script.

$Host.UI.RawUI.BackgroundColor = [System.ConsoleColor]::Black
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repo = $PSScriptRoot
Set-Location $repo

# --- Pre-flight checks -------------------------------------------------

function Fail([string]$msg) {
    Write-Host "[FAIL] $msg" -ForegroundColor Red
    exit 1
}

function Ok([string]$msg) {
    Write-Host "[ OK ] $msg" -ForegroundColor Green
}

function Warn([string]$msg) {
    Write-Host "[WARN] $msg" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==============================================================="
Write-Host " ACT 4060 (India) - US stocks + cross-class finetune"
Write-Host "==============================================================="

# 1. Python
try {
    $pyv = (& python --version 2>&1)
    Ok "python: $pyv"
} catch {
    Fail "python not found on PATH"
}

# 2. Alpaca creds
if (-not $env:APCA_API_KEY_ID) {
    Warn "APCA_API_KEY_ID not set - paper trading will fail. setx APCA_API_KEY_ID '...'"
} else {
    Ok "APCA_API_KEY_ID configured"
}

# 3. Tailscale SSH to the 5090
if (-not $env:ACT_5090_SSH_HOST) {
    Warn "ACT_5090_SSH_HOST not set - warm_store_sync will skip cycles"
} else {
    & ssh -o BatchMode=yes -o ConnectTimeout=5 $env:ACT_5090_SSH_HOST 'echo ok' 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { Ok "ssh $($env:ACT_5090_SSH_HOST) reachable" }
    else { Warn "ssh $($env:ACT_5090_SSH_HOST) NOT reachable - warm_store_sync will skip" }
}

# Persist env vars at BOTH process scope (immediate) AND user scope (setx,
# survives shell exit). Process-only writes were biting operators when
# Start-Process'd children somehow didn't inherit the parent shell env in
# certain Windows configurations - dual write guarantees the var lands.
# `try { setx ... }` because setx errors are non-fatal here.
function _SetEnvDual([string]$name, [string]$value) {
    Set-Item -Path ("env:" + $name) -Value $value
    try { setx $name $value | Out-Null } catch {}
}

# 4. Box role
if (-not $env:ACT_BOX_ROLE) {
    Warn "ACT_BOX_ROLE not set - defaulting to 'stocks' for the 4060"
    _SetEnvDual "ACT_BOX_ROLE" "stocks"
}

# 4b. Tailscale-routed analyst (THE 30B LIVES ON THE 5090, NOT HERE)
# The 4060 has 8GB VRAM and cannot host qwen3-coder:30b locally - that
# would OOM on first inference. Default the remote-analyst env so the
# dual_brain router calls the 5090 over Tailscale for analyst inference,
# while the 7b scanner stays local. Operator overrides win.
if (-not $env:OLLAMA_REMOTE_URL) {
    _SetEnvDual "OLLAMA_REMOTE_URL" "http://act5090:11434"
    Ok "OLLAMA_REMOTE_URL defaulted to http://act5090:11434 (Tailscale, persisted via setx)"
} else {
    Ok "OLLAMA_REMOTE_URL preset by operator: $env:OLLAMA_REMOTE_URL"
}
if (-not $env:OLLAMA_REMOTE_MODEL) {
    _SetEnvDual "OLLAMA_REMOTE_MODEL" "qwen3-coder:30b"
    Ok "OLLAMA_REMOTE_MODEL defaulted to qwen3-coder:30b (analyst, persisted via setx)"
} else {
    Ok "OLLAMA_REMOTE_MODEL preset by operator: $env:OLLAMA_REMOTE_MODEL"
}
if (-not $env:ACT_BRAIN_PROFILE) {
    _SetEnvDual "ACT_BRAIN_PROFILE" "moe_agentic"
    Ok "ACT_BRAIN_PROFILE defaulted to moe_agentic (7b local + 30b remote, persisted via setx)"
}
# Mirror the 5090's START_ALL.ps1 default - ACT_AGENTIC_LOOP=1 makes
# the agentic shadow loop fire on every per-asset tick. Without this
# env (and without config.yaml having the flag enabled), the executor's
# _run_agentic_shadow() returns early at agentic_loop_enabled()=False
# and zero LLM-driven trades fire on stocks/crypto. Operator can
# disable via ACT_DISABLE_AGENTIC_LOOP=1 (broader kill).
if (-not $env:ACT_AGENTIC_LOOP) {
    _SetEnvDual "ACT_AGENTIC_LOOP" "1"
    Ok "ACT_AGENTIC_LOOP=1 enabled (autonomous LLM trades on stocks + crypto)"
}
# LLM-sole-author: technical-lane (_evaluate_entry) becomes pure vote
# inputs to the LLM. The LLM compiles the only TradePlans that fire
# orders via submit_trade_plan. Operator directive 2026-04-30:
# all classical agents/math/genetics help the LLM, not parallel writers.
if (-not $env:ACT_LLM_SOLE_AUTHOR) {
    _SetEnvDual "ACT_LLM_SOLE_AUTHOR" "1"
    Ok "ACT_LLM_SOLE_AUTHOR=1 enabled (LLM is sole order author; agents/math/genetics feed it)"
}
# Match the 5090's bumped 16384 default — same reasoning as START_ALL.ps1:
# 8192 was too tight, prompts got truncated to 500 chars, agentic loop
# returned parse_failures. 16K gives the analyst real prompt budget.
if (-not $env:OLLAMA_NUM_CTX) {
    _SetEnvDual "OLLAMA_NUM_CTX" "16384"
    Ok "OLLAMA_NUM_CTX defaulted to 16384 (was 8192 - was truncating analyst prompts)"
}

# 4c. Probe the remote analyst before launch so silence has a stated reason
# Warn-only: if 5090 is unreachable, the bot still boots into shadow mode
# (LLM-GATE catches it). Hard-fail would be too brittle if 5090 reboots.
$probeUrl = $env:OLLAMA_REMOTE_URL
$probeModel = $env:OLLAMA_REMOTE_MODEL
try {
    $resp = Invoke-RestMethod -Uri "$probeUrl/api/tags" -TimeoutSec 5 -ErrorAction Stop
    $hasModel = $resp.models | Where-Object { $_.name -like "$probeModel*" }
    if ($hasModel) {
        Ok "remote analyst reachable: $probeModel @ $probeUrl"
    } else {
        Warn "remote $probeUrl reachable but $probeModel NOT pulled there - run 'ollama pull $probeModel' on the 5090"
    }
} catch {
    Warn "remote analyst unreachable at $probeUrl - check Tailscale + 5090 firewall (TCP 11434) + OLLAMA_HOST=0.0.0.0:11434 on the 5090"
}

# 4d. Local scanner pre-pull
# When the remote analyst is unreachable mid-day, src/main.py LLM-GATE
# auto-falls-back to the local scanner playing both roles (qwen2.5-coder:7b
# by default - fits in 8 GB VRAM). That fallback REQUIRES the scanner to
# actually be pulled locally. Without this pre-flight, a 4060 boot with a
# brand-new ollama install would see deepseek auto-downgrade fire instead.
# The list below is ranked: configured scanner first, then small fallbacks
# in order of "good enough for both roles in 8 GB". Pull whichever is
# missing AND likely to be needed.
# BOTH the local scanner AND the small backup analyst get pre-pulled.
# Without the backup pulled, LLM-GATE's tier-2 fallback can't fire when
# the remote 30b is unreachable, and the bot collapses to scanner-as-both
# (tier 3, single brain). Pulling llama3.2:3b ahead of time keeps the
# dual-brain dual even during 5090 outages. ~2 GB download; fits in
# 8 GB VRAM alongside qwen2.5-coder:7b (~5 GB).
$localScanner = if ($env:ACT_SCANNER_MODEL) { $env:ACT_SCANNER_MODEL } else { "qwen2.5-coder:7b" }
$backupAnalyst = if ($env:ACT_ANALYST_BACKUP_MODEL) { $env:ACT_ANALYST_BACKUP_MODEL } else { "llama3.2:3b" }
$localPullChain = @($localScanner, $backupAnalyst)
try {
    $localTags = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 5 -ErrorAction Stop
    $localModels = @($localTags.models | ForEach-Object { $_.name })
    foreach ($m in $localPullChain) {
        $present = $localModels | Where-Object { $_ -like "$m*" }
        if ($present) {
            Ok "local model present: $m"
        } else {
            $role = if ($m -eq $localScanner) { "scanner" } else { "backup analyst (LLM-GATE tier 2)" }
            Warn "local $role $m NOT pulled - pulling now"
            & ollama pull $m
            if ($LASTEXITCODE -eq 0) { Ok "pulled $m ($role)" } else { Warn "ollama pull $m failed - if this is the scanner, bot may boot in legacy-voter mode when remote also fails" }
        }
    }
} catch {
    Warn "local Ollama not reachable at 127.0.0.1:11434 - 'ollama serve' not running? Start it before launching the bot."
}

# 5. GPU (informational)
try {
    $gpu = (& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null)
    if ($gpu) { Ok "GPU: $gpu" } else { Warn "nvidia-smi empty - finetune router will gate on VRAM probe" }
} catch {
    Warn "nvidia-smi not on PATH - finetune router will likely fail VRAM gate"
}

Write-Host ""
Write-Host "Launching processes..."
Write-Host ""

# --- Launch parallel processes ---------------------------------------

function Start-Bg([string]$title, [string]$cmd) {
    $args = "/k", "title $title && cd /d ""$repo"" && set PYTHONUNBUFFERED=1 && $cmd"
    $p = Start-Process cmd.exe -ArgumentList $args -PassThru -WindowStyle Normal
    try { $p.PriorityClass = "BelowNormal" } catch {}
    Write-Host "  -> $title (PID=$($p.Id))"
    return $p
}

# 1. Live trading bot (stocks role)
Start-Bg "ACT - Stocks Live (4060)" "python -m src.main --role stocks" | Out-Null

# 2. Finetune router (cross-class, market-hours aware)
Start-Bg "ACT - Finetune Router (4060)" "python -m scripts.finetune_router_4060" | Out-Null

# 3. warm_store sync (bidirectional with 5090)
Start-Bg "ACT - warm_store sync (4060)" "python -m scripts.warm_store_sync" | Out-Null

# 4. Silence watchdog
Start-Bg "ACT - Silence watchdog (4060)" "python -m scripts.silence_watchdog" | Out-Null

# 5. Paper-exploration loop - guarantees trade activity even when LLM
#    is silent (Ollama endpoint flap, dead-quiet market, parse_failures).
#    Fires a quality-filtered momentum trade every 15 min via
#    paper_exploration_tick.py --relaxed. The script's internal
#    quiet-hours + daily-cap gates self-throttle so 15-min polling
#    fires at most 8 trades/UTC-day. Real-capital path (ACT_REAL_CAPITAL_ENABLED=1)
#    is hard-skipped inside the script. Operator opt-out via
#    ACT_DISABLE_PAPER_EXPLORATION=1 broader kill via ACT_DISABLE_AGENTIC_LOOP=1.
#    Auto-picks venue: alpaca crypto (24/7) when no APCA stocks open,
#    alpaca stocks during RTH, robinhood paper-sim as final fallback.
Start-Bg "ACT - Paper Exploration (4060)" "powershell -Command `"while (`$true) { python scripts\paper_exploration_tick.py --venue alpaca --relaxed; Start-Sleep -Seconds 900 }`"" | Out-Null

# 6. (optional) FastAPI dashboard
if ($env:ACT_4060_DASHBOARD -eq "1") {
    Start-Bg "ACT - Dashboard (4060)" "python -m uvicorn src.api.production_server:app --host 0.0.0.0 --port 8081" | Out-Null
}

Write-Host ""
Ok "All processes launched. Watch logs/ for activity."
Write-Host ""
Write-Host "Stop all: STOP_ALL_4060.ps1"
Write-Host "Status:   python -m src.skills.cli run status"
