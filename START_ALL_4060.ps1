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

# 4. Box role
if (-not $env:ACT_BOX_ROLE) {
    Warn "ACT_BOX_ROLE not set - defaulting to 'stocks' for the 4060"
    [Environment]::SetEnvironmentVariable("ACT_BOX_ROLE", "stocks", "Process")
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

# 5. (optional) FastAPI dashboard
if ($env:ACT_4060_DASHBOARD -eq "1") {
    Start-Bg "ACT - Dashboard (4060)" "python -m uvicorn src.api.production_server:app --host 0.0.0.0 --port 8081" | Out-Null
}

Write-Host ""
Ok "All processes launched. Watch logs/ for activity."
Write-Host ""
Write-Host "Stop all: STOP_ALL_4060.ps1"
Write-Host "Status:   python -m src.skills.cli run status"
