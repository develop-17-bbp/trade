# ===============================================================
#  ACT Trading System -- Full Self-Evolving Startup (RTX 5090)
#  Run with: powershell -ExecutionPolicy Bypass -File START_ALL.ps1
# ===============================================================

$Host.UI.RawUI.BackgroundColor = "Black"
Clear-Host

# ASCII Art with colors
Write-Host ""
Write-Host "     AAA   " -ForegroundColor Red -NoNewline
Write-Host "CCCCC" -ForegroundColor White -NoNewline
Write-Host "  TTTTT" -ForegroundColor Green
Write-Host "   A   A " -ForegroundColor Red -NoNewline
Write-Host "C      " -ForegroundColor White -NoNewline
Write-Host "    T  " -ForegroundColor Green
Write-Host "   AAAAA " -ForegroundColor Red -NoNewline
Write-Host "C      " -ForegroundColor White -NoNewline
Write-Host "    T  " -ForegroundColor Green -NoNewline
Write-Host "     ACT TRADING SYSTEM" -ForegroundColor Cyan
Write-Host "   A   A " -ForegroundColor Red -NoNewline
Write-Host "C      " -ForegroundColor White -NoNewline
Write-Host "    T  " -ForegroundColor Green -NoNewline
Write-Host "     RTX 5090 -- FULL STARTUP" -ForegroundColor Yellow
Write-Host "   A   A " -ForegroundColor Red -NoNewline
Write-Host "CCCCC" -ForegroundColor White -NoNewline
Write-Host "  TTTTT" -ForegroundColor Green
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Helper functions
function OK($msg)    { Write-Host "[OK] " -ForegroundColor Green -NoNewline; Write-Host $msg }
function WARN($msg)  { Write-Host "[WARN] " -ForegroundColor Yellow -NoNewline; Write-Host $msg }
function CHECK($msg) { Write-Host "[CHECK] " -ForegroundColor Yellow -NoNewline; Write-Host $msg }
function ERR($msg)   { Write-Host "[ERROR] " -ForegroundColor Red -NoNewline; Write-Host $msg }
function STEP($n,$msg) { Write-Host "[$n/7] " -ForegroundColor Yellow -NoNewline; Write-Host $msg -ForegroundColor Cyan }

# Step 0: Verify Ollama
CHECK "Verifying Ollama on localhost:11434..."
try {
    $r = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
    OK "Ollama running."
} catch {
    WARN "Ollama not detected! Starting Ollama..."
    Start-Process "ollama" "serve" -WindowStyle Minimized
    Start-Sleep 5
    try {
        Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop | Out-Null
        OK "Ollama started."
    } catch {
        ERR "Ollama failed to start. Please start it manually."
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Step 0b: Check models
CHECK "Checking Ollama models..."
$models = ollama list 2>$null
foreach ($m in @("mistral:latest", "llama3.2:latest")) {
    if ($models -notmatch $m.Split(":")[0]) {
        WARN "Downloading $m ..."
        ollama pull $m
    }
}
OK "Models ready: mistral + llama3.2"

# Step 0c: Load .env
CHECK "Loading .env credentials..."
$envFile = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path $envFile)) {
    ERR ".env file not found at $envFile"
    Read-Host "Press Enter to exit"
    exit 1
}
Get-Content $envFile | ForEach-Object {
    if ($_ -match "^\s*([^#][^=]+)=(.*)$") {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
    }
}
OK ".env loaded."

# Step 0d: Check cloudflared
CHECK "Checking cloudflared..."
if (-not (Get-Command "cloudflared" -ErrorAction SilentlyContinue)) {
    WARN "cloudflared not found. Downloading..."
    $cfPath = "$PSScriptRoot\cloudflared.exe"
    Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile $cfPath
    $env:PATH += ";$PSScriptRoot"
    OK "cloudflared downloaded."
} else {
    OK "cloudflared found."
}

# Cleanup
Write-Host ""
Write-Host "[CLEANUP] " -ForegroundColor Yellow -NoNewline
Write-Host "Stopping existing ACT processes..."
Get-Process | Where-Object { $_.MainWindowTitle -like "ACT*" } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep 2
OK "Cleanup done."

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  LAUNCHING ALL 7 SYSTEMS..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$dir = $PSScriptRoot

# Window 1: API Server
STEP 1 "Starting API Server (port 11007)..."
Start-Process "cmd" -ArgumentList "/k cd /d `"$dir`" && set PYTHONUNBUFFERED=1 && python -m src.api.production_server" -WindowStyle Normal
Start-Sleep 3
OK "API Server started on :11007"

# Window 2: Trading Bot
STEP 2 "Starting Trading Bot..."
Start-Process "cmd" -ArgumentList "/k cd /d `"$dir`" && set PYTHONUNBUFFERED=1 && python -m src.main" -WindowStyle Normal
Start-Sleep 3
OK "Trading Bot started."

# Window 3: Adaptation Loop
STEP 3 "Starting Adaptation Loop (every 1h)..."
Start-Process "cmd" -ArgumentList "/k cd /d `"$dir`" && set PYTHONUNBUFFERED=1 && python -m src.scripts.continuous_adapt --continuous --interval 1" -WindowStyle Normal
Start-Sleep 2
OK "Adaptation Loop started."

# Window 4: Autonomous Loop
STEP 4 "Starting Autonomous Loop (every 30min)..."
Start-Process "cmd" -ArgumentList "/k cd /d `"$dir`" && set PYTHONUNBUFFERED=1 && python -m src.scripts.autonomous_loop --interval 0.5" -WindowStyle Normal
Start-Sleep 2
OK "Autonomous Loop started."

# Window 5: Genetic Loop
STEP 5 "Starting Genetic Loop (pop=100, every 2h)..."
Start-Process "cmd" -ArgumentList "/k cd /d `"$dir`" && set PYTHONUNBUFFERED=1 && python -m src.scripts.genetic_loop --population_size 100 --interval 2" -WindowStyle Normal
Start-Sleep 2
OK "Genetic Loop started."

# Window 6: Frontend
STEP 6 "Starting Frontend (port 5173)..."
Start-Process "cmd" -ArgumentList "/k cd /d `"$dir\frontend`" && npm run dev" -WindowStyle Normal
Start-Sleep 4
OK "Frontend started on :5173"

# Window 7: Cloudflare Tunnel
STEP 7 "Starting Cloudflare Tunnel..."
Start-Process "cmd" -ArgumentList "/k cloudflared tunnel --url http://localhost:5173" -WindowStyle Normal
Start-Sleep 3
OK "Tunnel started — check ACT-Tunnel window for public URL."

# Summary
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "   ALL 7 SYSTEMS RUNNING  (RTX 5090 + CUDA)" -ForegroundColor Green
Write-Host ""
Write-Host "   1  " -NoNewline; Write-Host "API Server   " -ForegroundColor Cyan -NoNewline; Write-Host "http://localhost:11007"
Write-Host "   2  " -NoNewline; Write-Host "Trading Bot  " -ForegroundColor Red -NoNewline; Write-Host "Robinhood Paper (BTC/ETH)"
Write-Host "   3  " -NoNewline; Write-Host "Adapt Loop   " -ForegroundColor Cyan -NoNewline; Write-Host "Every 1h  -- retrain + fine-tune"
Write-Host "   4  " -NoNewline; Write-Host "Auto Loop    " -ForegroundColor Cyan -NoNewline; Write-Host "Every 30m -- self-heal + monitor"
Write-Host "   5  " -NoNewline; Write-Host "Genetic Loop " -ForegroundColor Green -NoNewline; Write-Host "Every 2h  -- evolve 100 DNA strategies"
Write-Host "   6  " -NoNewline; Write-Host "Frontend     " -ForegroundColor Cyan -NoNewline; Write-Host "http://localhost:5173"
Write-Host "   7  " -NoNewline; Write-Host "CF Tunnel    " -ForegroundColor Cyan -NoNewline; Write-Host "Check ACT-Tunnel window for URL"
Write-Host ""
Write-Host "   Ollama: localhost:11434  CUDA: RTX 5090" -ForegroundColor Yellow
Write-Host "   Models: mistral + llama3.2 + neural-chat" -ForegroundColor Yellow
Write-Host ""
Write-Host "   Open http://localhost:5173 to view dashboard" -ForegroundColor Green
Write-Host "   Check ACT-Tunnel window for remote access URL" -ForegroundColor Green
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Read-Host "Press Enter to close this window (all processes keep running)"
