# ACT Trading System -- Full Self-Evolving Startup (RTX 5090)
# Run: powershell -ExecutionPolicy Bypass -File START_ALL.ps1

$Host.UI.RawUI.BackgroundColor = [System.ConsoleColor]::Black
Clear-Host

Write-Host
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
Write-Host
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host

function OK($m)    { Write-Host "[OK] " -ForegroundColor Green -NoNewline; Write-Host $m }
function WARN($m)  { Write-Host "[WARN] " -ForegroundColor Yellow -NoNewline; Write-Host $m }
function CHECK($m) { Write-Host "[CHECK] " -ForegroundColor Yellow -NoNewline; Write-Host $m }
function ERR($m)   { Write-Host "[ERROR] " -ForegroundColor Red -NoNewline; Write-Host $m }
function STEP($n,$m) { Write-Host "[$n/7] " -ForegroundColor Yellow -NoNewline; Write-Host $m -ForegroundColor Cyan }

CHECK "Verifying Ollama on localhost:11434..."
try {
    Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop | Out-Null
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
        Start-Sleep 3
        exit 1
    }
}

CHECK "Checking Ollama models..."
$models = ollama list 2>$null
foreach ($mod in @("mistral:latest","llama3.2:latest")) {
    if ($models -notmatch $mod.Split(":")[0]) {
        WARN "Downloading $mod ..."
        ollama pull $mod
    }
}
OK "Models ready: mistral + llama3.2"

CHECK "Loading .env credentials..."
$envFile = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path $envFile)) {
    ERR ".env file not found at $envFile"
    Start-Sleep 3
    exit 1
}
Get-Content $envFile | ForEach-Object {
    if ($_ -match "^\s*([^#][^=]+)=(.*)$") {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
    }
}
OK ".env loaded."

CHECK "Checking cloudflared..."
if (-not (Get-Command "cloudflared" -ErrorAction SilentlyContinue)) {
    WARN "cloudflared not found. Downloading..."
    $cf = "$PSScriptRoot\cloudflared.exe"
    Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile $cf
    $env:PATH += ";$PSScriptRoot"
    OK "cloudflared downloaded."
} else {
    OK "cloudflared found."
}

Write-Host
Write-Host "[CLEANUP] " -ForegroundColor Yellow -NoNewline
Write-Host "Stopping existing ACT processes..."
Get-Process | Where-Object { $_.MainWindowTitle -like "ACT*" } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep 2
OK "Cleanup done."

Write-Host
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  LAUNCHING ALL 7 SYSTEMS..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host

$dir = $PSScriptRoot

STEP 1 "Starting API Server (port 11007)..."
Start-Process cmd -ArgumentList "/k cd /d " && set PYTHONUNBUFFERED=1 && python -m src.api.production_server" -WindowStyle Normal
Start-Sleep 3
OK "API Server started on :11007"

STEP 2 "Starting Trading Bot..."
Start-Process cmd -ArgumentList "/k cd /d " && set PYTHONUNBUFFERED=1 && python -m src.main" -WindowStyle Normal
Start-Sleep 3
OK "Trading Bot started."

STEP 3 "Starting Adaptation Loop (every 1h)..."
Start-Process cmd -ArgumentList "/k cd /d " && set PYTHONUNBUFFERED=1 && python -m src.scripts.continuous_adapt --continuous --interval 1" -WindowStyle Normal
Start-Sleep 2
OK "Adaptation Loop started."

STEP 4 "Starting Autonomous Loop (every 30min)..."
Start-Process cmd -ArgumentList "/k cd /d " && set PYTHONUNBUFFERED=1 && python -m src.scripts.autonomous_loop --interval 0.5" -WindowStyle Normal
Start-Sleep 2
OK "Autonomous Loop started."

STEP 5 "Starting Genetic Loop (pop=100, every 2h)..."
Start-Process cmd -ArgumentList "/k cd /d " && set PYTHONUNBUFFERED=1 && python -m src.scripts.genetic_loop --population_size 100 --interval 2" -WindowStyle Normal
Start-Sleep 2
OK "Genetic Loop started."

STEP 6 "Starting Frontend (port 5173)..."
Start-Process cmd -ArgumentList "/k cd /d " && npm run dev" -WindowStyle Normal
Start-Sleep 4
OK "Frontend started on :5173"

STEP 7 "Starting Cloudflare Tunnel..."
Start-Process cmd -ArgumentList "/k cloudflared tunnel --url http://localhost:5173" -WindowStyle Normal
Start-Sleep 3
OK "Tunnel started."

Write-Host
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host
Write-Host "   ALL 7 SYSTEMS RUNNING  (RTX 5090 + CUDA)" -ForegroundColor Green
Write-Host
Write-Host "   1  API Server    http://localhost:11007"
Write-Host "   2  Trading Bot   Robinhood Paper (BTC/ETH)"
Write-Host "   3  Adapt Loop    Every 1h  -- retrain + fine-tune"
Write-Host "   4  Auto Loop     Every 30m -- self-heal + monitor"
Write-Host "   5  Genetic Loop  Every 2h  -- 100 DNA strategies"
Write-Host "   6  Frontend      http://localhost:5173"
Write-Host "   7  CF Tunnel     Check ACT-Tunnel window for URL"
Write-Host
Write-Host "   Ollama: localhost:11434  CUDA: RTX 5090" -ForegroundColor Yellow
Write-Host "   Models: mistral + llama3.2 + neural-chat" -ForegroundColor Yellow
Write-Host
Write-Host "   Open http://localhost:5173 to view dashboard" -ForegroundColor Green
Write-Host "   Check ACT-Tunnel window for remote URL" -ForegroundColor Green
Write-Host
Write-Host "============================================================" -ForegroundColor Cyan
Start-Sleep 10
