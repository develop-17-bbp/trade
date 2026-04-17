# ===============================================================
#  ACT v8.0 — GPU-Optimized Parallel Process Manager
#  Auto-detects GPU and allocates processes accordingly.
#  Run: powershell -ExecutionPolicy Bypass -File START_ALL.ps1
# ===============================================================

$Host.UI.RawUI.BackgroundColor = [System.ConsoleColor]::Black
Clear-Host

# ── GPU/CPU Detection ──
$gpuName = "Unknown"
$gpuVRAM = 0
$cpuCores = [Environment]::ProcessorCount
$ramGB = [math]::Round((Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum).Sum / 1GB)
try {
    $gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" } | Select-Object -First 1
    if ($gpu) {
        $gpuName = $gpu.Name
        if ($gpuName -match "5090") { $gpuVRAM = 32 }
        elseif ($gpuName -match "4090") { $gpuVRAM = 24 }
        elseif ($gpuName -match "3090") { $gpuVRAM = 24 }
        elseif ($gpuName -match "4080") { $gpuVRAM = 16 }
        elseif ($gpuName -match "3080") { $gpuVRAM = 12 }
        elseif ($gpuName -match "3070") { $gpuVRAM = 8 }
        elseif ($gpuName -match "3060") { $gpuVRAM = 12 }
        elseif ($gpuName -match "4060") { $gpuVRAM = 8 }
        else { $gpuVRAM = [math]::Round($gpu.AdapterRAM / 1GB) }
    }
} catch {}

# ── Compute intervals by GPU tier ──
if ($gpuVRAM -ge 24) {
    $adaptInterval = 1; $autoInterval = 0.5; $geneticInterval = 1; $geneticPop = 150
    $ollamaModels = @("mistral:latest", "llama3.2:latest")
    $tier = "HIGH-END"
} elseif ($gpuVRAM -ge 12) {
    $adaptInterval = 2; $autoInterval = 1; $geneticInterval = 2; $geneticPop = 75
    $ollamaModels = @("mistral:latest", "llama3.2:latest")
    $tier = "MID-RANGE"
} elseif ($gpuVRAM -ge 6) {
    $adaptInterval = 4; $autoInterval = 2; $geneticInterval = 4; $geneticPop = 30
    $ollamaModels = @("llama3.2:latest")
    $tier = "ENTRY"
} else {
    $adaptInterval = 6; $autoInterval = 3; $geneticInterval = 6; $geneticPop = 15
    $ollamaModels = @("llama3.2:latest")
    $tier = "CPU-ONLY"
}

# ── ASCII Art ──
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
Write-Host "     ACT v8.0 TRADING SYSTEM" -ForegroundColor Cyan
Write-Host "   A   A " -ForegroundColor Red -NoNewline
Write-Host "C      " -ForegroundColor White -NoNewline
Write-Host "    T  " -ForegroundColor Green -NoNewline
Write-Host "     GPU-OPTIMIZED PARALLEL LAUNCH" -ForegroundColor Yellow
Write-Host "   A   A " -ForegroundColor Red -NoNewline
Write-Host "CCCCC" -ForegroundColor White -NoNewline
Write-Host "  TTTTT" -ForegroundColor Green
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  $tier : $gpuName ${gpuVRAM}GB | ${cpuCores} cores | ${ramGB}GB RAM" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

function OK($m) { Write-Host "[OK] " -ForegroundColor Green -NoNewline; Write-Host $m }
function WARN($m) { Write-Host "[WARN] " -ForegroundColor Yellow -NoNewline; Write-Host $m }
function CHECK($m) { Write-Host "[CHECK] " -ForegroundColor Yellow -NoNewline; Write-Host $m }
function STEP($n,$m) { Write-Host "[$n/7] " -ForegroundColor Yellow -NoNewline; Write-Host $m -ForegroundColor Cyan }

# ── Verify Ollama ──
CHECK "Verifying Ollama..."
try {
    Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop | Out-Null
    OK "Ollama running."
} catch {
    WARN "Starting Ollama..."
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Minimized
    Start-Sleep 5
    try {
        Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop | Out-Null
        OK "Ollama started."
    } catch {
        Write-Host "[ERROR] Ollama failed." -ForegroundColor Red; Start-Sleep 3; exit 1
    }
}

# ── Verify PyTorch CUDA ──
CHECK "Verifying PyTorch CUDA..."
$cudaCheck = python -c "import torch; print(f'CUDA={torch.cuda.is_available()} GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"} VRAM={torch.cuda.get_device_properties(0).total_mem // (1024**3) if torch.cuda.is_available() else 0}GB')" 2>&1
if ($cudaCheck -match "CUDA=True") {
    OK "PyTorch CUDA: $cudaCheck"
    # Update VRAM from torch (more accurate than WMI)
    if ($cudaCheck -match "VRAM=(\d+)GB") { $gpuVRAM = [int]$matches[1] }
} else {
    WARN "PyTorch CUDA NOT available: $cudaCheck"
    WARN "LLM fine-tuning will be slow (CPU only). Install: pip install torch --index-url https://download.pytorch.org/whl/cu128"
}

# ── Verify nvidia-smi ──
CHECK "Checking nvidia-smi..."
$smi = nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>&1
if ($LASTEXITCODE -eq 0) {
    OK "nvidia-smi: $smi"
} else {
    WARN "nvidia-smi not found (NVIDIA drivers may need install)"
}

# ── Pull models ──
CHECK "Checking models for $tier GPU..."
$models = ollama list 2>$null
foreach ($mod in $ollamaModels) {
    if ($models -notmatch $mod.Split(":")[0]) {
        WARN "Pulling $mod ..."
        ollama pull $mod
    }
}
OK "Models: $($ollamaModels -join ' + ')"

# ── Load .env ──
CHECK "Loading .env..."
$envFile = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path $envFile)) { Write-Host "[ERROR] .env missing" -ForegroundColor Red; Start-Sleep 3; exit 1 }
Get-Content $envFile | ForEach-Object {
    if ($_ -match "^\s*([^#=][^=]*)=(.*)$") {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
    }
}
OK ".env loaded."

# ── cloudflared ──
if (-not (Get-Command "cloudflared" -ErrorAction SilentlyContinue)) {
    WARN "Downloading cloudflared..."
    Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile "$PSScriptRoot\cloudflared.exe"
    $env:PATH += ";$PSScriptRoot"
    OK "cloudflared installed."
}

# ── Cleanup existing ──
Write-Host ""
Write-Host "[CLEANUP] Stopping existing ACT processes..." -ForegroundColor Yellow
Get-Process | Where-Object { $_.MainWindowTitle -like "ACT*" } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep 2
OK "Cleanup done."

$dir = $PSScriptRoot
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONPATH = $dir

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  LAUNCHING 7 PARALLEL PROCESSES ($tier)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ── 1: API Server (AboveNormal) ──
STEP 1 "API Server (port 11007) [AboveNormal]"
$p1 = Start-Process cmd.exe -ArgumentList "/k","title ACT - API Server && cd /d $dir && set PYTHONUNBUFFERED=1 && python -m src.api.production_server" -PassThru
try { $p1.PriorityClass = "AboveNormal" } catch {}
Start-Sleep 2
OK "API Server PID=$($p1.Id)"

# ── 2: Trading Bot (High — gets GPU priority for LLM) ──
STEP 2 "Trading Bot [HIGH priority - GPU primary]"
$p2 = Start-Process cmd.exe -ArgumentList "/k","title ACT - Trading Bot && cd /d $dir && set PYTHONUNBUFFERED=1 && set CUDA_VISIBLE_DEVICES=0 && python -m src.main" -PassThru
try { $p2.PriorityClass = "High" } catch {}
Start-Sleep 3
OK "Trading Bot PID=$($p2.Id)"

# ── 3: Adaptation (Normal — GPU for retraining) ──
STEP 3 "Adaptation Loop (${adaptInterval}h) [Normal - GPU retrain]"
$p3 = Start-Process cmd.exe -ArgumentList "/k","title ACT - Adaptation Loop && cd /d $dir && set PYTHONUNBUFFERED=1 && python -m src.scripts.continuous_adapt --continuous --interval $adaptInterval" -PassThru
Start-Sleep 1
OK "Adaptation PID=$($p3.Id)"

# ── 4: Autonomous (Normal — self-heal) ──
STEP 4 "Autonomous Loop (${autoInterval}h) [Normal]"
$p4 = Start-Process cmd.exe -ArgumentList "/k","title ACT - Autonomous Loop && cd /d $dir && set PYTHONUNBUFFERED=1 && python -m src.scripts.autonomous_loop --interval $autoInterval" -PassThru
Start-Sleep 1
OK "Autonomous PID=$($p4.Id)"

# ── 5: Genetic (BelowNormal — CPU heavy background) ──
STEP 5 "Genetic Loop (pop=$geneticPop, ${geneticInterval}h) [BelowNormal - CPU]"
$p5 = Start-Process cmd.exe -ArgumentList "/k","title ACT - Genetic Loop && cd /d $dir && set PYTHONUNBUFFERED=1 && python -m src.scripts.genetic_loop --population_size $geneticPop --interval $geneticInterval" -PassThru
try { $p5.PriorityClass = "BelowNormal" } catch {}
Start-Sleep 1
OK "Genetic PID=$($p5.Id)"

# ── 6: Frontend (BelowNormal) ──
STEP 6 "Frontend (port 5173) [BelowNormal]"
$p6 = Start-Process cmd.exe -ArgumentList "/k","title ACT - Frontend && cd /d $dir\frontend && npm run dev" -PassThru
try { $p6.PriorityClass = "BelowNormal" } catch {}
Start-Sleep 3
OK "Frontend PID=$($p6.Id)"

# ── 7: Tunnel (Idle) ──
STEP 7 "Cloudflare Tunnel [Idle priority]"
$p7 = Start-Process cmd.exe -ArgumentList "/k","title ACT - Tunnel && cloudflared tunnel --url http://localhost:5173" -PassThru
try { $p7.PriorityClass = "Idle" } catch {}
Start-Sleep 2
OK "Tunnel PID=$($p7.Id)"

# ── Summary ──
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "   ALL 7 SYSTEMS RUNNING ($tier)" -ForegroundColor Green
Write-Host ""
Write-Host "   #  Process        Priority      Interval  GPU Use" -ForegroundColor White
Write-Host "   1  API Server     AboveNormal   always    none" -ForegroundColor White
Write-Host "   2  Trading Bot    HIGH          always    LLM inference" -ForegroundColor Red
Write-Host "   3  Adapt Loop     Normal        ${adaptInterval}h        retrain+finetune" -ForegroundColor White
Write-Host "   4  Auto Loop      Normal        ${autoInterval}h      health check" -ForegroundColor White
Write-Host "   5  Genetic Loop   BelowNormal   ${geneticInterval}h        CPU backtest" -ForegroundColor White
Write-Host "   6  Frontend       BelowNormal   always    none" -ForegroundColor White
Write-Host "   7  Tunnel         Idle          always    none" -ForegroundColor White
Write-Host ""
Write-Host "   VRAM: Ollama ~$([math]::Min($gpuVRAM-2, 12))GB | LoRA ~$([math]::Max(2, $gpuVRAM-14))GB | System ~2GB" -ForegroundColor Yellow
Write-Host "   Dashboard: http://localhost:5173" -ForegroundColor Green
Write-Host "   API: http://localhost:11007" -ForegroundColor Green
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Start-Sleep 10
