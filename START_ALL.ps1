# ===============================================================
#  ACT v8.0 — GPU-Optimized Parallel Process Manager
#  Auto-detects GPU and allocates processes accordingly.
#  Run: powershell -ExecutionPolicy Bypass -File START_ALL.ps1
# ===============================================================

$Host.UI.RawUI.BackgroundColor = [System.ConsoleColor]::Black
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
try { chcp 65001 | Out-Null } catch {}

# ── Enable ANSI/VT color processing (persistent + current session) ──
try { reg add HKCU\Console /v VirtualTerminalLevel /t REG_DWORD /d 1 /f | Out-Null } catch {}
try {
    if (-not ('Win32.VT' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
namespace Win32 {
    public static class VT {
        [DllImport("kernel32.dll")] public static extern IntPtr GetStdHandle(int h);
        [DllImport("kernel32.dll")] public static extern bool GetConsoleMode(IntPtr h, out uint m);
        [DllImport("kernel32.dll")] public static extern bool SetConsoleMode(IntPtr h, uint m);
        public static void Enable() {
            IntPtr h = GetStdHandle(-11);
            uint m; GetConsoleMode(h, out m);
            SetConsoleMode(h, m | 0x0004);
        }
    }
}
'@
    }
    [Win32.VT]::Enable()
} catch {}

# ── ANSI color palette (bright) — works everywhere VT is enabled ──
$ESC = [char]27
$R   = "$ESC[91m"   # bright red
$G   = "$ESC[92m"   # bright green
$Y   = "$ESC[93m"   # bright yellow
$B   = "$ESC[94m"   # bright blue
$M   = "$ESC[95m"   # bright magenta
$C   = "$ESC[96m"   # bright cyan
$W   = "$ESC[97m"   # bright white
$DG  = "$ESC[90m"   # dark gray
$NC  = "$ESC[0m"    # reset

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

# ── Compute intervals DYNAMICALLY from GPU compute power ──
# Formula: compute_score = VRAM_GB * 2 + CPU_cores + RAM_GB/8
# Intervals scale inversely: more power = shorter intervals
# No hardcoded tiers — works from CPU-only to A100/H100+
$computeScore = ($gpuVRAM * 2) + $cpuCores + [math]::Floor($ramGB / 8)
# Score examples: RTX 5090 (32*2+16+8=88), A100 (80*2+64+64=272), RTX 3060 (12*2+8+4=40), CPU-only (0+8+4=12)

# Adapt interval: 6h at score=10, 15min at score=200+
$adaptInterval = [math]::Round([math]::Max(0.25, 6.0 / [math]::Max(1, $computeScore / 10)), 2)
# Auto interval: half of adapt
$autoInterval = [math]::Round([math]::Max(0.15, $adaptInterval / 2), 2)
# Genetic interval: same as adapt (compute-heavy)
$geneticInterval = [math]::Round([math]::Max(0.25, $adaptInterval), 2)
# Population: scales linearly with compute score
$geneticPop = [math]::Min(500, [math]::Max(10, [math]::Floor($computeScore * 2)))

# Models: load more if VRAM allows
if ($gpuVRAM -ge 40) {
    # A100/H100+ : can run multiple large models simultaneously
    $ollamaModels = @("mistral:latest", "llama3.2:latest", "neural-chat:latest", "codellama:latest")
} elseif ($gpuVRAM -ge 16) {
    $ollamaModels = @("mistral:latest", "llama3.2:latest", "neural-chat:latest")
} elseif ($gpuVRAM -ge 8) {
    $ollamaModels = @("mistral:latest", "llama3.2:latest")
} elseif ($gpuVRAM -ge 4) {
    $ollamaModels = @("llama3.2:latest")
} else {
    $ollamaModels = @("llama3.2:latest")
}

$tier = "SCORE=$computeScore"

# ── ACT Color Motion Banner (ANSI escape codes — works in all modern Windows terminals) ──
Write-Host ""
Write-Host "${C}  ══════════════════════════════════════════════════════════${NC}"
Write-Host ""
Write-Host "${R}       █████╗ ${NC}${W}   ██████╗ ${NC}${G}  ████████╗${NC}"
Write-Host "${R}      ██╔══██╗${NC}${W}  ██╔════╝ ${NC}${G}  ╚══██╔══╝${NC}"
Write-Host "${R}      ███████║${NC}${W}  ██║      ${NC}${G}     ██║   ${NC}"
Write-Host "${R}      ██╔══██║${NC}${W}  ██║      ${NC}${G}     ██║   ${NC}"
Write-Host "${R}      ██║  ██║${NC}${W}  ╚██████╗ ${NC}${G}     ██║   ${NC}"
Write-Host "${R}      ╚═╝  ╚═╝${NC}${W}   ╚═════╝ ${NC}${G}     ╚═╝   ${NC}"
Write-Host ""
Write-Host "         ${R}Autonomous${NC} ${DG}·${NC} ${W}Crypto${NC} ${DG}·${NC} ${G}Trader${NC}    ${Y}v8.0${NC}"
Write-Host "${DG}       GPU-Optimized · Self-Evolving · 12-Agent Consensus${NC}"
Write-Host ""

# Pulse effect: motion flash through 5 colors
$pulseText = "        [ ACT v8.0 BOOTING ]"
$pulseColors = @($R, $W, $G, $Y, $C)
foreach ($clr in $pulseColors) {
    Write-Host ("`r" + (' ' * 60)) -NoNewline
    Write-Host ("`r" + $clr + $pulseText + $NC) -NoNewline
    Start-Sleep -Milliseconds 90
}
Write-Host ("`r" + (' ' * 60))
Write-Host ""
Write-Host "$C  ══════════════════════════════════════════════════════════$NC"
Write-Host "$Y   $tier : $gpuName ${gpuVRAM}GB | ${cpuCores} cores | ${ramGB}GB RAM$NC"
Write-Host "$C  ══════════════════════════════════════════════════════════$NC"
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
# Write temp script to avoid PowerShell quote mangling
$pyCheck = Join-Path $PSScriptRoot "_check_cuda.py"
@"
import torch
cuda = torch.cuda.is_available()
if cuda:
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram = (getattr(props, 'total_memory', 0) or getattr(props, 'total_mem', 0)) // (1024**3)
    print('CUDA=True GPU=' + name + ' VRAM=' + str(vram) + 'GB')
else:
    print('CUDA=False GPU=none VRAM=0GB')
"@ | Set-Content -Path $pyCheck -Encoding UTF8
$cudaCheck = python $pyCheck 2>&1
Remove-Item $pyCheck -ErrorAction SilentlyContinue
if ("$cudaCheck" -match "CUDA=True") {
    OK "PyTorch CUDA: $cudaCheck"
    if ("$cudaCheck" -match "VRAM=(\d+)GB") { $gpuVRAM = [int]$matches[1] }
} else {
    WARN "PyTorch CUDA NOT available: $cudaCheck"
    WARN "Install CUDA PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu128"
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
