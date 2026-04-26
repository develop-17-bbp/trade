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

# ── Dual-brain models — default dense_r1 for 32GB RTX 5090 safety ──
# qwen3_r1 (qwen3:32b + deepseek-r1:32b) needs ~42 GB and OOMs on 32 GB.
# dense_r1 (deepseek-r1:7b + deepseek-r1:32b ~= 26 GB) fits.
# moe_agentic (qwen2.5-coder:7b + qwen3-coder:30b ~= 24 GB) also fits.
# Operator can override to qwen3_r1 on >40 GB hardware via
# `setx ACT_BRAIN_PROFILE qwen3_r1`.
$brainProfile = $env:ACT_BRAIN_PROFILE
if (-not $brainProfile) { $brainProfile = "dense_r1" }

switch ($brainProfile) {
    "qwen3_r1"             { $scannerModel = "qwen3:32b";          $analystModel = "deepseek-r1:32b" }
    "dense_r1"             { $scannerModel = "deepseek-r1:7b";     $analystModel = "deepseek-r1:32b" }
    "moe_agentic"          { $scannerModel = "qwen2.5-coder:7b";   $analystModel = "qwen3-coder:30b" }
    "devstral_qwen3coder"  { $scannerModel = "devstral:24b";       $analystModel = "qwen3-coder:30b" }
    default {
        WARN "Unknown ACT_BRAIN_PROFILE=$brainProfile -- defaulting to dense_r1"
        $brainProfile = "dense_r1"
        $scannerModel = "deepseek-r1:7b"; $analystModel = "deepseek-r1:32b"
    }
}

# ── Safety: reject profiles that won't fit 32 GB VRAM ──
# qwen3_r1 needs ~42 GB. On cards <40 GB VRAM, auto-downgrade to
# dense_r1 before the Python process even starts to prevent the
# "empty LLM response → parse_failures → zero trades" chain.
try {
    $vramLine = & nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    if ($vramLine) {
        $vramGiB = [int]$vramLine / 1024.0
        if ($brainProfile -eq "qwen3_r1" -and $vramGiB -lt 40) {
            WARN "Profile qwen3_r1 needs ~42 GB VRAM but only $([int]$vramGiB) GB detected - auto-downgrading to dense_r1"
            $brainProfile = "dense_r1"
            $scannerModel = "deepseek-r1:7b"
            $analystModel = "deepseek-r1:32b"
            _SetEnvPersistent "ACT_BRAIN_PROFILE" "dense_r1"
        }
    }
} catch {
    # nvidia-smi missing or failed; skip the safety check silently
}

# Per-role env overrides always win.
if ($env:ACT_SCANNER_MODEL) { $scannerModel = $env:ACT_SCANNER_MODEL }
if ($env:ACT_ANALYST_MODEL) { $analystModel = $env:ACT_ANALYST_MODEL }

$ollamaModels = @($analystModel, $scannerModel) | Select-Object -Unique

# ── Forbidden-model filter (ACT_FORBID_MODELS) ──────────────────
# When the operator has retired a model family, drop it from the
# pre-load list so START_ALL doesn't keep loading something
# downstream code is now configured to refuse. Same env var the
# python model_guard reads -- single source of truth.
$forbidRaw = $env:ACT_FORBID_MODELS
if ($forbidRaw) {
    $forbidList = $forbidRaw.ToLower().Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
    $filtered = @()
    foreach ($m in $ollamaModels) {
        $ml = $m.ToLower()
        $head = $ml.Split(":")[0]
        $blocked = $false
        foreach ($entry in $forbidList) {
            if ($entry -eq $ml -or $entry -eq $head -or $ml.Contains($entry)) {
                $blocked = $true; break
            }
        }
        if ($blocked) {
            Write-Host "[INFO] ACT_FORBID_MODELS dropped $m from pre-load" -ForegroundColor Yellow
        } else {
            $filtered += $m
        }
    }
    if ($filtered.Count -eq 0) {
        Write-Host "[ERROR] ACT_FORBID_MODELS blocked every profile model -- check your env" -ForegroundColor Red
        Start-Sleep 3; exit 1
    }
    $ollamaModels = $filtered
    if ($filtered -notcontains $analystModel) { $analystModel = $filtered[0] }
    if ($filtered -notcontains $scannerModel) { $scannerModel = $filtered[-1] }
}

# On very small GPUs (<8 GB), the 32B analyst won't fit even at Q4_K_M.
# Downshift the whole profile to 7B-only mode with a warning.
if ($gpuVRAM -lt 8 -and $gpuVRAM -gt 0) {
    WARN "gpuVRAM=$gpuVRAM GB — downshifting analyst from $analystModel to $scannerModel"
    $analystModel = $scannerModel
    $ollamaModels = @($scannerModel)
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
function STEP($n,$m) { Write-Host "[$n/8] " -ForegroundColor Yellow -NoNewline; Write-Host $m -ForegroundColor Cyan }

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

# ── Pre-load brain models into VRAM with keep_alive=-1 ──
# Without this, the FIRST scanner / analyst call after boot triggers
# a 30-60 sec disk-load → request times out → empty LLM response →
# parse_failures → bot fires zero trades for the first ~5 minutes.
# Pre-load both models with keep_alive=-1 (Ollama: pin in VRAM
# indefinitely until next request resets it) so Ollama has them
# warm before the trading bot even tries.
CHECK "Pre-loading brain models into VRAM (keep_alive=-1)..."
$ollamaUrl = if ($env:OLLAMA_BASE_URL) { $env:OLLAMA_BASE_URL.TrimEnd('/') } else { 'http://127.0.0.1:11434' }
# Pre-load context MUST match what the python bot uses (16384, set
# below as OLLAMA_NUM_CTX). If they differ, Ollama reloads the model
# at the larger context on the first python request and evicts the
# other resident model — leaving only one brain in VRAM, which is
# exactly the failure mode `ollama ps` reported (only 7B at ctx=32768).
$ctxNum = if ($env:OLLAMA_NUM_CTX) { $env:OLLAMA_NUM_CTX } else { '16384' }
foreach ($mod in $ollamaModels) {
    $payload = @{
        model       = $mod
        prompt      = 'ping'
        stream      = $false
        keep_alive  = -1
        options     = @{ num_ctx = [int]$ctxNum; num_predict = 4 }
    } | ConvertTo-Json -Depth 5 -Compress
    try {
        $resp = Invoke-WebRequest -Uri "$ollamaUrl/api/generate" `
            -Method Post -Body $payload -ContentType 'application/json' `
            -TimeoutSec 180 -ErrorAction Stop -UseBasicParsing
        if ($resp.StatusCode -eq 200) {
            OK "Pre-loaded $mod (resident in VRAM, keep_alive=-1)"
        } else {
            WARN "Pre-load $mod returned HTTP $($resp.StatusCode)"
        }
    } catch {
        WARN "Pre-load $mod failed: $($_.Exception.Message). Bot will retry on first tick."
    }
}

# ── Verify both models actually resident ──
try {
    $ps = Invoke-RestMethod -Uri "$ollamaUrl/api/ps" -TimeoutSec 5 -ErrorAction Stop
    if ($ps.models -and $ps.models.Count -gt 0) {
        $loaded = ($ps.models | ForEach-Object { $_.name }) -join ', '
        OK "Ollama VRAM: $loaded"
        if ($ps.models.Count -lt $ollamaModels.Count) {
            WARN "Only $($ps.models.Count) of $($ollamaModels.Count) models resident — set OLLAMA_MAX_LOADED_MODELS=$($ollamaModels.Count) and restart Ollama if both should fit."
        }
    } else {
        WARN "Ollama /api/ps reports no resident models — pre-load may have failed silently."
    }
} catch {
    WARN "Could not verify resident models via /api/ps: $($_.Exception.Message)"
}

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

# ── Resolve script directory EARLY so later paths (docker-compose, process
# launchers) can Join-Path against it. Without this, $dir is $null at line
# 225 and Join-Path throws "Cannot bind argument to parameter 'Path'".
$dir = $PSScriptRoot

# ── Phase 1/2 observability stack (Docker) ─────────────────────────
# Starts Redis, Prometheus, Grafana, Tempo, OTel Collector in WSL2 Docker
# via docker-compose. Bot reads ACT_REDIS_URL + OTLP endpoint on localhost.
# Skips cleanly if Docker Desktop isn't running — observability is optional;
# the bot boots without it.
CHECK "Starting observability stack (Grafana + Prometheus + Tempo + Redis + OTel)..."
$composeFile = Join-Path $dir "infra\docker-compose.yml"
$dockerOk = $false
try {
    docker info --format '{{.ServerVersion}}' 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $dockerOk = $true }
} catch {}
if ($dockerOk -and (Test-Path $composeFile)) {
    try {
        docker compose -f $composeFile up -d 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            # Give Grafana a moment to come up so the banner URL is actually live.
            Start-Sleep 3
            OK "Observability stack up. Grafana: http://localhost:3000  Prometheus: http://localhost:9090"
        } else {
            WARN "docker compose up returned non-zero — check 'docker compose logs' in infra/"
        }
    } catch {
        WARN "docker compose failed: $_ — skipping observability."
    }
} else {
    WARN "Docker not running (or compose file missing) — skipping Grafana/Prometheus stack."
    WARN "  To enable: start Docker Desktop, then re-run START_ALL.ps1."
}

# ── Cleanup existing ──
Write-Host ""
Write-Host "[CLEANUP] Stopping existing ACT processes..." -ForegroundColor Yellow
Get-Process | Where-Object { $_.MainWindowTitle -like "ACT*" } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep 2
OK "Cleanup done."

# $dir resolved earlier (before observability stack). Keep PYTHON* env setup here.
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONPATH = $dir

# ── Agentic-loop env (C17) ──
# Turn the shadow loop ON by default so the GPU box is producing shadow
# plans + brain_memory scan reports + graph edges from the moment the
# bot boots. Set ACT_DISABLE_AGENTIC_LOOP=1 in .env to force off.
#
# We use BOTH `$env:VAR` (in-process — child processes we launch here
# inherit it) AND `setx VAR` (persistent — survives to future terminals
# so `python -m src.skills.cli run status` run from a fresh cmd/pwsh
# window still sees the same flags). Without setx, operators running
# /status from a different terminal see "<unset>" because $env: is
# process-local.
function _SetEnvPersistent($name, $value) {
    Set-Item -Path ("env:" + $name) -Value $value
    try { setx $name $value | Out-Null } catch {}
}

if (-not $env:ACT_AGENTIC_LOOP) { _SetEnvPersistent "ACT_AGENTIC_LOOP" "1" }
if (-not $env:ACT_BRAIN_PROFILE) { _SetEnvPersistent "ACT_BRAIN_PROFILE" $brainProfile }
if (-not $env:ACT_SCANNER_MODEL) { _SetEnvPersistent "ACT_SCANNER_MODEL" $scannerModel }
if (-not $env:ACT_ANALYST_MODEL) { _SetEnvPersistent "ACT_ANALYST_MODEL" $analystModel }
# OLLAMA_MAX_LOADED_MODELS=2 keeps BOTH the scanner (7B) and the
# analyst (32B) warm in VRAM simultaneously. Default in older Ollama
# is 1 which causes scanner -> analyst -> scanner model swaps every
# tick (each swap ~20-60s, killing throughput). With max=2 + ctx=16384
# both models fit on a 32 GB RTX 5090 (~7 + ~22 = ~29 GB; ~3 GB
# headroom) and no swap fires per tick.
if (-not $env:OLLAMA_MAX_LOADED_MODELS) { _SetEnvPersistent "OLLAMA_MAX_LOADED_MODELS" "2" }
# OLLAMA_REMOTE_MODEL points the legacy LLMRouter `remote_gpu`
# provider at the analyst we just pinned. Without this, legacy code
# paths (agentic_strategist, TradingBrainV2 fallback) fall through to
# deepseek-r1:7b default, which evicts our pinned qwen pair on every
# call. Force-aligning prevents the eviction loop.
_SetEnvPersistent "OLLAMA_REMOTE_MODEL" $analystModel
# OLLAMA_NUM_PARALLEL controls concurrent requests per model (batch
# size, NOT model count). =1 means each model handles one request at
# a time — fine for ACT since scanner + analyst calls per tick are
# sequential by design. Crank to 2 only if you observe queue backlog
# in the scheduler.
if (-not $env:OLLAMA_NUM_PARALLEL) { _SetEnvPersistent "OLLAMA_NUM_PARALLEL" "1" }
# OLLAMA_NUM_CTX=16384 — large enough for the agentic loop's full
# prompt (system + evidence document + tool registry + multi-turn
# history can hit 4-8K tokens before the model starts generating).
# 8K was too tight: prompts got silently truncated -> garbled output
# -> parse_failures. 16K still fits on RTX 5090 (7B ~5 GB, 32B
# ~22 GB at 16K = ~27 GB total). Operators on smaller cards or
# lighter prompts can drop back to 8192 manually.
if (-not $env:OLLAMA_NUM_CTX) { _SetEnvPersistent "OLLAMA_NUM_CTX" "16384" }
# Generous timeouts for first-load of 32B from disk
if (-not $env:OLLAMA_READ_TIMEOUT_S) { _SetEnvPersistent "OLLAMA_READ_TIMEOUT_S" "180" }
OK "Agentic loop: ACT_AGENTIC_LOOP=$($env:ACT_AGENTIC_LOOP) profile=$($env:ACT_BRAIN_PROFILE) scanner=$scannerModel analyst=$analystModel MAX_LOADED=$($env:OLLAMA_MAX_LOADED_MODELS) PARALLEL=$($env:OLLAMA_NUM_PARALLEL) CTX=$($env:OLLAMA_NUM_CTX)"
OK "Env vars persisted via setx (open a new terminal to pick them up for diagnostics like /status)."

# ── Paper-soak-loose auto-enable ───────────────────────────────────
# Without the loose overlay, paper trades fire only on sniper-tier
# setups (>=2% expected move, >=4 strategies agreeing). On ranging
# days that means zero shadow rows for 24h+ and operators see "still
# no trades" -- which is exactly what just happened on this rig.
# The overlay loosens thresholds in PAPER MODE ONLY (real-capital
# gate is unaffected; the helper refuses to write when
# ACT_REAL_CAPITAL_ENABLED=1). Auto-enable keeps soak visibility
# high. Operators who want strict paper gates can either:
#   - set ACT_DISABLE_PAPER_SOAK_LOOSE=1 before launch, OR
#   - run `python -m src.skills.cli run paper-soak-loose enable=false`
# after launch.
$soakOverlay = Join-Path $dir "data\paper_soak_loose.json"
if ($env:ACT_REAL_CAPITAL_ENABLED -eq "1") {
    WARN "ACT_REAL_CAPITAL_ENABLED=1 -- skipping paper-soak-loose auto-enable (real money mode)."
} elseif ($env:ACT_DISABLE_PAPER_SOAK_LOOSE -eq "1") {
    WARN "ACT_DISABLE_PAPER_SOAK_LOOSE=1 -- skipping paper-soak-loose auto-enable (operator opt-out)."
} elseif (Test-Path $soakOverlay) {
    OK "paper-soak-loose overlay already present at data/paper_soak_loose.json (keeping operator's settings)."
} else {
    $overlayObj = [ordered]@{
        enabled_at = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffffffK")
        reason = "START_ALL auto-enable (override with ACT_DISABLE_PAPER_SOAK_LOOSE=1)"
        sniper = [ordered]@{
            min_score = 4
            min_expected_move_pct = 2.0
            min_confluence = 3
        }
        conviction = [ordered]@{
            min_normal_strategies_agreeing = 2
            bypass_macro_crisis = $true
        }
        cost_gate = [ordered]@{
            min_margin_pct = 0.3
        }
        requires_paper_mode = $true
    }
    try {
        if (-not (Test-Path (Split-Path $soakOverlay -Parent))) {
            New-Item -ItemType Directory -Force -Path (Split-Path $soakOverlay -Parent) | Out-Null
        }
        # PS 5.1 `-Encoding UTF8` writes a BOM that json.load rejects,
        # which makes the overlay invisible to the python reader and
        # silently disables paper_soak_loose on every tick. Use .NET
        # File.WriteAllText with a BOM-free UTF8Encoding instead.
        $jsonStr = $overlayObj | ConvertTo-Json -Depth 5
        $utf8NoBom = New-Object System.Text.UTF8Encoding $false
        [System.IO.File]::WriteAllText($soakOverlay, $jsonStr, $utf8NoBom)
        OK "paper-soak-loose ENABLED (sniper.min_score=4, min_move=2%, confluence=3)."
    } catch {
        WARN "Failed to write paper-soak-loose overlay: $($_.Exception.Message). Run manually: python -m src.skills.cli run paper-soak-loose enable=true"
    }
}

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
# Prefer the named tunnel at infra/cloudflared/config.yml (persistent hostnames,
# runs as a service, sits behind Cloudflare Access). Fall back to the legacy
# quick tunnel only for dev — quick tunnels rotate URL every restart and have
# NO AUTH. See infra/cloudflared/README.md for setup.
STEP 7 "Cloudflare Tunnel [Idle priority]"
$tunnelCfg = Join-Path $PSScriptRoot "infra\cloudflared\config.yml"
$namedTunnelService = Get-Service cloudflared -ErrorAction SilentlyContinue
if ($namedTunnelService -and $namedTunnelService.Status -eq 'Running') {
    OK "Named tunnel already running as a Windows service (preferred)"
    $p7 = $null
} elseif (Test-Path $tunnelCfg) {
    OK "Named tunnel config found at $tunnelCfg — launching via config"
    $p7 = Start-Process cmd.exe -ArgumentList "/k","title ACT - Tunnel (named) && cloudflared --config `"$tunnelCfg`" tunnel run" -PassThru
    try { $p7.PriorityClass = "Idle" } catch {}
    Start-Sleep 2
    OK "Named tunnel PID=$($p7.Id). Install as service for persistence: ./infra/cloudflared/setup_tunnel.ps1 -Domain <your-domain>"
} else {
    WARN "No named-tunnel config at $tunnelCfg — using QUICK tunnel (URL rotates on restart, NO AUTH)."
    WARN "Set up a named tunnel with: ./infra/cloudflared/setup_tunnel.ps1 -Domain <your-domain>"
    $p7 = Start-Process cmd.exe -ArgumentList "/k","title ACT - Tunnel (quick) && cloudflared tunnel --url http://localhost:5173" -PassThru
    try { $p7.PriorityClass = "Idle" } catch {}
    Start-Sleep 2
    OK "Quick tunnel PID=$($p7.Id)"
}

# 8: MCP server (Normal) — re-launches start_mcp.ps1 so STOP_ALL kills it
# cleanly and START_ALL brings it back. This closes the "restart_bot kills
# its own server" gap: previously STOP_ALL killed the MCP python process
# but START_ALL only relaunched the 7 trading processes, leaving MCP dead
# until the operator manually re-ran start_mcp.ps1.
STEP 8 "MCP Server (port 9100) [Normal]"
$mcpScript = Join-Path $PSScriptRoot "scripts\start_mcp.ps1"
if (Test-Path $mcpScript) {
    $p8 = Start-Process cmd.exe -ArgumentList "/k","title ACT - MCP Server && powershell -ExecutionPolicy Bypass -File `"$mcpScript`"" -PassThru
    Start-Sleep 2
    OK "MCP Server PID=$($p8.Id) (uvicorn on http://127.0.0.1:9100)"
} else {
    WARN "scripts\start_mcp.ps1 not found - MCP server NOT started. Claude Code MCP will be unreachable."
}

# ── Summary ──
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "   ALL 8 SYSTEMS RUNNING ($tier)" -ForegroundColor Green
Write-Host ""
Write-Host "   #  Process        Priority      Interval  GPU Use" -ForegroundColor White
Write-Host "   1  API Server     AboveNormal   always    none" -ForegroundColor White
Write-Host "   2  Trading Bot    HIGH          always    LLM inference" -ForegroundColor Red
Write-Host "   3  Adapt Loop     Normal        ${adaptInterval}h        retrain+finetune" -ForegroundColor White
Write-Host "   4  Auto Loop      Normal        ${autoInterval}h      health check" -ForegroundColor White
Write-Host "   5  Genetic Loop   BelowNormal   ${geneticInterval}h        CPU backtest" -ForegroundColor White
Write-Host "   6  Frontend       BelowNormal   always    none" -ForegroundColor White
Write-Host "   7  Tunnel         Idle          always    none" -ForegroundColor White
Write-Host "   8  MCP Server     Normal        always    none" -ForegroundColor White
Write-Host ""
Write-Host "   VRAM: Ollama ~$([math]::Min($gpuVRAM-2, 12))GB | LoRA ~$([math]::Max(2, $gpuVRAM-14))GB | System ~2GB" -ForegroundColor Yellow
Write-Host "   Dashboard:  http://localhost:5173" -ForegroundColor Green
Write-Host "   API:        http://localhost:11007" -ForegroundColor Green
if ($dockerOk) {
    Write-Host "   Grafana:    http://localhost:3000  (login: admin / `$env:GRAFANA_ADMIN_PASSWORD)" -ForegroundColor Green
    Write-Host "   Prometheus: http://localhost:9090  (localhost only — do NOT expose)" -ForegroundColor Green
    Write-Host "   Tempo:      http://localhost:3200  (queried via Grafana)" -ForegroundColor DarkGray
} else {
    Write-Host "   Grafana:    (not started — start Docker Desktop + re-run to enable)" -ForegroundColor DarkYellow
}
Write-Host "   Proc mon:   http://localhost:11007/api/v1/system/processes  (via tunnel once set up)" -ForegroundColor Green
if (Test-Path (Join-Path $PSScriptRoot "infra\cloudflared\config.yml")) {
    Write-Host "   Tunnel:     named tunnel active — see infra/cloudflared/README.md for hostnames" -ForegroundColor Green
} else {
    Write-Host "   Tunnel:     QUICK tunnel only (no auth, rotates URL). Run setup_tunnel.ps1 to upgrade." -ForegroundColor Yellow
}
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Start-Sleep 10
