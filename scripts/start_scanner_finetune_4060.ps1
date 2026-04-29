# start_scanner_finetune_4060.ps1
# Launches the scanner-only QLoRA fine-tune loop on the RTX 4060 box.
#
# Pre-flight: nvidia-smi check, SSH connectivity, env var sanity.
# Then runs scripts/finetune_scanner_4060.py forever (or with --once / --dry-run).
#
# Usage:
#   .\scripts\start_scanner_finetune_4060.ps1
#   .\scripts\start_scanner_finetune_4060.ps1 -Once
#   .\scripts\start_scanner_finetune_4060.ps1 -DryRun

param(
    [switch]$Once,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

# ── ANSI banner: A=orange, C=red, T=green ──
function Write-Banner {
    $orange = "$([char]27)[38;5;208m"
    $red    = "$([char]27)[31m"
    $green  = "$([char]27)[32m"
    $reset  = "$([char]27)[0m"
    Write-Host ""
    Write-Host "  ${orange}A${reset}${red}C${reset}${green}T${reset} - Scanner Fine-Tune [4060 India]"
    Write-Host "  ────────────────────────────────────────────────"
    Write-Host ""
}

function Fail($msg) {
    Write-Host "[FAIL] $msg" -ForegroundColor Red
    exit 1
}

function Ok($msg) {
    Write-Host "[ ok ] $msg" -ForegroundColor Green
}

function Warn($msg) {
    Write-Host "[warn] $msg" -ForegroundColor Yellow
}

Write-Banner

# ── Resolve repo root (script is in repo/scripts/) ──
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
Set-Location $repoRoot
Ok "repo root = $repoRoot"

# ── Env var sanity ──
if (-not $env:ACT_5090_SSH_HOST) {
    Fail "ACT_5090_SSH_HOST not set — see docs/finetune_two_box.md"
}
if (-not $env:ACT_5090_TRADE_DIR) {
    Fail "ACT_5090_TRADE_DIR not set — e.g. C:/Users/admin/trade on the 5090"
}
Ok "ACT_5090_SSH_HOST = $env:ACT_5090_SSH_HOST"
Ok "ACT_5090_TRADE_DIR = $env:ACT_5090_TRADE_DIR"

# ── nvidia-smi check ──
try {
    $smi = nvidia-smi --query-gpu=name,memory.free --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $smi) {
        if ($DryRun) {
            Warn "nvidia-smi unavailable — proceeding because --DryRun"
        } else {
            Fail "nvidia-smi failed — install NVIDIA driver / CUDA"
        }
    } else {
        $line = ($smi -split "`n")[0].Trim()
        Ok "GPU: $line"
        $freeMb = [int](($line -split ',')[1].Trim())
        $minVram = if ($env:ACT_SCANNER_MIN_VRAM_MB) { [int]$env:ACT_SCANNER_MIN_VRAM_MB } else { 7000 }
        if ($freeMb -lt $minVram -and -not $DryRun) {
            Fail "free VRAM $freeMb MB < $minVram MB threshold"
        }
    }
} catch {
    if (-not $DryRun) { Fail "GPU check exception: $_" }
}

# ── SSH connectivity probe ──
Write-Host "[ .. ] probing SSH to $env:ACT_5090_SSH_HOST ..."
$sshTest = ssh -o BatchMode=yes -o ConnectTimeout=10 $env:ACT_5090_SSH_HOST "echo ok" 2>&1
if ($LASTEXITCODE -ne 0 -or $sshTest -notmatch 'ok') {
    Fail "ssh ${env:ACT_5090_SSH_HOST} failed — set up keys per docs/finetune_two_box.md. Output: $sshTest"
}
Ok "SSH to $env:ACT_5090_SSH_HOST = ok"

# ── Verify remote warm_store exists ──
$remoteCheck = ssh -o BatchMode=yes $env:ACT_5090_SSH_HOST "powershell.exe -NoProfile -Command `"if (Test-Path '$env:ACT_5090_TRADE_DIR/data/warm_store.sqlite') { 'ok' } else { 'missing' }`"" 2>&1
if ($remoteCheck -notmatch 'ok') {
    Fail "remote warm_store.sqlite not found at $env:ACT_5090_TRADE_DIR/data/ — output: $remoteCheck"
}
Ok "remote warm_store.sqlite present"

# ── Build python args ──
$pyArgs = @('-m', 'scripts.finetune_scanner_4060')
if ($Once)   { $pyArgs += '--once' }
if ($DryRun) { $pyArgs += '--dry-run' }

Write-Host ""
Write-Host "[ go ] launching: python $($pyArgs -join ' ')" -ForegroundColor Cyan
Write-Host ""

$env:PYTHONUNBUFFERED = '1'
$env:CUDA_VISIBLE_DEVICES = '0'

# Run in-process so the operator sees logs in this terminal.
& python @pyArgs
exit $LASTEXITCODE
