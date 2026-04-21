# Shadow-retrain the meta model using LIVE trade outcomes accumulated in
# logs/meta_shadow.jsonl (written by the executor when ACT_META_SHADOW_MODE=1).
#
# Prereq: bot has run with shadow mode ON for enough paper trades (>= 100 joined).
#
# Usage:
#     powershell -ExecutionPolicy Bypass -File .\scripts\shadow_retrain.ps1
#     powershell -ExecutionPolicy Bypass -File .\scripts\shadow_retrain.ps1 -StatsOnly
#     powershell -ExecutionPolicy Bypass -File .\scripts\shadow_retrain.ps1 -MinJoined 150 -Force

[CmdletBinding()]
param(
    [int]$MinJoined = 100,
    [string[]]$Assets = @("BTC","ETH"),
    [switch]$Force,
    [switch]$StatsOnly
)

$ErrorActionPreference = "Continue"

function Section($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function OK($msg)      { Write-Host "  [OK] $msg" -ForegroundColor Green }
function WARN($msg)    { Write-Host "  [WARN] $msg" -ForegroundColor Yellow }

Section "Env"
Write-Host "  Shadow log path = logs\meta_shadow.jsonl"
Write-Host "  Assets          = $Assets"
Write-Host "  Min joined      = $MinJoined"
Write-Host "  Force           = $Force"
Write-Host "  Stats only      = $StatsOnly"

if ($StatsOnly) {
    Section "Shadow stats (no retrain)"
    python -m src.scripts.shadow_retrain --stats-only
    exit $LASTEXITCODE
}

foreach ($asset in $Assets) {
    Section "Shadow-retrain for $asset"
    $args = @("-m","src.scripts.shadow_retrain","--asset",$asset,"--min-joined",$MinJoined)
    if ($Force) { $args += "--force" }
    & python @args
    if ($LASTEXITCODE -eq 0) {
        OK "$asset shadow-retrained + deployed"
    } else {
        WARN "$asset retrain did not promote (see log above)"
    }
}

Section "Done"
Write-Host "  After a successful retrain, restart the bot:" -ForegroundColor Gray
Write-Host "    .\STOP_ALL.ps1 ; .\START_ALL.ps1" -ForegroundColor Gray
Write-Host "  The updated meta model will load at startup." -ForegroundColor Gray
