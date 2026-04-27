# ===============================================================
#  unblock_entries.ps1 — one-shot entry-rate unblock
# ===============================================================
# Sets the four env knobs that have been blocking trades (relaxed
# SMALL_BODY ratio + measured Robinhood spread/fees + safe brain
# profile), then stops + restarts all ACT processes so the new env
# is actually picked up. setx alone is not enough — running cmd
# windows have stale env baked in at launch.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\unblock_entries.ps1
#   powershell -ExecutionPolicy Bypass -File .\scripts\unblock_entries.ps1 -SmallBodyRatio 0.7
#
# All values default to Claude's recommended starting point. Override
# any of them via parameters if you want different settings.

[CmdletBinding()]
param(
    [double]$SmallBodyRatio       = 0.5,
    [double]$RobinhoodSpreadPct   = 1.0,
    [double]$RobinhoodFeesPct     = 0.30,
    [string]$BrainProfile         = "dense_r1"
)

$repoRoot = Split-Path -Parent $PSScriptRoot

function Section($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function OK($msg)      { Write-Host "  [OK] $msg"   -ForegroundColor Green }
function INFO($msg)    { Write-Host "  [..] $msg"   -ForegroundColor White }

Section "1/4  Set persistent env vars (registry)"
setx ACT_SMALL_BODY_RATIO     $SmallBodyRatio       | Out-Null
setx ACT_ROBINHOOD_SPREAD_PCT $RobinhoodSpreadPct   | Out-Null
setx ACT_ROBINHOOD_FEES_PCT   $RobinhoodFeesPct     | Out-Null
setx ACT_BRAIN_PROFILE        $BrainProfile         | Out-Null
OK "ACT_SMALL_BODY_RATIO     = $SmallBodyRatio"
OK "ACT_ROBINHOOD_SPREAD_PCT = $RobinhoodSpreadPct"
OK "ACT_ROBINHOOD_FEES_PCT   = $RobinhoodFeesPct"
OK "ACT_BRAIN_PROFILE        = $BrainProfile"

Section "2/4  Set in current process (so START_ALL children inherit immediately)"
$env:ACT_SMALL_BODY_RATIO     = $SmallBodyRatio
$env:ACT_ROBINHOOD_SPREAD_PCT = $RobinhoodSpreadPct
$env:ACT_ROBINHOOD_FEES_PCT   = $RobinhoodFeesPct
$env:ACT_BRAIN_PROFILE        = $BrainProfile
OK "current shell inherits the new values"

Section "3/4  Stop all running ACT processes"
$stopScript = Join-Path $repoRoot "STOP_ALL.ps1"
if (Test-Path $stopScript) {
    & powershell -ExecutionPolicy Bypass -File $stopScript
    OK "STOP_ALL.ps1 finished"
} else {
    Write-Host "  [WARN] $stopScript not found — skipping" -ForegroundColor Yellow
}

Section "4/4  Start fresh — new processes will read the new env"
$startScript = Join-Path $repoRoot "START_ALL.ps1"
if (Test-Path $startScript) {
    & powershell -ExecutionPolicy Bypass -File $startScript
    OK "START_ALL.ps1 launched"
} else {
    Write-Host "  [WARN] $startScript not found — skipping" -ForegroundColor Yellow
}

Section "Done"
Write-Host "  Verify with: " -NoNewline; Write-Host "Get-ChildItem env:ACT_*" -ForegroundColor Yellow
Write-Host "  Or via Claude: ask it to call MCP env_flags and confirm the four values are non-None."
Write-Host ""
Write-Host "  Within 5 minutes the bot's next decision tick should use the relaxed thresholds." -ForegroundColor Green
Write-Host "  Watch the Trading Bot cmd window for ENTRY lines, or pull recent_trades via MCP." -ForegroundColor Green
