# Train the rule-conditional meta-label model for BTC + ETH.
# Produces models/lgbm_{asset}_meta.txt + meta_calibration.json + meta_thresholds.json.
# The executor loads these at startup and uses the model as a VETO-ONLY gate AFTER
# rule signals fire.
#
# Usage (on the GPU box):
#     powershell -ExecutionPolicy Bypass -File .\scripts\train_meta_label.ps1
#     powershell -ExecutionPolicy Bypass -File .\scripts\train_meta_label.ps1 -Days 365

[CmdletBinding()]
param(
    [int]$Days = 180,
    [string]$PrimaryTf = "5m",
    [int]$MinScore = 4,
    [string[]]$Assets = @("BTC","ETH"),
    # Wipe the existing meta model first so the champion gate promotes freely.
    # Useful when prior label-schema changes left a stale low-F1 incumbent.
    [switch]$Force,
    # Spread-pct deducted from label. Default 0 - ML predicts direction-correctness;
    # spread is handled by the executor's exit strategy, not the model.
    [double]$SpreadPct = 0.0
)

$ErrorActionPreference = "Continue"

function Section($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function OK($msg)      { Write-Host "  [OK] $msg" -ForegroundColor Green }
function WARN($msg)    { Write-Host "  [WARN] $msg" -ForegroundColor Yellow }

Section "Env"
Write-Host "  ACT_LGBM_DEVICE = $env:ACT_LGBM_DEVICE"
Write-Host "  Days            = $Days"
Write-Host "  Primary TF      = $PrimaryTf"
Write-Host "  Min score       = $MinScore"
Write-Host "  Assets          = $Assets"

$forceFlag = if ($Force) { "--force" } else { "" }

foreach ($asset in $Assets) {
    Section "Training meta-label model for $asset ($Days days, $PrimaryTf, spread=$SpreadPct)"
    $args = @("-m","src.scripts.train_meta_label","--asset",$asset,"--days",$Days,
              "--primary-tf",$PrimaryTf,"--min-score",$MinScore,"--spread-pct",$SpreadPct)
    if ($Force) { $args += "--force" }
    & python @args
    if ($LASTEXITCODE -ne 0) {
        WARN "$asset meta training failed or rejected by champion gate"
    } else {
        OK "$asset meta model deployed"
    }
}

Section "Verification"
foreach ($asset in $Assets) {
    $p = "models/lgbm_$($asset.ToLower())_meta.txt"
    if (Test-Path $p) {
        $sz = (Get-Item $p).Length
        OK "$p ($sz bytes)"
    } else {
        WARN "MISSING: $p"
    }
    $cp = "models/lgbm_$($asset.ToLower())_meta_calibration.json"
    if (Test-Path $cp) { OK "$cp" } else { WARN "missing calibration: $cp" }
    $tp = "models/lgbm_$($asset.ToLower())_meta_thresholds.json"
    if (Test-Path $tp) { OK "$tp" } else { WARN "missing thresholds: $tp" }
}

Section "Done"
Write-Host "  After restart (STOP_ALL then START_ALL), executor log will show:" -ForegroundColor Gray
Write-Host "    [ML] LightGBM (BTC) META model loaded (rule-conditional, veto-only)" -ForegroundColor Gray
Write-Host "    [ML] META (BTC) calibrated - base_wr=... deltas=..." -ForegroundColor Gray
Write-Host "  Per-decision log lines:" -ForegroundColor Gray
Write-Host "    [BTC] META TAKE: prob=0.62 >= take=0.55" -ForegroundColor Gray
Write-Host "    [BTC] META VETO: prob=0.41 < take=0.55 - score -> 1" -ForegroundColor Gray
Write-Host ""
Write-Host "  The meta model can ONLY subtract from entry score (veto rule-approved trades)." -ForegroundColor Gray
Write-Host "  Set ACT_DISABLE_ML=1 to bypass the meta gate too if needed." -ForegroundColor Gray
