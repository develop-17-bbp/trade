# Force the first retrain cycle after deploying the ML-calibration + safe-entries
# commits. Writes the per-asset lgbm_{asset}_calibration.json files so the
# executor's calibrated-path (LGBM[cal]) activates instead of the hand-tuned
# fallback (LGBM[raw]).
#
# Usage (on the GPU box):
#     powershell -ExecutionPolicy Bypass -File .\scripts\first_retrain.ps1

$ErrorActionPreference = "Continue"

function Section($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function OK($msg)      { Write-Host "  [OK] $msg" -ForegroundColor Green }
function WARN($msg)    { Write-Host "  [WARN] $msg" -ForegroundColor Yellow }

Section "Checking env"
Write-Host "  ACT_LGBM_DEVICE    = $env:ACT_LGBM_DEVICE"
Write-Host "  ACT_SAFE_ENTRIES   = $env:ACT_SAFE_ENTRIES"
Write-Host "  DASHBOARD_API_KEY  = $([bool]$env:DASHBOARD_API_KEY)"

Section "Training BTC binary model (calibration + champion gate will fire)"
python -m src.scripts.train_all_models --asset BTC --bars 20000
if ($LASTEXITCODE -ne 0) {
    WARN "BTC train failed - check logs above"
}

Section "Training ETH binary model"
python -m src.scripts.train_all_models --asset ETH --bars 20000
if ($LASTEXITCODE -ne 0) {
    WARN "ETH train failed"
}

Section "Verification - calibration JSONs should now exist"
$expected = @(
    "models/lgbm_btc_calibration.json",
    "models/lgbm_eth_calibration.json"
)
foreach ($p in $expected) {
    if (Test-Path $p) {
        $size = (Get-Item $p).Length
        OK "$p ($size bytes)"
    } else {
        WARN "MISSING: $p"
    }
}

Section "Sanity-parse the calibration"
python -c "
import json
for asset in ('btc', 'eth'):
    path = 'models/lgbm_' + asset + '_calibration.json'
    try:
        b = json.load(open(path))
        print('  [OK] ' + asset + ': buckets=' + str(b['buckets']) + ' deltas=' + str(b['deltas']) + ' base_wr=' + ('%.3f' % b['baseline_win_rate']) + ' n=' + str(b['fit_n_samples']))
    except Exception as e:
        print('  [FAIL] ' + asset + ': ' + str(e))
"

Section "Done"
Write-Host "  After restart (STOP_ALL then START_ALL), executor log will show:" -ForegroundColor Gray
Write-Host "    [ML] LightGBM (BTC) calibrated - base_wr=... deltas=..." -ForegroundColor Gray
Write-Host "    [BTC] LGBM[cal]: TRADE conf=0.65 | score now=6   (instead of LGBM[raw])" -ForegroundColor Gray
