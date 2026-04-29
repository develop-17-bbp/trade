# ===============================================================
#  ACT 4060 (India) - stop all child processes launched by START_ALL_4060.ps1
# ===============================================================

$titles = @(
    "ACT - Stocks Live (4060)",
    "ACT - Finetune Router (4060)",
    "ACT - warm_store sync (4060)",
    "ACT - Silence watchdog (4060)",
    "ACT - Dashboard (4060)"
)

$found = 0
foreach ($t in $titles) {
    $procs = Get-Process | Where-Object { $_.MainWindowTitle -eq $t }
    foreach ($p in $procs) {
        try {
            Stop-Process -Id $p.Id -Force -ErrorAction Stop
            Write-Host "[ OK ] killed $t (PID=$($p.Id))" -ForegroundColor Green
            $found++
        } catch {
            Write-Host "[FAIL] could not kill $t (PID=$($p.Id)): $_" -ForegroundColor Red
        }
    }
}

if ($found -eq 0) {
    Write-Host "[INFO] no ACT 4060 processes found" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "Stopped $found processes."
}
