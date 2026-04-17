# ACT Trading System — Stop All Processes
# Run: powershell -ExecutionPolicy Bypass -File STOP_ALL.ps1

Write-Host ""
Write-Host "   STOPPING ALL ACT PROCESSES..." -ForegroundColor Red
Write-Host ""

# Kill ACT-titled windows
Get-Process | Where-Object { $_.MainWindowTitle -like "ACT*" } | ForEach-Object {
    Write-Host "  Stopping: $($_.MainWindowTitle)" -ForegroundColor Yellow
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}

# Kill Python processes
$py = Get-Process python -ErrorAction SilentlyContinue
if ($py) {
    Write-Host "  Stopping $($py.Count) Python process(es)..." -ForegroundColor Yellow
    $py | Stop-Process -Force -ErrorAction SilentlyContinue
}

# Kill Node processes (frontend)
$node = Get-Process node -ErrorAction SilentlyContinue
if ($node) {
    Write-Host "  Stopping $($node.Count) Node process(es)..." -ForegroundColor Yellow
    $node | Stop-Process -Force -ErrorAction SilentlyContinue
}

Start-Sleep 2
Write-Host ""
Write-Host "   ALL ACT PROCESSES STOPPED" -ForegroundColor Green
Write-Host ""
Start-Sleep 3
