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

# Stop the observability stack (Grafana + Prometheus + Tempo + Redis + OTel).
# Use `down` (not `down -v`) so Prometheus TSDB + Grafana dashboards + Redis
# AOF survive the restart.
$composeFile = Join-Path $PSScriptRoot "infra\docker-compose.yml"
if (Test-Path $composeFile) {
    try {
        docker info --format '{{.ServerVersion}}' 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Stopping observability stack (docker compose down)..." -ForegroundColor Yellow
            docker compose -f $composeFile down 2>&1 | Out-Null
        }
    } catch {}
}

Write-Host ""
Write-Host "   ALL ACT PROCESSES STOPPED" -ForegroundColor Green
Write-Host ""
Start-Sleep 3
