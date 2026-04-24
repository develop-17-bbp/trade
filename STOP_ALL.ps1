# ACT Trading System -- Stop All Processes
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

# Unload brain models from Ollama VRAM. START_ALL pinned them with
# keep_alive=-1; STOP_ALL releases by sending keep_alive=0 (ollama
# evicts immediately) for each currently-resident model. The Ollama
# service itself stays running (Windows background service);
# only the model weights are evicted from GPU memory so ACT is
# truly idle when stopped.
$ollamaUrl = if ($env:OLLAMA_BASE_URL) { $env:OLLAMA_BASE_URL.TrimEnd('/') } else { 'http://127.0.0.1:11434' }
try {
    $ps = Invoke-RestMethod -Uri "$ollamaUrl/api/ps" -TimeoutSec 5 -ErrorAction Stop
    $resident = @($ps.models | ForEach-Object { $_.name })
    if ($resident.Count -gt 0) {
        Write-Host "  Unloading $($resident.Count) Ollama model(s) from VRAM..." -ForegroundColor Yellow
        foreach ($m in $resident) {
            $payload = @{ model = $m; keep_alive = 0 } | ConvertTo-Json -Compress
            try {
                Invoke-WebRequest -Uri "$ollamaUrl/api/generate" `
                    -Method Post -Body $payload -ContentType 'application/json' `
                    -TimeoutSec 10 -ErrorAction Stop | Out-Null
                Write-Host "    Unloaded: $m" -ForegroundColor Yellow
            } catch {
                # Fallback to ollama CLI stop
                & ollama stop $m 2>$null
            }
        }
    } else {
        Write-Host "  No Ollama models resident -- skipping unload." -ForegroundColor Gray
    }
} catch {
    Write-Host "  Ollama not reachable for unload (fine if Ollama service is down)." -ForegroundColor Gray
}

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
Write-Host "   ALL ACT PROCESSES STOPPED -- VRAM RELEASED" -ForegroundColor Green
Write-Host ""
Start-Sleep 3
