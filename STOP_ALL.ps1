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
                    -TimeoutSec 10 -ErrorAction Stop -UseBasicParsing | Out-Null
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

# Forbidden-model purge. If ACT_PURGE_FORBIDDEN_MODELS=1 AND
# ACT_FORBID_MODELS is set, also `ollama rm` each forbidden model so
# it's truly removed from disk -- not just unloaded from VRAM. This
# completes the "totally remove deepseek-r1" workflow: set
#   ACT_FORBID_MODELS=deepseek-r1:7b,deepseek-r1:32b
#   ACT_PURGE_FORBIDDEN_MODELS=1
# then STOP_ALL once. Default off so a normal stop preserves the cache.
$forbidRaw = $env:ACT_FORBID_MODELS
if ($forbidRaw -and $env:ACT_PURGE_FORBIDDEN_MODELS -eq '1') {
    $forbidList = $forbidRaw.ToLower().Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
    Write-Host "  Purging forbidden Ollama models from disk..." -ForegroundColor Yellow
    try {
        $listed = & ollama list 2>$null
        foreach ($line in $listed) {
            $parts = $line.ToString().Trim() -split '\s+'
            if ($parts.Count -lt 1) { continue }
            $name = $parts[0].ToLower()
            if (-not $name -or $name -eq 'name') { continue }
            $head = $name.Split(":")[0]
            foreach ($entry in $forbidList) {
                if ($entry -eq $name -or $entry -eq $head -or $name.Contains($entry)) {
                    Write-Host "    Removing: $name" -ForegroundColor Yellow
                    & ollama rm $name 2>$null | Out-Null
                    break
                }
            }
        }
    } catch {
        Write-Host "    Purge skipped: $($_.Exception.Message)" -ForegroundColor Gray
    }
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
