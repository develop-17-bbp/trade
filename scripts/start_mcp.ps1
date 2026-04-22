# Start the ACT MCP server on the GPU box.
# Exposes inspection tools for Claude Code sessions over HTTP.
# Routed via Cloudflare tunnel (mcp.<domain>) with Access auth in front.
#
# Usage:
#     powershell -ExecutionPolicy Bypass -File .\scripts\start_mcp.ps1
#     powershell -ExecutionPolicy Bypass -File .\scripts\start_mcp.ps1 -Port 9100
#     powershell -ExecutionPolicy Bypass -File .\scripts\start_mcp.ps1 -AllowMutations
#
# Before first run:
#   1. Install mcp SDK:  pip install mcp
#   2. Set a token so the server rejects unauth requests even if CF Access is down:
#        setx ACT_MCP_TOKEN "<strong-random-string>"
#   3. Update infra/cloudflared/config.yml to add an ingress rule for mcp.<domain>
#      -> http://localhost:9100 (see infra/cloudflared/README.md)

[CmdletBinding()]
param(
    [int]$Port = 9100,
    # Renamed from $Host - PowerShell's $Host is a read-only automatic variable
    # and using it as a parameter throws "Cannot overwrite variable Host".
    [string]$BindHost = "127.0.0.1",
    [switch]$AllowMutations
)

$ErrorActionPreference = "Continue"

function Section($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function OK($msg)      { Write-Host "  [OK] $msg" -ForegroundColor Green }
function WARN($msg)    { Write-Host "  [WARN] $msg" -ForegroundColor Yellow }

Section "Env check"
$tokenSet = [bool]$env:ACT_MCP_TOKEN
Write-Host "  ACT_MCP_TOKEN set:       $tokenSet"
Write-Host "  ACT_MCP_ALLOW_MUTATIONS: $env:ACT_MCP_ALLOW_MUTATIONS"
Write-Host "  Port:                    $Port"
Write-Host "  Bind host:               $BindHost"

if (-not $tokenSet) {
    WARN "ACT_MCP_TOKEN is not set. The server will rely only on Cloudflare Access for auth."
    WARN "For belt-and-suspenders, run: setx ACT_MCP_TOKEN `"<strong-random-string>`""
}

if ($AllowMutations) {
    $env:ACT_MCP_ALLOW_MUTATIONS = "1"
    WARN "MUTATIONS ENABLED for this session. restart_bot + trigger_retrain callable."
}

Section "Verify mcp SDK"
python -c "from mcp.server.fastmcp import FastMCP; print('OK')" 2>&1
if ($LASTEXITCODE -ne 0) {
    WARN "mcp SDK missing. Installing..."
    python -m pip install mcp
    if ($LASTEXITCODE -ne 0) {
        Write-Error "mcp install failed. Install manually: pip install mcp"
        exit 1
    }
}

Section "Starting MCP server"
Write-Host "  URL (via tunnel): https://mcp.<yourdomain>/"
Write-Host "  Local URL:        http://${BindHost}:${Port}/mcp"
Write-Host ""
Write-Host "  Press Ctrl+C to stop."
Write-Host ""
python -m src.mcp_server.act_mcp --host $BindHost --port $Port
