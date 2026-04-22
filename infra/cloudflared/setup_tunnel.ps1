# infra/cloudflared/setup_tunnel.ps1
# ----------------------------------------------------------------------------
# Interactive one-time setup for the ACT GPU box's named Cloudflare tunnel.
# Replaces the ephemeral `cloudflared tunnel --url ...` quick tunnel with a
# persistent named tunnel that:
#   * has a stable hostname (grafana.<domain>, api.<domain>)
#   * runs as a Windows service (auto-starts on reboot)
#   * survives cloudflared restarts without changing URL
#
# Prerequisites (MANUAL - cannot be automated):
#   * A Cloudflare account with a domain (free tier OK)
#   * cloudflared.exe in PATH (START_ALL.ps1 downloads it if missing)
#
# Usage (run from an elevated PowerShell on the GPU box):
#     powershell -ExecutionPolicy Bypass -File .\infra\cloudflared\setup_tunnel.ps1 -Domain yourdomain.com
#
# The script will:
#   1. cloudflared tunnel login             (opens browser, auth once)
#   2. cloudflared tunnel create act-gpu    (prints UUID + creds path)
#   3. Fill config.yml from template using those values + your domain
#   4. cloudflared tunnel route dns for each hostname
#   5. Install cloudflared as a Windows service
#
# Defense in depth - DO BEFORE SENDING THE URLS TO ANYONE:
#   a) Go to Cloudflare Zero Trust dashboard (https://one.dash.cloudflare.com/)
#   b) Access -> Applications -> Add an Application -> Self-hosted
#   c) For each hostname (grafana.<domain>, api.<domain>):
#        - Application domain: the hostname
#        - Session duration: 24 hours
#   d) Add a policy: "Allow", Include: emails in { your email }
#   e) Save. Now every request to the tunnel hits Cloudflare Access first
#      and requires an email-OTP / SSO login before reaching the GPU box.

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$Domain,

    [string]$TunnelName = "act-gpu",

    [switch]$SkipService
)

$ErrorActionPreference = "Stop"

function OK   ($m) { Write-Host "  [OK] $m" -ForegroundColor Green }
function WARN ($m) { Write-Host "  [WARN] $m" -ForegroundColor Yellow }
function STEP ($n, $m) { Write-Host "`n=== Step $n : $m ===" -ForegroundColor Cyan }

$ScriptDir = Split-Path -Parent $PSCommandPath
$TemplatePath = Join-Path $ScriptDir "config.yml.template"
$ConfigPath   = Join-Path $ScriptDir "config.yml"

if (-not (Test-Path $TemplatePath)) {
    Write-Error "config.yml.template not found at $TemplatePath"
    exit 1
}

# Verify cloudflared is available
$cfd = Get-Command cloudflared.exe -ErrorAction SilentlyContinue
if (-not $cfd) {
    WARN "cloudflared.exe not in PATH. Run START_ALL.ps1 once (it will download cloudflared) or install manually."
    exit 1
}
OK "cloudflared found at $($cfd.Path)"

# ------ Step 1: login ------
STEP 1 "Cloudflare account login (opens browser)"
Write-Host "A browser window will open. Pick the domain $Domain from the dropdown and Authorize."
Read-Host "Press ENTER to continue"
cloudflared tunnel login
if ($LASTEXITCODE -ne 0) {
    Write-Error "cloudflared tunnel login failed"
    exit 1
}
OK "Login cert saved to %USERPROFILE%\.cloudflared\cert.pem"

# ------ Step 2: create tunnel ------
STEP 2 "Create named tunnel '$TunnelName'"
$CreateOutput = cloudflared tunnel create $TunnelName 2>&1 | Out-String
Write-Host $CreateOutput

$UUID = ($CreateOutput | Select-String -Pattern '([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})').Matches |
        ForEach-Object { $_.Value } | Select-Object -First 1
if (-not $UUID) {
    # Tunnel may already exist - fetch its UUID instead
    $ListOutput = cloudflared tunnel list 2>&1 | Out-String
    $UUID = ($ListOutput | Select-String -Pattern "([0-9a-f-]{36})\s+$TunnelName").Matches |
            ForEach-Object { $_.Groups[1].Value } | Select-Object -First 1
}
if (-not $UUID) {
    Write-Error "Could not determine tunnel UUID. Manual: cloudflared tunnel list"
    exit 1
}
OK "Tunnel UUID: $UUID"

$CredsPath = Join-Path $env:USERPROFILE ".cloudflared\$UUID.json"
if (-not (Test-Path $CredsPath)) {
    WARN "Credentials file not at expected path $CredsPath - search for $UUID.json under .cloudflared/"
    exit 1
}
OK "Credentials file: $CredsPath"

# ------ Step 3: render config.yml from template ------
STEP 3 "Write config.yml from template"
$Tmpl = Get-Content $TemplatePath -Raw
$Rendered = $Tmpl `
    -replace 'REPLACE_ME_TUNNEL_UUID', $UUID `
    -replace 'REPLACE_ME_CREDENTIALS_PATH', ($CredsPath -replace '\\','\\\\') `
    -replace 'REPLACE_ME_DOMAIN', $Domain
Set-Content -Path $ConfigPath -Value $Rendered -Encoding UTF8
OK "config.yml written to $ConfigPath"

# ------ Step 4: route DNS (one CNAME per hostname) ------
STEP 4 "Route DNS for each hostname"
foreach ($Sub in @("grafana", "api", "mcp")) {
    $Host1 = "$Sub.$Domain"
    Write-Host "  Routing $Host1 -> $UUID"
    cloudflared tunnel route dns $TunnelName $Host1
    if ($LASTEXITCODE -ne 0) {
        WARN "DNS route for $Host1 failed (may already exist). Continuing."
    } else {
        OK "DNS: $Host1"
    }
}

# ------ Step 5: install as Windows service ------
if ($SkipService) {
    WARN "Skipping service install (--SkipService set)"
} else {
    STEP 5 "Install cloudflared as Windows service"
    Write-Host "This needs Administrator privileges. If it fails, re-run the script from an elevated PowerShell."
    cloudflared --config $ConfigPath service install
    if ($LASTEXITCODE -eq 0) {
        OK "Service installed. Starting..."
        Start-Service cloudflared -ErrorAction SilentlyContinue
        $svc = Get-Service cloudflared -ErrorAction SilentlyContinue
        if ($svc) { OK "Service status: $($svc.Status)" }
    } else {
        WARN "Service install failed. Manual: run this script as Administrator, or: cloudflared --config $ConfigPath service install"
    }
}

# ------ Done ------
Write-Host "`n=============================================================" -ForegroundColor Green
Write-Host " Tunnel setup complete" -ForegroundColor Green
Write-Host "=============================================================" -ForegroundColor Green
Write-Host " URLs (after DNS propagates, 1-2 minutes):"
Write-Host "   https://grafana.$Domain"
Write-Host "   https://api.$Domain/api/v1/system/processes"
Write-Host ""
Write-Host " NEXT - Cloudflare Access (Zero Trust) is REQUIRED before sharing these URLs:"
Write-Host "   1. Open https://one.dash.cloudflare.com/"
Write-Host "   2. Access -> Applications -> Add -> Self-hosted"
Write-Host "   3. Add grafana.$Domain and api.$Domain (one policy each)"
Write-Host "   4. Include rule: emails { your email address }"
Write-Host "   5. Save. Every request will then require SSO/OTP login."
Write-Host ""
Write-Host " To check the tunnel:    cloudflared tunnel info $TunnelName"
Write-Host " To restart the service: Restart-Service cloudflared"
Write-Host ""
