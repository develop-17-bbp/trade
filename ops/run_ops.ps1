# Windows Task Scheduler wrapper for ops/act_ops.py
# Schedule with:
#   schtasks /Create /TN "ACT Ops Agent" /TR "powershell -ExecutionPolicy Bypass -File C:\Users\admin\trade\ops\run_ops.ps1" /SC MINUTE /MO 15 /RU "%USERNAME%"
#
# Or open Task Scheduler GUI and import act_ops_task.xml.

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

# Load .env into the current process so checks see the live env state
$envPath = Join-Path $repoRoot ".env"
if (Test-Path $envPath) {
    Get-Content $envPath | ForEach-Object {
        if ($_ -match "^\s*([^#=][^=]*)=(.*)$") {
            [System.Environment]::SetEnvironmentVariable(
                $matches[1].Trim(), $matches[2].Trim(), "Process"
            )
        }
    }
}

# Run the cycle. Output goes to logs\act_ops_cron.log; the structured
# JSONL audit trail lives in logs\act_ops.jsonl.
$logPath = Join-Path $repoRoot "logs\act_ops_cron.log"
$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"--- $ts cycle start ---" | Out-File -FilePath $logPath -Append -Encoding utf8

try {
    python -m ops.act_ops 2>&1 | Out-File -FilePath $logPath -Append -Encoding utf8
    "--- $ts cycle ok ---" | Out-File -FilePath $logPath -Append -Encoding utf8
} catch {
    "--- $ts cycle ERROR: $($_.Exception.Message) ---" | Out-File -FilePath $logPath -Append -Encoding utf8
    exit 1
}
