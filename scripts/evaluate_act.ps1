# Run the ACT evaluation report. Prints a colored terminal view of
# component state + attribution + recommendations.
#
# Usage:
#     powershell -ExecutionPolicy Bypass -File .\scripts\evaluate_act.ps1
#     powershell -ExecutionPolicy Bypass -File .\scripts\evaluate_act.ps1 -Json
#     powershell -ExecutionPolicy Bypass -File .\scripts\evaluate_act.ps1 -NoColors

[CmdletBinding()]
param(
    [switch]$Json,
    [switch]$NoColors
)

$ErrorActionPreference = "Continue"

$scriptArgs = @("scripts/evaluate_act.py")
if ($Json)      { $scriptArgs += "--json" }
if ($NoColors)  { $scriptArgs += "--no-colors" }

python @scriptArgs
exit $LASTEXITCODE
