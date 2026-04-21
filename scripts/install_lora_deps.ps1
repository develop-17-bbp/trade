# Install LoRA / fine-tuning dependencies that are currently missing on the GPU box.
# Fixes the "Scanner fine-tune failed: No module named 'peft'" warnings in
# logs/autonomous_loop.log. Also installs yfinance which three economic-layer
# modules depend on.
#
# Usage (on the GPU box, in any PowerShell):
#     powershell -ExecutionPolicy Bypass -File .\scripts\install_lora_deps.ps1

$ErrorActionPreference = "Continue"

function Section($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Info($msg)    { Write-Host "  $msg" -ForegroundColor Gray }
function OK($msg)      { Write-Host "  [OK] $msg" -ForegroundColor Green }
function WARN($msg)    { Write-Host "  [WARN] $msg" -ForegroundColor Yellow }

Section "Upgrading pip first"
python -m pip install --upgrade pip

Section "Core LoRA stack (peft + bitsandbytes + accelerate)"
python -m pip install peft bitsandbytes accelerate
if ($LASTEXITCODE -eq 0) { OK "LoRA core installed" } else { WARN "LoRA core failed - check GPU/CUDA match" }

Section "Unsloth (optional but preferred - 2x training speed, 60 pct less VRAM)"
python -m pip install "unsloth[cu121-torch241] @ git+https://github.com/unslothai/unsloth.git" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    WARN "unsloth install failed - trainer will fall back to standard HuggingFace (slower but works)"
} else {
    OK "unsloth installed"
}

Section "yfinance (3 economic layers depend on it)"
python -m pip install yfinance
if ($LASTEXITCODE -eq 0) { OK "yfinance installed" } else { WARN "yfinance failed" }

Section "Verification"
python -c "
import importlib
checks = [('peft','PEFT'),('bitsandbytes','bnb'),('accelerate','accelerate'),('yfinance','yfinance'),('unsloth','unsloth (optional)')]
for mod, label in checks:
    try:
        importlib.import_module(mod)
        print('  [OK] ' + label)
    except Exception as e:
        print('  [MISS] ' + label + ': ' + str(e))
"

Section "Done"
Write-Host "  If any [MISS] above - that is what is still needed. Otherwise the LoRA" -ForegroundColor Gray
Write-Host "  trainer will now run cleanly next cycle." -ForegroundColor Gray
