Param(
    [switch]$Dev,
    [switch]$Verify
)
# DIX VISION v42.2 — Windows run wrapper
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $root

if (-not (Test-Path ".venv")) {
    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    if (Test-Path "requirements-windows.txt") {
        .\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
    }
}

$py = ".\.venv\Scripts\python.exe"
if ($Verify) {
    & $py dix.py verify
} elseif ($Dev) {
    & $py main.py --dev
} else {
    & $py main.py
}
