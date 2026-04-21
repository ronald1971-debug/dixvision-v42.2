Param(
    [string]$OutDir = "dist"
)
# DIX VISION v42.2 — Windows build wrapper
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $root

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$stamp = Get-Date -Format "yyyyMMddHHmmss"
$zip = Join-Path $OutDir "dix_vision_v42.2_$stamp.zip"

Write-Host "[BUILD] Packaging to $zip"
$exclude = @(".venv", "__pycache__", "data\sqlite", "data\logs", "data\snapshots\full", "$OutDir")
$staging = Join-Path $env:TEMP "dix_vision_stage_$stamp"
New-Item -ItemType Directory -Force -Path $staging | Out-Null
robocopy $root $staging /MIR /XD @($exclude) /XF "*.pyc" | Out-Null
Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $zip -Force
Remove-Item -Recurse -Force $staging
Write-Host "[BUILD] OK → $zip"
