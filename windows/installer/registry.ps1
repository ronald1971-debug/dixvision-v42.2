# registry.ps1
# Registers DIX VISION v42.2 in HKCU so tray + updater can locate it.

param(
    [string]$InstallDir = "C:\dix_vision_v42_2",
    [string]$Version    = "42.2.0"
)

$ErrorActionPreference = "Stop"

$key = "HKCU:\Software\DIX_VISION\v42"
New-Item -Path $key -Force | Out-Null
New-ItemProperty -Path $key -Name "InstallDir" -Value $InstallDir -PropertyType String -Force | Out-Null
New-ItemProperty -Path $key -Name "Version"    -Value $Version    -PropertyType String -Force | Out-Null
Write-Host "Registered DIX_VISION v42.2 at $key" -ForegroundColor Green
