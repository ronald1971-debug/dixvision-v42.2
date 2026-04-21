# uninstall.ps1
# DIX VISION v42.2 — clean uninstaller

param(
    [string]$InstallDir = "C:\dix_vision_v42_2",
    [switch]$KeepData
)

$ErrorActionPreference = "Stop"

# Stop + remove service if present
try { nssm stop "DIX_VISION" | Out-Null } catch {}
try { nssm remove "DIX_VISION" confirm | Out-Null } catch {}

# Remove registry key
Remove-Item -Path "HKCU:\Software\DIX_VISION\v42" -Recurse -Force -ErrorAction SilentlyContinue

if (-not $KeepData) {
    Write-Host "Removing $InstallDir ..." -ForegroundColor Yellow
    Remove-Item -Path $InstallDir -Recurse -Force -ErrorAction SilentlyContinue
} else {
    Write-Host "Keeping $InstallDir\data (use -KeepData:$false to remove)" -ForegroundColor Yellow
}
Write-Host "Uninstall complete." -ForegroundColor Green
