# env_setup.ps1
# DIX VISION v42.2 — Environment bootstrap
# Configures Windows system PATH + data directories for the current user.

param(
    [string]$InstallDir = "C:\dix_vision_v42_2"
)

$ErrorActionPreference = "Stop"

Write-Host "Configuring environment for DIX VISION v42.2" -ForegroundColor Cyan

$dirs = @("data\sqlite","data\snapshots\full","data\snapshots\incremental",
           "data\logs","data\caches","data\oss")
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Force -Path "$InstallDir\$d" | Out-Null
}

[Environment]::SetEnvironmentVariable("DIX_VISION_HOME", $InstallDir, "User")
Write-Host "DIX_VISION_HOME set to $InstallDir" -ForegroundColor Green
