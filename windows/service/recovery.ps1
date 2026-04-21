# recovery.ps1
# DIX VISION v42.2 — NSSM recovery policy: auto-restart up to 5 times,
# then fail-closed (service stays stopped so operator must intervene).

param(
    [string]$ServiceName = "DIX_VISION"
)

$ErrorActionPreference = "Stop"

nssm set $ServiceName AppExit Default Restart
nssm set $ServiceName AppRestartDelay 5000
nssm set $ServiceName AppThrottle 30000
nssm set $ServiceName AppStdoutCreationDisposition 4
nssm set $ServiceName AppStderrCreationDisposition 4
nssm set $ServiceName AppStopMethodSkip 0
nssm set $ServiceName AppStopMethodConsole 20000
nssm set $ServiceName AppExit 0 Exit

Write-Host "Recovery policy applied to $ServiceName" -ForegroundColor Green
