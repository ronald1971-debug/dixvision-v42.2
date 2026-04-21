# DIX VISION v42.2 — Watchdog (PowerShell)
# Run as scheduled task every 1 minute
$ServiceName = "DIX_VISION"
$MaxRestarts = 3

$svc = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($svc -and $svc.Status -ne "Running") {
    Write-EventLog -LogName Application -Source "DIX_VISION" -EventId 1 `
        -EntryType Warning -Message "DIX_VISION service not running. Attempting restart."
    Start-Service -Name $ServiceName
}
