<#
.SYNOPSIS
    Installs a "DIX MEME" shortcut on the current user's desktop pointing
    at scripts\windows\start_dixvision_meme.bat in this repository clone.

.DESCRIPTION
    Idempotent — re-running overwrites any existing "DIX MEME.lnk" on the
    desktop. Working directory is set to the repo root so relative paths
    (registry/, ui/, .venv/, dash_meme/) resolve correctly when the
    shortcut is double-clicked.

    This is the SECOND, separate shortcut: DIX VISION launches the cockpit
    (/dash2/), DIX MEME launches the DEXtools-styled memecoin dashboard
    (/meme/). Both run against the same backend harness — closing either
    browser window does NOT stop the system.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\windows\install_desktop_shortcut_meme.ps1
#>

[CmdletBinding()]
param(
    [string] $ShortcutName = "DIX MEME",
    [switch] $AllUsers,
    [switch] $Quiet
)

$ErrorActionPreference = "Stop"

function Write-Status($msg) {
    if (-not $Quiet) { Write-Host $msg }
}

# Resolve repo root (this script lives in scripts/windows/).
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot  = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$Launcher  = Join-Path $RepoRoot "scripts\windows\start_dixvision_meme.bat"

if (-not (Test-Path $Launcher)) {
    throw "Launcher not found: $Launcher"
}

if ($AllUsers) {
    $DesktopDir = [Environment]::GetFolderPath("CommonDesktopDirectory")
} else {
    $DesktopDir = [Environment]::GetFolderPath("Desktop")
}
$LinkPath = Join-Path $DesktopDir ($ShortcutName + ".lnk")

# Pick an icon: Python's pythonw.exe inside the venv if it exists, otherwise
# fall back to the system shell icon.
$VenvPyW = Join-Path $RepoRoot ".venv\Scripts\pythonw.exe"
if (Test-Path $VenvPyW) {
    $IconLocation = $VenvPyW + ",0"
} else {
    $IconLocation = "$env:SystemRoot\System32\shell32.dll,13"
}

Write-Status ""
Write-Status "=== DIX MEME desktop shortcut ==="
Write-Status "Repo:     $RepoRoot"
Write-Status "Launcher: $Launcher"
Write-Status "Shortcut: $LinkPath"
Write-Status ""

$Shell = New-Object -ComObject WScript.Shell
$Link  = $Shell.CreateShortcut($LinkPath)
$Link.TargetPath       = $Launcher
$Link.WorkingDirectory = $RepoRoot
$Link.IconLocation     = $IconLocation
$Link.Description      = "Launch the DIX MEME memecoin dashboard (DEXtools-styled)"
$Link.WindowStyle      = 1  # Normal window so uvicorn logs are visible
$Link.Save()

Write-Status "Shortcut created. Double-click 'DIX MEME' on your desktop to launch."
