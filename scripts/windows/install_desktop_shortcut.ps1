<#
.SYNOPSIS
    Installs a "DIX VISION" shortcut on the current user's desktop pointing
    at scripts\windows\start_dixvision.bat in this repository clone.

.DESCRIPTION
    Idempotent — re-running overwrites any existing "DIX VISION.lnk" on the
    desktop. Working directory is set to the repo root so relative paths
    (registry/, ui/, .venv/) resolve correctly when the shortcut is double-
    clicked.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\windows\install_desktop_shortcut.ps1

.NOTES
    The shortcut launches start_dixvision.bat which itself handles venv
    creation, pip install, and opening http://127.0.0.1:8080/ in the default
    browser. See scripts\windows\start_dixvision.bat for details.
#>

[CmdletBinding()]
param(
    [string] $ShortcutName = "DIX VISION",
    [switch] $AllUsers
)

$ErrorActionPreference = "Stop"

# Resolve repo root (this script lives in scripts/windows/).
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot  = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$Launcher  = Join-Path $RepoRoot "scripts\windows\start_dixvision.bat"

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

Write-Host ""
Write-Host "=== DIX VISION desktop shortcut ==="
Write-Host "Repo:     $RepoRoot"
Write-Host "Launcher: $Launcher"
Write-Host "Shortcut: $LinkPath"
Write-Host ""

$Shell = New-Object -ComObject WScript.Shell
$Link  = $Shell.CreateShortcut($LinkPath)
$Link.TargetPath       = $Launcher
$Link.WorkingDirectory = $RepoRoot
$Link.IconLocation     = $IconLocation
$Link.Description      = "Launch the DIX VISION v42.2 control-plane dashboard"
$Link.WindowStyle      = 1  # Normal window so uvicorn logs are visible
$Link.Save()

Write-Host "Shortcut created. Double-click 'DIX VISION' on your desktop to launch."
