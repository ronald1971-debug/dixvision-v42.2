# DIX VISION v42.2 Windows Installer
# Run as Administrator: PowerShell -ExecutionPolicy Bypass -File setup.ps1

param(
    [string]$InstallDir = "C:\dix_vision_v42_2",
    [string]$PythonPath = "C:\Python311\python.exe"
)

Write-Host "DIX VISION v42.2 Installer" -ForegroundColor Cyan

# Create virtual environment
Write-Host "Creating virtual environment..."
& $PythonPath -m venv "$InstallDir\venv"

# Install requirements
Write-Host "Installing dependencies..."
& "$InstallDir\venv\Scripts\pip.exe" install -r "$InstallDir\requirements.txt"

# Create data directories
$dirs = @("data\sqlite","data\snapshots","data\logs","data\caches")
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Force -Path "$InstallDir\$d" | Out-Null
}

# Generate foundation hash
Write-Host "Generating integrity hash..."
& "$InstallDir\venv\Scripts\python.exe" "$InstallDir\scripts\generate_hash.py"

Write-Host "Installation complete." -ForegroundColor Green
Write-Host "Start with: python main.py" -ForegroundColor Yellow
