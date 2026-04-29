@echo off
REM ============================================================================
REM DIX VISION v42.2 — clean shutdown for the Windows launcher.
REM Kills any process bound to the dashboard port (default 8080).
REM ============================================================================

setlocal enabledelayedexpansion
set "DASH_PORT=8080"

echo Looking for processes on port %DASH_PORT%...

set "FOUND="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%DASH_PORT% " ^| findstr "LISTENING"') do (
    set "FOUND=1"
    echo   killing PID %%P
    taskkill /PID %%P /F >nul 2>&1
)

if not defined FOUND (
    echo No DIX VISION process listening on %DASH_PORT%.
) else (
    echo DIX VISION stopped.
)

endlocal
