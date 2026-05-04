@echo off
REM ============================================================================
REM DIX VISION + DIX MEME — combined Windows launcher
REM ============================================================================
REM Single .bat that boots the FastAPI harness once and opens BOTH dashboards
REM in the operator's default browser:
REM
REM   1) http://127.0.0.1:8080/dash2/   — DIX VISION cockpit (operator)
REM   2) http://127.0.0.1:8080/meme/    — DIX MEME (memecoin) cockpit
REM
REM Both dashboards are mounted on the same uvicorn process — only ONE harness
REM runs, so closing the browser does not stop the system. The LEARNING,
REM SENSING, GOVERNANCE, AUDIT layers all live in the harness, not the browser.
REM
REM Usage: double-click this .bat directly, or wire it into a desktop shortcut.
REM
REM See start_dixvision.bat / start_dixvision_meme.bat for single-dashboard
REM equivalents — the boot sequence below is identical to those, just with two
REM browser tabs opened instead of one.
REM ============================================================================

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\.."
pushd "%REPO_ROOT%" >nul

set "VENV_DIR=%REPO_ROOT%\.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "VENV_MARKER=%VENV_DIR%\.dixvision_installed"
set "DASH2_DIR=%REPO_ROOT%\dashboard2026"
set "DASH2_DIST=%DASH2_DIR%\dist\index.html"
set "MEME_DIR=%REPO_ROOT%\dash_meme"
set "MEME_DIST=%MEME_DIR%\dist\index.html"
set "DASH2_URL=http://127.0.0.1:8080/dash2/"
set "MEME_URL=http://127.0.0.1:8080/meme/"
set "DASH_PORT=8080"
set "LAUNCHER_LOG=%REPO_ROOT%\launcher_both.log"

REM --- AUDIT-P0.3 / Sprint-1 "Trust the Ledger" ---------------------------------
REM Same governance ledger as the single-dashboard launchers — both surfaces
REM read/write the SQLite store under ``%APPDATA%\DIX VISION\governance.db``.
REM The harness refuses to boot without ``DIXVISION_LEDGER_PATH``; we default
REM it here unless the operator has exported it themselves before launch.
if not defined DIXVISION_LEDGER_PATH (
    set "DIXVISION_LEDGER_DIR=%APPDATA%\DIX VISION"
    if not exist "%APPDATA%\DIX VISION" mkdir "%APPDATA%\DIX VISION" >nul 2>&1
    set "DIXVISION_LEDGER_PATH=%APPDATA%\DIX VISION\governance.db"
)

break > "%LAUNCHER_LOG%" 2>nul

echo.
echo === DIX VISION + DIX MEME — combined Windows launcher ===
echo Repo:       %REPO_ROOT%
echo Venv:       %VENV_DIR%
echo Cockpit:    %DASH2_URL%
echo Memecoin:   %MEME_URL%
echo Log:        %LAUNCHER_LOG%
echo Ledger:     %DIXVISION_LEDGER_PATH%
echo.
echo Both dashboards run on the SAME harness process. Closing the browser
echo does NOT stop the system; sensing/learning/governance/audit keep running.
echo.

REM --- find a usable Python interpreter ---------------------------------------
set "PY_CMD="
where py >nul 2>&1
if %errorlevel%==0 (
    py -3.12 -c "import sys" >nul 2>&1
    if !errorlevel!==0 set "PY_CMD=py -3.12"
)
if not defined PY_CMD (
    where python >nul 2>&1
    if !errorlevel!==0 (
        for /f "tokens=2 delims= " %%v in ('python -V 2^>^&1') do set "PY_VER=%%v"
        echo Found python !PY_VER!
        set "PY_CMD=python"
    )
)
if not defined PY_CMD (
    echo [ERROR] Python 3.12+ not found.
    echo         Install from https://www.python.org/downloads/windows/ ^(check "Add to PATH"^)
    echo         or run: winget install Python.Python.3.12
    pause
    popd >nul
    exit /b 1
)

REM --- create venv on first run ------------------------------------------------
if not exist "%VENV_PY%" (
    echo Creating virtual environment...
    %PY_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] venv creation failed.
        pause
        popd >nul
        exit /b 1
    )
)

REM --- install / sync dependencies (same logic as start_dixvision.bat) ---------
if not exist "%VENV_MARKER%" (
    echo Installing dependencies ^(first-run, ~2-3 minutes^)...
    "%VENV_PY%" -m pip install --upgrade pip
    "%VENV_PY%" -m pip install -r requirements-dev.txt
    if errorlevel 1 (
        echo [ERROR] pip install -r requirements-dev.txt failed.
        pause
        popd >nul
        exit /b 1
    )
    "%VENV_PY%" -m pip install -e .
    if errorlevel 1 (
        echo [ERROR] pip install -e . failed.
        pause
        popd >nul
        exit /b 1
    )
    echo. > "%VENV_MARKER%"
    echo Dependencies installed.
) else (
    echo Syncing dependencies ^(quiet; only installs if requirements drifted^)...
    "%VENV_PY%" -m pip install -q --disable-pip-version-check -r requirements-dev.txt
    if errorlevel 1 (
        echo [ERROR] pip install -r requirements-dev.txt failed.
        pause
        popd >nul
        exit /b 1
    )
    "%VENV_PY%" -m pip install -q --disable-pip-version-check -e .
    if errorlevel 1 (
        echo [ERROR] pip install -e . failed.
        pause
        popd >nul
        exit /b 1
    )
)

REM --- build BOTH React dashboards ---------------------------------------------
where npm >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] npm not found ^(install Node 20+ from https://nodejs.org/^).
    echo         /dash2/ and /meme/ will fall back to the stub harness.
    goto :start_server
)

REM Build dashboard2026 (DIX VISION cockpit -> /dash2/).
if exist "%DASH2_DIR%\package.json" (
    echo Building cockpit dashboard ^(dashboard2026^)...
    pushd "%DASH2_DIR%" >nul
    if not exist "node_modules" (
        call npm ci --silent
        if errorlevel 1 (
            echo [WARN] npm ci failed in dashboard2026/; /dash2/ may 404.
            popd >nul
            goto :build_meme
        )
    )
    call npm run build --silent
    if errorlevel 1 (
        echo [WARN] dashboard2026 build failed; /dash2/ may 404.
    ) else (
        echo Cockpit dashboard built: %DASH2_DIST%
    )
    popd >nul
)

:build_meme
REM Build dash_meme (DIX MEME -> /meme/).
if exist "%MEME_DIR%\package.json" (
    echo Building DIX MEME ^(dash_meme^)...
    pushd "%MEME_DIR%" >nul
    if not exist "node_modules" (
        call npm ci --silent
        if errorlevel 1 (
            echo [WARN] npm ci failed in dash_meme/; /meme/ may 404.
            popd >nul
            goto :start_server
        )
    )
    call npm run build --silent
    if errorlevel 1 (
        echo [WARN] dash_meme build failed; /meme/ may 404.
    ) else (
        echo DIX MEME built: %MEME_DIST%
    )
    popd >nul
)

:start_server
REM --- launch BOTH dashboards in the default browser after a short delay ------
REM Stagger the second open by ~1.5s so the first tab takes focus and the
REM second opens beside it, not on top of it.
start "" /b cmd /c "timeout /t 3 /nobreak >nul && start "" "%DASH2_URL%""
start "" /b cmd /c "timeout /t 4 /nobreak >nul && start "" "%MEME_URL%""

echo.
echo Starting FastAPI harness on http://127.0.0.1:%DASH_PORT%/ ^(Ctrl+C to stop^)
echo Cockpit:  %DASH2_URL%
echo Memecoin: %MEME_URL%
echo All output is tee'd to %LAUNCHER_LOG% — paste it back if you hit issues.
echo Tip: run scripts\windows\stop_dixvision.bat to force-kill if needed.
echo.

where powershell >nul 2>&1
if %errorlevel%==0 (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%VENV_PY%' -m uvicorn ui.server:app --host 127.0.0.1 --port %DASH_PORT% 2>&1 | Tee-Object -FilePath '%LAUNCHER_LOG%'; exit $LASTEXITCODE"
    set "EXITCODE=!errorlevel!"
) else (
    echo [WARN] powershell not found; output is captured to %LAUNCHER_LOG% only.
    "%VENV_PY%" -m uvicorn ui.server:app --host 127.0.0.1 --port %DASH_PORT% > "%LAUNCHER_LOG%" 2>&1
    set "EXITCODE=!errorlevel!"
)

echo.
if not "!EXITCODE!"=="0" (
    echo [ERROR] uvicorn exited with code !EXITCODE!. Full log: %LAUNCHER_LOG%
) else (
    echo Harness process exited cleanly. Full log: %LAUNCHER_LOG%
)

echo.
echo Press any key to close this window...
pause >nul

popd >nul
endlocal & exit /b %EXITCODE%
