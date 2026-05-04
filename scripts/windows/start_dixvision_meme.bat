@echo off
REM ============================================================================
REM DIX MEME — Windows launcher (separate from DIX VISION cockpit)
REM ============================================================================
REM Same harness, separate dashboard surface. The FastAPI control-plane is
REM mounted at http://127.0.0.1:8080/, and the DEXtools-styled DIX MEME
REM dashboard is served from /meme/ off the same harness. The LEARNING,
REM SENSING, GOVERNANCE, AUDIT layers are all in the harness — closing
REM the browser does NOT stop the system.
REM
REM Boot order:
REM   1) Bootstrap venv + python deps (shared with DIX VISION).
REM   2) Build dash_meme/ React app to dash_meme/dist (mounted at /meme/).
REM   3) Build dashboard2026/ React app too so /dash2/ keeps working.
REM   4) Start uvicorn ui.server:app (single process, both /meme/ + /dash2/).
REM   5) Open http://127.0.0.1:8080/meme/ in the default browser.
REM
REM Usage: double-click "DIX MEME.lnk" on the desktop (created by
REM        install_desktop_shortcut_meme.ps1), or run this .bat directly.
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
set "DASH_URL=http://127.0.0.1:8080/meme/"
set "DASH_PORT=8080"
set "LAUNCHER_LOG=%REPO_ROOT%\launcher_meme.log"

REM --- AUDIT-P0.3 / Sprint-1 "Trust the Ledger" ---------------------------------
REM Same governance ledger as start_dixvision.bat — both launchers
REM share the SQLite store under ``%APPDATA%\DIX VISION\governance.db``
REM so DIX VISION and DIX MEME read/write the same chain. The harness
REM refuses to boot without ``DIXVISION_LEDGER_PATH``; we default it
REM here unless the operator has exported it themselves before launch.
if not defined DIXVISION_LEDGER_PATH (
    set "DIXVISION_LEDGER_DIR=%APPDATA%\DIX VISION"
    if not exist "%APPDATA%\DIX VISION" mkdir "%APPDATA%\DIX VISION" >nul 2>&1
    set "DIXVISION_LEDGER_PATH=%APPDATA%\DIX VISION\governance.db"
)

break > "%LAUNCHER_LOG%" 2>nul

echo.
echo === DIX MEME — Windows launcher ===
echo Repo:   %REPO_ROOT%
echo Venv:   %VENV_DIR%
echo URL:    %DASH_URL%
echo Log:    %LAUNCHER_LOG%
echo Ledger: %DIXVISION_LEDGER_PATH%
echo.
echo Governance ledger is SQLite-backed and shared with DIX VISION.
echo Set DIXVISION_LEDGER_PATH before launch to override the default.
echo.
echo NOTE: The harness keeps running (sensing, learning, governance,
echo       audit ledger) regardless of which dashboard is open or
echo       whether the browser is closed.
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

REM --- AUDIT-P2.5 credential pre-flight ---------------------------------------
REM Surface missing API keys on the operator console BEFORE uvicorn starts
REM so the operator does not have to dig through launcher.log to discover
REM that an external feed is unauthenticated. Advisory only -- exits 0
REM even when keys are missing, mirroring the harness permissive default.
echo.
echo --- Credential pre-flight ^(missing keys only, advisory^) -------------------
"%VENV_PY%" -m scripts.check_credentials --missing-only
echo ----------------------------------------------------------------------------
echo.

REM --- build the React dashboards ----------------------------------------------
where npm >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] npm not found ^(install Node 20+ from https://nodejs.org/^).
    echo         /meme/ will fall back to the harness root if dash_meme/dist is missing.
    goto :install_shortcut
)

REM Build dash_meme (this dashboard's primary surface).
if exist "%MEME_DIR%\package.json" (
    echo Building DIX MEME ^(dash_meme^)...
    pushd "%MEME_DIR%" >nul
    if not exist "node_modules" (
        call npm ci --silent
        if errorlevel 1 (
            echo [WARN] npm ci failed in dash_meme/; continuing without /meme/ build.
            popd >nul
            goto :build_dash2
        )
    )
    call npm run build --silent
    if errorlevel 1 (
        echo [WARN] dash_meme build failed; /meme/ will not be served.
        popd >nul
        goto :build_dash2
    )
    popd >nul
    echo DIX MEME built: %MEME_DIST%
) else (
    echo [WARN] dash_meme/package.json missing.
)

:build_dash2
REM Also build dashboard2026 so /dash2/ keeps working alongside /meme/.
if exist "%DASH2_DIR%\package.json" (
    echo Building cockpit dashboard ^(dashboard2026^)...
    pushd "%DASH2_DIR%" >nul
    if not exist "node_modules" (
        call npm ci --silent
        if errorlevel 1 (
            echo [WARN] npm ci failed in dashboard2026/; /dash2/ may 404.
            popd >nul
            goto :install_shortcut
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

:install_shortcut
REM --- ensure the desktop shortcut is in place (idempotent, self-healing) ------
where powershell >nul 2>&1
if %errorlevel%==0 (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%install_desktop_shortcut_meme.ps1" -Quiet
    if errorlevel 1 (
        echo [WARN] DIX MEME desktop shortcut install failed ^(continuing^).
    ) else (
        echo Desktop shortcut: installed/refreshed.
    )
)

:start_server
REM --- launch the dashboard in the default browser after a short delay ---------
start "" /b cmd /c "timeout /t 3 /nobreak >nul && start "" "%DASH_URL%""

echo.
echo Starting FastAPI harness on http://127.0.0.1:%DASH_PORT%/ ^(Ctrl+C to stop^)
echo DIX MEME UI:        %DASH_URL%
echo DIX VISION cockpit: http://127.0.0.1:%DASH_PORT%/dash2/
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
    echo Dashboard process exited cleanly. Full log: %LAUNCHER_LOG%
)

echo.
echo Press any key to close this window...
pause >nul

popd >nul
endlocal & exit /b %EXITCODE%
