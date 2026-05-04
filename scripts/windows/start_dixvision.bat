@echo off
REM ============================================================================
REM DIX VISION v42.2 — Windows launcher
REM ============================================================================
REM Idempotent: first run creates a venv and installs dependencies; subsequent
REM runs just (re)start the FastAPI control-plane harness on http://127.0.0.1:8080
REM and open the dashboard in the default browser.
REM
REM Usage: double-click "DIX VISION.lnk" on the desktop (created by
REM        install_desktop_shortcut.ps1), or run this .bat directly.
REM ============================================================================

setlocal enabledelayedexpansion

REM --- repo root is two levels above this script -------------------------------
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\.."
pushd "%REPO_ROOT%" >nul

set "VENV_DIR=%REPO_ROOT%\.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "VENV_MARKER=%VENV_DIR%\.dixvision_installed"
set "DASH2_DIR=%REPO_ROOT%\dashboard2026"
set "DASH2_DIST=%DASH2_DIR%\dist\index.html"
REM DASH_URL stays at / on purpose: ui/server.py's GET / handler 307s
REM to /dash2/ when the React build is present, and falls back to the
REM Phase E1 stub when it is not. Pointing the launcher directly at
REM /dash2/ would 404 in the npm-missing / build-failed branches above.
set "DASH_URL=http://127.0.0.1:8080/"
set "DASH_PORT=8080"
set "LAUNCHER_LOG=%REPO_ROOT%\launcher.log"

REM --- AUDIT-P0.3 / Sprint-1 "Trust the Ledger" ---------------------------------
REM The harness refuses to boot without ``DIXVISION_LEDGER_PATH`` so a
REM crash-recoverable SQLite governance ledger is always mounted in
REM production. We default it under ``%APPDATA%\DIX VISION\`` (a
REM per-user, roaming-profile-friendly location that survives venv
REM resets) unless the operator has explicitly exported the env var
REM themselves before launching.
if not defined DIXVISION_LEDGER_PATH (
    set "DIXVISION_LEDGER_DIR=%APPDATA%\DIX VISION"
    if not exist "%APPDATA%\DIX VISION" mkdir "%APPDATA%\DIX VISION" >nul 2>&1
    set "DIXVISION_LEDGER_PATH=%APPDATA%\DIX VISION\governance.db"
)

REM Truncate the previous launcher.log so each run starts clean and the
REM operator (or Devin Review) can paste the latest failure verbatim.
break > "%LAUNCHER_LOG%" 2>nul

echo.
echo === DIX VISION v42.2 — Windows launcher ===
echo Repo:   %REPO_ROOT%
echo Venv:   %VENV_DIR%
echo URL:    %DASH_URL%
echo Log:    %LAUNCHER_LOG%
echo Ledger: %DIXVISION_LEDGER_PATH%
echo.
echo Governance ledger is SQLite-backed and persists across restarts.
echo Set DIXVISION_LEDGER_PATH before launch to override the default.
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

REM --- install / update dependencies -------------------------------------------
REM Auto-installs everything from requirements-dev.txt (which transitively
REM pulls in requirements.txt — runtime deps + dev tooling). The package
REM itself is then installed editable so `ui.server`, `core.contracts`,
REM `system_engine.scvs`, etc. resolve from the repo source.
REM
REM On the FIRST run we print a verbose progress message (this can take
REM ~2-3 minutes). On every SUBSEQUENT run we still re-run pip in quiet
REM mode so a `git pull` that introduces a new top-level import (e.g.
REM ``langchain-core`` after PR #85) does not silently leave the venv
REM behind and crash uvicorn at module-load time. pip is idempotent and
REM completes in 2-3s when nothing has changed, so this is cheap.
if not exist "%VENV_MARKER%" (
    echo Installing dependencies ^(first-run, ~2-3 minutes^)...
    "%VENV_PY%" -m pip install --upgrade pip
    "%VENV_PY%" -m pip install -r requirements-dev.txt
    if errorlevel 1 (
        echo [ERROR] pip install -r requirements-dev.txt failed.
        echo         If wheels failed to build, install MSVC Build Tools:
        echo            winget install Microsoft.VisualStudio.2022.BuildTools
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
        echo         Try deleting the .venv folder and re-running this script.
        pause
        popd >nul
        exit /b 1
    )
    "%VENV_PY%" -m pip install -q --disable-pip-version-check -e .
    if errorlevel 1 (
        echo [ERROR] pip install -e . failed.
        echo         Try deleting the .venv folder and re-running this script.
        pause
        popd >nul
        exit /b 1
    )
)

REM --- build the React SPA (dashboard2026) if Node is installed ----------------
REM ``dashboard2026/dist/`` is gitignored, so a fresh clone has only the
REM Phase E1 stub at ``/`` until the React SPA is built. We build it on
REM every launch so a ``git pull`` that updates the SPA picks up cleanly.
REM If Node isn't installed we warn and continue — the FastAPI server's
REM ``/`` route falls back to the stub harness instead of 404'ing.
where npm >nul 2>&1
if %errorlevel%==0 (
    if exist "%DASH2_DIR%\package.json" (
        echo Building React dashboard ^(dashboard2026^)...
        pushd "%DASH2_DIR%" >nul
        if not exist "node_modules" (
            call npm ci --silent
            if errorlevel 1 (
                echo [WARN] npm ci failed; falling back to stub harness.
                popd >nul
                goto :skip_dash2_build
            )
        )
        call npm run build --silent
        if errorlevel 1 (
            echo [WARN] npm run build failed; falling back to stub harness.
            popd >nul
            goto :skip_dash2_build
        )
        popd >nul
        echo Dashboard built: %DASH2_DIST%
    ) else (
        echo [WARN] dashboard2026/package.json missing; using stub harness.
    )
) else (
    echo [WARN] npm not found ^(install Node 20+ from https://nodejs.org/^).
    echo         Falling back to the Phase E1 stub harness at /.
)
:skip_dash2_build

REM --- ensure the desktop shortcut is in place (idempotent, self-healing) ------
REM Calls install_desktop_shortcut.ps1 in -Quiet mode every launch. The PS
REM script overwrites any existing "DIX VISION.lnk" on the user's desktop,
REM so re-running fixes a deleted shortcut. Failures here are non-fatal; the
REM launcher still proceeds to start the dashboard.
where powershell >nul 2>&1
if %errorlevel%==0 (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%install_desktop_shortcut.ps1" -Quiet
    if errorlevel 1 (
        echo [WARN] Desktop shortcut install failed ^(continuing^).
    ) else (
        echo Desktop shortcut: installed/refreshed.
    )
) else (
    echo [WARN] powershell not found; skipping desktop shortcut install.
)

REM --- launch the dashboard in the default browser after a short delay ---------
REM ('start "" /b' detaches without opening a new shell window)
start "" /b cmd /c "timeout /t 3 /nobreak >nul && start "" "%DASH_URL%""

echo.
echo Starting FastAPI harness on %DASH_URL% ^(Ctrl+C to stop^)
echo All output is tee'd to %LAUNCHER_LOG% — paste it back if you hit issues.
echo Tip: run scripts\windows\stop_dixvision.bat to force-kill if needed.
echo.

REM --- run uvicorn in the foreground so logs are visible -----------------------
REM PowerShell's Tee-Object mirrors stdout+stderr to the operator's
REM console *and* to launcher.log so a crash that closes the cmd
REM window still leaves a paste-able artefact behind. ``$LASTEXITCODE``
REM is the only safe way to surface uvicorn's exit code through a
REM PowerShell pipeline (the parent ``%errorlevel%`` would always be 0).
where powershell >nul 2>&1
if %errorlevel%==0 (
    REM The `|` lives *inside* the double-quoted -Command string, so cmd
    REM does not need it escaped. Earlier we used ``^|`` (cmd-style pipe
    REM escape), which leaked a literal ``^`` into PowerShell's argv and
    REM caused uvicorn to reject ``^`` as an unexpected positional
    REM argument ("Got unexpected extra argument (^)"). Plain ``|`` is
    REM what PowerShell needs to see; cmd leaves it alone within ``"…"``.
    powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%VENV_PY%' -m uvicorn ui.server:app --host 127.0.0.1 --port %DASH_PORT% 2>&1 | Tee-Object -FilePath '%LAUNCHER_LOG%'; exit $LASTEXITCODE"
    set "EXITCODE=!errorlevel!"
) else (
    REM Fallback: no PowerShell available. Capture to log file only;
    REM real-time visibility is sacrificed but a paste-able artefact
    REM still exists.
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

REM --- always pause so the cmd window cannot close before the operator -------
REM has a chance to read the output. This is the single biggest UX fix
REM in this script: prior versions exited silently if uvicorn failed at
REM import time, which made remote diagnosis impossible.
echo.
echo Press any key to close this window...
pause >nul

popd >nul
endlocal & exit /b %EXITCODE%
