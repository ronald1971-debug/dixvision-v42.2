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
set "DASH_URL=http://127.0.0.1:8080/"
set "DASH_PORT=8080"
set "LAUNCHER_LOG=%REPO_ROOT%\launcher.log"

REM Truncate the previous launcher.log so each run starts clean and the
REM operator (or Devin Review) can paste the latest failure verbatim.
break > "%LAUNCHER_LOG%" 2>nul

echo.
echo === DIX VISION v42.2 — Windows launcher ===
echo Repo:  %REPO_ROOT%
echo Venv:  %VENV_DIR%
echo URL:   %DASH_URL%
echo Log:   %LAUNCHER_LOG%
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

REM --- install / update dependencies on first run ------------------------------
REM Auto-installs everything from requirements-dev.txt (which transitively
REM pulls in requirements.txt — runtime deps + dev tooling). The package
REM itself is then installed editable so `ui.server`, `core.contracts`,
REM `system_engine.scvs`, etc. resolve from the repo source.
if not exist "%VENV_MARKER%" (
    echo Installing dependencies ^(first-run only, ~2-3 minutes^)...
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
)

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
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "& '%VENV_PY%' -m uvicorn ui.server:app --host 127.0.0.1 --port %DASH_PORT% 2>&1 ^| Tee-Object -FilePath '%LAUNCHER_LOG%'; exit $LASTEXITCODE"
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
