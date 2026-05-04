"""Static regression guards for the Windows one-click launcher.

The launcher is a Windows-only set of .bat / .ps1 files that we cannot
execute on the Linux CI runner. These tests instead pin the contract of
the scripts as text, so a future edit that breaks the auto-install
behaviour (or accidentally drops the install-shortcut step) trips CI on
every platform.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WIN_DIR = REPO_ROOT / "scripts" / "windows"

START_BAT = WIN_DIR / "start_dixvision.bat"
START_MEME_BAT = WIN_DIR / "start_dixvision_meme.bat"
STOP_BAT = WIN_DIR / "stop_dixvision.bat"
INSTALL_PS1 = WIN_DIR / "install_desktop_shortcut.ps1"


def test_launcher_files_present() -> None:
    assert START_BAT.is_file(), START_BAT
    assert STOP_BAT.is_file(), STOP_BAT
    assert INSTALL_PS1.is_file(), INSTALL_PS1


def test_start_bat_auto_installs_desktop_shortcut() -> None:
    """First-run UX: the .bat must invoke install_desktop_shortcut.ps1."""
    body = START_BAT.read_text(encoding="utf-8")
    assert "install_desktop_shortcut.ps1" in body, (
        "start_dixvision.bat must call install_desktop_shortcut.ps1 so the "
        "shortcut lands on the desktop on first run"
    )
    assert "-ExecutionPolicy Bypass" in body, (
        "PowerShell invocation must bypass execution policy"
    )
    assert "-Quiet" in body, (
        "auto-install path must run the PS script in -Quiet mode to avoid "
        "polluting the launcher console"
    )
    assert "where powershell" in body, (
        "must guard PS invocation behind a `where powershell` probe so the "
        "launcher does not crash if PowerShell is missing"
    )


def test_start_bat_does_not_abort_on_shortcut_failure() -> None:
    """Shortcut install failures are non-fatal — the dashboard must still start."""
    body = START_BAT.read_text(encoding="utf-8")
    assert "[WARN] Desktop shortcut install failed" in body, (
        "shortcut failures must surface a [WARN] line, not abort the launcher"
    )
    assert "Desktop shortcut: installed/refreshed." in body, (
        "successful install must surface a confirmation line"
    )


def test_install_ps1_supports_quiet_flag() -> None:
    """The auto-install path passes -Quiet; the PS script must accept it."""
    body = INSTALL_PS1.read_text(encoding="utf-8")
    assert "[switch] $Quiet" in body, "install_desktop_shortcut.ps1 must accept -Quiet"
    assert "function Write-Status" in body, (
        "install_desktop_shortcut.ps1 must route output through Write-Status so "
        "-Quiet actually suppresses chatter"
    )


def test_install_ps1_idempotent_overwrite() -> None:
    """Re-running must overwrite an existing shortcut (self-healing)."""
    body = INSTALL_PS1.read_text(encoding="utf-8")
    assert "CreateShortcut(" in body, "must use WScript.Shell CreateShortcut"
    assert "$Link.Save()" in body, "must save the shortcut"
    assert 'GetFolderPath("Desktop")' in body, (
        "must resolve Desktop via Environment.GetFolderPath so OneDrive-redirected "
        "desktops are handled correctly"
    )


def test_stop_bat_targets_dashboard_port() -> None:
    body = STOP_BAT.read_text(encoding="utf-8")
    assert "DASH_PORT=8080" in body
    assert "taskkill" in body


def test_start_bats_run_credential_preflight() -> None:
    """AUDIT-P2.5: both .bat launchers must invoke check_credentials --missing-only.

    The pre-flight runs *after* the venv is hydrated (so the import resolves)
    and *before* uvicorn starts (so the operator sees the missing-key list on
    the console, not buried in launcher.log).
    """
    for bat in (START_BAT, START_MEME_BAT):
        body = bat.read_text(encoding="utf-8")
        assert "scripts.check_credentials" in body, (
            f"{bat.name} must invoke `python -m scripts.check_credentials` as a "
            "pre-flight before uvicorn"
        )
        assert "--missing-only" in body, (
            f"{bat.name} must pass --missing-only so the pre-flight stays advisory"
        )

        preflight_idx = body.index("scripts.check_credentials")
        uvicorn_idx = body.rindex("uvicorn ui.server:app")
        assert preflight_idx < uvicorn_idx, (
            f"{bat.name}: credential pre-flight must run before uvicorn starts"
        )

        venv_marker = "pip install -q --disable-pip-version-check -e ."
        venv_idx = body.index(venv_marker)
        assert venv_idx < preflight_idx, (
            f"{bat.name}: credential pre-flight must run after the venv is hydrated"
        )
