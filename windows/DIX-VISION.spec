# PyInstaller spec -- produces a single portable DIX-VISION.exe.
#
#   pip install pyinstaller==6.6.0
#   pyinstaller --noconfirm --clean windows/DIX-VISION.spec
#
# The resulting dist/DIX-VISION.exe is ~35 MB, bundles Python 3.11 + all deps,
# and on first run unpacks an `%LOCALAPPDATA%\DIX VISION\` data dir, generates
# a cockpit token, launches `python -m cockpit`, and opens the browser.

# ruff: noqa
# type: ignore

import os
from pathlib import Path

from PyInstaller.building.api import EXE, PYZ
from PyInstaller.building.build_main import Analysis
from PyInstaller.building.datastruct import Tree

repo = Path(SPECPATH).resolve().parent

a = Analysis(
    [str(repo / "windows" / "launcher_entry.py")],
    pathex=[str(repo)],
    binaries=[],
    datas=[
        (str(repo / "cockpit" / "static"), "cockpit/static"),
        (str(repo / "dix_manifest"), "dix_manifest"),
        (str(repo / "DIX VISION v42.2 \u2013 CANONICAL SYSTEM MANIFEST.txt"), "."),
    ],
    hiddenimports=[
        "cockpit",
        "cockpit.app",
        "cockpit.launcher",
        "cockpit.pairing",
        "cockpit.qr",
        "mind",
        "governance",
        "security",
        "system_monitor",
        "execution",
        "risk",
        "state",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "matplotlib", "tkinter", "PyQt5", "PySide2", "notebook",
        "scipy", "IPython", "pytest", "mypy",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="DIX-VISION",
    icon=str(repo / "windows" / "app.ico") if (repo / "windows" / "app.ico").is_file() else None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version=str(repo / "windows" / "version_info.txt") if (repo / "windows" / "version_info.txt").is_file() else None,
    onefile=True,
    strip=False,
    upx=False,
)
