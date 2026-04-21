"""
windows/updater/rollback_update.py
Rollback helper: swaps the current install with the last-known-good backup.
"""
from __future__ import annotations

import shutil
from pathlib import Path


def rollback(install_dir: Path, backup_dir: Path) -> None:
    install_dir = Path(install_dir)
    backup_dir = Path(backup_dir)
    if not backup_dir.exists():
        raise FileNotFoundError(f"no_backup_at:{backup_dir}")
    if install_dir.exists():
        tombstone = install_dir.with_suffix(".rolledback")
        if tombstone.exists():
            shutil.rmtree(tombstone)
        shutil.move(str(install_dir), str(tombstone))
    shutil.copytree(backup_dir, install_dir)
