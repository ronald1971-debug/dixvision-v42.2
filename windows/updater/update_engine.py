"""
windows/updater/update_engine.py
Staged update application: downloads payload to a staging dir, verifies hash,
then asks the operator to flip by calling apply_staged(). Never auto-applies.
"""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path


def stage_download(url: str, staging: Path, sha256: str | None = None,
                   timeout: float = 30.0) -> Path:
    staging.mkdir(parents=True, exist_ok=True)
    target = staging / "payload.zip"
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(target, "wb") as f:
        shutil.copyfileobj(resp, f)
    if sha256:
        h = hashlib.sha256(target.read_bytes()).hexdigest()
        if h.lower() != sha256.lower():
            target.unlink(missing_ok=True)
            raise ValueError(f"sha256_mismatch expected={sha256} got={h}")
    return target


def apply_staged(staged_zip: Path, install_dir: Path) -> None:
    import zipfile

    install_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(staged_zip, "r") as z:
        z.extractall(install_dir)
