"""
windows/updater/version_check.py
Reports the running version and (optionally) queries an update channel URL
for the latest published version. Never auto-applies.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path


def current_version(path: Path = Path("VERSION")) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return "unknown"


def fetch_latest(url: str, timeout: float = 5.0) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "DIX-Vision-Updater"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8")
        try:
            payload = json.loads(data)
            if isinstance(payload, dict):
                return str(payload.get("version") or payload.get("latest") or "")
        except Exception:
            return data.strip()
    except Exception:
        return None
    return None
