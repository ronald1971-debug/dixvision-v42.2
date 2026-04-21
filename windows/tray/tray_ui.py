"""
windows/tray/tray_ui.py
Minimal text-based UI. The real Windows tray uses ``tray_app.py`` with
pystray/Qt; this module is the cross-platform fallback for dev/headless.
"""
from __future__ import annotations

import json

from . import tray_actions


def render_status() -> str:
    snap = tray_actions.status()
    return json.dumps(snap, indent=2, default=str)


def main() -> int:
    print(render_status())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
