"""
start.py
DIX VISION v42.2 — Simplified entrypoint wrapper.

Equivalent to `python main.py` but keeps the process name deterministic
for service supervisors (NSSM, systemd) that expect `start.py`.
"""
from __future__ import annotations

import sys

if __name__ == "__main__":
    from main import main
    sys.exit(main() or 0)
