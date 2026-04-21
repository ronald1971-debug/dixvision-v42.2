"""
windows/launcher_entry.py -- PyInstaller entry point for the portable .exe.

Flow when a user double-clicks DIX-VISION.exe:
    1. Resolve %LOCALAPPDATA%\\DIX VISION\\  (created on first run).
    2. Ensure data/ exists and cockpit_token.txt is populated.
    3. Set env vars so the cockpit writes to the per-user data dir.
    4. Open the browser on http://127.0.0.1:8765/?token=<token>.
    5. Start `python -m cockpit --mode desktop`.

No admin elevation, no registry writes, no install wizard. Uninstall is just
deleting the folder.
"""
from __future__ import annotations

import os
import secrets
import sys
import threading
import time
import webbrowser
from pathlib import Path


def _data_root() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    root = Path(base) / "DIX VISION"
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    return root


def _ensure_token(data_dir: Path) -> str:
    token_file = data_dir / "cockpit_token.txt"
    if token_file.is_file():
        t = token_file.read_text(encoding="utf-8").strip()
        if t:
            return t
    t = secrets.token_urlsafe(32)
    token_file.write_text(t, encoding="utf-8")
    return t


def _open_browser(port: int, token: str) -> None:
    # Give uvicorn a beat to bind the socket before opening the browser.
    time.sleep(1.5)
    webbrowser.open(f"http://127.0.0.1:{port}/?token={token}", new=2)


def main() -> int:
    root = _data_root()
    data_dir = root / "data"
    token = _ensure_token(data_dir)
    port = int(os.environ.get("DIX_PORT", "8765"))

    os.environ.setdefault("DIX_MODE", "desktop")
    os.environ.setdefault("DIX_BIND_HOST", "127.0.0.1")
    os.environ["DIX_PORT"] = str(port)
    os.environ["DIX_COCKPIT_TOKEN"] = token
    os.environ["DIX_COCKPIT_TOKEN_FILE"] = str(data_dir / "cockpit_token.txt")
    os.environ["DIX_PAIRING_DB"] = str(data_dir / "pairing.sqlite")
    os.environ["DIX_LEDGER_DB"] = str(data_dir / "ledger.sqlite")
    os.environ["DIX_EPISODIC_DB"] = str(data_dir / "episodes.sqlite")
    os.environ["DIX_WALLET_POLICY_DB"] = str(data_dir / "wallet_policy.sqlite")

    threading.Thread(target=_open_browser, args=(port, token), daemon=True).start()

    from cockpit.launcher import main as cockpit_main

    sys.argv = ["DIX-VISION", "--mode", "desktop",
                "--host", "127.0.0.1", "--port", str(port)]
    try:
        cockpit_main()
    except KeyboardInterrupt:
        return 0
    except Exception as exc:                                                    # noqa: BLE001
        err = root / "logs" / "startup_error.txt"
        err.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
