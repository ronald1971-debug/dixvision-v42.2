"""
core/single_instance.py
DIX VISION v42.2 — Single-Instance Lock

Prevents multiple processes from booting concurrently. Uses an OS file lock
(fcntl on POSIX, msvcrt on Windows). Held for the lifetime of the process.
"""
from __future__ import annotations

import atexit
import os
import sys
from pathlib import Path

_LOCK_HANDLE: object | None = None
_LOCK_PATH: Path | None = None


class AlreadyRunningError(RuntimeError):
    """Raised when another DIX VISION instance holds the lock."""


def _default_lock_path() -> Path:
    root = Path(os.environ.get("DIX_ROOT", "."))
    p = root / "data" / "dix_vision.lock"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def acquire(lock_path: Path | None = None) -> None:
    """Acquire the single-instance lock. Raises AlreadyRunningError if held."""
    global _LOCK_HANDLE, _LOCK_PATH
    if _LOCK_HANDLE is not None:
        return
    path = lock_path or _default_lock_path()
    f = open(path, "a+")
    try:
        if sys.platform == "win32":
            import msvcrt

            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as e:
                f.close()
                raise AlreadyRunningError(f"another instance holds {path}") from e
        else:
            import fcntl

            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                f.close()
                raise AlreadyRunningError(f"another instance holds {path}") from e
        f.seek(0)
        f.truncate(0)
        f.write(f"{os.getpid()}\n")
        f.flush()
    except AlreadyRunningError:
        raise
    except Exception:
        try:
            f.close()
        except Exception:
            pass
        raise
    _LOCK_HANDLE = f
    _LOCK_PATH = path
    atexit.register(release)


def release() -> None:
    """Release the single-instance lock."""
    global _LOCK_HANDLE, _LOCK_PATH
    if _LOCK_HANDLE is None:
        return
    f = _LOCK_HANDLE
    try:
        if sys.platform == "win32":
            import msvcrt

            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
            except OSError:
                pass
        else:
            import fcntl

            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
            except OSError:
                pass
    finally:
        try:
            f.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        _LOCK_HANDLE = None
        if _LOCK_PATH and _LOCK_PATH.exists():
            try:
                _LOCK_PATH.unlink()
            except OSError:
                pass
        _LOCK_PATH = None


def is_held() -> bool:
    return _LOCK_HANDLE is not None
