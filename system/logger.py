"""
system/logger.py
DIX VISION v42.2 — Structured Async JSON Logger

Non-blocking queue-based. UTC + monotonic seq on every record.
Writes to stdout AND data/logs/system.log.

All ``Logger`` instances share a single file handle + write lock to
prevent file-descriptor proliferation and to serialise writes.
"""
from __future__ import annotations

import atexit
import json
from pathlib import Path
from queue import Queue
from threading import RLock, Thread
from typing import Any, IO

from system.time_source import now

_FILE_LOCK = RLock()
_SHARED_FILE: IO[str] | None = None
_SHARED_FILE_PATH: Path | None = None


def _get_shared_file(log_dir: str) -> IO[str]:
    """Return a process-wide append-mode handle to ``<log_dir>/system.log``.

    Opened exactly once per process; subsequent calls return the same
    file object.  A single write lock (``_FILE_LOCK``) serialises
    writes so concurrent Logger threads do not interleave bytes.
    """
    global _SHARED_FILE, _SHARED_FILE_PATH
    with _FILE_LOCK:
        if _SHARED_FILE is None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            _SHARED_FILE_PATH = Path(log_dir) / "system.log"
            _SHARED_FILE = open(_SHARED_FILE_PATH, "a", encoding="utf-8")
            atexit.register(_close_shared_file)
        return _SHARED_FILE


def _close_shared_file() -> None:
    global _SHARED_FILE
    with _FILE_LOCK:
        if _SHARED_FILE is not None:
            try:
                _SHARED_FILE.flush()
                _SHARED_FILE.close()
            except Exception:
                pass
            _SHARED_FILE = None


class Logger:
    def __init__(self, name: str, log_dir: str = "data/logs") -> None:
        self.name = name
        self._q: Queue = Queue()
        # Share one file handle + write lock across all Logger
        # instances to avoid fd proliferation and interleaved writes.
        self._f = _get_shared_file(log_dir)
        self._t = Thread(target=self._run, daemon=True, name=f"Logger-{name}")
        self._t.start()
        atexit.register(self._flush)

    def _run(self) -> None:
        while True:
            rec = self._q.get()
            try:
                if rec is None:
                    break
                line = json.dumps(rec, default=str)
                print(line)
                with _FILE_LOCK:
                    self._f.write(line + "\n")
                    self._f.flush()
            finally:
                self._q.task_done()

    def _flush(self) -> None:
        # Pair every ``put()`` with a ``task_done()`` in ``_run`` so
        # ``join()`` cannot block indefinitely on shutdown.
        self._q.join()

    def _emit(self, level: str, msg: str, **kw: Any) -> None:
        ts = now()
        self._q.put({"utc": ts.utc_time.isoformat(), "seq": ts.sequence,
                     "mono_ns": ts.monotonic_ns, "logger": self.name,
                     "level": level, "message": msg, **kw})

    def debug(self, msg: str, **kw: Any) -> None: self._emit("DEBUG", msg, **kw)
    def info(self, msg: str, **kw: Any) -> None: self._emit("INFO", msg, **kw)
    def warning(self, msg: str, **kw: Any) -> None: self._emit("WARN", msg, **kw)
    def error(self, msg: str, **kw: Any) -> None: self._emit("ERROR", msg, **kw)
    def critical(self, msg: str, **kw: Any) -> None: self._emit("CRITICAL", msg, **kw)

_loggers: dict[str, Logger] = {}
_lock = RLock()

def get_logger(name: str = "default") -> Logger:
    with _lock:
        if name not in _loggers:
            _loggers[name] = Logger(name)
        return _loggers[name]
