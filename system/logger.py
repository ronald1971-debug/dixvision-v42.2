"""
system/logger.py
DIX VISION v42.2 — Structured Async JSON Logger

Non-blocking queue-based. UTC + monotonic seq on every record.
Writes to stdout AND data/logs/system.log.
"""
from __future__ import annotations

import atexit
import json
from pathlib import Path
from queue import Queue
from threading import RLock, Thread
from typing import Any

from system.time_source import now


class Logger:
    def __init__(self, name: str, log_dir: str = "data/logs") -> None:
        self.name = name
        self._q: Queue = Queue()
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._f = open(Path(log_dir) / "system.log", "a", encoding="utf-8")
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
