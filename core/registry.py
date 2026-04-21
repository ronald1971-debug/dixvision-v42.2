"""
core/registry.py
DIX VISION v42.2 — Lazy Component Registry (LOCKED after boot)
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

class Registry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable] = {}
        self._instances: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._locked = False

    def register(self, name: str, factory: Callable[[], Any]) -> None:
        with self._lock:
            if self._locked:
                raise RuntimeError("Registry is locked after boot")
            self._factories[name] = factory

    def get(self, name: str) -> Any:
        with self._lock:
            if name not in self._instances:
                if name not in self._factories:
                    raise KeyError(f"Component '{name}' not registered")
                self._instances[name] = self._factories[name]()
            return self._instances[name]

    def lock(self) -> None:
        with self._lock:
            self._locked = True

    def resolve(self, t: type[T]) -> T:
        try:
            return self.get(t.__name__)
        except KeyError:
            return _stub(t)

def _stub(t: type) -> Any:
    class S:
        def __getattr__(self, n):
            return lambda *a, **k: None
    s = S()
    s.__class__.__name__ = f"Stub<{t.__name__}>"
    return s

_r: Registry | None = None
_rlock = threading.Lock()

def get_registry() -> Registry:
    global _r
    if _r is None:
        with _rlock:
            if _r is None:
                _r = Registry()
    return _r

registry = get_registry()
