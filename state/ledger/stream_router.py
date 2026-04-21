"""
state/ledger/stream_router.py
Logical separation of event streams by event_type → subscriber list.

Producers: EventStore.append (indirectly via this router)
Consumers: projectors, governance kernel, observability pipelines

Thread-safe. Non-blocking. Handlers are called in registration order.
Exceptions inside a handler never affect other handlers.
"""
from __future__ import annotations

import threading
from collections.abc import Callable

Handler = Callable[[dict], None]


class StreamRouter:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subs: dict[str, list[Handler]] = {}
        self._all: list[Handler] = []

    def subscribe(self, event_type: str, handler: Handler) -> None:
        with self._lock:
            self._subs.setdefault(event_type.upper(), []).append(handler)

    def subscribe_all(self, handler: Handler) -> None:
        with self._lock:
            self._all.append(handler)

    def publish(self, event: dict) -> None:
        et = str(event.get("event_type", "")).upper()
        with self._lock:
            specific = list(self._subs.get(et, []))
            universal = list(self._all)
        for h in specific + universal:
            try:
                h(event)
            except Exception:
                continue


_router: StreamRouter | None = None
_lock = threading.Lock()


def get_stream_router() -> StreamRouter:
    global _router
    if _router is None:
        with _lock:
            if _router is None:
                _router = StreamRouter()
    return _router
