"""
mind.sources.rate_limiter — tiny token-bucket rate limiter, per-provider.

Safe across threads. Non-blocking by default (``acquire`` returns False
if no capacity); callers either skip or sleep themselves.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class TokenBucket:
    rate_per_sec: float       # tokens refilled per second
    capacity: float           # max tokens
    _tokens: float = 0.0
    _last: float = 0.0
    _lock: threading.Lock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, n: float = 1.0) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_sec)
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False

    def wait_and_acquire(self, n: float = 1.0, timeout_s: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.acquire(n):
                return True
            time.sleep(0.05)
        return False
