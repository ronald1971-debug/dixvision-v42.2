"""Per-task watchdog — flags a task as STALLED if the bump counter
fails to advance within ``timeout_ns``.
"""

from __future__ import annotations


class Watchdog:
    """Single-task stall detector. Determinism via caller-provided ts_ns."""

    name: str = "watchdog"
    spec_id: str = "SYS-HEALTH-WD-01"

    __slots__ = ("_timeout_ns", "_last_bump_ns", "_armed")

    def __init__(self, timeout_ns: int = 2_000_000_000) -> None:
        if timeout_ns <= 0:
            raise ValueError("timeout_ns must be positive")
        self._timeout_ns = timeout_ns
        self._last_bump_ns: int | None = None
        self._armed = False

    def bump(self, ts_ns: int) -> None:
        if self._last_bump_ns is not None and ts_ns < self._last_bump_ns:
            raise ValueError("watchdog ts_ns must be monotonic")
        self._last_bump_ns = ts_ns
        self._armed = False

    def is_stalled(self, ts_ns: int) -> bool:
        """Return True at most once per stall episode."""

        if self._last_bump_ns is None or self._armed:
            return False
        if ts_ns - self._last_bump_ns < self._timeout_ns:
            return False
        self._armed = True
        return True


__all__ = ["Watchdog"]
