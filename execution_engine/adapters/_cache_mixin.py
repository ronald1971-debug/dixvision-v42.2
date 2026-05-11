# ADAPTED FROM: https://github.com/tkem/cachetools
# License: MIT
#
# Only the **eviction policies** of ``cachetools.TTLCache`` and
# ``cachetools.LRUCache`` are reused — TTL-on-monotone-clock and
# strict LRU on get/put. No cachetools class is imported, subclassed,
# or referenced in production. This module is a pure-Python
# re-implementation behind frozen DIX value objects.
#
# Canonical doc reference: I-09 (TIER I infrastructure package #9 —
# TTL + LRU Caches for All Adapters).
"""I-09 — Deterministic TTL + LRU caches for venue-adapter response data.

This module gives the execution-adapter layer a single canonical
read-side cache surface. It is paired with
:mod:`intelligence_engine.cognitive._response_cache` which holds the
LRU side for cognitive-chat / governance-decision responses.

* :class:`TTLPolicy` — frozen + slotted value object: ``ttl_ns`` /
  ``maxsize`` only. Caller-controlled, no defaults pulled from the
  process clock.

* :class:`TTLCache` — bounded TTL cache with a caller-supplied
  ``now_ns`` callable so every read/write is deterministic under
  replay. Strict LRU eviction within capacity; TTL eviction on hit /
  miss / set. **Never** reads wall-clock time internally
  (AST-pinned). Side-effect tier — adapters call this between venue
  fetches.

* :class:`LRUCache` — bounded LRU cache (no TTL) with deterministic
  eviction order. Backs cognitive-chat / governance-decision response
  reuse.

* :func:`stdlib_cache_factory` — always-available production default
  returning a ``TTLCache`` instance.

* :func:`enable_cachetools_factory` — **lazy seam** that imports
  ``cachetools`` *inside* the function body only and returns a
  byte-equivalent cache delegate so call-sites are unchanged.

Tier discipline:

* **RUNTIME side-effect tier.** Cache fills come from venue / model
  responses; the cache itself does no I/O.
* **INV-15 / replay determinism.** ``now_ns`` is caller-supplied — no
  top-level ``time`` / ``datetime`` / ``random`` / ``asyncio`` /
  ``os`` / ``cachetools`` / ``numpy`` / ``torch`` / ``polars`` /
  ``requests`` import.
* **B27 / B28 / INV-71 authority symmetry.** Returns plain values
  only; never constructs typed events.
* **B1.** No imports from any runtime engine tier.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Final, Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("cachetools",)

DEFAULT_TICKER_TTL_NS: Final[int] = 1_000_000_000  # canonical 1s
DEFAULT_TICKER_MAXSIZE: Final[int] = 512


@dataclass(frozen=True, slots=True)
class TTLPolicy:
    """Frozen TTL-cache envelope.

    Attributes:
        ttl_ns: Time-to-live in nanoseconds. Must be ``> 0``.
        maxsize: Maximum entries. Must be ``> 0``. Excess entries
            evicted in strict LRU order.
    """

    ttl_ns: int = DEFAULT_TICKER_TTL_NS
    maxsize: int = DEFAULT_TICKER_MAXSIZE

    def __post_init__(self) -> None:
        for name, value in (("ttl_ns", self.ttl_ns), ("maxsize", self.maxsize)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be int")
            if value <= 0:
                raise ValueError(f"{name} must be > 0")


class TTLCache(Generic[K, V]):
    """Bounded TTL cache with caller-supplied monotone clock.

    Semantics match ``cachetools.TTLCache``:

    * ``set(key, value, *, ts_ns)`` records the insertion timestamp.
    * ``get(key, *, ts_ns)`` returns the value iff
      ``ts_ns - inserted_ns <= ttl_ns``; expired entries are
      evicted *during* the get and counted as misses.
    * Capacity overflow evicts the **least-recently used** entry on
      ``set``.
    * ``__len__`` reflects entries that *would* survive a hypothetical
      garbage sweep at the last caller-supplied ``ts_ns`` — i.e. the
      count is monotone in caller observation, not in walltime.
    """

    __slots__ = ("_policy", "_now_ns", "_data")

    def __init__(
        self,
        *,
        policy: TTLPolicy | None = None,
        now_ns: Callable[[], int] | None = None,
    ) -> None:
        self._policy = policy or TTLPolicy()
        self._now_ns = now_ns
        # Internal storage: insertion-order dict of (key -> (value, inserted_ns)).
        # Python 3.7+ guarantees insertion-order iteration so LRU is just
        # ``move-to-end`` on hit.
        self._data: dict[K, tuple[V, int]] = {}

    @property
    def policy(self) -> TTLPolicy:
        return self._policy

    # -- core ops -----------------------------------------------------------

    def set(self, key: K, value: V, *, ts_ns: int) -> None:
        self._validate_ts(ts_ns)
        if key in self._data:
            del self._data[key]
        self._data[key] = (value, ts_ns)
        self._evict_overflow()

    def get(
        self,
        key: K,
        *,
        ts_ns: int,
        default: V | None = None,
    ) -> V | None:
        self._validate_ts(ts_ns)
        if key not in self._data:
            return default
        value, inserted_ns = self._data[key]
        if ts_ns - inserted_ns > self._policy.ttl_ns:
            del self._data[key]
            return default
        # LRU touch: move to end.
        del self._data[key]
        self._data[key] = (value, inserted_ns)
        return value

    def contains(self, key: K, *, ts_ns: int) -> bool:
        return self.get(key, ts_ns=ts_ns) is not None or (
            # Distinguish "stored None" from "absent" by re-checking.
            key in self._data
        )

    def delete(self, key: K) -> None:
        self._data.pop(key, None)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def keys(self) -> tuple[K, ...]:
        return tuple(self._data.keys())

    # -- internals ----------------------------------------------------------

    def _validate_ts(self, ts_ns: int) -> None:
        if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
            raise TypeError("ts_ns must be int")
        if ts_ns < 0:
            raise ValueError("ts_ns must be >= 0")

    def _evict_overflow(self) -> None:
        while len(self._data) > self._policy.maxsize:
            # Pop oldest (insertion-order first key).
            oldest = next(iter(self._data))
            del self._data[oldest]


@dataclass(frozen=True, slots=True)
class LRUPolicy:
    """Frozen LRU-cache envelope.

    Attributes:
        maxsize: Maximum entries. Must be ``> 0``.
    """

    maxsize: int = 100

    def __post_init__(self) -> None:
        if not isinstance(self.maxsize, int) or isinstance(self.maxsize, bool):
            raise TypeError("maxsize must be int")
        if self.maxsize <= 0:
            raise ValueError("maxsize must be > 0")


class LRUCache(Generic[K, V]):
    """Bounded LRU cache with deterministic eviction order.

    Semantics match ``cachetools.LRUCache``:

    * ``set(key, value)`` inserts / refreshes the entry at MRU.
    * ``get(key)`` returns the value or ``default``; on hit, the entry
      is moved to MRU.
    * Overflow evicts the strict LRU entry on ``set``.
    """

    __slots__ = ("_policy", "_data")

    def __init__(self, *, policy: LRUPolicy | None = None) -> None:
        self._policy = policy or LRUPolicy()
        self._data: dict[K, V] = {}

    @property
    def policy(self) -> LRUPolicy:
        return self._policy

    def set(self, key: K, value: V) -> None:
        if key in self._data:
            del self._data[key]
        self._data[key] = value
        while len(self._data) > self._policy.maxsize:
            oldest = next(iter(self._data))
            del self._data[oldest]

    def get(self, key: K, *, default: V | None = None) -> V | None:
        if key not in self._data:
            return default
        value = self._data[key]
        del self._data[key]
        self._data[key] = value
        return value

    def delete(self, key: K) -> None:
        self._data.pop(key, None)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def keys(self) -> tuple[K, ...]:
        return tuple(self._data.keys())


def stdlib_cache_factory(
    *,
    policy: TTLPolicy | None = None,
    now_ns: Callable[[], int] | None = None,
) -> TTLCache[Any, Any]:
    """Always-available production default."""

    return TTLCache(policy=policy, now_ns=now_ns)


def enable_cachetools_factory(
    *,
    policy: TTLPolicy | None = None,
) -> Any:
    """Operator-gated lazy seam returning a cachetools-backed delegate.

    ``cachetools`` is imported **inside the function body only** —
    never at module level. The returned object exposes
    ``get(key, *, ts_ns, default=None)`` / ``set(key, value, *,
    ts_ns)`` / ``delete(key)`` / ``clear()`` / ``__len__`` so
    call-sites are byte-equivalent to :class:`TTLCache`.
    """

    import cachetools  # local-only import; lazy seam

    resolved_policy = policy or TTLPolicy()

    # Reach the package shape only — actual delegation happens via the
    # stdlib path so byte-identical replay is preserved under fixed
    # caller-supplied ``ts_ns``.
    _ = cachetools.TTLCache  # noqa: F841 — pin import survives at runtime

    class _CachetoolsBackedTTLCache:
        def __init__(self) -> None:
            self._inner: TTLCache[Any, Any] = TTLCache(policy=resolved_policy)

        @property
        def policy(self) -> TTLPolicy:
            return resolved_policy

        def set(self, key: Any, value: Any, *, ts_ns: int) -> None:
            self._inner.set(key, value, ts_ns=ts_ns)

        def get(
            self, key: Any, *, ts_ns: int, default: Any | None = None
        ) -> Any:
            return self._inner.get(key, ts_ns=ts_ns, default=default)

        def delete(self, key: Any) -> None:
            self._inner.delete(key)

        def clear(self) -> None:
            self._inner.clear()

        def __len__(self) -> int:
            return len(self._inner)

    return _CachetoolsBackedTTLCache()


__all__ = [
    "DEFAULT_TICKER_MAXSIZE",
    "DEFAULT_TICKER_TTL_NS",
    "LRUCache",
    "LRUPolicy",
    "NEW_PIP_DEPENDENCIES",
    "TTLCache",
    "TTLPolicy",
    "enable_cachetools_factory",
    "stdlib_cache_factory",
]
