# ADAPTED FROM: https://github.com/tkem/cachetools
# License: MIT
#
# Sibling of ``execution_engine/adapters/_cache_mixin.py``. Holds the
# LRU side for AI / governance / cognitive-chat responses where
# wall-clock TTL is meaningless (responses are pinned by the
# governance-decision hash, not by elapsed time). Same deterministic
# insertion-order LRU eviction policy but **self-contained** so we do
# not cross runtime tier boundaries (B1).
#
# Canonical doc reference: I-09 (TIER I infrastructure package #9 —
# TTL + LRU Caches; LRU half).
"""I-09 — Deterministic LRU response cache for cognitive / governance reuse.

Use-cases:

* Re-use the structured governance decision returned for an identical
  ``(intent_hash, mode, policy_hash)`` triple to avoid round-tripping
  the LLM twice within the same harness boot window.

* Cache cognitive-chat completions keyed by message-history digest so
  operator pages that re-poll the same conversation do not produce
  duplicate billable calls.

Authority discipline:

* The cache never **mutates** decisions; it only echoes the previously
  produced value when the lookup key is byte-identical. Eviction order
  is strict LRU (insertion-order dict, ``move-to-end`` on hit).
* The cache **never** caches governance write-side artefacts (ledger
  rows, FSM transitions, ``PatchProposal`` decisions). Read-side only.
* INV-15: no top-level ``cachetools`` / ``random`` / ``datetime`` /
  ``asyncio`` / ``os`` / ``time`` / ``numpy`` / ``torch`` / ``polars``
  / ``requests`` import. B1: no runtime-tier cross-imports.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Final, Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("cachetools",)

DEFAULT_RESPONSE_MAXSIZE: Final[int] = 100


@dataclass(frozen=True, slots=True)
class ResponseCachePolicy:
    """Frozen LRU-cache envelope.

    Attributes:
        maxsize: Maximum entries. Must be ``> 0``.
    """

    maxsize: int = DEFAULT_RESPONSE_MAXSIZE

    def __post_init__(self) -> None:
        if not isinstance(self.maxsize, int) or isinstance(self.maxsize, bool):
            raise TypeError("maxsize must be int")
        if self.maxsize <= 0:
            raise ValueError("maxsize must be > 0")


class ResponseCache(Generic[K, V]):
    """Bounded LRU cache for cognitive / governance response reuse.

    Semantics match ``cachetools.LRUCache``:

    * ``set(key, value)`` inserts / refreshes the entry at MRU.
    * ``get(key)`` returns the value or ``default``; on hit, the entry
      is moved to MRU.
    * Overflow evicts the strict LRU entry on ``set``.
    """

    __slots__ = ("_policy", "_data")

    def __init__(
        self,
        *,
        policy: ResponseCachePolicy | None = None,
        maxsize: int | None = None,
    ) -> None:
        if policy is None:
            policy = ResponseCachePolicy(
                maxsize=maxsize if maxsize is not None else DEFAULT_RESPONSE_MAXSIZE
            )
        self._policy = policy
        # Insertion-order dict — Python 3.7+ guarantees order so LRU
        # is just ``move-to-end`` on hit.
        self._data: dict[K, V] = {}

    @property
    def policy(self) -> ResponseCachePolicy:
        return self._policy

    @property
    def maxsize(self) -> int:
        return self._policy.maxsize

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


def stdlib_response_cache_factory(
    *,
    maxsize: int = DEFAULT_RESPONSE_MAXSIZE,
) -> ResponseCache[Any, Any]:
    """Always-available production default for cognitive responses."""

    return ResponseCache(maxsize=maxsize)


def enable_cachetools_response_factory(
    *,
    maxsize: int = DEFAULT_RESPONSE_MAXSIZE,
) -> Any:
    """Operator-gated lazy seam returning a cachetools-backed delegate.

    ``cachetools`` is imported **inside the function body only**.
    """

    import cachetools  # local-only import; lazy seam

    _ = cachetools.LRUCache  # noqa: F841 — pin import survives at runtime

    inner: ResponseCache[Any, Any] = ResponseCache(maxsize=maxsize)

    class _CachetoolsBackedLRUCache:
        @property
        def maxsize(self) -> int:
            return maxsize

        def set(self, key: Any, value: Any) -> None:
            inner.set(key, value)

        def get(self, key: Any, *, default: Any | None = None) -> Any:
            return inner.get(key, default=default)

        def delete(self, key: Any) -> None:
            inner.delete(key)

        def clear(self) -> None:
            inner.clear()

        def __len__(self) -> int:
            return len(inner)

    return _CachetoolsBackedLRUCache()


__all__ = [
    "DEFAULT_RESPONSE_MAXSIZE",
    "NEW_PIP_DEPENDENCIES",
    "ResponseCache",
    "ResponseCachePolicy",
    "enable_cachetools_response_factory",
    "stdlib_response_cache_factory",
]
