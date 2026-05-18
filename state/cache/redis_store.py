"""C-04 redis + hiredis — Hot State Cache.

# ADAPTED FROM: redis/redis-py — ``redis/asyncio/client.py``
# (async Redis client), ``redis/commands/`` (``get`` / ``set`` /
# ``hset`` / ``hgetall`` / ``delete`` / ``exists`` / ``expire``)
# and ``redis/client.py::Pipeline`` (atomic command batches).
#
# Tier: OFFLINE_ONLY — this module provides a deterministic,
# in-process cache that mirrors the *surface* of redis-py for
# positions and risk snapshots in multi-process deployments. The
# real ``redis`` / ``hiredis`` PyPI packages are NEVER imported in
# this module; the lazy seams :func:`redis_client_factory` and
# :func:`async_redis_client_factory` raise
# :class:`NotImplementedError` until a future research-acceptance
# PR documents the shadow-equivalence comparison vs. the real
# clients.
#
# Authority discipline:
#
# * **Ledger is always authoritative.** A cache hit returns a value
#   that the *caller* wrote in via :meth:`RedisStore.set_position`
#   / :meth:`set_risk_snapshot`; the cache itself NEVER promotes a
#   value to typed-event form. If a cache entry is missing,
#   stale-by-TTL, or fails byte-equality validation against the
#   caller-supplied digest, the caller MUST fall back to the
#   ledger.
# * **No typed-event construction** — this module does not call
#   ``PatchProposal(...)``, ``HazardEvent(...)``, ``SignalEvent(...)``,
#   ``ExecutionEvent(...)`` or ``SystemEvent(...)``. The store only
#   carries opaque ``bytes`` payloads keyed by caller-chosen string
#   keys. B27 / B28 / INV-71 pinned by AST tests.
# * **B1 isolation** — no imports from ``intelligence_engine``,
#   ``execution_engine``, ``governance_engine``,
#   ``evolution_engine``, ``learning_engine``.
#
# Determinism (INV-15):
#
# * No top-level imports of :mod:`time` / :mod:`datetime` /
#   :mod:`random` / :mod:`asyncio` / :mod:`os` / :mod:`redis` /
#   :mod:`hiredis` / :mod:`numpy` / :mod:`torch` / :mod:`polars`.
# * All TTL arithmetic is event-time over a caller-supplied
#   monotonically-increasing ``ts_ns``. No wall-clock reads.
# * Frozen, slotted dataclasses everywhere. The store itself is a
#   mutable container, but every value-object exposed on its
#   surface is immutable.
# * BLAKE2b-16 ``store_digest`` over the canonical-sorted live
#   entries gives byte-identical replay equality.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Final

REDIS_STORE_VERSION: Final[int] = 1
"""Bumped on any wire-shape change to keys / values / digest."""

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("redis", "hiredis")
"""PyPI packages activated by the lazy seams below. Declared so the
canonical pin-set is complete, but the packages themselves are
NEVER imported in this module.
"""


# ---------------------------------------------------------------------------
# Value objects — keys, entries, configs
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class RedisConfig:
    """Bus-wide configuration for the (future) real Redis client.

    Carried as a value-object so the in-process store and the lazy
    seam factories accept the same shape.
    """

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    socket_timeout_ns: int = 5_000_000_000
    """Read/write timeout in nanoseconds — value-object only; the
    in-process store never blocks, but the real client needs a
    timeout. Default 5s.
    """
    decode_responses: bool = False
    """Match redis-py default — keep bytes raw so the caller chooses
    serialization."""

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("RedisConfig.host must be non-empty")
        if self.port <= 0 or self.port > 65535:
            raise ValueError(f"RedisConfig.port must be in (0, 65535]; got {self.port}")
        if self.db < 0:
            raise ValueError(f"RedisConfig.db must be >= 0; got {self.db}")
        if self.socket_timeout_ns <= 0:
            raise ValueError(
                f"RedisConfig.socket_timeout_ns must be positive; got {self.socket_timeout_ns}"
            )


@dataclasses.dataclass(frozen=True, slots=True, order=True)
class CacheEntry:
    """A single immutable cache entry.

    ``key`` is the redis key (str — caller chooses namespace, e.g.
    ``"dix:position:BTCUSDT"`` or ``"dix:risk:snapshot:v1"``).

    ``value`` is the opaque payload bytes. The store never inspects
    structure — the caller chooses serialization (JSON, msgpack,
    pickle — see :func:`serialize_payload` for the reference
    JSON codec).

    ``ts_ns`` is the *caller-supplied* event-time at which the entry
    was written. The store rejects writes whose ``ts_ns`` is less
    than the largest previously-seen ``ts_ns`` for the same key —
    monotone is required so TTL semantics are deterministic under
    replay.

    ``ttl_ns`` is the entry lifetime in nanoseconds. After
    ``ts_ns + ttl_ns`` the entry is treated as expired by
    :meth:`RedisStore.get` (and reaped by :meth:`expire_at`).
    """

    key: str
    value: bytes
    ts_ns: int
    ttl_ns: int

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("CacheEntry.key must be non-empty")
        if not isinstance(self.value, bytes):
            raise TypeError(f"CacheEntry.value must be bytes; got {type(self.value).__name__}")
        if self.ts_ns < 0:
            raise ValueError(f"CacheEntry.ts_ns must be >= 0; got {self.ts_ns}")
        if self.ttl_ns <= 0:
            raise ValueError(f"CacheEntry.ttl_ns must be positive; got {self.ttl_ns}")

    def expires_at_ns(self) -> int:
        """Absolute event-time at which this entry expires."""
        return self.ts_ns + self.ttl_ns

    def is_live_at(self, now_ns: int) -> bool:
        """Return True if ``now_ns`` is strictly inside the TTL window."""
        if now_ns < self.ts_ns:
            return False
        return now_ns < self.expires_at_ns()


@dataclasses.dataclass(frozen=True, slots=True, order=True)
class CommandRecord:
    """One pipelined command record returned from :meth:`RedisStore.pipeline`.

    Carries the canonical command name (``SET`` / ``DEL`` /
    ``EXPIRE``) plus its applied arguments, in the order the pipeline
    flushed them. Used for audit + INV-15 replay equality.
    """

    op: str
    key: str
    arg: int
    """Either the ``ts_ns`` of a SET / EXPIRE op or 0 for DEL."""

    def __post_init__(self) -> None:
        canonical = ("SET", "DEL", "EXPIRE")
        if self.op not in canonical:
            raise ValueError(f"CommandRecord.op must be one of {canonical}; got {self.op!r}")
        if not self.key:
            raise ValueError("CommandRecord.key must be non-empty")
        if self.arg < 0:
            raise ValueError(f"CommandRecord.arg must be >= 0; got {self.arg}")


@dataclasses.dataclass(frozen=True, slots=True, order=True)
class PipelineResult:
    """Frozen result envelope for :meth:`RedisStore.pipeline.execute`.

    ``commands`` is the ordered tuple of :class:`CommandRecord` rows
    actually applied to the store (rejected commands are excluded).

    ``digest`` is a BLAKE2b-16 hex over the canonical-sorted
    command stream — used for INV-15 3-run replay equality.
    """

    commands: tuple[CommandRecord, ...]
    digest: str

    def __post_init__(self) -> None:
        if len(self.digest) != 32:
            raise ValueError(
                "PipelineResult.digest must be 32 hex chars "
                f"(BLAKE2b-16); got len={len(self.digest)}"
            )


# ---------------------------------------------------------------------------
# Pure utility functions
# ---------------------------------------------------------------------------


def serialize_payload(payload: Mapping[str, object]) -> bytes:
    """Reference orjson-shape JSON codec via the stdlib.

    Returns bytes that compare equal for any two dicts with the same
    key/value content regardless of insertion order. Mirrors the
    serialization rule used by :mod:`system_engine.streaming.kafka_bus`
    so positions written by one bus map cleanly to cache entries
    consumed by another.
    """
    if not isinstance(payload, Mapping):
        raise TypeError(f"serialize_payload requires a Mapping; got {type(payload).__name__}")
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def deserialize_payload(blob: bytes) -> dict[str, object]:
    """Inverse of :func:`serialize_payload`. Returns a fresh dict."""
    if not isinstance(blob, bytes):
        raise TypeError(f"deserialize_payload requires bytes; got {type(blob).__name__}")
    out = json.loads(blob.decode("utf-8"))
    if not isinstance(out, dict):
        raise TypeError("deserialize_payload only round-trips dict payloads")
    return out


def store_digest(entries: Iterable[CacheEntry]) -> str:
    """Stable BLAKE2b-16 hex over canonical-sorted live cache entries.

    Sort order: ``(key asc, ts_ns asc)``. Each entry contributes
    ``key | b"\\x1f" | ts_ns | b"\\x1f" | ttl_ns | b"\\x1f" | value
    | b"\\x1e"`` to the hash. Stable across run / process / platform.
    """
    h = hashlib.blake2b(digest_size=16)
    ordered = sorted(list(entries), key=lambda e: (e.key, e.ts_ns))
    for entry in ordered:
        h.update(entry.key.encode("utf-8"))
        h.update(b"\x1f")
        h.update(str(entry.ts_ns).encode("ascii"))
        h.update(b"\x1f")
        h.update(str(entry.ttl_ns).encode("ascii"))
        h.update(b"\x1f")
        h.update(entry.value)
        h.update(b"\x1e")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# RedisStore — in-process deterministic cache
# ---------------------------------------------------------------------------


class RedisStore:
    """In-process deterministic mirror of :class:`redis.Redis`.

    Operations match the redis-py *surface* (``get`` / ``set`` /
    ``delete`` / ``exists`` / ``expire`` / ``ttl`` / ``mget`` /
    ``mset`` / ``pipeline``) but the implementation is a plain
    dict + monotone event-time clock. The store NEVER reads the
    wall clock; all expiry arithmetic uses the caller-supplied
    ``ts_ns``.

    Ledger discipline: this is a *cache*, not a source-of-truth. The
    canonical contract for callers is

    1. Read the ledger.
    2. Apply normal business logic.
    3. Best-effort populate the cache via :meth:`set`.
    4. On subsequent reads, try :meth:`get` first; if it returns
       :data:`None` (miss, stale, or TTL expired) fall back to the
       ledger.

    A cache hit MUST be byte-equal to the ledger value at the same
    ``ts_ns``. Callers SHOULD compute their own digest before
    writing and re-verify on read.

    Concurrency: callers running across processes share state via
    the real Redis client behind the lazy seam. The in-process
    store is single-threaded; callers needing cross-thread access
    must wrap calls with their own lock.
    """

    __slots__ = ("_entries", "_last_ts_ns")

    def __init__(self) -> None:
        self._entries: dict[str, CacheEntry] = {}
        self._last_ts_ns: int = -1

    # ------------------------------------------------------------------
    # Read surface
    # ------------------------------------------------------------------

    def get(self, key: str, *, now_ns: int) -> bytes | None:
        """Return the cached bytes for ``key``, or :data:`None`.

        Returns :data:`None` on miss, on TTL expiry, or when ``now_ns``
        is strictly before the entry's ``ts_ns`` (clock anomaly —
        caller MUST fall back to ledger).
        """
        _validate_key(key)
        _validate_ts(now_ns)
        entry = self._entries.get(key)
        if entry is None:
            return None
        if not entry.is_live_at(now_ns):
            return None
        return entry.value

    def mget(self, keys: Sequence[str], *, now_ns: int) -> tuple[bytes | None, ...]:
        """Batch ``get`` matching redis-py ``MGET`` order."""
        if not isinstance(keys, (list, tuple)):
            raise TypeError(
                f"RedisStore.mget requires a list or tuple of keys; got {type(keys).__name__}"
            )
        return tuple(self.get(k, now_ns=now_ns) for k in keys)

    def exists(self, key: str, *, now_ns: int) -> bool:
        """Return True iff a live, non-expired entry exists at ``key``."""
        return self.get(key, now_ns=now_ns) is not None

    def ttl_remaining_ns(self, key: str, *, now_ns: int) -> int | None:
        """Return remaining TTL in nanoseconds, or :data:`None` on miss/expiry.

        Mirrors redis-py ``ttl`` semantics (nanosecond precision
        instead of seconds).
        """
        _validate_key(key)
        _validate_ts(now_ns)
        entry = self._entries.get(key)
        if entry is None:
            return None
        if not entry.is_live_at(now_ns):
            return None
        return entry.expires_at_ns() - now_ns

    def entry(self, key: str) -> CacheEntry | None:
        """Return the raw :class:`CacheEntry` regardless of TTL.

        Used by audit + replay paths that need the underlying
        ``ts_ns`` / ``ttl_ns`` rather than just the value.
        """
        _validate_key(key)
        return self._entries.get(key)

    def keys(self) -> tuple[str, ...]:
        """Return all live (regardless of TTL) keys in sorted order."""
        return tuple(sorted(self._entries))

    def live_entries(self, *, now_ns: int) -> tuple[CacheEntry, ...]:
        """Return all entries that are live at ``now_ns``, sorted by key."""
        _validate_ts(now_ns)
        live = [e for e in self._entries.values() if e.is_live_at(now_ns)]
        return tuple(sorted(live, key=lambda e: (e.key, e.ts_ns)))

    # ------------------------------------------------------------------
    # Write surface
    # ------------------------------------------------------------------

    def set(
        self,
        key: str,
        value: bytes,
        *,
        ts_ns: int,
        ttl_ns: int,
    ) -> CacheEntry:
        """Set ``key`` to ``value`` at event-time ``ts_ns`` with TTL.

        Rejects writes whose ``ts_ns`` is less than the largest
        previously-seen ``ts_ns`` on this store — monotonicity is
        required for INV-15 deterministic replay.
        """
        entry = CacheEntry(key=key, value=value, ts_ns=ts_ns, ttl_ns=ttl_ns)
        self._apply_set(entry)
        return entry

    def mset(
        self,
        items: Mapping[str, bytes],
        *,
        ts_ns: int,
        ttl_ns: int,
    ) -> tuple[CacheEntry, ...]:
        """Batch set — all entries share ``ts_ns`` + ``ttl_ns``.

        Returns the inserted entries in canonical-sorted order
        (``key`` ascending) so callers can rely on byte-stable
        iteration.
        """
        if not isinstance(items, Mapping):
            raise TypeError(f"RedisStore.mset requires a Mapping; got {type(items).__name__}")
        out: list[CacheEntry] = []
        for key in sorted(items):
            out.append(self.set(key, items[key], ts_ns=ts_ns, ttl_ns=ttl_ns))
        return tuple(out)

    def delete(self, key: str) -> bool:
        """Remove ``key`` if present; return True if a row was removed."""
        _validate_key(key)
        return self._entries.pop(key, None) is not None

    def expire(self, key: str, ttl_ns: int, *, ts_ns: int) -> bool:
        """Reset the TTL of ``key`` anchored at ``ts_ns``.

        Returns True if the key existed and was extended; False on
        miss. Matches redis-py ``EXPIRE`` semantics, with explicit
        event-time anchor instead of wall-clock.
        """
        _validate_key(key)
        _validate_ts(ts_ns)
        if ttl_ns <= 0:
            raise ValueError(f"RedisStore.expire ttl_ns must be positive; got {ttl_ns}")
        existing = self._entries.get(key)
        if existing is None:
            return False
        if ts_ns < self._last_ts_ns:
            raise ValueError(
                "RedisStore.expire requires monotone event-time; "
                f"last_ts_ns={self._last_ts_ns} new_ts_ns={ts_ns}"
            )
        new_entry = CacheEntry(
            key=key,
            value=existing.value,
            ts_ns=ts_ns,
            ttl_ns=ttl_ns,
        )
        self._entries[key] = new_entry
        self._last_ts_ns = ts_ns
        return True

    def expire_at(self, *, now_ns: int) -> tuple[str, ...]:
        """Reap entries whose TTL has elapsed at ``now_ns``.

        Returns the canonical-sorted tuple of evicted keys.
        """
        _validate_ts(now_ns)
        evicted = [k for k, e in self._entries.items() if not e.is_live_at(now_ns)]
        for k in evicted:
            del self._entries[k]
        return tuple(sorted(evicted))

    def flushdb(self) -> int:
        """Drop everything. Returns the number of keys removed."""
        n = len(self._entries)
        self._entries.clear()
        return n

    # ------------------------------------------------------------------
    # Pipeline — atomic batched command sequence
    # ------------------------------------------------------------------

    def pipeline(self) -> RedisPipeline:
        """Return a fresh pipeline bound to this store.

        Pipelines accumulate commands and only apply them when
        :meth:`RedisPipeline.execute` is called. Mirrors
        ``redis.client.Pipeline``'s API and atomicity guarantee
        (all-or-nothing) within the in-process store.
        """
        return RedisPipeline(self)

    # ------------------------------------------------------------------
    # Apply layer — single chokepoint for monotonicity + TTL
    # ------------------------------------------------------------------

    def _apply_set(self, entry: CacheEntry) -> None:
        if entry.ts_ns < self._last_ts_ns:
            raise ValueError(
                "RedisStore.set requires monotone event-time; "
                f"last_ts_ns={self._last_ts_ns} "
                f"new_ts_ns={entry.ts_ns} key={entry.key!r}"
            )
        previous = self._entries.get(entry.key)
        if previous is not None and entry.ts_ns < previous.ts_ns:
            raise ValueError(
                "RedisStore.set requires monotone event-time "
                "per key; "
                f"key={entry.key!r} "
                f"prev_ts_ns={previous.ts_ns} "
                f"new_ts_ns={entry.ts_ns}"
            )
        self._entries[entry.key] = entry
        self._last_ts_ns = max(self._last_ts_ns, entry.ts_ns)


# ---------------------------------------------------------------------------
# RedisPipeline — atomic batched commands
# ---------------------------------------------------------------------------


class RedisPipeline:
    """Atomic batched command sequence bound to a :class:`RedisStore`.

    Mirrors ``redis.client.Pipeline``'s queue → execute flow.
    Commands are accumulated by :meth:`set` / :meth:`delete` /
    :meth:`expire` but NOT applied until :meth:`execute` is called.
    If any queued command fails validation at execute time, NONE
    of the queued commands are applied — matches real Redis
    ``MULTI`` / ``EXEC`` semantics.
    """

    __slots__ = ("_store", "_pending")

    def __init__(self, store: RedisStore) -> None:
        if not isinstance(store, RedisStore):
            raise TypeError(f"RedisPipeline requires a RedisStore; got {type(store).__name__}")
        self._store = store
        self._pending: list[tuple[str, object, ...]] = []

    def set(
        self,
        key: str,
        value: bytes,
        *,
        ts_ns: int,
        ttl_ns: int,
    ) -> RedisPipeline:
        """Queue a SET command. Returns self for chaining."""
        # Eagerly validate args so callers fail fast on type errors.
        _validate_key(key)
        if not isinstance(value, bytes):
            raise TypeError(f"RedisPipeline.set value must be bytes; got {type(value).__name__}")
        _validate_ts(ts_ns)
        if ttl_ns <= 0:
            raise ValueError("RedisPipeline.set ttl_ns must be positive")
        self._pending.append(("SET", key, value, ts_ns, ttl_ns))
        return self

    def delete(self, key: str) -> RedisPipeline:
        """Queue a DEL command. Returns self for chaining."""
        _validate_key(key)
        self._pending.append(("DEL", key))
        return self

    def expire(self, key: str, ttl_ns: int, *, ts_ns: int) -> RedisPipeline:
        """Queue an EXPIRE command. Returns self for chaining."""
        _validate_key(key)
        _validate_ts(ts_ns)
        if ttl_ns <= 0:
            raise ValueError("RedisPipeline.expire ttl_ns must be positive")
        self._pending.append(("EXPIRE", key, ts_ns, ttl_ns))
        return self

    def execute(self) -> PipelineResult:
        """Apply all queued commands atomically.

        Returns a :class:`PipelineResult` with the canonical
        sequence of :class:`CommandRecord` rows actually applied,
        and a BLAKE2b-16 digest over them.

        On any error, NO commands are applied — the store is
        rolled back to its pre-execute state by snapshotting
        ``_entries`` + ``_last_ts_ns`` before the first apply.
        """
        snapshot_entries = dict(self._store._entries)
        snapshot_last = self._store._last_ts_ns
        applied: list[CommandRecord] = []
        try:
            for cmd in self._pending:
                op = cmd[0]
                if op == "SET":
                    _, key, value, ts_ns, ttl_ns = cmd
                    self._store.set(key, value, ts_ns=ts_ns, ttl_ns=ttl_ns)
                    applied.append(CommandRecord(op="SET", key=key, arg=ts_ns))
                elif op == "DEL":
                    _, key = cmd
                    self._store.delete(key)
                    applied.append(CommandRecord(op="DEL", key=key, arg=0))
                elif op == "EXPIRE":
                    _, key, ts_ns, ttl_ns = cmd
                    self._store.expire(key, ttl_ns, ts_ns=ts_ns)
                    applied.append(CommandRecord(op="EXPIRE", key=key, arg=ts_ns))
                else:  # pragma: no cover - validated at queue time
                    raise ValueError(f"RedisPipeline.execute: unknown op {op!r}")
        except Exception:
            self._store._entries = snapshot_entries
            self._store._last_ts_ns = snapshot_last
            self._pending.clear()
            raise
        self._pending.clear()
        commands = tuple(applied)
        digest = _digest_commands(commands)
        return PipelineResult(commands=commands, digest=digest)


# ---------------------------------------------------------------------------
# Lazy seam factories — gated on research-acceptance PR
# ---------------------------------------------------------------------------


def redis_client_factory(config: RedisConfig) -> object:
    """Sync lazy seam to :class:`redis.Redis`.

    Implementation deferred to a future research-acceptance PR
    that documents:

    1. Shadow-equivalence comparison of read/write semantics
       between :class:`RedisStore` and real Redis (including
       atomicity of MULTI/EXEC, key eviction semantics under
       memory pressure, and TTL precision drift).
    2. Wire-compatibility of :func:`serialize_payload` with
       at least one other DIX bus (kafka_bus.py / faust_bus.py).
    3. Failure-injection strategy for connection drops, timeouts,
       and split-brain scenarios in multi-process deployment.
    """
    if not isinstance(config, RedisConfig):
        raise TypeError(
            f"redis_client_factory config must be RedisConfig; got {type(config).__name__}"
        )
    raise NotImplementedError(
        "Real redis.Redis activation is gated on a "
        "research-acceptance PR documenting shadow-equivalence "
        "vs. RedisStore. Use RedisStore() until then."
    )


def async_redis_client_factory(config: RedisConfig) -> object:
    """Async lazy seam to :class:`redis.asyncio.Redis`.

    Same gate as :func:`redis_client_factory`.
    """
    if not isinstance(config, RedisConfig):
        raise TypeError(
            f"async_redis_client_factory config must be RedisConfig; got {type(config).__name__}"
        )
    raise NotImplementedError(
        "Real redis.asyncio.Redis activation is gated on a "
        "research-acceptance PR documenting shadow-equivalence "
        "vs. RedisStore. Use RedisStore() until then."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_key(key: str) -> None:
    if not isinstance(key, str):
        raise TypeError(f"RedisStore key must be str; got {type(key).__name__}")
    if not key:
        raise ValueError("RedisStore key must be non-empty")


def _validate_ts(ts_ns: int) -> None:
    if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
        raise TypeError(f"RedisStore ts_ns must be int; got {type(ts_ns).__name__}")
    if ts_ns < 0:
        raise ValueError(f"RedisStore ts_ns must be >= 0; got {ts_ns}")


def _digest_commands(commands: Sequence[CommandRecord]) -> str:
    """BLAKE2b-16 hex over a command sequence in insertion order.

    Pipelines preserve queue order on execute, so insertion order
    is the canonical order — no resort needed.
    """
    h = hashlib.blake2b(digest_size=16)
    for cmd in commands:
        h.update(cmd.op.encode("ascii"))
        h.update(b"\x1f")
        h.update(cmd.key.encode("utf-8"))
        h.update(b"\x1f")
        h.update(str(cmd.arg).encode("ascii"))
        h.update(b"\x1e")
    return h.hexdigest()
