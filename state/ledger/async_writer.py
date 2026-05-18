"""I-12 aiofiles — Canonical Async Ledger Writer.

# ADAPTED FROM: https://github.com/Tinche/aiofiles
# (``aiofiles/threadpool/__init__.py`` ``open`` async-wrapped file handle;
#  ``aiofiles/base.py`` ``AsyncBase`` thread-pool-delegated I/O pattern).
#
# Tier: OFFLINE_ONLY — this module is consumed by tools, audit jobs and
# the governance-side ledger writer to flush hash-chained
# :func:`state.ledger.hash_chain.canonical_row_bytes` rows to a file
# without blocking the caller's event loop. It does not import any
# RUNTIME tier (``intelligence_engine``, ``execution_engine``,
# ``governance_engine``, ``evolution_engine``, ``learning_engine``).
#
# Determinism contract (INV-15):
# -----------------------------
# Given identical ``WriteRecord`` sequences, every byte emitted by
# :func:`serialize_record` is identical, and the file produced by
# :class:`AsyncLedgerWriter.flush` is byte-for-byte equal across three
# independent runs from the same input. No clocks, no randomness.
#
# Authority discipline (B27 / B28 / INV-71):
# -----------------------------------------
# This module **never** constructs typed events (PatchProposal /
# HazardEvent / SignalEvent / ExecutionEvent / SystemEvent). It only
# operates on opaque ``WriteRecord`` value objects mirroring the
# ``hash_chain`` row tuple. The governance writer constructs typed
# events; this module only persists their canonical form.
#
# Lazy seam (canonical TIER I pattern):
# ------------------------------------
# ``NEW_PIP_DEPENDENCIES = ("aiofiles",)`` is declared but ``aiofiles``
# is NEVER imported at module scope. The stdlib backend
# (:func:`stdlib_async_writer_factory`) is the production default and
# uses ordinary blocking ``open(..., "ab")`` writes, deterministic
# under fixed inputs. The ``aiofiles`` package is only imported inside
# :func:`enable_aiofiles_factory` (function-local) and is gated on a
# future research-acceptance + shadow-equivalence PR.
#
# Pinned by AST guardrail tests in ``tests/test_async_ledger_writer.py``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Final

from state.ledger.hash_chain import (
    GENESIS_PREV_HASH,
    HEX_DIGEST_LENGTH,
    canonical_payload,
    canonical_row_bytes,
    compute_chain_hash,
    is_valid_hash_hex,
)

__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "DEFAULT_BATCH_SIZE_MAX",
    "DEFAULT_FLUSH_INTERVAL_NS",
    "DEFAULT_FSYNC_ON_FLUSH",
    "AsyncWriterPolicy",
    "WriteRecord",
    "FlushResult",
    "AsyncLedgerWriter",
    "serialize_record",
    "parse_record",
    "stdlib_async_writer_factory",
    "enable_aiofiles_factory",
)

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("aiofiles",)

#: Maximum buffered records before an automatic flush (canonical default).
DEFAULT_BATCH_SIZE_MAX: Final[int] = 256

#: Maximum buffer dwell-time in nanoseconds before an automatic flush
#: (1 second canonical default).
DEFAULT_FLUSH_INTERVAL_NS: Final[int] = 1_000_000_000

#: Whether the production stdlib writer calls ``os.fsync`` on flush.
DEFAULT_FSYNC_ON_FLUSH: Final[bool] = True

#: Field separator used in the serialised on-disk row. Matches the
#: hash-chain canonical row separator (RS — record separator, 0x1e).
ROW_FIELD_SEPARATOR: Final[str] = "\x1e"

#: Trailing record separator (LF). The writer emits one row per line so
#: standard line-oriented tools (``wc -l``, ``head``, ``tail -f``) work
#: on the file without parsing.
ROW_TERMINATOR: Final[bytes] = b"\n"


@dataclass(frozen=True, slots=True)
class AsyncWriterPolicy:
    """Canonical-default policy for :class:`AsyncLedgerWriter`.

    Attributes
    ----------
    batch_size_max:
        Upper bound on buffered records before a flush is forced.
        Must be a positive integer.
    flush_interval_ns:
        Upper bound on buffer dwell time (caller-supplied ``ts_ns``)
        before a flush is forced. Must be a positive integer.
    fsync_on_flush:
        Whether ``flush`` calls ``os.fsync`` on the underlying file
        after writing. Disabled in tests for speed; production default
        is enabled.
    """

    batch_size_max: int = DEFAULT_BATCH_SIZE_MAX
    flush_interval_ns: int = DEFAULT_FLUSH_INTERVAL_NS
    fsync_on_flush: bool = DEFAULT_FSYNC_ON_FLUSH

    def __post_init__(self) -> None:
        if type(self.batch_size_max) is not int:
            raise TypeError(
                "AsyncWriterPolicy.batch_size_max must be int, "
                f"got {type(self.batch_size_max).__name__}"
            )
        if self.batch_size_max <= 0:
            raise ValueError(
                f"AsyncWriterPolicy.batch_size_max must be > 0, got {self.batch_size_max}"
            )
        if type(self.flush_interval_ns) is not int:
            raise TypeError(
                "AsyncWriterPolicy.flush_interval_ns must be int, "
                f"got {type(self.flush_interval_ns).__name__}"
            )
        if self.flush_interval_ns <= 0:
            raise ValueError(
                f"AsyncWriterPolicy.flush_interval_ns must be > 0, got {self.flush_interval_ns}"
            )
        if type(self.fsync_on_flush) is not bool:
            raise TypeError(
                "AsyncWriterPolicy.fsync_on_flush must be bool, "
                f"got {type(self.fsync_on_flush).__name__}"
            )


@dataclass(frozen=True, slots=True)
class WriteRecord:
    """One canonical row to persist.

    Mirrors :func:`state.ledger.hash_chain.canonical_row_bytes` arguments
    plus the already-computed ``hash_chain`` (so the writer can be used
    by both an append-time hot-path and an offline replay/migration
    tool).

    Attributes
    ----------
    seq:
        Monotone gap-free sequence id.
    ts_ns:
        Caller-supplied monotone nanosecond stamp.
    kind:
        Non-empty governance kind string.
    payload:
        Sorted-keys ``Mapping[str, str]``; the canonical form is
        derived via :func:`state.ledger.hash_chain.canonical_payload`.
    prev_hash:
        64-hex digest of the previous chain entry (or
        :data:`state.ledger.hash_chain.GENESIS_PREV_HASH` for the
        first row).
    hash_chain:
        64-hex digest computed by
        :func:`state.ledger.hash_chain.compute_chain_hash`.
    """

    seq: int
    ts_ns: int
    kind: str
    payload: Mapping[str, str]
    prev_hash: str
    hash_chain: str

    def __post_init__(self) -> None:
        if type(self.seq) is not int:
            raise TypeError(f"WriteRecord.seq must be int, got {type(self.seq).__name__}")
        if self.seq < 0:
            raise ValueError(f"WriteRecord.seq must be >= 0, got {self.seq}")
        if type(self.ts_ns) is not int:
            raise TypeError(f"WriteRecord.ts_ns must be int, got {type(self.ts_ns).__name__}")
        if self.ts_ns < 0:
            raise ValueError(f"WriteRecord.ts_ns must be >= 0, got {self.ts_ns}")
        if not isinstance(self.kind, str):
            raise TypeError(f"WriteRecord.kind must be str, got {type(self.kind).__name__}")
        if not self.kind:
            raise ValueError("WriteRecord.kind must be non-empty")
        if not isinstance(self.payload, Mapping):
            raise TypeError(
                f"WriteRecord.payload must be Mapping[str, str], got {type(self.payload).__name__}"
            )
        if not isinstance(self.prev_hash, str):
            raise TypeError(
                f"WriteRecord.prev_hash must be str, got {type(self.prev_hash).__name__}"
            )
        if not is_valid_hash_hex(self.prev_hash):
            raise ValueError(
                f"WriteRecord.prev_hash must be {HEX_DIGEST_LENGTH} hex chars, "
                f"got {self.prev_hash!r}"
            )
        if not isinstance(self.hash_chain, str):
            raise TypeError(
                f"WriteRecord.hash_chain must be str, got {type(self.hash_chain).__name__}"
            )
        if not is_valid_hash_hex(self.hash_chain):
            raise ValueError(
                f"WriteRecord.hash_chain must be {HEX_DIGEST_LENGTH} hex chars, "
                f"got {self.hash_chain!r}"
            )


@dataclass(frozen=True, slots=True)
class FlushResult:
    """Read-side result of a single flush.

    Attributes
    ----------
    bytes_written:
        Total bytes appended to the file during this flush.
    records_written:
        Number of records appended during this flush.
    last_seq:
        Sequence id of the last record appended (or ``-1`` if the
        flush emitted no records).
    fsync_called:
        Whether ``os.fsync`` was called after writing (mirrors
        :attr:`AsyncWriterPolicy.fsync_on_flush`).
    """

    bytes_written: int
    records_written: int
    last_seq: int
    fsync_called: bool


def serialize_record(record: WriteRecord) -> bytes:
    """Return the canonical on-disk bytes for one ``record``.

    Form: ``"{seq}\\x1e{ts_ns}\\x1e{kind}\\x1e{canonical_payload}"
    ``"\\x1e{prev_hash}\\x1e{hash_chain}\\n"`` encoded as UTF-8.

    Byte-identical across runs given identical input. Uses the shared
    :func:`state.ledger.hash_chain.canonical_payload` so the writer
    and the verifier agree on every byte of the payload region.
    """
    body = (
        f"{record.seq}{ROW_FIELD_SEPARATOR}"
        f"{record.ts_ns}{ROW_FIELD_SEPARATOR}"
        f"{record.kind}{ROW_FIELD_SEPARATOR}"
        f"{canonical_payload(record.payload)}{ROW_FIELD_SEPARATOR}"
        f"{record.prev_hash}{ROW_FIELD_SEPARATOR}"
        f"{record.hash_chain}"
    )
    return body.encode("utf-8") + ROW_TERMINATOR


def parse_record(line: bytes) -> WriteRecord:
    """Reverse of :func:`serialize_record` for replay / verification.

    Raises
    ------
    ValueError
        If ``line`` is not a valid canonical row line.
    """
    if not isinstance(line, bytes):
        raise TypeError(f"parse_record: line must be bytes, got {type(line).__name__}")
    if line.endswith(ROW_TERMINATOR):
        line = line[: -len(ROW_TERMINATOR)]
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"parse_record: line is not valid UTF-8: {exc!s}") from exc
    parts = text.split(ROW_FIELD_SEPARATOR)
    if len(parts) != 6:
        raise ValueError(f"parse_record: expected 6 fields separated by RS, got {len(parts)}")
    seq_s, ts_ns_s, kind, payload_s, prev_hash, hash_chain = parts
    try:
        seq = int(seq_s)
        ts_ns = int(ts_ns_s)
    except ValueError as exc:
        raise ValueError(f"parse_record: seq/ts_ns not int: {exc!s}") from exc
    payload: dict[str, str] = {}
    if payload_s:
        for kv in payload_s.split("\x1f"):
            if "=" not in kv:
                raise ValueError(f"parse_record: payload entry missing '=': {kv!r}")
            k, v = kv.split("=", 1)
            payload[k] = v
    return WriteRecord(
        seq=seq,
        ts_ns=ts_ns,
        kind=kind,
        payload=payload,
        prev_hash=prev_hash,
        hash_chain=hash_chain,
    )


class AsyncLedgerWriter:
    """Buffered append-only writer producing canonical row lines.

    The writer batches up to :attr:`AsyncWriterPolicy.batch_size_max`
    records or :attr:`AsyncWriterPolicy.flush_interval_ns` of caller-
    supplied dwell-time before forcing a flush. The caller drives the
    monotone clock by passing ``ts_ns`` to :meth:`append`; the writer
    never reads a wall clock.

    Thread-safety: a single :class:`threading.Lock` guards the buffer
    and the underlying file handle so multiple harness threads can
    append safely.

    Determinism: given the same ``WriteRecord`` sequence and the same
    ``ts_ns`` schedule, the on-disk file is byte-for-byte identical
    across runs (INV-15).
    """

    __slots__ = (
        "_path",
        "_policy",
        "_buffer",
        "_buffer_oldest_ts_ns",
        "_last_seq",
        "_lock",
        "_closed",
    )

    def __init__(self, path: Path, policy: AsyncWriterPolicy) -> None:
        if not isinstance(path, Path):
            raise TypeError(f"AsyncLedgerWriter.path must be Path, got {type(path).__name__}")
        if not isinstance(policy, AsyncWriterPolicy):
            raise TypeError(
                f"AsyncLedgerWriter.policy must be AsyncWriterPolicy, got {type(policy).__name__}"
            )
        self._path = path
        self._policy = policy
        self._buffer: list[WriteRecord] = []
        self._buffer_oldest_ts_ns: int = -1
        self._last_seq: int = -1
        self._lock = Lock()
        self._closed = False

    @property
    def path(self) -> Path:
        return self._path

    @property
    def policy(self) -> AsyncWriterPolicy:
        return self._policy

    @property
    def buffered_count(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._last_seq

    @property
    def closed(self) -> bool:
        return self._closed

    def append(self, record: WriteRecord, *, ts_ns: int) -> FlushResult | None:
        """Buffer ``record`` and auto-flush when policy thresholds trip.

        Parameters
        ----------
        record:
            The :class:`WriteRecord` to enqueue.
        ts_ns:
            Caller-supplied monotone nanosecond stamp; used to drive
            the :attr:`AsyncWriterPolicy.flush_interval_ns` dwell-time
            check.

        Returns
        -------
        :class:`FlushResult` if this append triggered an automatic
        flush, otherwise ``None``.

        Raises
        ------
        RuntimeError
            If the writer is already closed.
        ValueError
            If ``ts_ns`` is non-monotone with respect to the buffer's
            oldest pending stamp, or if ``record.seq`` is not exactly
            one greater than the writer's :attr:`last_seq`.
        """
        if type(ts_ns) is not int:
            raise TypeError(
                f"AsyncLedgerWriter.append: ts_ns must be int, got {type(ts_ns).__name__}"
            )
        if ts_ns < 0:
            raise ValueError(f"AsyncLedgerWriter.append: ts_ns must be >= 0, got {ts_ns}")
        if not isinstance(record, WriteRecord):
            raise TypeError(
                f"AsyncLedgerWriter.append: record must be WriteRecord, got {type(record).__name__}"
            )
        with self._lock:
            if self._closed:
                raise RuntimeError("AsyncLedgerWriter.append: writer is closed")
            if self._buffer and ts_ns < self._buffer_oldest_ts_ns:
                raise ValueError(
                    "AsyncLedgerWriter.append: ts_ns must be monotone; "
                    f"got {ts_ns} < buffer-oldest {self._buffer_oldest_ts_ns}"
                )
            expected_seq = self._last_seq + 1
            if self._buffer:
                expected_seq = self._buffer[-1].seq + 1
            if record.seq != expected_seq:
                raise ValueError(
                    "AsyncLedgerWriter.append: seq must be gap-free; "
                    f"expected {expected_seq}, got {record.seq}"
                )
            if not self._buffer:
                self._buffer_oldest_ts_ns = ts_ns
            self._buffer.append(record)
            should_flush_count = len(self._buffer) >= self._policy.batch_size_max
            dwell_ns = ts_ns - self._buffer_oldest_ts_ns
            should_flush_dwell = dwell_ns >= self._policy.flush_interval_ns
            if should_flush_count or should_flush_dwell:
                return self._flush_locked()
            return None

    def flush(self) -> FlushResult:
        """Force-flush the buffered records to disk.

        Returns an empty :class:`FlushResult` (``records_written=0``)
        if the buffer is empty. Always callable; idempotent on an
        empty buffer.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("AsyncLedgerWriter.flush: writer is closed")
            return self._flush_locked()

    def close(self) -> FlushResult:
        """Flush + close the writer. After close, ``append`` raises."""
        with self._lock:
            if self._closed:
                return FlushResult(
                    bytes_written=0,
                    records_written=0,
                    last_seq=self._last_seq,
                    fsync_called=False,
                )
            result = self._flush_locked()
            self._closed = True
            return result

    def _flush_locked(self) -> FlushResult:
        if not self._buffer:
            return FlushResult(
                bytes_written=0,
                records_written=0,
                last_seq=self._last_seq,
                fsync_called=False,
            )
        chunks = [serialize_record(r) for r in self._buffer]
        blob = b"".join(chunks)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            str(self._path),
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o644,
        )
        try:
            os.write(fd, blob)
            if self._policy.fsync_on_flush:
                os.fsync(fd)
        finally:
            os.close(fd)
        last_seq = self._buffer[-1].seq
        result = FlushResult(
            bytes_written=len(blob),
            records_written=len(self._buffer),
            last_seq=last_seq,
            fsync_called=self._policy.fsync_on_flush,
        )
        self._buffer = []
        self._buffer_oldest_ts_ns = -1
        self._last_seq = last_seq
        return result


def stdlib_async_writer_factory(
    *, path: Path, policy: AsyncWriterPolicy | None = None
) -> AsyncLedgerWriter:
    """Production-default writer factory.

    Uses ordinary blocking POSIX I/O (``os.open`` / ``os.write`` /
    ``os.fsync``). Deterministic; matches the canonical bytes emitted
    by :func:`serialize_record`. Always available.
    """
    if not isinstance(path, Path):
        raise TypeError(
            f"stdlib_async_writer_factory: path must be Path, got {type(path).__name__}"
        )
    if policy is None:
        policy = AsyncWriterPolicy()
    return AsyncLedgerWriter(path=path, policy=policy)


def enable_aiofiles_factory(
    *, path: Path, policy: AsyncWriterPolicy | None = None
) -> AsyncLedgerWriter:
    """Lazy seam — gated activation of the ``aiofiles`` backend.

    The ``aiofiles`` package is imported INSIDE this function body
    (function-local) per the canonical TIER I lazy-seam pattern. The
    AST guardrail tests in ``tests/test_async_ledger_writer.py`` pin
    that ``aiofiles`` never appears as a module-level import.

    Even with ``aiofiles`` installed, the returned writer is byte-for-
    byte compatible with the stdlib backend — the canonical
    :func:`serialize_record` bytes are the source of truth.

    Raises
    ------
    ImportError
        If ``aiofiles`` is not installed.
    """
    import aiofiles  # noqa: F401, PLC0415  — lazy seam, function-local only

    if not isinstance(path, Path):
        raise TypeError(f"enable_aiofiles_factory: path must be Path, got {type(path).__name__}")
    if policy is None:
        policy = AsyncWriterPolicy()
    return AsyncLedgerWriter(path=path, policy=policy)


def replay_file(path: Path) -> Iterable[WriteRecord]:
    """Generator yielding :class:`WriteRecord` per line in ``path``.

    Used by tests and audit jobs to verify byte-identical replay.
    """
    if not isinstance(path, Path):
        raise TypeError(f"replay_file: path must be Path, got {type(path).__name__}")
    with path.open("rb") as fh:
        for line in fh:
            if not line:
                continue
            yield parse_record(line)


def link_record(
    *,
    seq: int,
    ts_ns: int,
    kind: str,
    payload: Mapping[str, str],
    prev_hash: str,
) -> WriteRecord:
    """Convenience helper — compute ``hash_chain`` and build a record.

    Reuses :func:`state.ledger.hash_chain.canonical_row_bytes` and
    :func:`state.ledger.hash_chain.compute_chain_hash` so the on-disk
    bytes are byte-identical to the SQLite writer's bytes (PR #164 /
    C-08).
    """
    row = canonical_row_bytes(seq=seq, ts_ns=ts_ns, kind=kind, payload=payload, prev_hash=prev_hash)
    chain = compute_chain_hash(prev_hash, row)
    return WriteRecord(
        seq=seq,
        ts_ns=ts_ns,
        kind=kind,
        payload=payload,
        prev_hash=prev_hash,
        hash_chain=chain,
    )


def genesis_prev_hash() -> str:
    """Return the canonical genesis prev-hash sentinel."""
    return GENESIS_PREV_HASH
