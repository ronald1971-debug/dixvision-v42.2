"""A-18.2 — Offline parquet snapshots of the canonical event ledger.

# ADAPTED FROM: apache/arrow (pyarrow/parquet/__init__.py write_table /
# read_table / ParquetWriter)

This module ships an **OFFLINE_ONLY** writer/reader pair that serialises
batches of canonical :class:`core.contracts.events.Event` values to
columnar Apache Parquet files for analytics, replay, and long-term
cold storage. The on-disk schema mirrors the field names of the four
typed events exactly so that replay tooling (and the future polyglot
Lane B port) can rebuild byte-identical :class:`Event` instances
without an intermediate translation table.

Tier discipline
---------------

* OFFLINE_ONLY: writes are never invoked on the runtime hot path.
  Authority lint forbids the hot-path tiers
  (``execution_engine`` / ``governance_engine`` / ``system_engine`` /
  ``intelligence_engine`` / ``evolution_engine``) from importing this
  module; only offline callers (``learning_engine.*``,
  ``simulation.*``, ``tools.*``, ``scripts.*``) may. The B1 / L1 / L2
  cross-engine lint rules already enforce this for the existing
  ``state.ledger.reader`` module; this module sits in the same package
  and reuses the allow-list semantics.

* Deterministic by construction (INV-15). Encoding is a pure
  function of the input event sequence: rows are written in input
  order, ``meta`` mappings are projected with **sorted keys** before
  serialisation, and no clock / random / uuid / process id is read at
  any point. The on-disk parquet bytes are byte-identical across
  re-runs (3-run equality pinned by the test suite). The writer
  passes ``compression=None``, ``use_dictionary=False``,
  ``write_statistics=False``, and overrides Arrow's default writer-
  version metadata to a fixed constant so encoder drift between
  pyarrow point-releases cannot leak into the bytes.

* No clock / random / IO context other than ``Path.write_bytes``
  (writer) and ``Path.read_bytes`` (reader). The pyarrow library is
  **lazy-imported** only inside the factory function bodies — the
  module top-level has zero pyarrow references so callers without
  pyarrow installed can still import :mod:`state.ledger.snapshots` to
  inspect the schema definition / sentinel constants.

* No engine cross-imports. The schema is keyed off the field names
  declared in :mod:`core.contracts.events` (the canonical 4-event
  contract); the writer and reader take a ``Sequence[Event]`` directly
  and never touch ``governance_engine`` / ``execution_engine`` /
  ``system_engine`` / ``intelligence_engine`` / ``evolution_engine``.

* No typed-event construction inside the writer. The writer only
  consumes events; the reader **does** construct events but only as
  the canonical replay-projection step (mirrors :func:`replay_l2`
  pattern from A-18.1 — the reader is the documented re-materialiser
  for offline tooling). Receivers that want to round-trip into the
  runtime ledger must route through the existing
  ``LedgerAuthorityWriter`` chokepoint; the reader's outputs are
  pure value objects for analytics, not bus traffic.

Schema
------

Each :class:`core.contracts.events.Event` variant occupies its own
typed parquet table. The four schemas are exposed as the module-level
constants below, and helper functions (:func:`event_to_signal_row` /
``...`` ) project each variant into a row. Mapping types (``meta`` and
``payload``) are encoded as **sorted-key JSON** strings so the on-disk
ordering is deterministic even for inputs whose Python ``dict``
insertion order differs.

Pip dependency
--------------

``NEW_PIP_DEPENDENCIES = ("pyarrow",)`` — declared at the module level
for ops tooling; lazy-imported only inside the factory bodies.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol

from core.contracts.events import (
    Event,
    EventKind,
    ExecutionEvent,
    HazardEvent,
    SignalEvent,
    SystemEvent,
)

if TYPE_CHECKING:  # pragma: no cover - type-only
    import pyarrow as _pa  # noqa: F401

__all__ = (
    "MAX_BATCH_ROWS",
    "NEW_PIP_DEPENDENCIES",
    "PARQUET_WRITER_VERSION",
    "SCHEMA_VERSION",
    "EXECUTION_COLUMN_NAMES",
    "HAZARD_COLUMN_NAMES",
    "SIGNAL_COLUMN_NAMES",
    "SYSTEM_COLUMN_NAMES",
    "SnapshotReadResult",
    "SnapshotSummary",
    "SnapshotTransport",
    "LedgerSnapshotWriter",
    "LedgerSnapshotReader",
    "execution_event_to_row",
    "hazard_event_to_row",
    "parse_meta_json",
    "signal_event_to_row",
    "split_events_by_kind",
    "system_event_to_row",
    "pyarrow_snapshot_writer_factory",
    "pyarrow_snapshot_reader_factory",
)

# ---------------------------------------------------------------------------
# Public sentinels
# ---------------------------------------------------------------------------

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("pyarrow",)
"""Lazy pip dependency surfaced for tooling — never imported at top level."""

SCHEMA_VERSION: Final[str] = "dix-events-v1"
"""Stamped into the file-level metadata so future schema migrations are
detectable without parsing the column list."""

PARQUET_WRITER_VERSION: Final[str] = "2.6"
"""Pinned parquet logical-version to defeat encoder drift between
pyarrow point releases (INV-15)."""

MAX_BATCH_ROWS: Final[int] = 1_048_576
"""Upper bound on a single :meth:`LedgerSnapshotWriter.write` call.
Keeps a snapshot file bounded to <= 2^20 rows per typed kind so
analytic readers can mmap a file without page-faulting the entire
ledger history."""

SIGNAL_COLUMN_NAMES: Final[tuple[str, ...]] = (
    "ts_ns",
    "symbol",
    "side",
    "confidence",
    "plugin_chain",
    "meta_json",
    "produced_by_engine",
    "signal_trust",
    "signal_source",
)
EXECUTION_COLUMN_NAMES: Final[tuple[str, ...]] = (
    "ts_ns",
    "symbol",
    "side",
    "qty",
    "price",
    "status",
    "venue",
    "order_id",
    "meta_json",
    "produced_by_engine",
)
SYSTEM_COLUMN_NAMES: Final[tuple[str, ...]] = (
    "ts_ns",
    "sub_kind",
    "source",
    "payload_json",
    "meta_json",
    "produced_by_engine",
    "proposed",
)
HAZARD_COLUMN_NAMES: Final[tuple[str, ...]] = (
    "ts_ns",
    "code",
    "severity",
    "source",
    "detail",
    "meta_json",
    "produced_by_engine",
)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SnapshotSummary:
    """Outcome record returned by :meth:`LedgerSnapshotWriter.write`."""

    path: Path
    signal_rows: int
    execution_rows: int
    system_rows: int
    hazard_rows: int

    @property
    def total_rows(self) -> int:
        return self.signal_rows + self.execution_rows + self.system_rows + self.hazard_rows


@dataclass(frozen=True, slots=True)
class SnapshotReadResult:
    """Outcome record returned by :meth:`LedgerSnapshotReader.read`.

    The reader **never constructs typed bus events** (Triad Lock /
    INV-56). It returns the raw projection rows segregated by EVT-01..04
    kind in stable insertion order; callers that need to re-materialise
    typed events must do so themselves through the canonical producing
    engine (``intelligence_engine.*`` for ``SignalEvent``,
    ``execution_engine.*`` for ``ExecutionEvent``, etc).
    """

    signal_rows: tuple[Mapping[str, object], ...]
    execution_rows: tuple[Mapping[str, object], ...]
    system_rows: tuple[Mapping[str, object], ...]
    hazard_rows: tuple[Mapping[str, object], ...]

    @property
    def total_rows(self) -> int:
        return (
            len(self.signal_rows)
            + len(self.execution_rows)
            + len(self.system_rows)
            + len(self.hazard_rows)
        )


# ---------------------------------------------------------------------------
# Pure projections — used by tests and by the writer/reader pair
# ---------------------------------------------------------------------------


def _sorted_meta_json(meta: Mapping[str, str]) -> str:
    """Return a sorted-key JSON projection of ``meta`` (INV-15).

    Empty mappings collapse to ``"{}"`` so the on-disk bytes are
    deterministic regardless of caller dict-insertion order.
    """
    if not meta:
        return "{}"
    return json.dumps(dict(sorted(meta.items())), separators=(",", ":"))


def parse_meta_json(blob: str) -> Mapping[str, str]:
    """Inverse of :func:`_sorted_meta_json` returning a frozen view.

    Exposed publicly because callers that re-materialise events from a
    :class:`SnapshotReadResult` need to recover the original mapping.
    """
    if not blob or blob == "{}":
        return MappingProxyType({})
    raw = json.loads(blob)
    if not isinstance(raw, dict):
        raise ValueError(
            f"snapshots.parse_meta_json: expected JSON object, got {type(raw).__name__}"
        )
    return MappingProxyType({str(k): str(v) for k, v in raw.items()})


def signal_event_to_row(ev: SignalEvent) -> dict[str, object]:
    """Pure projection ``SignalEvent`` → parquet row (sorted-key meta)."""
    return {
        "ts_ns": int(ev.ts_ns),
        "symbol": str(ev.symbol),
        "side": ev.side.value,
        "confidence": float(ev.confidence),
        "plugin_chain": list(ev.plugin_chain),
        "meta_json": _sorted_meta_json(ev.meta),
        "produced_by_engine": str(ev.produced_by_engine),
        "signal_trust": ev.signal_trust.value,
        "signal_source": str(ev.signal_source),
    }


def execution_event_to_row(ev: ExecutionEvent) -> dict[str, object]:
    """Pure projection ``ExecutionEvent`` → parquet row."""
    return {
        "ts_ns": int(ev.ts_ns),
        "symbol": str(ev.symbol),
        "side": ev.side.value,
        "qty": float(ev.qty),
        "price": float(ev.price),
        "status": ev.status.value,
        "venue": str(ev.venue),
        "order_id": str(ev.order_id),
        "meta_json": _sorted_meta_json(ev.meta),
        "produced_by_engine": str(ev.produced_by_engine),
    }


def system_event_to_row(ev: SystemEvent) -> dict[str, object]:
    """Pure projection ``SystemEvent`` → parquet row (sorted-key payload+meta)."""
    return {
        "ts_ns": int(ev.ts_ns),
        "sub_kind": ev.sub_kind.value,
        "source": str(ev.source),
        "payload_json": _sorted_meta_json(ev.payload),
        "meta_json": _sorted_meta_json(ev.meta),
        "produced_by_engine": str(ev.produced_by_engine),
        "proposed": bool(ev.proposed),
    }


def hazard_event_to_row(ev: HazardEvent) -> dict[str, object]:
    """Pure projection ``HazardEvent`` → parquet row."""
    return {
        "ts_ns": int(ev.ts_ns),
        "code": str(ev.code),
        "severity": ev.severity.value,
        "source": str(ev.source),
        "detail": str(ev.detail),
        "meta_json": _sorted_meta_json(ev.meta),
        "produced_by_engine": str(ev.produced_by_engine),
    }


def split_events_by_kind(
    events: Sequence[Event],
) -> tuple[
    tuple[SignalEvent, ...],
    tuple[ExecutionEvent, ...],
    tuple[SystemEvent, ...],
    tuple[HazardEvent, ...],
]:
    """Bucket ``events`` by EVT-01..04 kind, preserving input order."""
    signals: list[SignalEvent] = []
    executions: list[ExecutionEvent] = []
    systems: list[SystemEvent] = []
    hazards: list[HazardEvent] = []
    for ev in events:
        if ev.kind is EventKind.SIGNAL:
            assert isinstance(ev, SignalEvent)
            signals.append(ev)
        elif ev.kind is EventKind.EXECUTION:
            assert isinstance(ev, ExecutionEvent)
            executions.append(ev)
        elif ev.kind is EventKind.SYSTEM:
            assert isinstance(ev, SystemEvent)
            systems.append(ev)
        elif ev.kind is EventKind.HAZARD:
            assert isinstance(ev, HazardEvent)
            hazards.append(ev)
        else:
            raise ValueError(f"snapshots.split_events_by_kind: unknown EventKind {ev.kind!r}")
    return tuple(signals), tuple(executions), tuple(systems), tuple(hazards)


# ---------------------------------------------------------------------------
# Transport seam
# ---------------------------------------------------------------------------


class SnapshotTransport(Protocol):
    """Pluggable parquet I/O surface (caller-supplied, lazy)."""

    def write_tables(
        self,
        *,
        path: Path,
        signals_rows: Sequence[Mapping[str, object]],
        executions_rows: Sequence[Mapping[str, object]],
        systems_rows: Sequence[Mapping[str, object]],
        hazards_rows: Sequence[Mapping[str, object]],
    ) -> None: ...

    def read_tables(
        self,
        *,
        path: Path,
    ) -> tuple[
        Sequence[Mapping[str, object]],
        Sequence[Mapping[str, object]],
        Sequence[Mapping[str, object]],
        Sequence[Mapping[str, object]],
    ]: ...


# ---------------------------------------------------------------------------
# Writer / reader
# ---------------------------------------------------------------------------


class LedgerSnapshotWriter:
    """OFFLINE_ONLY parquet snapshot writer over a :class:`SnapshotTransport`.

    Construction is decoupled from pyarrow so unit tests can drop in an
    in-memory transport stub. The canonical pyarrow-backed transport is
    built by :func:`pyarrow_snapshot_writer_factory` which lazy-imports
    pyarrow only inside the factory body.
    """

    __slots__ = ("_transport",)

    def __init__(self, *, transport: SnapshotTransport) -> None:
        self._transport = transport

    def write(
        self,
        *,
        path: Path,
        events: Sequence[Event],
    ) -> SnapshotSummary:
        if len(events) > MAX_BATCH_ROWS:
            raise ValueError(
                "LedgerSnapshotWriter.write: batch exceeds "
                f"MAX_BATCH_ROWS={MAX_BATCH_ROWS} (got {len(events)})"
            )
        signals, executions, systems, hazards = split_events_by_kind(events)
        signals_rows = tuple(signal_event_to_row(ev) for ev in signals)
        executions_rows = tuple(execution_event_to_row(ev) for ev in executions)
        systems_rows = tuple(system_event_to_row(ev) for ev in systems)
        hazards_rows = tuple(hazard_event_to_row(ev) for ev in hazards)
        self._transport.write_tables(
            path=path,
            signals_rows=signals_rows,
            executions_rows=executions_rows,
            systems_rows=systems_rows,
            hazards_rows=hazards_rows,
        )
        return SnapshotSummary(
            path=path,
            signal_rows=len(signals_rows),
            execution_rows=len(executions_rows),
            system_rows=len(systems_rows),
            hazard_rows=len(hazards_rows),
        )


class LedgerSnapshotReader:
    """OFFLINE_ONLY parquet snapshot reader over a :class:`SnapshotTransport`.

    Re-materialises the four typed event lists from a snapshot path. The
    reader concatenates the four lists in fixed EVT-01..04 order so the
    output is deterministic; callers that need a globally-sorted view
    can apply a stable sort on ``ts_ns`` themselves.
    """

    __slots__ = ("_transport",)

    def __init__(self, *, transport: SnapshotTransport) -> None:
        self._transport = transport

    def read(self, *, path: Path) -> SnapshotReadResult:
        (
            signals_rows,
            executions_rows,
            systems_rows,
            hazards_rows,
        ) = self._transport.read_tables(path=path)
        return SnapshotReadResult(
            signal_rows=tuple(signals_rows),
            execution_rows=tuple(executions_rows),
            system_rows=tuple(systems_rows),
            hazard_rows=tuple(hazards_rows),
        )


# ---------------------------------------------------------------------------
# pyarrow transport (lazy)
# ---------------------------------------------------------------------------


def pyarrow_snapshot_writer_factory() -> LedgerSnapshotWriter:
    """Build a :class:`LedgerSnapshotWriter` over the pyarrow transport.

    Lazy-imports ``pyarrow`` and ``pyarrow.parquet`` only inside this
    function body. Raises :class:`RuntimeError` (never propagating the
    underlying :class:`ImportError`) when the package is not installed
    so callers can fail-soft to an alternative transport.
    """
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow_snapshot_writer_factory: pyarrow is not installed; "
            "install via `pip install pyarrow`"
        ) from exc
    transport = _PyarrowTransport(pa=pa, pq=pq)
    return LedgerSnapshotWriter(transport=transport)


def pyarrow_snapshot_reader_factory() -> LedgerSnapshotReader:
    """Build a :class:`LedgerSnapshotReader` over the pyarrow transport.

    Mirror of :func:`pyarrow_snapshot_writer_factory` for read side.
    """
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow_snapshot_reader_factory: pyarrow is not installed; "
            "install via `pip install pyarrow`"
        ) from exc
    transport = _PyarrowTransport(pa=pa, pq=pq)
    return LedgerSnapshotReader(transport=transport)


class _PyarrowTransport:
    """Internal pyarrow-backed :class:`SnapshotTransport` implementation.

    Constructed only by the two factories above; never imported at
    module top level. Each typed kind is persisted as its own parquet
    file alongside the user-supplied ``path`` (``<path>``,
    ``<path>.executions``, ``<path>.systems``, ``<path>.hazards``).
    Keeping each kind in its own file means readers can mmap only the
    columns they need.
    """

    __slots__ = ("_pa", "_pq")

    def __init__(self, *, pa: object, pq: object) -> None:
        self._pa = pa
        self._pq = pq

    def _suffix(self, path: Path, kind: str) -> Path:
        if kind == "signals":
            return path
        return path.with_name(f"{path.name}.{kind}")

    def _write_one(
        self,
        *,
        path: Path,
        rows: Sequence[Mapping[str, object]],
        columns: tuple[str, ...],
    ) -> None:
        pa = self._pa
        pq = self._pq
        if rows:
            arrays = {col: [r[col] for r in rows] for col in columns}
            table = pa.table(arrays)  # type: ignore[attr-defined]
        else:
            empty_arrays = {col: [] for col in columns}
            table = pa.table(empty_arrays)  # type: ignore[attr-defined]
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "dix.schema_version": SCHEMA_VERSION,
            "dix.parquet_writer_version": PARQUET_WRITER_VERSION,
        }
        table = table.replace_schema_metadata(meta)
        pq.write_table(  # type: ignore[attr-defined]
            table,
            str(path),
            compression="none",
            use_dictionary=False,
            write_statistics=False,
        )

    def write_tables(
        self,
        *,
        path: Path,
        signals_rows: Sequence[Mapping[str, object]],
        executions_rows: Sequence[Mapping[str, object]],
        systems_rows: Sequence[Mapping[str, object]],
        hazards_rows: Sequence[Mapping[str, object]],
    ) -> None:
        self._write_one(
            path=self._suffix(path, "signals"),
            rows=signals_rows,
            columns=SIGNAL_COLUMN_NAMES,
        )
        self._write_one(
            path=self._suffix(path, "executions"),
            rows=executions_rows,
            columns=EXECUTION_COLUMN_NAMES,
        )
        self._write_one(
            path=self._suffix(path, "systems"),
            rows=systems_rows,
            columns=SYSTEM_COLUMN_NAMES,
        )
        self._write_one(
            path=self._suffix(path, "hazards"),
            rows=hazards_rows,
            columns=HAZARD_COLUMN_NAMES,
        )

    def _read_one(
        self,
        *,
        path: Path,
        columns: tuple[str, ...],
    ) -> Sequence[Mapping[str, object]]:
        pq = self._pq
        if not path.exists():
            return ()
        table = pq.read_table(str(path))  # type: ignore[attr-defined]
        data = table.to_pydict()
        n = len(data[columns[0]]) if columns else 0
        rows: list[Mapping[str, object]] = []
        for i in range(n):
            row = {col: data[col][i] for col in columns}
            rows.append(MappingProxyType(row))
        return tuple(rows)

    def read_tables(
        self,
        *,
        path: Path,
    ) -> tuple[
        Sequence[Mapping[str, object]],
        Sequence[Mapping[str, object]],
        Sequence[Mapping[str, object]],
        Sequence[Mapping[str, object]],
    ]:
        return (
            self._read_one(
                path=self._suffix(path, "signals"),
                columns=SIGNAL_COLUMN_NAMES,
            ),
            self._read_one(
                path=self._suffix(path, "executions"),
                columns=EXECUTION_COLUMN_NAMES,
            ),
            self._read_one(
                path=self._suffix(path, "systems"),
                columns=SYSTEM_COLUMN_NAMES,
            ),
            self._read_one(
                path=self._suffix(path, "hazards"),
                columns=HAZARD_COLUMN_NAMES,
            ),
        )
