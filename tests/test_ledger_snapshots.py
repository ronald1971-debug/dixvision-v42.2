"""Tests for A-18.2 — parquet ledger snapshots.

Covers:
- Pure row projection round-trips for all 4 EVT-01..04 variants
- sorted-key JSON projection of meta / payload mappings (INV-15)
- :func:`split_events_by_kind` bucketing
- :class:`LedgerSnapshotWriter` / :class:`LedgerSnapshotReader` over a
  stub transport (reader returns :class:`SnapshotReadResult` raw rows;
  re-materialisation into typed events lives in this test module only,
  because the Triad Lock forbids ``state.ledger.snapshots`` from
  constructing ``SignalEvent`` / ``ExecutionEvent``)
- 3-run byte-identical replay equality with pyarrow transport
- ``MAX_BATCH_ROWS`` enforcement
- :func:`pyarrow_snapshot_writer_factory` lazy-import / RuntimeError path
- AST guards: no top-level pyarrow import, lazy pyarrow only inside
  factory bodies, no clock / random / engine cross-imports, no
  typed-event construction anywhere in the module (B21 / B22 / INV-56
  Triad Lock), ``# ADAPTED FROM:`` header
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import cast

import pytest

from core.contracts.events import (
    Event,
    EventKind,
    ExecutionEvent,
    ExecutionStatus,
    HazardEvent,
    HazardSeverity,
    Side,
    SignalEvent,
    SystemEvent,
    SystemEventKind,
)
from core.contracts.signal_trust import SignalTrust
from state.ledger import snapshots as mod
from state.ledger.snapshots import (
    EXECUTION_COLUMN_NAMES,
    HAZARD_COLUMN_NAMES,
    MAX_BATCH_ROWS,
    NEW_PIP_DEPENDENCIES,
    PARQUET_WRITER_VERSION,
    SCHEMA_VERSION,
    SIGNAL_COLUMN_NAMES,
    SYSTEM_COLUMN_NAMES,
    LedgerSnapshotReader,
    LedgerSnapshotWriter,
    SnapshotReadResult,
    SnapshotTransport,
    execution_event_to_row,
    hazard_event_to_row,
    parse_meta_json,
    pyarrow_snapshot_reader_factory,
    pyarrow_snapshot_writer_factory,
    signal_event_to_row,
    split_events_by_kind,
    system_event_to_row,
)

# ---------------------------------------------------------------------------
# Row -> typed event re-materialisers.
#
# These helpers live in the test module — never in ``state.ledger.snapshots`` —
# because the Triad Lock (INV-56) forbids any non-producer module from
# constructing ``SignalEvent`` / ``ExecutionEvent``. The ``tests/`` path is
# explicitly exempt from B21 / B22 (authority_lint TRIAD_CONSTRUCTOR_TEST_EXEMPT).
# ---------------------------------------------------------------------------


def _row_to_signal_event(row: Mapping[str, object]) -> SignalEvent:
    return SignalEvent(
        ts_ns=int(row["ts_ns"]),  # type: ignore[arg-type]
        symbol=str(row["symbol"]),
        side=Side(str(row["side"])),
        confidence=float(row["confidence"]),  # type: ignore[arg-type]
        plugin_chain=tuple(str(p) for p in row["plugin_chain"]),  # type: ignore[union-attr]
        meta=parse_meta_json(str(row["meta_json"])),
        produced_by_engine=str(row["produced_by_engine"]),
        signal_trust=SignalTrust(str(row["signal_trust"])),
        signal_source=str(row["signal_source"]),
    )


def _row_to_execution_event(row: Mapping[str, object]) -> ExecutionEvent:
    return ExecutionEvent(
        ts_ns=int(row["ts_ns"]),  # type: ignore[arg-type]
        symbol=str(row["symbol"]),
        side=Side(str(row["side"])),
        qty=float(row["qty"]),  # type: ignore[arg-type]
        price=float(row["price"]),  # type: ignore[arg-type]
        status=ExecutionStatus(str(row["status"])),
        venue=str(row["venue"]),
        order_id=str(row["order_id"]),
        meta=parse_meta_json(str(row["meta_json"])),
        produced_by_engine=str(row["produced_by_engine"]),
    )


def _row_to_system_event(row: Mapping[str, object]) -> SystemEvent:
    return SystemEvent(
        ts_ns=int(row["ts_ns"]),  # type: ignore[arg-type]
        sub_kind=SystemEventKind(str(row["sub_kind"])),
        source=str(row["source"]),
        payload=parse_meta_json(str(row["payload_json"])),
        meta=parse_meta_json(str(row["meta_json"])),
        produced_by_engine=str(row["produced_by_engine"]),
        proposed=bool(row["proposed"]),
    )


def _row_to_hazard_event(row: Mapping[str, object]) -> HazardEvent:
    return HazardEvent(
        ts_ns=int(row["ts_ns"]),  # type: ignore[arg-type]
        code=str(row["code"]),
        severity=HazardSeverity(str(row["severity"])),
        source=str(row["source"]),
        detail=str(row["detail"]),
        meta=parse_meta_json(str(row["meta_json"])),
        produced_by_engine=str(row["produced_by_engine"]),
    )


def _materialise_result(result: SnapshotReadResult) -> tuple[Event, ...]:
    signals = [_row_to_signal_event(r) for r in result.signal_rows]
    executions = [_row_to_execution_event(r) for r in result.execution_rows]
    systems = [_row_to_system_event(r) for r in result.system_rows]
    hazards = [_row_to_hazard_event(r) for r in result.hazard_rows]
    return tuple(signals + executions + systems + hazards)


# ---------------------------------------------------------------------------
# Sample events
# ---------------------------------------------------------------------------


def _signal(
    *,
    ts_ns: int = 1_000,
    symbol: str = "BTCUSDT",
    meta: Mapping[str, str] | None = None,
) -> SignalEvent:
    return SignalEvent(
        ts_ns=ts_ns,
        symbol=symbol,
        side=Side.BUY,
        confidence=0.75,
        plugin_chain=("p1", "p2"),
        meta=meta if meta is not None else {"k": "v"},
        produced_by_engine="intelligence_engine",
        signal_trust=SignalTrust.INTERNAL,
        signal_source="microstructure_v1",
    )


def _execution(*, ts_ns: int = 2_000) -> ExecutionEvent:
    return ExecutionEvent(
        ts_ns=ts_ns,
        symbol="BTCUSDT",
        side=Side.SELL,
        qty=0.5,
        price=42_000.0,
        status=ExecutionStatus.FILLED,
        venue="binance",
        order_id="ORD-1",
        meta={"latency_us": "150"},
        produced_by_engine="execution_engine",
    )


def _system(*, ts_ns: int = 3_000) -> SystemEvent:
    return SystemEvent(
        ts_ns=ts_ns,
        sub_kind=SystemEventKind.UPDATE_PROPOSED,
        source="learning",
        payload={"version": "v42"},
        meta={"trace": "abc"},
        produced_by_engine="learning_engine",
        proposed=True,
    )


def _hazard(*, ts_ns: int = 4_000) -> HazardEvent:
    return HazardEvent(
        ts_ns=ts_ns,
        code="HAZ-01",
        severity=HazardSeverity.HIGH,
        source="system",
        detail="data-staleness",
        meta={"cls": "A"},
        produced_by_engine="system_engine",
    )


# ---------------------------------------------------------------------------
# Public sentinels
# ---------------------------------------------------------------------------


class TestPublicSentinels:
    def test_pip_dependencies(self) -> None:
        assert NEW_PIP_DEPENDENCIES == ("pyarrow",)

    def test_schema_version_pinned(self) -> None:
        assert SCHEMA_VERSION == "dix-events-v1"

    def test_parquet_writer_version_pinned(self) -> None:
        assert PARQUET_WRITER_VERSION == "2.6"

    def test_max_batch_rows(self) -> None:
        assert MAX_BATCH_ROWS == 1_048_576

    def test_signal_column_names_match_dataclass(self) -> None:
        # Mirrors SignalEvent fields except ``kind`` (the discriminator
        # is reconstructed from the column file separation, not stored).
        assert SIGNAL_COLUMN_NAMES == (
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

    def test_execution_column_names(self) -> None:
        assert EXECUTION_COLUMN_NAMES == (
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

    def test_system_column_names(self) -> None:
        assert SYSTEM_COLUMN_NAMES == (
            "ts_ns",
            "sub_kind",
            "source",
            "payload_json",
            "meta_json",
            "produced_by_engine",
            "proposed",
        )

    def test_hazard_column_names(self) -> None:
        assert HAZARD_COLUMN_NAMES == (
            "ts_ns",
            "code",
            "severity",
            "source",
            "detail",
            "meta_json",
            "produced_by_engine",
        )


# ---------------------------------------------------------------------------
# Pure projections
# ---------------------------------------------------------------------------


class TestPureProjections:
    def test_signal_event_round_trip(self) -> None:
        ev = _signal()
        row = signal_event_to_row(ev)
        assert row["ts_ns"] == 1_000
        assert row["side"] == "BUY"
        assert row["signal_trust"] == "INTERNAL"
        recovered = _row_to_signal_event(row)
        assert recovered == ev

    def test_execution_event_round_trip(self) -> None:
        ev = _execution()
        row = execution_event_to_row(ev)
        assert row["side"] == "SELL"
        assert row["status"] == "FILLED"
        recovered = _row_to_execution_event(row)
        assert recovered == ev

    def test_system_event_round_trip(self) -> None:
        ev = _system()
        row = system_event_to_row(ev)
        assert row["sub_kind"] == "UPDATE_PROPOSED"
        assert row["proposed"] is True
        recovered = _row_to_system_event(row)
        assert recovered == ev

    def test_hazard_event_round_trip(self) -> None:
        ev = _hazard()
        row = hazard_event_to_row(ev)
        assert row["code"] == "HAZ-01"
        assert row["severity"] == "HIGH"
        recovered = _row_to_hazard_event(row)
        assert recovered == ev

    def test_meta_json_is_sorted_key(self) -> None:
        # Caller insertion-order: a, c, b — on-disk: a, b, c
        ev = _signal(meta={"a": "1", "c": "3", "b": "2"})
        row = signal_event_to_row(ev)
        assert row["meta_json"] == '{"a":"1","b":"2","c":"3"}'

    def test_meta_json_empty_collapses(self) -> None:
        ev = _signal(meta={})
        row = signal_event_to_row(ev)
        assert row["meta_json"] == "{}"

    def test_meta_round_trip_yields_frozen_view(self) -> None:
        ev = _signal(meta={"a": "1"})
        recovered = _row_to_signal_event(signal_event_to_row(ev))
        assert isinstance(recovered.meta, MappingProxyType)
        with pytest.raises(TypeError):
            recovered.meta["x"] = "y"  # type: ignore[index]

    def test_meta_round_trip_insertion_order_invariant(self) -> None:
        ev_a = _signal(meta={"a": "1", "b": "2"})
        ev_b = _signal(meta={"b": "2", "a": "1"})
        # The two inputs are equal because dict equality is unordered…
        assert ev_a == ev_b
        # …but more importantly the serialised row bytes are byte-identical.
        assert signal_event_to_row(ev_a)["meta_json"] == signal_event_to_row(ev_b)["meta_json"]


# ---------------------------------------------------------------------------
# split_events_by_kind
# ---------------------------------------------------------------------------


class TestSplitEventsByKind:
    def test_buckets_preserve_order(self) -> None:
        a = _signal(ts_ns=1)
        b = _execution(ts_ns=2)
        c = _signal(ts_ns=3)
        d = _system(ts_ns=4)
        e = _hazard(ts_ns=5)
        signals, executions, systems, hazards = split_events_by_kind((a, b, c, d, e))
        assert signals == (a, c)
        assert executions == (b,)
        assert systems == (d,)
        assert hazards == (e,)

    def test_empty_sequence(self) -> None:
        signals, executions, systems, hazards = split_events_by_kind(())
        assert signals == executions == systems == hazards == ()


# ---------------------------------------------------------------------------
# Stub transport — exercises writer/reader without pyarrow
# ---------------------------------------------------------------------------


class _InMemoryTransport:
    """SnapshotTransport stub that buffers rows in process memory."""

    def __init__(self) -> None:
        self._store: dict[Path, dict[str, Sequence[Mapping[str, object]]]] = {}

    def write_tables(
        self,
        *,
        path: Path,
        signals_rows: Sequence[Mapping[str, object]],
        executions_rows: Sequence[Mapping[str, object]],
        systems_rows: Sequence[Mapping[str, object]],
        hazards_rows: Sequence[Mapping[str, object]],
    ) -> None:
        self._store[path] = {
            "signals": signals_rows,
            "executions": executions_rows,
            "systems": systems_rows,
            "hazards": hazards_rows,
        }

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
        bucket = self._store[path]
        return (
            bucket["signals"],
            bucket["executions"],
            bucket["systems"],
            bucket["hazards"],
        )


class TestWriterReaderStub:
    def test_writer_reader_round_trip(self) -> None:
        transport: SnapshotTransport = _InMemoryTransport()
        writer = LedgerSnapshotWriter(transport=transport)
        reader = LedgerSnapshotReader(transport=transport)
        events: tuple[Event, ...] = (
            _signal(),
            _execution(),
            _system(),
            _hazard(),
            _signal(ts_ns=10_000),
        )
        path = Path("/snap/test")
        summary = writer.write(path=path, events=events)
        assert summary.signal_rows == 2
        assert summary.execution_rows == 1
        assert summary.system_rows == 1
        assert summary.hazard_rows == 1
        assert summary.total_rows == 5
        assert summary.path == path
        result = reader.read(path=path)
        assert isinstance(result, SnapshotReadResult)
        assert result.total_rows == 5
        recovered = _materialise_result(result)
        # Reader orders by EVT-01..04 kind, then preserves input order
        # within kind.
        assert recovered == (
            events[0],
            events[4],
            events[1],
            events[2],
            events[3],
        )

    def test_empty_batch_writes_zero_rows(self) -> None:
        transport: SnapshotTransport = _InMemoryTransport()
        writer = LedgerSnapshotWriter(transport=transport)
        summary = writer.write(path=Path("/snap/empty"), events=())
        assert summary.total_rows == 0

    def test_max_batch_rows_rejected(self) -> None:
        transport: SnapshotTransport = _InMemoryTransport()
        writer = LedgerSnapshotWriter(transport=transport)
        # Build a length-list without actually allocating MAX_BATCH_ROWS+1
        # SignalEvent objects (which would blow memory at 1M+ records).
        events = cast("Sequence[Event]", [_signal()] * (MAX_BATCH_ROWS + 1))
        with pytest.raises(ValueError, match="MAX_BATCH_ROWS"):
            writer.write(path=Path("/snap/big"), events=events)


# ---------------------------------------------------------------------------
# Pyarrow transport — end-to-end 3-run determinism
# ---------------------------------------------------------------------------


pyarrow_required = pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is None,
    reason="pyarrow not installed",
)


@pyarrow_required
class TestPyarrowTransport:
    def _events(self) -> tuple[Event, ...]:
        return (
            _signal(ts_ns=1_000),
            _execution(ts_ns=2_000),
            _system(ts_ns=3_000),
            _hazard(ts_ns=4_000),
            _signal(ts_ns=5_000, meta={"x": "1", "a": "2"}),
        )

    def test_pyarrow_write_then_read_round_trip(self) -> None:
        writer = pyarrow_snapshot_writer_factory()
        reader = pyarrow_snapshot_reader_factory()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "snap.parquet"
            writer.write(path=base, events=self._events())
            result = reader.read(path=base)
        assert isinstance(result, SnapshotReadResult)
        recovered = _materialise_result(result)
        # Two SignalEvents come first, then ExecutionEvent, SystemEvent,
        # HazardEvent (split_events_by_kind preserves input order within
        # kind).
        assert len(recovered) == 5
        assert recovered[0] == self._events()[0]
        assert recovered[1] == self._events()[4]
        assert recovered[2] == self._events()[1]
        assert recovered[3] == self._events()[2]
        assert recovered[4] == self._events()[3]

    def test_pyarrow_3run_byte_identical(self) -> None:
        """INV-15 — 3 independent writes of identical inputs must produce
        byte-identical parquet bytes on every supported suffix file."""
        writer = pyarrow_snapshot_writer_factory()
        events = self._events()
        with tempfile.TemporaryDirectory() as tmp:
            base1 = Path(tmp) / "run1.parquet"
            base2 = Path(tmp) / "run2.parquet"
            base3 = Path(tmp) / "run3.parquet"
            writer.write(path=base1, events=events)
            writer.write(path=base2, events=events)
            writer.write(path=base3, events=events)
            for suffix in ("", ".executions", ".systems", ".hazards"):
                f1 = base1.with_name(f"{base1.name}{suffix}")
                f2 = base2.with_name(f"{base2.name}{suffix}")
                f3 = base3.with_name(f"{base3.name}{suffix}")
                b1 = f1.read_bytes()
                b2 = f2.read_bytes()
                b3 = f3.read_bytes()
                assert b1 == b2 == b3, f"INV-15 violation on suffix={suffix!r}: bytes diverged"

    def test_pyarrow_meta_insertion_order_invariant(self) -> None:
        """Two events with the same meta but different Python dict
        insertion order must serialise to byte-identical parquet."""
        writer = pyarrow_snapshot_writer_factory()
        ev_a = _signal(meta={"a": "1", "b": "2", "c": "3"})
        ev_b = _signal(meta={"c": "3", "a": "1", "b": "2"})
        with tempfile.TemporaryDirectory() as tmp:
            ba = Path(tmp) / "a.parquet"
            bb = Path(tmp) / "b.parquet"
            writer.write(path=ba, events=(ev_a,))
            writer.write(path=bb, events=(ev_b,))
            assert ba.read_bytes() == bb.read_bytes()

    def test_pyarrow_reader_on_missing_file_returns_empty(self) -> None:
        reader = pyarrow_snapshot_reader_factory()
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.parquet"
            result = reader.read(path=missing)
        assert isinstance(result, SnapshotReadResult)
        assert result.total_rows == 0
        assert result.signal_rows == ()
        assert result.execution_rows == ()
        assert result.system_rows == ()
        assert result.hazard_rows == ()


class TestPyarrowFactoryImportError:
    """Lazy-import failure path — pyarrow missing must raise RuntimeError."""

    def test_writer_factory_raises_runtime_error_when_pyarrow_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_pa = sys.modules.pop("pyarrow", None)
        original_pq = sys.modules.pop("pyarrow.parquet", None)
        monkeypatch.setitem(sys.modules, "pyarrow", None)
        try:
            with pytest.raises(RuntimeError, match="pyarrow is not installed"):
                pyarrow_snapshot_writer_factory()
        finally:
            if original_pa is not None:
                sys.modules["pyarrow"] = original_pa
            if original_pq is not None:
                sys.modules["pyarrow.parquet"] = original_pq

    def test_reader_factory_raises_runtime_error_when_pyarrow_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_pa = sys.modules.pop("pyarrow", None)
        original_pq = sys.modules.pop("pyarrow.parquet", None)
        monkeypatch.setitem(sys.modules, "pyarrow", None)
        try:
            with pytest.raises(RuntimeError, match="pyarrow is not installed"):
                pyarrow_snapshot_reader_factory()
        finally:
            if original_pa is not None:
                sys.modules["pyarrow"] = original_pa
            if original_pq is not None:
                sys.modules["pyarrow.parquet"] = original_pq


# ---------------------------------------------------------------------------
# AST guards (architectural invariants)
# ---------------------------------------------------------------------------


_MODULE_SRC = Path(mod.__file__).read_text(encoding="utf-8")
_MODULE_AST = ast.parse(_MODULE_SRC)


def _top_level_imports(tree: ast.Module) -> list[str]:
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imports.append(node.module)
    return imports


def _inside_function_imports(tree: ast.Module) -> list[tuple[str, str]]:
    """Return ``(function_name, module_name)`` pairs for imports inside
    function bodies (excludes ``TYPE_CHECKING`` branch)."""
    pairs: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Import):
                    for a in inner.names:
                        pairs.append((node.name, a.name))
                elif isinstance(inner, ast.ImportFrom):
                    if inner.module is not None:
                        pairs.append((node.name, inner.module))
    return pairs


class TestASTGuards:
    def test_no_top_level_pyarrow_import(self) -> None:
        for name in _top_level_imports(_MODULE_AST):
            assert not name.startswith("pyarrow"), (
                f"snapshots.py must not import {name!r} at top level "
                "(lazy-only inside factory bodies)"
            )

    def test_pyarrow_imported_only_inside_factory_bodies(self) -> None:
        pyarrow_imports = [
            (fn, mod_name)
            for (fn, mod_name) in _inside_function_imports(_MODULE_AST)
            if mod_name.startswith("pyarrow")
        ]
        # Both pyarrow imports (pa + pq) must live inside the two
        # factory functions only.
        allowed_factories = {
            "pyarrow_snapshot_writer_factory",
            "pyarrow_snapshot_reader_factory",
        }
        for fn, mod_name in pyarrow_imports:
            assert fn in allowed_factories, (
                f"pyarrow import {mod_name!r} found inside function {fn!r}; "
                "only the two factory functions may import pyarrow"
            )

    def test_no_clock_imports(self) -> None:
        banned = {
            "time",
            "datetime",
            "random",
            "uuid",
            "asyncio",
            "os",
            "secrets",
        }
        for name in _top_level_imports(_MODULE_AST):
            root = name.split(".")[0]
            assert root not in banned, (
                f"snapshots.py must not import banned module {name!r} (INV-15)"
            )

    def test_no_engine_cross_imports(self) -> None:
        banned_engines = {
            "execution_engine",
            "governance_engine",
            "system_engine",
            "intelligence_engine",
            "evolution_engine",
        }
        for name in _top_level_imports(_MODULE_AST):
            root = name.split(".")[0]
            assert root not in banned_engines, (
                f"snapshots.py must not import {name!r} (B1 isolation)"
            )

    def test_adapted_from_header_present(self) -> None:
        assert "# ADAPTED FROM: apache/arrow" in _MODULE_SRC, (
            "snapshots.py must carry the # ADAPTED FROM: apache/arrow header"
        )

    def test_no_typed_event_construction_in_module(self) -> None:
        """Triad Lock (INV-56 / B21 / B22) — ``state.ledger.snapshots``
        is not a producing engine, so it must **never** construct
        ``SignalEvent`` / ``ExecutionEvent`` / ``SystemEvent`` /
        ``HazardEvent`` anywhere. Re-materialisation lives in callers
        (e.g. the test module's ``_row_to_*_event`` helpers).

        Also pins B27 / B28 / INV-71 authority symmetry: no
        governance-side typed event (``PatchProposal`` etc) may be
        constructed either.
        """
        banned_construction = {
            "SignalEvent",
            "ExecutionEvent",
            "SystemEvent",
            "HazardEvent",
            "PatchProposal",
            "GovernanceDecision",
            "OperatorDirective",
            "ExecutionIntent",
        }
        for node in ast.walk(_MODULE_AST):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                name = node.func.id
                assert name not in banned_construction, (
                    f"snapshots.py must not construct {name!r} (Triad Lock / authority symmetry)"
                )


# ---------------------------------------------------------------------------
# Authority symmetry — re-asserted via the inverse direction
# ---------------------------------------------------------------------------


class TestEventDiscriminatorMirrorsContract:
    """Every EventKind enum member must be handled by
    :func:`split_events_by_kind`; otherwise the module silently drops new
    contract additions."""

    def test_all_event_kinds_buckets(self) -> None:
        kinds_in_split_source = {
            "SIGNAL",
            "EXECUTION",
            "SYSTEM",
            "HAZARD",
        }
        assert {k.name for k in EventKind} == kinds_in_split_source
