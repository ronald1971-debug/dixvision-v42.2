# ADAPTED FROM: duckdb/duckdb tools/pythonpkg/duckdb/__init__.py
#   - duckdb.connect(database, read_only) — in-process analytical
#     connection factory
#   - DuckDBPyConnection.execute(sql, parameters) — parameterised
#     prepared-statement surface
#   - DuckDBPyConnection.fetchall() / fetchone() — eager row
#     materialisation
#   - read_parquet(path) / read_csv(path) — SQL table-function file
#     sources
# MIT license; no DuckDB source is reproduced verbatim — only the
# public Python-API call signatures and return shapes (cursor-style
# fetch over a typed connection, parquet-table SQL function) are
# mirrored.
"""A-14 duckdb → offline ledger analytics service.

This module is the **OFFLINE_ONLY** analytical SQL surface that
``learning_engine/`` uses to slice the durable ledger that
``state/ledger/`` writes (PR #164) and the parquet snapshots that
``state/ledger/snapshots.py`` will export (A-18). It is the canonical
adaptation of the DuckDB Python API into DIX — a coordinator that
opens an in-process DuckDB connection, runs analytical SQL against a
parquet/SQLite source, and returns frozen :class:`QueryResult` value
objects suitable for replay verification.

Tier
----
**OFFLINE_ONLY.** ``learning_engine/analytics/`` is the high-throughput
slow-cadence analytics tier. DuckDB must never be imported from any
runtime tier (``execution_engine/``, ``governance_engine/``,
``system_engine/``, ``core/``, or
``intelligence_engine/meta_controller/hot_path.py``). The S-10
``B-POLARS`` lint precedent (PR #289) establishes the runtime-tier
ban pattern for batch-analytical libraries; this module follows the
same convention.

Design constraints
------------------
* **Lazy import.** ``import duckdb`` lives **inside**
  :func:`duckdb_backend_factory` so this module imports cleanly in
  environments without ``duckdb`` installed (mirrors the S-01 ccxt /
  S-05 firecrawl / S-10 polars precedent). Pinned by AST test.
* **Read-only contract.** :class:`LedgerAnalytics` exposes only
  ``count`` / ``aggregate`` / ``percentile`` / ``group_by`` /
  ``fetch_rows`` methods — no ``execute_write`` / ``insert`` /
  ``update`` surface. The DuckDB connection is opened in-memory by
  default; when a file path is supplied it is opened ``read_only=True``
  so the analytics tier physically cannot mutate the ledger or its
  parquet exports.
* **Protocol seam.** :class:`AnalyticsBackend` is a ``Protocol`` with
  one method (``execute``) that returns a tuple of rows; the
  in-process pure-Python backend :class:`InProcessAnalyticsBackend`
  walks a tuple of ``Mapping[str, object]`` rows in memory so the
  test suite can pin invariants without ``duckdb`` installed. The
  DuckDB-backed implementation is delivered by
  :func:`duckdb_backend_factory`.
* **Frozen contracts.** :class:`QueryRequest`, :class:`QueryResult`,
  :class:`AggregateSpec`, and :class:`GroupBySpec` are
  ``@dataclass(frozen=True, slots=True)`` with eager validation in
  ``__post_init__``.
* **INV-15 byte-identical.** Inputs to every query are sorted before
  the backend is called; outputs are sorted by ``group_keys`` before
  being projected into :class:`QueryResult`. Pinned by 3-run equality
  test against the in-process backend. Backends contractually return
  rows in a deterministic order — DuckDB callers must wrap their SQL
  with ``ORDER BY``.
* **No new pip deps at module-import time.**
  :data:`NEW_PIP_DEPENDENCIES` declares ``("duckdb",)`` so the
  pip-dep audit picks it up, but the module body never imports
  duckdb at toplevel.

Authority symmetry (B27 / B28 / INV-71)
---------------------------------------
This module **never** constructs typed bus events
(:class:`PatchProposal`, :class:`SignalEvent`,
:class:`GovernanceDecision`). It returns advisory analytical reports
only; higher-level coordinators on the evolution-engine side are
responsible for projecting analytics into typed proposals. Pinned by
AST test.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Final, Literal, Protocol, runtime_checkable

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("duckdb",)
"""A-14 introduces a single new pip dep: ``duckdb``.

DuckDB is **lazy-imported inside** :func:`duckdb_backend_factory` so
this module imports cleanly without it. ``tools/authority_lint.py``
should ban ``import duckdb`` from RUNTIME tiers in a follow-up
sub-PR (mirroring S-10.4's ``B-POLARS`` rule).
"""

# ----------------------------------------------------------------------
# Validation primitives
# ----------------------------------------------------------------------

_MAX_LIMIT: Final[int] = 1_000_000
_MAX_GROUP_KEYS: Final[int] = 16
_MAX_AGGS: Final[int] = 32
_MAX_COLUMNS: Final[int] = 256
_MAX_NAME_LEN: Final[int] = 128

_VALID_AGG_OPS: Final[frozenset[str]] = frozenset({"count", "sum", "avg", "min", "max"})
_VALID_PERCENTILES: Final[frozenset[float]] = frozenset({0.5, 0.9, 0.95, 0.99})


def _check_name(name: str, *, kind: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"{kind} must be str, got {type(name).__name__!r}")
    if not name:
        raise ValueError(f"{kind} must not be empty")
    if len(name) > _MAX_NAME_LEN:
        raise ValueError(f"{kind}={name!r} exceeds max length {_MAX_NAME_LEN}")
    if not name.replace("_", "").isalnum():
        raise ValueError(f"{kind}={name!r} must be alphanumeric (underscores allowed)")
    return name


# ----------------------------------------------------------------------
# Backend protocol + in-process implementation
# ----------------------------------------------------------------------


@runtime_checkable
class AnalyticsBackend(Protocol):
    """Pluggable analytics backend.

    The in-process backend walks an in-memory tuple of rows; the
    DuckDB-backed implementation (delivered by
    :func:`duckdb_backend_factory`) issues SQL against a DuckDB
    connection. The contract is intentionally narrow — a single
    ``execute`` method that returns rows as
    :class:`Mapping`[str, object] tuples.
    """

    def execute(
        self,
        request: QueryRequest,
    ) -> tuple[Mapping[str, object], ...]:
        """Run *request* and return its rows in deterministic order."""


@dataclasses.dataclass(frozen=True, slots=True)
class InProcessAnalyticsBackend:
    """Pure-Python analytics backend over an in-memory row tuple.

    Used by the test suite (no DuckDB required) and as a reference
    implementation that pins the contract for the DuckDB-backed
    impl. Rows are stored as a sorted tuple of mappings; queries
    walk the tuple once per call (O(n × log n) for sorted output).
    """

    rows: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.rows, tuple):
            raise TypeError("rows must be a tuple")
        for index, row in enumerate(self.rows):
            if not isinstance(row, Mapping):
                raise TypeError(f"rows[{index}] must be Mapping, got {type(row).__name__!r}")

    def execute(
        self,
        request: QueryRequest,
    ) -> tuple[Mapping[str, object], ...]:
        request._validate_against_columns(self._columns())

        # 1. WHERE-style filter.
        filtered: list[Mapping[str, object]] = []
        for row in self.rows:
            if _row_matches(row, request.filters):
                filtered.append(dict(row))

        # 2. ORDER BY for INV-15 byte-identical replay.
        filtered.sort(key=lambda r: tuple(_sort_key(r.get(col)) for col in request._sort_columns))

        # 3. LIMIT.
        if request.limit is not None:
            filtered = filtered[: request.limit]

        return tuple(filtered)

    def _columns(self) -> frozenset[str]:
        seen: set[str] = set()
        for row in self.rows:
            seen.update(row.keys())
        return frozenset(seen)


def _row_matches(
    row: Mapping[str, object],
    filters: Mapping[str, object],
) -> bool:
    for key, expected in filters.items():
        actual = row.get(key)
        if actual != expected:
            return False
    return True


def _sort_key(value: object) -> tuple[int, object]:
    """Stable cross-type sort key.

    Tuple of ``(type_rank, value)`` so heterogeneous columns sort in a
    deterministic order across runs. The type rank is itself
    deterministic.
    """
    if value is None:
        return (0, 0)
    if isinstance(value, bool):
        return (1, int(value))
    if isinstance(value, int):
        return (2, value)
    if isinstance(value, float):
        return (3, value)
    if isinstance(value, str):
        return (4, value)
    if isinstance(value, (bytes, bytearray)):
        return (5, bytes(value))
    return (6, repr(value))


# ----------------------------------------------------------------------
# Request / result value objects
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class QueryRequest:
    """One analytical query — projection + filter + limit.

    Backends consume this directly; the in-process backend interprets
    it in Python and the DuckDB backend compiles it into SQL with
    parametrised arguments to avoid injection.
    """

    table: str
    columns: tuple[str, ...]
    filters: Mapping[str, object] = dataclasses.field(default_factory=dict)
    order_by: tuple[str, ...] = ()
    limit: int | None = None

    def __post_init__(self) -> None:
        _check_name(self.table, kind="table")

        if not isinstance(self.columns, tuple):
            raise TypeError("columns must be a tuple")
        if not self.columns:
            raise ValueError("columns must not be empty")
        if len(self.columns) > _MAX_COLUMNS:
            raise ValueError(f"columns length {len(self.columns)} > max {_MAX_COLUMNS}")
        for col in self.columns:
            _check_name(col, kind="column")

        if not isinstance(self.filters, Mapping):
            raise TypeError("filters must be a Mapping")
        for key in self.filters:
            _check_name(key, kind="filter key")

        if not isinstance(self.order_by, tuple):
            raise TypeError("order_by must be a tuple")
        for col in self.order_by:
            _check_name(col, kind="order_by")

        if self.limit is not None:
            if not isinstance(self.limit, int) or isinstance(self.limit, bool):
                raise TypeError("limit must be int or None")
            if self.limit < 0:
                raise ValueError("limit must be >= 0")
            if self.limit > _MAX_LIMIT:
                raise ValueError(f"limit > {_MAX_LIMIT}")

    @property
    def _sort_columns(self) -> tuple[str, ...]:
        """Deterministic sort key — uses ``order_by`` else all columns."""
        return self.order_by if self.order_by else self.columns

    def _validate_against_columns(
        self,
        available: frozenset[str],
    ) -> None:
        unknown = set(self.columns) - available
        if unknown and available:
            raise ValueError(f"query references unknown columns: {sorted(unknown)}")
        filter_unknown = set(self.filters.keys()) - available
        if filter_unknown and available:
            raise ValueError(f"query filters reference unknown columns: {sorted(filter_unknown)}")


@dataclasses.dataclass(frozen=True, slots=True)
class QueryResult:
    """Frozen result of one analytical query.

    Columns are recorded explicitly; rows are a tuple-of-tuples (one
    per matching record) ordered as declared by the request. A
    BLAKE2b-16 ``result_digest`` is computed over the canonical
    sorted-key JSON projection so two runs of the same query
    produce byte-identical digests for INV-15 replay.
    """

    request_table: str
    request_columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]
    result_digest: str

    def __post_init__(self) -> None:
        _check_name(self.request_table, kind="request_table")
        if not isinstance(self.request_columns, tuple):
            raise TypeError("request_columns must be a tuple")
        if not isinstance(self.rows, tuple):
            raise TypeError("rows must be a tuple")
        for index, row in enumerate(self.rows):
            if not isinstance(row, tuple):
                raise TypeError(f"rows[{index}] must be tuple")
            if len(row) != len(self.request_columns):
                raise ValueError(
                    f"rows[{index}] has {len(row)} fields, expected {len(self.request_columns)}"
                )
        if not isinstance(self.result_digest, str) or len(self.result_digest) != 32:
            raise ValueError("result_digest must be a 32-char hex string")

    def row_count(self) -> int:
        return len(self.rows)


# ----------------------------------------------------------------------
# Aggregate / group-by specs
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class AggregateSpec:
    """One aggregation column — ``op(column) AS alias``.

    Five supported operations: ``count`` / ``sum`` / ``avg`` /
    ``min`` / ``max`` (deterministic, no clock, no random).
    """

    op: Literal["count", "sum", "avg", "min", "max"]
    column: str
    alias: str

    def __post_init__(self) -> None:
        if self.op not in _VALID_AGG_OPS:
            raise ValueError(f"op={self.op!r} not in {sorted(_VALID_AGG_OPS)}")
        _check_name(self.column, kind="aggregate column")
        _check_name(self.alias, kind="aggregate alias")


@dataclasses.dataclass(frozen=True, slots=True)
class GroupBySpec:
    """One group-by query — ``SELECT group_keys, aggs FROM table``."""

    table: str
    group_keys: tuple[str, ...]
    aggregates: tuple[AggregateSpec, ...]
    filters: Mapping[str, object] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_name(self.table, kind="table")
        if not isinstance(self.group_keys, tuple):
            raise TypeError("group_keys must be a tuple")
        if len(self.group_keys) > _MAX_GROUP_KEYS:
            raise ValueError(f"group_keys length {len(self.group_keys)} > max {_MAX_GROUP_KEYS}")
        for key in self.group_keys:
            _check_name(key, kind="group_keys")
        if not isinstance(self.aggregates, tuple):
            raise TypeError("aggregates must be a tuple")
        if not self.aggregates:
            raise ValueError("aggregates must not be empty")
        if len(self.aggregates) > _MAX_AGGS:
            raise ValueError(f"aggregates length {len(self.aggregates)} > max {_MAX_AGGS}")
        if not isinstance(self.filters, Mapping):
            raise TypeError("filters must be a Mapping")
        for key in self.filters:
            _check_name(key, kind="filter key")


# ----------------------------------------------------------------------
# Coordinator
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class LedgerAnalytics:
    """OFFLINE_ONLY ledger analytics coordinator.

    Wraps an :class:`AnalyticsBackend` and exposes a high-level
    surface — ``count`` / ``fetch_rows`` / ``group_by`` /
    ``percentile`` — all returning frozen :class:`QueryResult`
    advisory records. Never mutates the ledger or any of its
    parquet/SQLite exports.
    """

    backend: AnalyticsBackend

    def __post_init__(self) -> None:
        if not isinstance(self.backend, AnalyticsBackend):
            raise TypeError("backend must implement AnalyticsBackend Protocol")

    # ------------------------------------------------------------------
    # Primitive query — projection + filter + limit
    # ------------------------------------------------------------------

    def fetch_rows(self, request: QueryRequest) -> QueryResult:
        rows = self.backend.execute(request)
        materialised = tuple(tuple(row.get(col) for col in request.columns) for row in rows)
        return QueryResult(
            request_table=request.table,
            request_columns=request.columns,
            rows=materialised,
            result_digest=_digest_rows(
                table=request.table,
                columns=request.columns,
                rows=materialised,
            ),
        )

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    def count(
        self,
        table: str,
        *,
        filters: Mapping[str, object] | None = None,
    ) -> int:
        """Return number of rows matching ``filters`` in ``table``."""
        return self._count_via_backend(table=table, filters=filters or {})

    def _count_via_backend(
        self,
        *,
        table: str,
        filters: Mapping[str, object],
    ) -> int:
        # Pick one filter column or any cached column to keep the
        # request well-formed.
        column = next(iter(filters), "id")
        request = QueryRequest(
            table=table,
            columns=(column,),
            filters=dict(filters),
        )
        rows = self.backend.execute(request)
        return len(rows)

    def percentile(
        self,
        request: QueryRequest,
        *,
        column: str,
        percentile: float,
    ) -> float | None:
        """Return ``column`` value at the given percentile.

        Pure deterministic — exact nearest-rank ('lower') percentile.
        """
        _check_name(column, kind="percentile column")
        if percentile not in _VALID_PERCENTILES:
            raise ValueError(f"percentile={percentile} not in {sorted(_VALID_PERCENTILES)}")
        rows = self.backend.execute(request)
        values = sorted(
            row[column]
            for row in rows
            if isinstance(row.get(column), (int, float)) and not isinstance(row.get(column), bool)
        )
        if not values:
            return None
        rank = int(percentile * (len(values) - 1))
        return float(values[rank])

    def group_by(self, spec: GroupBySpec) -> QueryResult:
        """Run a group-by query against the backend.

        The in-process implementation walks rows once and accumulates
        per-group aggregates in a sorted dict so the output order is
        deterministic.
        """
        # Issue a single fetch covering all required columns.
        needed = list(spec.group_keys)
        for agg in spec.aggregates:
            if agg.column not in needed:
                needed.append(agg.column)
        request = QueryRequest(
            table=spec.table,
            columns=tuple(needed),
            filters=dict(spec.filters),
        )
        rows = self.backend.execute(request)

        groups: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
        for row in rows:
            key = tuple(row.get(k) for k in spec.group_keys)
            groups.setdefault(key, []).append(row)

        result_columns = spec.group_keys + tuple(agg.alias for agg in spec.aggregates)
        result_rows: list[tuple[object, ...]] = []
        for key in sorted(groups, key=lambda k: tuple(_sort_key(v) for v in k)):
            bucket = groups[key]
            agg_values: list[object] = []
            for agg in spec.aggregates:
                values = [row.get(agg.column) for row in bucket if row.get(agg.column) is not None]
                agg_values.append(_apply_agg(agg.op, values))
            result_rows.append(tuple(key) + tuple(agg_values))

        materialised = tuple(result_rows)
        return QueryResult(
            request_table=spec.table,
            request_columns=result_columns,
            rows=materialised,
            result_digest=_digest_rows(
                table=spec.table,
                columns=result_columns,
                rows=materialised,
            ),
        )


def _apply_agg(op: str, values: Iterable[object]) -> object:
    materialised = [v for v in values if v is not None]
    if op == "count":
        return len(materialised)
    if not materialised:
        return None
    numeric: list[float] = []
    for v in materialised:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        numeric.append(float(v))
    if op == "min":
        return min(numeric) if numeric else min(materialised, key=repr)
    if op == "max":
        return max(numeric) if numeric else max(materialised, key=repr)
    if op == "sum":
        return sum(numeric)
    if op == "avg":
        return sum(numeric) / len(numeric) if numeric else None
    raise ValueError(f"unknown aggregate op {op!r}")


def _digest_rows(
    *,
    table: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> str:
    """BLAKE2b-16 digest over a canonical sorted-key JSON projection."""
    payload = {
        "table": table,
        "columns": list(columns),
        "rows": [_jsonable_row(row) for row in rows],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()


def _jsonable_row(row: Sequence[object]) -> list[object]:
    out: list[object] = []
    for value in row:
        if value is None or isinstance(value, (bool, int, float, str)):
            out.append(value)
        elif isinstance(value, (bytes, bytearray)):
            out.append(bytes(value).hex())
        else:
            out.append(repr(value))
    return out


# ----------------------------------------------------------------------
# DuckDB-backed implementation (lazy import)
# ----------------------------------------------------------------------


def duckdb_backend_factory(
    *,
    database: str | None = None,
    read_only: bool = True,
) -> AnalyticsBackend:
    """Build a DuckDB-backed :class:`AnalyticsBackend`.

    The DuckDB module is imported **inside the factory body** so this
    file imports cleanly when ``duckdb`` is not installed (mirrors the
    S-01 ccxt / S-05 firecrawl / S-10 polars precedents).

    Parameters
    ----------
    database:
        Path to a DuckDB / parquet / SQLite file. ``None`` opens an
        in-memory connection that callers can populate with
        ``read_parquet(...)`` / ``read_csv(...)`` table functions.
    read_only:
        Open the file in read-only mode (``True`` by default).
        OFFLINE_ONLY tier — analytics must never mutate the ledger.
    """
    import duckdb  # noqa: PLC0415 — lazy import is intentional

    connect_kwargs: dict[str, object] = {}
    if database is not None:
        connect_kwargs["database"] = database
        connect_kwargs["read_only"] = read_only
    connection = duckdb.connect(**connect_kwargs)
    return _DuckDBBackend(connection=connection)


class _DuckDBBackend:
    """DuckDB-backed :class:`AnalyticsBackend` impl.

    Holds a single DuckDB connection (lazy-imported by
    :func:`duckdb_backend_factory`). Each ``execute`` call issues a
    parameterised ``SELECT`` against the configured table /
    parquet-table-function source.
    """

    __slots__ = ("_connection",)

    def __init__(self, *, connection: object) -> None:
        # ``connection`` is duckdb.DuckDBPyConnection; we hold it as
        # ``object`` so this module type-checks without duckdb
        # installed.
        self._connection = connection

    def execute(
        self,
        request: QueryRequest,
    ) -> tuple[Mapping[str, object], ...]:
        columns = ", ".join(request.columns)
        where_clause = ""
        params: list[object] = []
        if request.filters:
            sorted_keys = sorted(request.filters)
            clauses = [f"{key} = ?" for key in sorted_keys]
            where_clause = " WHERE " + " AND ".join(clauses)
            params.extend(request.filters[key] for key in sorted_keys)
        order_columns = request.order_by if request.order_by else request.columns
        order_clause = " ORDER BY " + ", ".join(order_columns)
        limit_clause = ""
        if request.limit is not None:
            limit_clause = f" LIMIT {int(request.limit)}"
        sql = f"SELECT {columns} FROM {request.table}{where_clause}{order_clause}{limit_clause}"
        cursor = self._connection.execute(sql, params)
        try:
            description = cursor.description or []
            col_names = [col[0] for col in description]
            rows = cursor.fetchall()
        finally:
            # DuckDBPyConnection doesn't expose .close() per row; the
            # caller can close the connection when done.
            pass
        return tuple(dict(zip(col_names, row, strict=False)) for row in rows)
