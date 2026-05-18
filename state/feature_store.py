# ADAPTED FROM: feast-dev/feast Python SDK
# (sdk/python/feast/feature_store.py — FeatureStore class,
#  get_online_features / get_historical_features;
#  sdk/python/feast/feature_view.py — FeatureView / Entity definitions;
#  sdk/python/feast/infra/online_stores/sqlite.py — local SQLite
#  online-store pattern (no Redis / DynamoDB / external server).)
"""DIX feature store — feast-style SQLite backend (S-09).

Pure-stdlib reproduction of feast's ``FeatureStore`` /
``get_online_features`` / ``get_historical_features`` surface behind
the :class:`FeatureStoreBase` Protocol.

The feast SDK is a much larger system (feature engineering, registry
sync, multi-backend infra). For DIX we adapt the **single-process
SQLite path** only — the pattern feast itself ships as the default
``infra/online_stores/sqlite.py`` backend. That keeps the dependency
surface at zero (``sqlite3`` is stdlib) and pins INV-15 byte-stable
replays via deterministic ``ORDER BY`` clauses.

Algorithmic surface ported from feast:

* ``feast.FeatureStore.apply([fv])`` →
  :meth:`FeatureStore.register_view`
* ``feast.FeatureStore.write_to_online_store(...)`` →
  :meth:`FeatureStore.ingest`
* ``feast.FeatureStore.get_online_features(...).to_dict()`` →
  :meth:`FeatureStore.get_online_features`
* ``feast.FeatureStore.get_historical_features(entity_df,
  features=...).to_df()`` → :meth:`FeatureStore.get_historical_features`

Authority constraints (S-09 spec, lines 628–636):

* **OFFLINE write tier** — :meth:`register_view` and :meth:`ingest`
  are never called from the hot path. Writers live in
  ``learning_engine`` / ``evolution_engine`` (training pipelines,
  feature backfill). Authority-lint will not let an execution-tier
  module call those methods (read-side ``get_online_features`` is the
  RUNTIME-SAFE half).
* **RUNTIME-SAFE read tier** — :meth:`get_online_features` is the
  point-in-time-safe lookup that hot-path engines (Indira plugins,
  governance gates) may call. The contract is a per-row latest-value
  fold over a single SQLite ``SELECT … ORDER BY ts_ns DESC LIMIT 1``,
  which finishes in microseconds for typical N.
* **INV-15 byte-identical replay** — every read path orders by
  ``(entity_key, feature_name, ts_ns DESC)`` (or by ``feature_name``
  when collapsing into the per-row dict) so identical inputs produce
  identical outputs across runs / machines / Python instances.
* **Time fed by caller** — every method takes an explicit ``ts_ns``
  argument. The store never reads the wall clock. Point-in-time
  lookups filter on ``ts_ns <= caller_ts_ns`` so back-tests can
  replay the exact feature snapshot a live decision saw.
* **Pure stdlib** — only ``sqlite3``, ``dataclasses``, ``json``,
  ``math``. ``NEW_PIP_DEPENDENCIES = ()``. No feast / pandas /
  pyarrow / fsspec import.
* **No clock, no PRNG, no IO outside the caller-supplied path**.

Schema (single SQLite file, two tables)::

    CREATE TABLE feature_views (
        name              TEXT PRIMARY KEY,
        entity_keys_json  TEXT NOT NULL,
        feature_names_json TEXT NOT NULL,
        registered_ts_ns  INTEGER NOT NULL
    );
    CREATE TABLE feature_values (
        view_name    TEXT NOT NULL,
        entity_key   TEXT NOT NULL,
        feature_name TEXT NOT NULL,
        ts_ns        INTEGER NOT NULL,
        value        REAL NOT NULL,
        PRIMARY KEY (view_name, entity_key, feature_name, ts_ns)
    );
    CREATE INDEX idx_fv_lookup
        ON feature_values (view_name, entity_key, feature_name, ts_ns DESC);

The composite primary key is the upsert ``ON CONFLICT … DO UPDATE``
target — re-ingesting the same ``(view, entity, feature, ts_ns)``
overwrites the value rather than duplicating the row. This matches
feast's online-store contract where each ``(entity, feature)`` row is
addressable by timestamp.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ()
_SCHEMA_VERSION = 1

_DDL = (
    """
    CREATE TABLE IF NOT EXISTS feature_views (
        name               TEXT PRIMARY KEY,
        entity_keys_json   TEXT NOT NULL,
        feature_names_json TEXT NOT NULL,
        registered_ts_ns   INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS feature_values (
        view_name    TEXT NOT NULL,
        entity_key   TEXT NOT NULL,
        feature_name TEXT NOT NULL,
        ts_ns        INTEGER NOT NULL,
        value        REAL NOT NULL,
        PRIMARY KEY (view_name, entity_key, feature_name, ts_ns)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_fv_lookup
        ON feature_values (view_name, entity_key, feature_name, ts_ns DESC)
    """,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_name(name: str, *, label: str) -> None:
    if not isinstance(name, str):
        raise TypeError(f"{label} must be str, got {type(name).__name__}")
    if not name:
        raise ValueError(f"{label} must be non-empty")


def _validate_value(value: float, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite number, got {type(value).__name__}")
    fv = float(value)
    if not math.isfinite(fv):
        raise ValueError(f"{label} must be finite, got {fv!r}")
    return fv


def _frozen_str_tuple(values: Iterable[str], *, label: str) -> tuple[str, ...]:
    out: list[str] = []
    for i, v in enumerate(values):
        _validate_name(v, label=f"{label}[{i}]")
        out.append(v)
    if len(set(out)) != len(out):
        raise ValueError(f"{label} must have unique entries: {out!r}")
    if not out:
        raise ValueError(f"{label} must have at least one entry")
    return tuple(out)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FeatureView:
    """Schema definition for a related set of features.

    Mirrors feast's ``FeatureView`` minus the entity-join / source plumbing
    we don't need — DIX feature ingestion is push-only, the writer holds
    the upstream join.
    """

    name: str
    entity_keys: tuple[str, ...]
    feature_names: tuple[str, ...]
    ts_ns: int

    def __post_init__(self) -> None:
        _validate_name(self.name, label="FeatureView.name")
        if not isinstance(self.entity_keys, tuple):
            raise TypeError(
                f"FeatureView.entity_keys must be tuple, got {type(self.entity_keys).__name__}"
            )
        _frozen_str_tuple(self.entity_keys, label="FeatureView.entity_keys")
        if not isinstance(self.feature_names, tuple):
            raise TypeError(
                f"FeatureView.feature_names must be tuple, got {type(self.feature_names).__name__}"
            )
        _frozen_str_tuple(self.feature_names, label="FeatureView.feature_names")
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise TypeError(f"FeatureView.ts_ns must be int, got {type(self.ts_ns).__name__}")
        if self.ts_ns <= 0:
            raise ValueError(f"FeatureView.ts_ns must be positive, got {self.ts_ns!r}")


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    """One row of features for a single (view, entity) at a given ts.

    Mirrors feast's ``write_to_online_store`` row — a dense map from
    feature name → finite float value.
    """

    ts_ns: int
    view_name: str
    entity_key: str
    values: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise TypeError(f"FeatureRecord.ts_ns must be int, got {type(self.ts_ns).__name__}")
        if self.ts_ns <= 0:
            raise ValueError(f"FeatureRecord.ts_ns must be positive, got {self.ts_ns!r}")
        _validate_name(self.view_name, label="FeatureRecord.view_name")
        _validate_name(self.entity_key, label="FeatureRecord.entity_key")
        if not isinstance(self.values, Mapping):
            raise TypeError(
                f"FeatureRecord.values must be Mapping, got {type(self.values).__name__}"
            )
        if not self.values:
            raise ValueError("FeatureRecord.values must be non-empty")
        for k, v in self.values.items():
            _validate_name(k, label="FeatureRecord.values key")
            _validate_value(v, label=f"FeatureRecord.values[{k!r}]")


@dataclass(frozen=True, slots=True)
class FeatureRequest:
    """A single online-feature lookup at a caller-supplied ``ts_ns``.

    Mirrors feast's ``get_online_features(features=…, entity_rows=…)``
    call, narrowed to a single entity row to keep the read path
    point-in-time and clock-free.
    """

    ts_ns: int
    view_name: str
    entity_key: str
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise TypeError(f"FeatureRequest.ts_ns must be int, got {type(self.ts_ns).__name__}")
        if self.ts_ns <= 0:
            raise ValueError(f"FeatureRequest.ts_ns must be positive, got {self.ts_ns!r}")
        _validate_name(self.view_name, label="FeatureRequest.view_name")
        _validate_name(self.entity_key, label="FeatureRequest.entity_key")
        if not isinstance(self.feature_names, tuple):
            raise TypeError(
                "FeatureRequest.feature_names must be tuple, "
                f"got {type(self.feature_names).__name__}"
            )
        _frozen_str_tuple(self.feature_names, label="FeatureRequest.feature_names")


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    """Result of an online-feature lookup.

    ``values`` is the dense per-feature dict the caller asked for. Each
    entry is the latest stored value at or before ``request.ts_ns``;
    missing features are absent (callers must check
    ``set(values) == set(request.feature_names)`` if the caller treats
    missing as fatal). ``observed_ts_ns_per_feature`` exposes the actual
    ``ts_ns`` of each returned value so replay-correctness checks can
    pin point-in-time alignment with the calling decision.
    """

    ts_ns: int
    view_name: str
    entity_key: str
    values: Mapping[str, float] = field(default_factory=dict)
    observed_ts_ns_per_feature: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise TypeError(f"FeatureSnapshot.ts_ns must be int, got {type(self.ts_ns).__name__}")
        if self.ts_ns <= 0:
            raise ValueError(f"FeatureSnapshot.ts_ns must be positive, got {self.ts_ns!r}")
        _validate_name(self.view_name, label="FeatureSnapshot.view_name")
        _validate_name(self.entity_key, label="FeatureSnapshot.entity_key")
        if not isinstance(self.values, Mapping):
            raise TypeError(
                f"FeatureSnapshot.values must be Mapping, got {type(self.values).__name__}"
            )
        if not isinstance(self.observed_ts_ns_per_feature, Mapping):
            raise TypeError(
                "FeatureSnapshot.observed_ts_ns_per_feature must be Mapping, "
                f"got {type(self.observed_ts_ns_per_feature).__name__}"
            )
        for k, v in self.values.items():
            _validate_name(k, label="FeatureSnapshot.values key")
            _validate_value(v, label=f"FeatureSnapshot.values[{k!r}]")
        for k, v in self.observed_ts_ns_per_feature.items():
            _validate_name(k, label="FeatureSnapshot.observed_ts_ns_per_feature key")
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ValueError(
                    "FeatureSnapshot.observed_ts_ns_per_feature["
                    f"{k!r}] must be positive int, got {v!r}"
                )
        # observed-ts keys must be a subset of values keys
        if set(self.observed_ts_ns_per_feature) - set(self.values):
            raise ValueError("FeatureSnapshot.observed_ts_ns_per_feature has keys not in values")


@dataclass(frozen=True, slots=True)
class HistoricalRow:
    """One ``(entity, ts_ns)`` row in a point-in-time entity dataframe.

    Mirrors feast's ``entity_df`` row — the caller hands the store a
    list of these and asks for the features the value of which the
    decision saw at the row's ``ts_ns``.
    """

    ts_ns: int
    view_name: str
    entity_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise TypeError(f"HistoricalRow.ts_ns must be int, got {type(self.ts_ns).__name__}")
        if self.ts_ns <= 0:
            raise ValueError(f"HistoricalRow.ts_ns must be positive, got {self.ts_ns!r}")
        _validate_name(self.view_name, label="HistoricalRow.view_name")
        _validate_name(self.entity_key, label="HistoricalRow.entity_key")


# ---------------------------------------------------------------------------
# Protocol surface
# ---------------------------------------------------------------------------


@runtime_checkable
class FeatureStoreBase(Protocol):
    """Minimal Protocol that every feature-store backend must satisfy.

    Mirrors the slice of feast's ``FeatureStore`` we adapt: register a
    schema, push a row, fetch per-entity online features, fetch
    point-in-time historical features.
    """

    def register_view(self, view: FeatureView) -> None: ...
    def list_views(self) -> tuple[FeatureView, ...]: ...
    def ingest(self, record: FeatureRecord) -> None: ...
    def get_online_features(self, request: FeatureRequest) -> FeatureSnapshot: ...
    def get_historical_features(
        self,
        rows: Sequence[HistoricalRow],
        feature_names: Sequence[str],
    ) -> tuple[FeatureSnapshot, ...]: ...


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------


class FeatureStore:
    """SQLite-backed feature store.

    Two operating modes:

    * **In-memory (default)** — ``path=":memory:"`` keeps everything
      inside the connection. Ideal for tests + replays.
    * **File-backed** — pass an absolute filesystem path. The backend
      opens the file in WAL mode so multiple readers can coexist with
      a single writer (mirrors feast's SQLite online-store config).

    The store never reads the wall clock. Every method takes
    ``ts_ns`` from the caller. Replays therefore see the same
    snapshot the live decision saw.
    """

    __slots__ = ("_path", "_conn")

    def __init__(self, *, path: str = ":memory:") -> None:
        if not isinstance(path, str):
            raise TypeError(f"FeatureStore.path must be str, got {type(path).__name__}")
        if not path:
            raise ValueError("FeatureStore.path must be non-empty")
        self._path = path
        self._conn = sqlite3.connect(path, isolation_level=None)
        self._conn.execute("PRAGMA foreign_keys = ON")
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        for ddl in _DDL:
            self._conn.execute(ddl)

    @property
    def path(self) -> str:
        return self._path

    # ----- writers (OFFLINE tier) ----------------------------------------

    def register_view(self, view: FeatureView) -> None:
        """Persist a feature view's schema. Idempotent for identical views."""
        if not isinstance(view, FeatureView):
            raise TypeError(f"register_view expects FeatureView, got {type(view).__name__}")
        ek = json.dumps(list(view.entity_keys), sort_keys=True)
        fn = json.dumps(list(view.feature_names), sort_keys=True)
        existing = self._conn.execute(
            "SELECT entity_keys_json, feature_names_json, registered_ts_ns "
            "FROM feature_views WHERE name = ?",
            (view.name,),
        ).fetchone()
        if existing is not None:
            if existing[0] != ek or existing[1] != fn:
                raise ValueError(
                    f"FeatureView {view.name!r} already registered with a different schema"
                )
            return
        self._conn.execute(
            "INSERT INTO feature_views "
            "(name, entity_keys_json, feature_names_json, registered_ts_ns) "
            "VALUES (?, ?, ?, ?)",
            (view.name, ek, fn, view.ts_ns),
        )

    def list_views(self) -> tuple[FeatureView, ...]:
        """Return every registered view, ordered by ``name`` ascending."""
        rows = self._conn.execute(
            "SELECT name, entity_keys_json, feature_names_json, registered_ts_ns "
            "FROM feature_views ORDER BY name ASC"
        ).fetchall()
        out: list[FeatureView] = []
        for name, ek_json, fn_json, ts in rows:
            out.append(
                FeatureView(
                    name=name,
                    entity_keys=tuple(json.loads(ek_json)),
                    feature_names=tuple(json.loads(fn_json)),
                    ts_ns=int(ts),
                )
            )
        return tuple(out)

    def ingest(self, record: FeatureRecord) -> None:
        """Push one row of features. Upsert by ``(view, entity, feature, ts_ns)``."""
        if not isinstance(record, FeatureRecord):
            raise TypeError(f"ingest expects FeatureRecord, got {type(record).__name__}")
        view_row = self._conn.execute(
            "SELECT feature_names_json FROM feature_views WHERE name = ?",
            (record.view_name,),
        ).fetchone()
        if view_row is None:
            raise ValueError(f"FeatureRecord.view_name {record.view_name!r} not registered")
        known = set(json.loads(view_row[0]))
        unknown = set(record.values) - known
        if unknown:
            raise ValueError(
                f"FeatureRecord.values has features not in view "
                f"{record.view_name!r}: {sorted(unknown)!r}"
            )
        params = [
            (
                record.view_name,
                record.entity_key,
                feat,
                record.ts_ns,
                float(value),
            )
            # Iterate in name-sorted order so replay byte-prints stay
            # stable even if Python dict iteration order ever changed.
            for feat, value in sorted(record.values.items())
        ]
        self._conn.executemany(
            "INSERT INTO feature_values "
            "(view_name, entity_key, feature_name, ts_ns, value) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(view_name, entity_key, feature_name, ts_ns) "
            "DO UPDATE SET value = excluded.value",
            params,
        )

    # ----- readers (RUNTIME-SAFE tier) -----------------------------------

    def get_online_features(self, request: FeatureRequest) -> FeatureSnapshot:
        """Return the latest value per feature at-or-before ``request.ts_ns``.

        Missing features are simply absent from the resulting
        ``values`` dict. The caller decides how to handle absence
        (fatal in pinned-feature pipelines, soft-default in others).
        """
        if not isinstance(request, FeatureRequest):
            raise TypeError(
                f"get_online_features expects FeatureRequest, got {type(request).__name__}"
            )
        view_row = self._conn.execute(
            "SELECT feature_names_json FROM feature_views WHERE name = ?",
            (request.view_name,),
        ).fetchone()
        if view_row is None:
            raise ValueError(f"FeatureRequest.view_name {request.view_name!r} not registered")
        known = set(json.loads(view_row[0]))
        unknown = set(request.feature_names) - known
        if unknown:
            raise ValueError(
                f"FeatureRequest.feature_names has features not in view "
                f"{request.view_name!r}: {sorted(unknown)!r}"
            )
        values: dict[str, float] = {}
        observed: dict[str, int] = {}
        for feat in sorted(request.feature_names):
            row = self._conn.execute(
                "SELECT value, ts_ns FROM feature_values "
                "WHERE view_name = ? AND entity_key = ? AND feature_name = ? "
                "  AND ts_ns <= ? "
                "ORDER BY ts_ns DESC LIMIT 1",
                (
                    request.view_name,
                    request.entity_key,
                    feat,
                    request.ts_ns,
                ),
            ).fetchone()
            if row is None:
                continue
            values[feat] = float(row[0])
            observed[feat] = int(row[1])
        return FeatureSnapshot(
            ts_ns=request.ts_ns,
            view_name=request.view_name,
            entity_key=request.entity_key,
            values=values,
            observed_ts_ns_per_feature=observed,
        )

    def get_historical_features(
        self,
        rows: Sequence[HistoricalRow],
        feature_names: Sequence[str],
    ) -> tuple[FeatureSnapshot, ...]:
        """Point-in-time-correct join: latest values at-or-before each row's ts.

        Returns one :class:`FeatureSnapshot` per input row, in input
        order. The caller's responsibility to keep the input order
        deterministic; we never reorder.
        """
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise TypeError(
                "get_historical_features expects a Sequence[HistoricalRow], "
                f"got {type(rows).__name__}"
            )
        if not isinstance(feature_names, Sequence) or isinstance(feature_names, (str, bytes)):
            raise TypeError(
                "get_historical_features expects a Sequence[str] for "
                f"feature_names, got {type(feature_names).__name__}"
            )
        feats = _frozen_str_tuple(feature_names, label="feature_names")
        out: list[FeatureSnapshot] = []
        for row in rows:
            if not isinstance(row, HistoricalRow):
                raise TypeError(
                    f"get_historical_features rows must be HistoricalRow, got {type(row).__name__}"
                )
            req = FeatureRequest(
                ts_ns=row.ts_ns,
                view_name=row.view_name,
                entity_key=row.entity_key,
                feature_names=feats,
            )
            out.append(self.get_online_features(req))
        return tuple(out)

    # ----- lifecycle ------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection. Idempotent."""
        try:
            self._conn.close()
        except sqlite3.ProgrammingError:
            pass

    def __enter__(self) -> FeatureStore:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "FeatureView",
    "FeatureRecord",
    "FeatureRequest",
    "FeatureSnapshot",
    "HistoricalRow",
    "FeatureStoreBase",
    "FeatureStore",
)
