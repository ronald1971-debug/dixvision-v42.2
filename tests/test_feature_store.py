"""Tests for state/feature_store.py (S-09 feast adaptation)."""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path

import pytest

from state.feature_store import (
    NEW_PIP_DEPENDENCIES,
    FeatureRecord,
    FeatureRequest,
    FeatureSnapshot,
    FeatureStore,
    FeatureStoreBase,
    FeatureView,
    HistoricalRow,
)

_MODULE_PATH = Path(__file__).resolve().parents[1] / "state" / "feature_store.py"
_FORBIDDEN_IMPORTS = (
    "datetime",
    "time",
    "asyncio",
    "threading",
    "subprocess",
    "socket",
    "logging",
    "feast",
    "pandas",
    "pyarrow",
    "numpy",
    "fsspec",
    "random",
    "secrets",
)


# ---------------------------------------------------------------------------
# Module metadata / authority lint
# ---------------------------------------------------------------------------


def test_module_has_no_pip_dependencies() -> None:
    assert NEW_PIP_DEPENDENCIES == ()


def test_module_has_adapted_from_header() -> None:
    text = _MODULE_PATH.read_text(encoding="utf-8")
    assert "# ADAPTED FROM: feast-dev/feast" in text


def test_module_has_no_forbidden_imports() -> None:
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _FORBIDDEN_IMPORTS:
                    bad.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            if root in _FORBIDDEN_IMPORTS:
                bad.append(node.module)
    assert bad == [], f"Forbidden imports in feature_store.py: {bad!r}"


def test_module_has_no_clock_substrings() -> None:
    text = _MODULE_PATH.read_text(encoding="utf-8")
    for needle in ("time.time(", "time.monotonic(", "datetime.now(", "datetime.utcnow("):
        assert needle not in text, f"feature_store.py contains {needle!r}"


# ---------------------------------------------------------------------------
# FeatureView
# ---------------------------------------------------------------------------


def _view(name: str = "v") -> FeatureView:
    return FeatureView(
        name=name,
        entity_keys=("symbol",),
        feature_names=("vol", "spread"),
        ts_ns=1,
    )


def test_feature_view_minimal() -> None:
    v = _view()
    assert v.name == "v"
    assert v.entity_keys == ("symbol",)
    assert v.feature_names == ("vol", "spread")


def test_feature_view_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        FeatureView(
            name="",
            entity_keys=("e",),
            feature_names=("f",),
            ts_ns=1,
        )


def test_feature_view_rejects_non_tuple_entity_keys() -> None:
    with pytest.raises(TypeError):
        FeatureView(
            name="v",
            entity_keys=["e"],  # type: ignore[arg-type]
            feature_names=("f",),
            ts_ns=1,
        )


def test_feature_view_rejects_non_tuple_feature_names() -> None:
    with pytest.raises(TypeError):
        FeatureView(
            name="v",
            entity_keys=("e",),
            feature_names=["f"],  # type: ignore[arg-type]
            ts_ns=1,
        )


def test_feature_view_rejects_empty_feature_names() -> None:
    with pytest.raises(ValueError):
        FeatureView(
            name="v",
            entity_keys=("e",),
            feature_names=(),
            ts_ns=1,
        )


def test_feature_view_rejects_duplicate_feature_names() -> None:
    with pytest.raises(ValueError):
        FeatureView(
            name="v",
            entity_keys=("e",),
            feature_names=("f", "f"),
            ts_ns=1,
        )


def test_feature_view_rejects_zero_ts_ns() -> None:
    with pytest.raises(ValueError):
        FeatureView(
            name="v",
            entity_keys=("e",),
            feature_names=("f",),
            ts_ns=0,
        )


def test_feature_view_rejects_bool_ts_ns() -> None:
    with pytest.raises(TypeError):
        FeatureView(
            name="v",
            entity_keys=("e",),
            feature_names=("f",),
            ts_ns=True,  # type: ignore[arg-type]
        )


def test_feature_view_is_frozen() -> None:
    v = _view()
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FeatureRecord
# ---------------------------------------------------------------------------


def test_feature_record_minimal() -> None:
    r = FeatureRecord(ts_ns=10, view_name="v", entity_key="BTC", values={"vol": 1.0})
    assert r.values == {"vol": 1.0}


def test_feature_record_rejects_non_finite_value() -> None:
    with pytest.raises(ValueError):
        FeatureRecord(
            ts_ns=1,
            view_name="v",
            entity_key="BTC",
            values={"vol": float("inf")},
        )


def test_feature_record_rejects_nan_value() -> None:
    with pytest.raises(ValueError):
        FeatureRecord(
            ts_ns=1,
            view_name="v",
            entity_key="BTC",
            values={"vol": float("nan")},
        )


def test_feature_record_rejects_bool_value() -> None:
    with pytest.raises(TypeError):
        FeatureRecord(
            ts_ns=1,
            view_name="v",
            entity_key="BTC",
            values={"vol": True},  # type: ignore[dict-item]
        )


def test_feature_record_rejects_empty_values() -> None:
    with pytest.raises(ValueError):
        FeatureRecord(ts_ns=1, view_name="v", entity_key="BTC", values={})


def test_feature_record_rejects_zero_ts_ns() -> None:
    with pytest.raises(ValueError):
        FeatureRecord(ts_ns=0, view_name="v", entity_key="BTC", values={"f": 1.0})


def test_feature_record_rejects_empty_entity_key() -> None:
    with pytest.raises(ValueError):
        FeatureRecord(ts_ns=1, view_name="v", entity_key="", values={"f": 1.0})


# ---------------------------------------------------------------------------
# FeatureRequest
# ---------------------------------------------------------------------------


def test_feature_request_minimal() -> None:
    q = FeatureRequest(ts_ns=5, view_name="v", entity_key="BTC", feature_names=("vol",))
    assert q.feature_names == ("vol",)


def test_feature_request_rejects_empty_feature_names() -> None:
    with pytest.raises(ValueError):
        FeatureRequest(ts_ns=1, view_name="v", entity_key="BTC", feature_names=())


def test_feature_request_rejects_non_tuple_feature_names() -> None:
    with pytest.raises(TypeError):
        FeatureRequest(
            ts_ns=1,
            view_name="v",
            entity_key="BTC",
            feature_names=["vol"],  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# FeatureSnapshot
# ---------------------------------------------------------------------------


def test_feature_snapshot_minimal() -> None:
    s = FeatureSnapshot(
        ts_ns=5,
        view_name="v",
        entity_key="BTC",
        values={"vol": 1.5},
        observed_ts_ns_per_feature={"vol": 4},
    )
    assert s.values["vol"] == 1.5
    assert s.observed_ts_ns_per_feature["vol"] == 4


def test_feature_snapshot_rejects_observed_keys_not_in_values() -> None:
    with pytest.raises(ValueError):
        FeatureSnapshot(
            ts_ns=5,
            view_name="v",
            entity_key="BTC",
            values={"vol": 1.0},
            observed_ts_ns_per_feature={"spread": 4},
        )


def test_feature_snapshot_rejects_non_finite_value() -> None:
    with pytest.raises(ValueError):
        FeatureSnapshot(
            ts_ns=5,
            view_name="v",
            entity_key="BTC",
            values={"vol": float("nan")},
        )


# ---------------------------------------------------------------------------
# HistoricalRow
# ---------------------------------------------------------------------------


def test_historical_row_minimal() -> None:
    r = HistoricalRow(ts_ns=10, view_name="v", entity_key="BTC")
    assert r.ts_ns == 10


def test_historical_row_rejects_zero_ts_ns() -> None:
    with pytest.raises(ValueError):
        HistoricalRow(ts_ns=0, view_name="v", entity_key="BTC")


# ---------------------------------------------------------------------------
# FeatureStore — register / list
# ---------------------------------------------------------------------------


def test_store_constructs_with_default_in_memory_path() -> None:
    s = FeatureStore()
    assert s.path == ":memory:"
    s.close()


def test_store_rejects_empty_path() -> None:
    with pytest.raises(ValueError):
        FeatureStore(path="")


def test_store_rejects_non_str_path() -> None:
    with pytest.raises(TypeError):
        FeatureStore(path=123)  # type: ignore[arg-type]


def test_store_register_view_minimal() -> None:
    s = FeatureStore()
    v = _view()
    s.register_view(v)
    assert s.list_views() == (v,)
    s.close()


def test_store_register_view_idempotent_on_identical_schema() -> None:
    s = FeatureStore()
    v = _view()
    s.register_view(v)
    s.register_view(v)
    assert s.list_views() == (v,)
    s.close()


def test_store_register_view_rejects_schema_drift() -> None:
    s = FeatureStore()
    s.register_view(_view())
    with pytest.raises(ValueError):
        s.register_view(
            FeatureView(
                name="v",
                entity_keys=("symbol",),
                feature_names=("vol",),  # missing 'spread'
                ts_ns=2,
            )
        )
    s.close()


def test_store_register_view_rejects_non_view() -> None:
    s = FeatureStore()
    with pytest.raises(TypeError):
        s.register_view("not a view")  # type: ignore[arg-type]
    s.close()


def test_store_list_views_sorted_by_name() -> None:
    s = FeatureStore()
    s.register_view(FeatureView(name="b", entity_keys=("e",), feature_names=("f",), ts_ns=1))
    s.register_view(FeatureView(name="a", entity_keys=("e",), feature_names=("f",), ts_ns=1))
    names = [v.name for v in s.list_views()]
    assert names == ["a", "b"]
    s.close()


# ---------------------------------------------------------------------------
# FeatureStore — ingest
# ---------------------------------------------------------------------------


def test_store_ingest_rejects_unknown_view() -> None:
    s = FeatureStore()
    with pytest.raises(ValueError):
        s.ingest(FeatureRecord(ts_ns=1, view_name="missing", entity_key="BTC", values={"vol": 1.0}))
    s.close()


def test_store_ingest_rejects_unknown_feature() -> None:
    s = FeatureStore()
    s.register_view(_view())
    with pytest.raises(ValueError):
        s.ingest(
            FeatureRecord(
                ts_ns=1,
                view_name="v",
                entity_key="BTC",
                values={"unknown": 1.0},
            )
        )
    s.close()


def test_store_ingest_rejects_non_record() -> None:
    s = FeatureStore()
    s.register_view(_view())
    with pytest.raises(TypeError):
        s.ingest("not a record")  # type: ignore[arg-type]
    s.close()


def test_store_ingest_upserts_same_ts_ns() -> None:
    s = FeatureStore()
    s.register_view(_view())
    s.ingest(FeatureRecord(ts_ns=10, view_name="v", entity_key="BTC", values={"vol": 1.0}))
    s.ingest(FeatureRecord(ts_ns=10, view_name="v", entity_key="BTC", values={"vol": 2.0}))
    snap = s.get_online_features(
        FeatureRequest(ts_ns=11, view_name="v", entity_key="BTC", feature_names=("vol",))
    )
    assert snap.values["vol"] == 2.0
    s.close()


# ---------------------------------------------------------------------------
# FeatureStore — get_online_features
# ---------------------------------------------------------------------------


def _seeded_store() -> FeatureStore:
    s = FeatureStore()
    s.register_view(_view())
    s.ingest(
        FeatureRecord(
            ts_ns=5,
            view_name="v",
            entity_key="BTC",
            values={"vol": 1.0, "spread": 0.1},
        )
    )
    s.ingest(
        FeatureRecord(
            ts_ns=10,
            view_name="v",
            entity_key="BTC",
            values={"vol": 2.0, "spread": 0.2},
        )
    )
    s.ingest(
        FeatureRecord(
            ts_ns=20,
            view_name="v",
            entity_key="BTC",
            values={"vol": 3.0, "spread": 0.3},
        )
    )
    return s


def test_store_get_online_features_returns_latest_at_or_before_ts() -> None:
    s = _seeded_store()
    snap = s.get_online_features(
        FeatureRequest(
            ts_ns=10,
            view_name="v",
            entity_key="BTC",
            feature_names=("vol", "spread"),
        )
    )
    assert snap.values == {"vol": 2.0, "spread": 0.2}
    assert snap.observed_ts_ns_per_feature == {"vol": 10, "spread": 10}
    s.close()


def test_store_get_online_features_skips_future_ts() -> None:
    s = _seeded_store()
    snap = s.get_online_features(
        FeatureRequest(
            ts_ns=15,
            view_name="v",
            entity_key="BTC",
            feature_names=("vol",),
        )
    )
    assert snap.values == {"vol": 2.0}  # ts=20 row excluded by point-in-time
    s.close()


def test_store_get_online_features_returns_empty_if_no_history() -> None:
    s = _seeded_store()
    snap = s.get_online_features(
        FeatureRequest(ts_ns=4, view_name="v", entity_key="BTC", feature_names=("vol",))
    )
    assert snap.values == {}
    assert snap.observed_ts_ns_per_feature == {}
    s.close()


def test_store_get_online_features_unknown_entity_returns_empty() -> None:
    s = _seeded_store()
    snap = s.get_online_features(
        FeatureRequest(ts_ns=20, view_name="v", entity_key="ETH", feature_names=("vol",))
    )
    assert snap.values == {}
    s.close()


def test_store_get_online_features_rejects_unknown_view() -> None:
    s = _seeded_store()
    with pytest.raises(ValueError):
        s.get_online_features(
            FeatureRequest(
                ts_ns=20,
                view_name="missing",
                entity_key="BTC",
                feature_names=("vol",),
            )
        )
    s.close()


def test_store_get_online_features_rejects_unknown_feature() -> None:
    s = _seeded_store()
    with pytest.raises(ValueError):
        s.get_online_features(
            FeatureRequest(
                ts_ns=20,
                view_name="v",
                entity_key="BTC",
                feature_names=("unknown",),
            )
        )
    s.close()


def test_store_get_online_features_rejects_non_request() -> None:
    s = _seeded_store()
    with pytest.raises(TypeError):
        s.get_online_features("not a request")  # type: ignore[arg-type]
    s.close()


def test_store_get_online_features_partial_history() -> None:
    """Not every feature has a past row at the requested ts."""
    s = FeatureStore()
    s.register_view(_view())
    s.ingest(
        FeatureRecord(
            ts_ns=10,
            view_name="v",
            entity_key="BTC",
            values={"vol": 1.0},  # spread has no row
        )
    )
    snap = s.get_online_features(
        FeatureRequest(
            ts_ns=20,
            view_name="v",
            entity_key="BTC",
            feature_names=("vol", "spread"),
        )
    )
    assert snap.values == {"vol": 1.0}
    assert "spread" not in snap.values
    s.close()


# ---------------------------------------------------------------------------
# FeatureStore — get_historical_features (point-in-time join)
# ---------------------------------------------------------------------------


def test_store_get_historical_features_point_in_time() -> None:
    s = _seeded_store()
    rows = (
        HistoricalRow(ts_ns=5, view_name="v", entity_key="BTC"),
        HistoricalRow(ts_ns=15, view_name="v", entity_key="BTC"),
        HistoricalRow(ts_ns=25, view_name="v", entity_key="BTC"),
    )
    out = s.get_historical_features(rows, ("vol",))
    assert len(out) == 3
    assert out[0].values == {"vol": 1.0}
    assert out[1].values == {"vol": 2.0}
    assert out[2].values == {"vol": 3.0}
    s.close()


def test_store_get_historical_features_preserves_input_order() -> None:
    s = _seeded_store()
    rows = (
        HistoricalRow(ts_ns=25, view_name="v", entity_key="BTC"),
        HistoricalRow(ts_ns=5, view_name="v", entity_key="BTC"),
        HistoricalRow(ts_ns=15, view_name="v", entity_key="BTC"),
    )
    out = s.get_historical_features(rows, ("vol",))
    vols = [snap.values["vol"] for snap in out]
    assert vols == [3.0, 1.0, 2.0]
    s.close()


def test_store_get_historical_features_rejects_non_sequence_rows() -> None:
    s = _seeded_store()
    with pytest.raises(TypeError):
        s.get_historical_features("rows", ("vol",))  # type: ignore[arg-type]
    s.close()


def test_store_get_historical_features_rejects_str_features() -> None:
    s = _seeded_store()
    with pytest.raises(TypeError):
        s.get_historical_features(
            (HistoricalRow(ts_ns=5, view_name="v", entity_key="BTC"),),
            "vol",  # type: ignore[arg-type]
        )
    s.close()


def test_store_get_historical_features_rejects_non_historical_row() -> None:
    s = _seeded_store()
    with pytest.raises(TypeError):
        s.get_historical_features(("not a row",), ("vol",))  # type: ignore[arg-type]
    s.close()


# ---------------------------------------------------------------------------
# Replay determinism (INV-15)
# ---------------------------------------------------------------------------


def _seed_independent() -> FeatureStore:
    s = FeatureStore()
    s.register_view(
        FeatureView(
            name="v",
            entity_keys=("symbol",),
            feature_names=("vol", "spread"),
            ts_ns=1,
        )
    )
    for i in range(50):
        s.ingest(
            FeatureRecord(
                ts_ns=i + 1,
                view_name="v",
                entity_key="BTC" if i % 2 == 0 else "ETH",
                values={"vol": float(i), "spread": float(i) * 0.5},
            )
        )
    return s


def test_get_online_features_deterministic_across_runs() -> None:
    snaps = []
    for _ in range(3):
        s = _seed_independent()
        snap = s.get_online_features(
            FeatureRequest(
                ts_ns=100,
                view_name="v",
                entity_key="BTC",
                feature_names=("vol", "spread"),
            )
        )
        snaps.append(snap)
        s.close()
    assert snaps[0] == snaps[1] == snaps[2]


def test_get_historical_features_deterministic_across_runs() -> None:
    rows = tuple(HistoricalRow(ts_ns=20 + i, view_name="v", entity_key="BTC") for i in range(5))
    outs = []
    for _ in range(3):
        s = _seed_independent()
        outs.append(s.get_historical_features(rows, ("vol",)))
        s.close()
    assert outs[0] == outs[1] == outs[2]


def test_list_views_deterministic_across_runs() -> None:
    views = []
    for _ in range(3):
        s = FeatureStore()
        for name in ("z", "m", "a"):
            s.register_view(
                FeatureView(
                    name=name,
                    entity_keys=("e",),
                    feature_names=("f",),
                    ts_ns=1,
                )
            )
        views.append(s.list_views())
        s.close()
    assert views[0] == views[1] == views[2]


# ---------------------------------------------------------------------------
# File-backed mode
# ---------------------------------------------------------------------------


def test_store_persists_across_connections_on_file_backend(tmp_path) -> None:
    path = str(tmp_path / "features.db")
    s1 = FeatureStore(path=path)
    s1.register_view(_view())
    s1.ingest(FeatureRecord(ts_ns=10, view_name="v", entity_key="BTC", values={"vol": 1.5}))
    s1.close()

    s2 = FeatureStore(path=path)
    snap = s2.get_online_features(
        FeatureRequest(ts_ns=20, view_name="v", entity_key="BTC", feature_names=("vol",))
    )
    assert snap.values == {"vol": 1.5}
    s2.close()


def test_store_file_backend_uses_wal(tmp_path) -> None:
    path = str(tmp_path / "features.db")
    s = FeatureStore(path=path)
    mode = s._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"
    s.close()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_store_close_is_idempotent() -> None:
    s = FeatureStore()
    s.close()
    s.close()  # should not raise


def test_store_context_manager() -> None:
    with FeatureStore() as s:
        s.register_view(_view())
    # Re-use after close should fail
    with pytest.raises(sqlite_error_class()):
        s.register_view(_view())


def sqlite_error_class():
    import sqlite3

    return sqlite3.ProgrammingError


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_feature_store_satisfies_base_protocol() -> None:
    s = FeatureStore()
    assert isinstance(s, FeatureStoreBase)
    s.close()


def test_full_round_trip_via_protocol_facade() -> None:
    backend: FeatureStoreBase = FeatureStore()
    backend.register_view(_view())
    backend.ingest(FeatureRecord(ts_ns=10, view_name="v", entity_key="BTC", values={"vol": 1.0}))
    snap = backend.get_online_features(
        FeatureRequest(ts_ns=11, view_name="v", entity_key="BTC", feature_names=("vol",))
    )
    assert snap.values == {"vol": 1.0}
    backend.close()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Registry YAML sanity (S-09 spec line 639)
# ---------------------------------------------------------------------------


def test_registry_yaml_exists_and_pins_sqlite() -> None:
    yaml_path = Path(__file__).resolve().parents[1] / "registry" / "feast" / "feature_store.yaml"
    assert yaml_path.is_file()
    text = yaml_path.read_text(encoding="utf-8")
    # Pin the choices we adapted away from feast — provider must be local
    # and the only online_store type allowed is sqlite.
    assert re.search(r"^\s*provider:\s*local\b", text, re.MULTILINE)
    assert re.search(r"^\s*type:\s*sqlite\b", text, re.MULTILINE)


# ---------------------------------------------------------------------------
# Defensive guards
# ---------------------------------------------------------------------------


def test_feature_view_rejects_empty_entity_keys() -> None:
    with pytest.raises(ValueError):
        FeatureView(
            name="v",
            entity_keys=(),
            feature_names=("f",),
            ts_ns=1,
        )


def test_feature_record_rejects_empty_view_name() -> None:
    with pytest.raises(ValueError):
        FeatureRecord(ts_ns=1, view_name="", entity_key="BTC", values={"f": 1.0})


def test_feature_request_rejects_empty_view_name() -> None:
    with pytest.raises(ValueError):
        FeatureRequest(ts_ns=1, view_name="", entity_key="BTC", feature_names=("f",))


def test_feature_request_rejects_zero_ts_ns() -> None:
    with pytest.raises(ValueError):
        FeatureRequest(ts_ns=0, view_name="v", entity_key="BTC", feature_names=("f",))
