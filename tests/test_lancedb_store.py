"""Tests for C-24 lancedb embedded zero-config vector store."""

from __future__ import annotations

import ast
import pathlib

import pytest

from state.memory_tensor.contracts import (
    Episode,
    MemoryHit,
    MemoryQuery,
    MemoryResult,
    MemoryStoreBase,
)
from state.memory_tensor.lancedb_store import (
    LANCEDB_ADAPTER_VERSION,
    NEW_PIP_DEPENDENCIES,
    IndexParams,
    IndexType,
    LanceDBStore,
    LanceDBStoreError,
    MetricType,
    WhereClause,
    lancedb_connect,
)

_THIS_MODULE = pathlib.Path("state/memory_tensor/lancedb_store.py").resolve()


# ---------------------------------------------------------------------------
# Module identity
# ---------------------------------------------------------------------------
def test_new_pip_dependencies():
    assert NEW_PIP_DEPENDENCIES == ("lancedb",)


def test_version():
    assert LANCEDB_ADAPTER_VERSION == "1"


def test_metric_type_values():
    assert MetricType.L2.value == "l2"
    assert MetricType.COSINE.value == "cosine"
    assert MetricType.DOT.value == "dot"


def test_index_type_values():
    assert IndexType.IVF_PQ.value == "IVF_PQ"
    assert IndexType.BTREE.value == "BTREE"
    assert IndexType.BITMAP.value == "BITMAP"


# ---------------------------------------------------------------------------
# WhereClause primitives
# ---------------------------------------------------------------------------
def test_where_equals_match():
    w = WhereClause(equals={"strategy": "alpha"})
    assert w.matches({"strategy": "alpha"})
    assert not w.matches({"strategy": "beta"})


def test_where_not_equals_match():
    w = WhereClause(not_equals={"strategy": "alpha"})
    assert not w.matches({"strategy": "alpha"})
    assert w.matches({"strategy": "beta"})


def test_where_combined():
    w = WhereClause(
        equals={"asset": "BTC"},
        not_equals={"banned": "yes"},
    )
    assert w.matches({"asset": "BTC"})
    assert not w.matches({"asset": "BTC", "banned": "yes"})


def test_where_keys_sorted():
    w = WhereClause(equals={"z": "1", "a": "2"})
    assert w.equals == (("a", "2"), ("z", "1"))


def test_where_rejects_non_mapping():
    with pytest.raises(TypeError):
        WhereClause(equals=[("a", "b")])  # type: ignore[arg-type]


def test_where_rejects_non_str_keys():
    with pytest.raises(TypeError):
        WhereClause(equals={1: "x"})  # type: ignore[dict-item]


def test_where_rejects_non_str_values():
    with pytest.raises(TypeError):
        WhereClause(equals={"a": 1})  # type: ignore[dict-item]


def test_where_equality_and_hash():
    w1 = WhereClause(equals={"a": "b"})
    w2 = WhereClause(equals={"a": "b"})
    assert w1 == w2
    assert hash(w1) == hash(w2)


def test_where_repr():
    w = WhereClause(equals={"a": "b"})
    assert "WhereClause(" in repr(w)


# ---------------------------------------------------------------------------
# IndexParams primitives
# ---------------------------------------------------------------------------
def test_index_params_construction():
    p = IndexParams(
        index_type=IndexType.IVF_PQ,
        metric_type=MetricType.COSINE,
        params={"num_partitions": 256, "num_sub_vectors": 16},
    )
    assert p.index_type is IndexType.IVF_PQ
    assert ("num_partitions", 256) in p.params
    assert ("num_sub_vectors", 16) in p.params


def test_index_params_default_params_empty():
    p = IndexParams(
        index_type=IndexType.BTREE,
        metric_type=MetricType.L2,
    )
    assert p.params == ()


def test_index_params_rejects_non_enum():
    with pytest.raises(TypeError):
        IndexParams(
            index_type="IVF_PQ",  # type: ignore[arg-type]
            metric_type=MetricType.COSINE,
        )
    with pytest.raises(TypeError):
        IndexParams(
            index_type=IndexType.IVF_PQ,
            metric_type="cosine",  # type: ignore[arg-type]
        )


def test_index_params_rejects_non_positive_value():
    with pytest.raises(ValueError):
        IndexParams(
            index_type=IndexType.IVF_PQ,
            metric_type=MetricType.COSINE,
            params={"num_partitions": 0},
        )


# ---------------------------------------------------------------------------
# Store construction
# ---------------------------------------------------------------------------
def test_store_construction_defaults():
    store = LanceDBStore(dim=4, max_size=10)
    assert store.dim == 4
    assert store.max_size == 10
    assert store.metric_type is MetricType.COSINE
    assert store.table_name == "dix_lancedb"
    assert store.index_params is None
    assert len(store) == 0


def test_store_rejects_zero_dim():
    with pytest.raises(ValueError):
        LanceDBStore(dim=0, max_size=10)


def test_store_rejects_negative_max_size():
    with pytest.raises(ValueError):
        LanceDBStore(dim=4, max_size=-1)


def test_store_rejects_non_metric():
    with pytest.raises(TypeError):
        LanceDBStore(dim=4, max_size=10, metric_type="l2")  # type: ignore[arg-type]


def test_store_rejects_empty_table_name():
    with pytest.raises(ValueError):
        LanceDBStore(dim=4, max_size=10, table_name="")


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------
def test_store_satisfies_protocol():
    store = LanceDBStore(dim=2, max_size=4)
    assert isinstance(store, MemoryStoreBase)


# ---------------------------------------------------------------------------
# Add / contains / iter
# ---------------------------------------------------------------------------
def _ep(eid: str, vec: tuple[float, ...], ts: int = 1, **payload: str) -> Episode:
    return Episode(
        ts_ns=ts,
        episode_id=eid,
        embedding=vec,
        payload=payload,
    )


def test_add_and_contains():
    store = LanceDBStore(dim=2, max_size=4)
    store.add(_ep("a", (1.0, 0.0)))
    assert len(store) == 1
    assert "a" in store
    assert "b" not in store


def test_add_rejects_non_episode():
    store = LanceDBStore(dim=2, max_size=4)
    with pytest.raises(TypeError):
        store.add("not-an-episode")  # type: ignore[arg-type]


def test_add_rejects_wrong_dim():
    store = LanceDBStore(dim=2, max_size=4)
    e = Episode(ts_ns=1, episode_id="a", embedding=(1.0, 0.0, 0.0), payload={})
    with pytest.raises(ValueError):
        store.add(e)


def test_add_rejects_duplicate_id():
    store = LanceDBStore(dim=2, max_size=4)
    store.add(_ep("a", (1.0, 0.0)))
    with pytest.raises(ValueError):
        store.add(_ep("a", (0.0, 1.0)))


def test_iter_sorted_by_episode_id():
    store = LanceDBStore(dim=2, max_size=4)
    store.add(_ep("z", (1.0, 0.0)))
    store.add(_ep("a", (0.0, 1.0)))
    ids = [e.episode_id for e in store]
    assert ids == ["a", "z"]


def test_contains_non_str_returns_false():
    store = LanceDBStore(dim=2, max_size=4)
    assert (123 in store) is False  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------
def test_evict_oldest_by_ts():
    store = LanceDBStore(dim=2, max_size=2)
    store.add(_ep("a", (1.0, 0.0), ts=10))
    store.add(_ep("b", (0.0, 1.0), ts=20))
    store.add(_ep("c", (1.0, 1.0), ts=30))
    assert "a" not in store
    assert "b" in store
    assert "c" in store


def test_evict_tie_breaks_by_episode_id():
    store = LanceDBStore(dim=2, max_size=2)
    store.add(_ep("z", (1.0, 0.0), ts=10))
    store.add(_ep("a", (0.0, 1.0), ts=10))
    store.add(_ep("m", (1.0, 1.0), ts=20))
    assert "a" not in store
    assert "z" in store
    assert "m" in store


# ---------------------------------------------------------------------------
# Insert / upsert / delete / to_list
# ---------------------------------------------------------------------------
def test_insert_batch():
    store = LanceDBStore(dim=2, max_size=4)
    store.insert([_ep("a", (1.0, 0.0)), _ep("b", (0.0, 1.0))])
    assert len(store) == 2


def test_insert_raises_on_duplicate():
    store = LanceDBStore(dim=2, max_size=4)
    store.add(_ep("a", (1.0, 0.0)))
    with pytest.raises(ValueError):
        store.insert([_ep("a", (0.0, 1.0))])


def test_delete_returns_true_when_present():
    store = LanceDBStore(dim=2, max_size=4)
    store.add(_ep("a", (1.0, 0.0)))
    assert store.delete("a") is True
    assert "a" not in store


def test_delete_returns_false_when_absent():
    store = LanceDBStore(dim=2, max_size=4)
    assert store.delete("a") is False


def test_delete_rejects_non_str():
    store = LanceDBStore(dim=2, max_size=4)
    with pytest.raises(TypeError):
        store.delete(123)  # type: ignore[arg-type]


def test_upsert_overwrites_existing():
    store = LanceDBStore(dim=2, max_size=4)
    store.add(_ep("a", (1.0, 0.0), ts=10))
    store.upsert([_ep("a", (0.0, 1.0), ts=20)])
    assert len(store) == 1
    hit = next(iter(store))
    assert hit.embedding == (0.0, 1.0)


def test_to_list_unfiltered():
    store = LanceDBStore(dim=2, max_size=4)
    store.add(_ep("a", (1.0, 0.0), ts=10))
    store.add(_ep("b", (0.0, 1.0), ts=20))
    result = store.to_list()
    assert [e.episode_id for e in result] == ["a", "b"]


def test_to_list_filtered():
    store = LanceDBStore(dim=2, max_size=4)
    store.add(_ep("a", (1.0, 0.0), strategy="alpha"))
    store.add(_ep("b", (1.0, 0.0), strategy="beta"))
    result = store.to_list(
        where=WhereClause(equals={"strategy": "alpha"}),
    )
    assert [e.episode_id for e in result] == ["a"]


def test_to_list_rejects_non_where():
    store = LanceDBStore(dim=2, max_size=4)
    with pytest.raises(TypeError):
        store.to_list(where="strategy = alpha")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Create index (audit-only)
# ---------------------------------------------------------------------------
def test_create_index_records_params():
    store = LanceDBStore(dim=2, max_size=4)
    params = IndexParams(
        index_type=IndexType.IVF_PQ,
        metric_type=MetricType.COSINE,
        params={"num_partitions": 256, "num_sub_vectors": 16},
    )
    store.create_index(params)
    assert store.index_params == params


def test_create_index_rejects_non_index_params():
    store = LanceDBStore(dim=2, max_size=4)
    with pytest.raises(TypeError):
        store.create_index({"index_type": "IVF_PQ"})  # type: ignore[arg-type]


def test_create_index_rejects_metric_mismatch():
    store = LanceDBStore(dim=2, max_size=4, metric_type=MetricType.COSINE)
    params = IndexParams(
        index_type=IndexType.IVF_PQ,
        metric_type=MetricType.L2,
    )
    with pytest.raises(ValueError):
        store.create_index(params)


# ---------------------------------------------------------------------------
# Cosine search
# ---------------------------------------------------------------------------
def test_search_cosine_identical_returns_zero_distance():
    store = LanceDBStore(dim=2, max_size=4)
    store.add(_ep("a", (1.0, 0.0)))
    q = MemoryQuery(ts_ns=100, query_id="q1", embedding=(1.0, 0.0), k=1)
    result = store.search(q)
    assert result.hits[0].distance == pytest.approx(0.0)


def test_search_cosine_orthogonal_returns_one():
    store = LanceDBStore(dim=2, max_size=4)
    store.add(_ep("a", (1.0, 0.0)))
    q = MemoryQuery(ts_ns=100, query_id="q1", embedding=(0.0, 1.0), k=1)
    result = store.search(q)
    assert result.hits[0].distance == pytest.approx(1.0)


def test_search_cosine_zero_norm_falls_back():
    store = LanceDBStore(dim=2, max_size=4)
    store.add(_ep("a", (0.0, 0.0)))
    q = MemoryQuery(ts_ns=100, query_id="q1", embedding=(1.0, 0.0), k=1)
    result = store.search(q)
    assert result.hits[0].distance == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# L2 search
# ---------------------------------------------------------------------------
def test_search_l2_zero_when_identical():
    store = LanceDBStore(dim=2, max_size=4, metric_type=MetricType.L2)
    store.add(_ep("a", (3.0, 4.0)))
    q = MemoryQuery(ts_ns=100, query_id="q1", embedding=(3.0, 4.0), k=1)
    result = store.search(q)
    assert result.hits[0].distance == pytest.approx(0.0)


def test_search_l2_pythagoras():
    store = LanceDBStore(dim=2, max_size=4, metric_type=MetricType.L2)
    store.add(_ep("a", (0.0, 0.0)))
    q = MemoryQuery(ts_ns=100, query_id="q1", embedding=(3.0, 4.0), k=1)
    result = store.search(q)
    assert result.hits[0].distance == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# DOT search
# ---------------------------------------------------------------------------
def test_search_dot_non_negative():
    store = LanceDBStore(dim=2, max_size=4, metric_type=MetricType.DOT)
    store.add(_ep("a", (1.0, 0.0)))
    store.add(_ep("b", (0.5, 0.5)))
    q = MemoryQuery(ts_ns=100, query_id="q1", embedding=(1.0, 0.0), k=2)
    result = store.search(q)
    for hit in result.hits:
        assert hit.distance >= 0.0


def test_search_dot_picks_highest_ip():
    store = LanceDBStore(dim=2, max_size=4, metric_type=MetricType.DOT)
    store.add(_ep("a", (1.0, 0.0)))
    store.add(_ep("b", (0.5, 0.5)))
    q = MemoryQuery(ts_ns=100, query_id="q1", embedding=(1.0, 0.0), k=2)
    result = store.search(q)
    assert result.hits[0].episode_id == "a"


# ---------------------------------------------------------------------------
# Filtered search
# ---------------------------------------------------------------------------
def test_filtered_search_equals_filters_out():
    store = LanceDBStore(dim=2, max_size=4)
    store.add(_ep("a", (1.0, 0.0), strategy="alpha"))
    store.add(_ep("b", (1.0, 0.0), strategy="beta"))
    q = MemoryQuery(ts_ns=100, query_id="q1", embedding=(1.0, 0.0), k=2)
    result = store.search_with_filter(q, where=WhereClause(equals={"strategy": "alpha"}))
    assert [h.episode_id for h in result.hits] == ["a"]


def test_filtered_search_returns_empty_when_no_match():
    store = LanceDBStore(dim=2, max_size=4)
    store.add(_ep("a", (1.0, 0.0), strategy="alpha"))
    q = MemoryQuery(ts_ns=100, query_id="q1", embedding=(1.0, 0.0), k=2)
    result = store.search_with_filter(q, where=WhereClause(equals={"strategy": "missing"}))
    assert result.hits == ()


def test_search_filter_none_returns_all():
    store = LanceDBStore(dim=2, max_size=4)
    store.add(_ep("a", (1.0, 0.0), strategy="alpha"))
    store.add(_ep("b", (0.0, 1.0), strategy="beta"))
    q = MemoryQuery(ts_ns=100, query_id="q1", embedding=(1.0, 0.0), k=2)
    result = store.search_with_filter(q, where=None)
    assert len(result.hits) == 2


def test_search_rejects_non_query():
    store = LanceDBStore(dim=2, max_size=4)
    with pytest.raises(TypeError):
        store.search("not-a-query")  # type: ignore[arg-type]


def test_search_rejects_dim_mismatch():
    store = LanceDBStore(dim=2, max_size=4)
    q = MemoryQuery(ts_ns=100, query_id="q1", embedding=(1.0, 0.0, 0.0), k=1)
    with pytest.raises(ValueError):
        store.search(q)


def test_search_rejects_non_where():
    store = LanceDBStore(dim=2, max_size=4)
    q = MemoryQuery(ts_ns=100, query_id="q1", embedding=(1.0, 0.0), k=1)
    with pytest.raises(TypeError):
        store.search_with_filter(q, where="strategy = alpha")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Hit ordering / k limit
# ---------------------------------------------------------------------------
def test_search_hits_sorted_by_distance_asc():
    store = LanceDBStore(dim=2, max_size=4)
    store.add(_ep("a", (1.0, 0.0)))
    store.add(_ep("b", (0.5, 0.5)))
    store.add(_ep("c", (0.0, 1.0)))
    q = MemoryQuery(ts_ns=100, query_id="q1", embedding=(1.0, 0.0), k=3)
    result = store.search(q)
    distances = [h.distance for h in result.hits]
    assert distances == sorted(distances)


def test_search_k_limits():
    store = LanceDBStore(dim=2, max_size=4)
    for i, eid in enumerate(["a", "b", "c", "d"]):
        store.add(_ep(eid, (1.0 - i * 0.1, 0.0), ts=10 + i))
    q = MemoryQuery(ts_ns=100, query_id="q1", embedding=(1.0, 0.0), k=2)
    result = store.search(q)
    assert len(result.hits) == 2


def test_search_tie_breaks_by_ts_then_episode_id():
    store = LanceDBStore(dim=2, max_size=4)
    store.add(_ep("z", (1.0, 0.0), ts=10))
    store.add(_ep("a", (1.0, 0.0), ts=10))
    store.add(_ep("m", (1.0, 0.0), ts=20))
    q = MemoryQuery(ts_ns=100, query_id="q1", embedding=(1.0, 0.0), k=3)
    result = store.search(q)
    assert [h.episode_id for h in result.hits] == ["a", "z", "m"]


def test_search_result_echoes_ts_and_query_id():
    store = LanceDBStore(dim=2, max_size=4)
    store.add(_ep("a", (1.0, 0.0)))
    q = MemoryQuery(ts_ns=12345, query_id="qx", embedding=(1.0, 0.0), k=1)
    result = store.search(q)
    assert result.ts_ns == 12345
    assert result.query_id == "qx"


def test_search_result_type():
    store = LanceDBStore(dim=2, max_size=4)
    store.add(_ep("a", (1.0, 0.0)))
    q = MemoryQuery(ts_ns=100, query_id="q1", embedding=(1.0, 0.0), k=1)
    result = store.search(q)
    assert isinstance(result, MemoryResult)
    assert all(isinstance(h, MemoryHit) for h in result.hits)


def test_search_empty_store_returns_no_hits():
    store = LanceDBStore(dim=2, max_size=4)
    q = MemoryQuery(ts_ns=100, query_id="q1", embedding=(1.0, 0.0), k=3)
    result = store.search(q)
    assert result.hits == ()


# ---------------------------------------------------------------------------
# Serialise / Deserialise
# ---------------------------------------------------------------------------
def test_serialize_byte_stable():
    store = LanceDBStore(dim=2, max_size=4)
    store.add(_ep("a", (1.0, 0.0), ts=10, strategy="alpha"))
    store.add(_ep("b", (0.0, 1.0), ts=20))
    blob1 = store.serialize()
    blob2 = store.serialize()
    assert blob1 == blob2


def test_serialize_independent_of_insert_order():
    s1 = LanceDBStore(dim=2, max_size=4)
    s1.add(_ep("a", (1.0, 0.0), ts=10))
    s1.add(_ep("b", (0.0, 1.0), ts=20))
    s2 = LanceDBStore(dim=2, max_size=4)
    s2.add(_ep("b", (0.0, 1.0), ts=20))
    s2.add(_ep("a", (1.0, 0.0), ts=10))
    assert s1.serialize() == s2.serialize()


def test_deserialize_round_trip():
    store = LanceDBStore(dim=2, max_size=4)
    store.add(_ep("a", (1.0, 0.0), ts=10, strategy="alpha"))
    store.add(_ep("b", (0.0, 1.0), ts=20))
    blob = store.serialize()
    restored = LanceDBStore.deserialize(blob)
    assert restored.dim == 2
    assert restored.max_size == 4
    assert len(restored) == 2
    assert "a" in restored
    assert restored.serialize() == blob


def test_deserialize_round_trip_with_index():
    store = LanceDBStore(dim=2, max_size=4)
    store.add(_ep("a", (1.0, 0.0), ts=10))
    store.create_index(
        IndexParams(
            index_type=IndexType.IVF_PQ,
            metric_type=MetricType.COSINE,
            params={"num_partitions": 256},
        )
    )
    blob = store.serialize()
    restored = LanceDBStore.deserialize(blob)
    assert restored.index_params is not None
    assert restored.index_params.index_type is IndexType.IVF_PQ
    assert restored.serialize() == blob


def test_deserialize_rejects_non_bytes():
    with pytest.raises(TypeError):
        LanceDBStore.deserialize("not-bytes")  # type: ignore[arg-type]


def test_deserialize_rejects_corrupt():
    with pytest.raises(LanceDBStoreError):
        LanceDBStore.deserialize(b"{not json")


def test_deserialize_rejects_wrong_version():
    blob = (
        b'{"version":999,"table":"x","dim":2,'
        b'"max_size":4,"metric":"cosine","dot_offset":0.0,'
        b'"index":null,"episodes":[]}'
    )
    with pytest.raises(LanceDBStoreError):
        LanceDBStore.deserialize(blob)


def test_deserialize_rejects_bad_metric():
    blob = (
        b'{"version":1,"table":"x","dim":2,'
        b'"max_size":4,"metric":"bogus","dot_offset":0.0,'
        b'"index":null,"episodes":[]}'
    )
    with pytest.raises(LanceDBStoreError):
        LanceDBStore.deserialize(blob)


# ---------------------------------------------------------------------------
# INV-15 byte-identical 3-run replay
# ---------------------------------------------------------------------------
def test_replay_byte_identical_three_runs():
    def run() -> bytes:
        store = LanceDBStore(dim=3, max_size=8)
        store.add(_ep("a", (1.0, 0.0, 0.0), ts=10))
        store.add(_ep("b", (0.0, 1.0, 0.0), ts=20))
        store.add(_ep("c", (0.0, 0.0, 1.0), ts=30, strategy="alpha"))
        store.create_index(
            IndexParams(
                index_type=IndexType.IVF_PQ,
                metric_type=MetricType.COSINE,
                params={"num_partitions": 256},
            )
        )
        q = MemoryQuery(ts_ns=100, query_id="q1", embedding=(0.5, 0.5, 0.0), k=3)
        result = store.search(q)
        return store.serialize() + b"||" + repr(result.hits).encode("ascii")

    b1 = run()
    b2 = run()
    b3 = run()
    assert b1 == b2 == b3


def test_replay_byte_identical_filtered():
    def run() -> bytes:
        store = LanceDBStore(dim=2, max_size=4)
        store.add(_ep("a", (1.0, 0.0), ts=10, strategy="alpha"))
        store.add(_ep("b", (0.0, 1.0), ts=20, strategy="beta"))
        q = MemoryQuery(ts_ns=100, query_id="q1", embedding=(1.0, 0.0), k=2)
        result = store.search_with_filter(q, where=WhereClause(equals={"strategy": "alpha"}))
        return repr(result.hits).encode("ascii")

    assert run() == run() == run()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def test_factory_rejects_empty_uri():
    with pytest.raises(ValueError):
        lancedb_connect(uri="")


def test_factory_raises_when_dep_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "lancedb" or name.startswith("lancedb."):
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(LanceDBStoreError):
        lancedb_connect(uri="/tmp/db")


# ---------------------------------------------------------------------------
# AST guards
# ---------------------------------------------------------------------------
def _parse() -> ast.Module:
    return ast.parse(_THIS_MODULE.read_text(encoding="utf-8"))


def _top_level_imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_no_top_level_lancedb_import():
    assert "lancedb" not in _top_level_imports(_parse())


def test_no_top_level_clock_or_random_imports():
    forbidden = {"time", "datetime", "random", "secrets", "os", "asyncio"}
    assert not (forbidden & _top_level_imports(_parse()))


def test_no_top_level_numerical_lib_imports():
    forbidden = {"numpy", "torch", "polars", "pandas", "scipy"}
    assert not (forbidden & _top_level_imports(_parse()))


def test_no_engine_cross_imports():
    tree = _parse()
    forbidden = {
        "governance_engine",
        "execution_engine",
        "evolution_engine",
        "intelligence_engine",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            head = node.module.split(".")[0]
            assert head not in forbidden, f"forbidden import: {node.module}"


def test_no_typed_event_constructions():
    tree = _parse()
    forbidden_names = {
        "SignalEvent",
        "ExecutionEvent",
        "SystemEvent",
        "HazardEvent",
        "GovernanceDecision",
        "PatchProposal",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in forbidden_names:
                pytest.fail(f"forbidden constructor call: {func.id}")
            if isinstance(func, ast.Attribute) and func.attr in forbidden_names:
                pytest.fail(f"forbidden constructor call: {func.attr}")


def test_adapted_from_header_present():
    text = _THIS_MODULE.read_text(encoding="utf-8")
    assert "# ADAPTED FROM: lancedb/lancedb" in text


def test_lancedb_import_confined_to_factory():
    tree = _parse()
    factory_fn: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "lancedb_connect":
            factory_fn = node
            break
    assert factory_fn is not None
    factory_imports: set[str] = set()
    for child in ast.walk(factory_fn):
        if isinstance(child, ast.ImportFrom) and child.module:
            factory_imports.add(child.module.split(".")[0])
        elif isinstance(child, ast.Import):
            for alias in child.names:
                factory_imports.add(alias.name.split(".")[0])
    assert "lancedb" in factory_imports
