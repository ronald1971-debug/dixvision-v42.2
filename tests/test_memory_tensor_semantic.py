"""Tests for state.memory_tensor.semantic.SemanticMemoryStore (S-08.3)."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from state.memory_tensor.contracts import (
    Episode,
    MemoryHit,
    MemoryQuery,
    MemoryResult,
    MemoryStoreBase,
)
from state.memory_tensor.semantic import (
    NEW_PIP_DEPENDENCIES,
    SemanticMemoryStore,
)

_SEMANTIC_PATH = Path(__file__).resolve().parents[1] / "state" / "memory_tensor" / "semantic.py"

_FORBIDDEN_TOP_LEVEL_IMPORTS = frozenset(
    {
        "time",
        "datetime",
        "os",
        "asyncio",
        "threading",
        "subprocess",
        "socket",
        "logging",
        "numpy",
        "faiss",
        "qdrant_client",
        "random",
        "secrets",
    }
)


# ---------------------------------------------------------------------------
# Module metadata
# ---------------------------------------------------------------------------


def test_module_declares_no_new_pip_deps() -> None:
    assert NEW_PIP_DEPENDENCIES == ()


def test_adapted_from_header_present() -> None:
    text = _SEMANTIC_PATH.read_text(encoding="utf-8")
    assert "ADAPTED FROM: facebookresearch/faiss" in text


def test_no_forbidden_top_level_imports() -> None:
    tree = ast.parse(_SEMANTIC_PATH.read_text(encoding="utf-8"))
    seen: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                seen.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            seen.add(node.module.split(".")[0])
    assert _FORBIDDEN_TOP_LEVEL_IMPORTS.isdisjoint(seen), seen


def test_no_clock_substrings() -> None:
    text = _SEMANTIC_PATH.read_text(encoding="utf-8")
    for needle in (
        "time.time(",
        "time.monotonic(",
        "time.perf_counter(",
        "datetime.now(",
        "datetime.utcnow(",
    ):
        assert needle not in text


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_constructor_minimal() -> None:
    s = SemanticMemoryStore(dim=4, max_size=8)
    assert s.dim == 4
    assert s.max_size == 8
    assert s.nprobe == 1
    assert s.centroids == ()
    assert s.is_ivf is False
    assert len(s) == 0


def test_constructor_rejects_non_int_dim() -> None:
    with pytest.raises(TypeError):
        SemanticMemoryStore(dim=4.0, max_size=8)  # type: ignore[arg-type]


def test_constructor_rejects_zero_dim() -> None:
    with pytest.raises(ValueError):
        SemanticMemoryStore(dim=0, max_size=8)


def test_constructor_rejects_negative_dim() -> None:
    with pytest.raises(ValueError):
        SemanticMemoryStore(dim=-1, max_size=8)


def test_constructor_rejects_non_int_max_size() -> None:
    with pytest.raises(TypeError):
        SemanticMemoryStore(dim=4, max_size=8.0)  # type: ignore[arg-type]


def test_constructor_rejects_zero_max_size() -> None:
    with pytest.raises(ValueError):
        SemanticMemoryStore(dim=4, max_size=0)


def test_constructor_rejects_non_int_nprobe() -> None:
    with pytest.raises(TypeError):
        SemanticMemoryStore(
            dim=4,
            max_size=8,
            centroids=((1.0, 0.0, 0.0, 0.0),),
            nprobe=1.0,  # type: ignore[arg-type]
        )


def test_constructor_rejects_zero_nprobe() -> None:
    with pytest.raises(ValueError):
        SemanticMemoryStore(
            dim=4,
            max_size=8,
            centroids=((1.0, 0.0, 0.0, 0.0),),
            nprobe=0,
        )


def test_constructor_rejects_nprobe_above_centroid_count() -> None:
    with pytest.raises(ValueError):
        SemanticMemoryStore(
            dim=2,
            max_size=8,
            centroids=((1.0, 0.0), (0.0, 1.0)),
            nprobe=3,
        )


def test_constructor_rejects_centroid_dim_mismatch() -> None:
    with pytest.raises(ValueError):
        SemanticMemoryStore(
            dim=3,
            max_size=8,
            centroids=((1.0, 0.0),),  # dim=2 ≠ store dim=3
        )


def test_constructor_rejects_non_tuple_centroid_row() -> None:
    with pytest.raises(TypeError):
        SemanticMemoryStore(
            dim=2,
            max_size=8,
            centroids=([1.0, 0.0],),  # type: ignore[arg-type]
        )


def test_constructor_has_slots() -> None:
    assert hasattr(SemanticMemoryStore, "__slots__")
    assert "_dim" in SemanticMemoryStore.__slots__
    assert "_centroids" in SemanticMemoryStore.__slots__
    assert "_buckets" in SemanticMemoryStore.__slots__


def test_constructor_ivf_mode_recognised() -> None:
    s = SemanticMemoryStore(
        dim=2,
        max_size=8,
        centroids=((1.0, 0.0), (0.0, 1.0)),
        nprobe=1,
    )
    assert s.is_ivf is True
    assert s.centroids == ((1.0, 0.0), (0.0, 1.0))
    assert s.nprobe == 1


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------


def test_add_increments_len() -> None:
    s = SemanticMemoryStore(dim=2, max_size=8)
    s.add(Episode(ts_ns=1, episode_id="a", embedding=(1.0, 0.0)))
    assert len(s) == 1
    s.add(Episode(ts_ns=2, episode_id="b", embedding=(0.0, 1.0)))
    assert len(s) == 2


def test_add_rejects_non_episode() -> None:
    s = SemanticMemoryStore(dim=2, max_size=8)
    with pytest.raises(TypeError):
        s.add("not an episode")  # type: ignore[arg-type]


def test_add_rejects_dim_mismatch() -> None:
    s = SemanticMemoryStore(dim=3, max_size=8)
    with pytest.raises(ValueError):
        s.add(Episode(ts_ns=1, episode_id="a", embedding=(1.0, 0.0)))


def test_add_rejects_duplicate_id() -> None:
    s = SemanticMemoryStore(dim=2, max_size=8)
    s.add(Episode(ts_ns=1, episode_id="a", embedding=(1.0, 0.0)))
    with pytest.raises(ValueError):
        s.add(Episode(ts_ns=2, episode_id="a", embedding=(0.0, 1.0)))


def test_add_evicts_oldest_when_full() -> None:
    s = SemanticMemoryStore(dim=2, max_size=2)
    s.add(Episode(ts_ns=1, episode_id="a", embedding=(1.0, 0.0)))
    s.add(Episode(ts_ns=2, episode_id="b", embedding=(0.0, 1.0)))
    s.add(Episode(ts_ns=3, episode_id="c", embedding=(1.0, 1.0)))
    assert "a" not in s
    assert "b" in s
    assert "c" in s
    assert len(s) == 2


def test_add_eviction_breaks_ts_ties_by_episode_id() -> None:
    s = SemanticMemoryStore(dim=2, max_size=2)
    s.add(Episode(ts_ns=10, episode_id="z", embedding=(1.0, 0.0)))
    s.add(Episode(ts_ns=10, episode_id="m", embedding=(0.0, 1.0)))
    s.add(Episode(ts_ns=11, episode_id="n", embedding=(1.0, 1.0)))
    # both "z" and "m" have ts_ns=10; "m" < "z" so "m" is the oldest.
    assert "m" not in s
    assert "z" in s
    assert "n" in s


def test_add_does_not_evict_when_below_cap() -> None:
    s = SemanticMemoryStore(dim=2, max_size=4)
    for i in range(3):
        s.add(
            Episode(
                ts_ns=i + 1,
                episode_id=f"e{i}",
                embedding=(float(i + 1), 0.0),
            )
        )
    assert len(s) == 3
    assert {"e0", "e1", "e2"}.issubset({ep.episode_id for ep in s})


def test_add_buckets_episodes_in_ivf_mode() -> None:
    centroids = ((1.0, 0.0), (0.0, 1.0))
    s = SemanticMemoryStore(dim=2, max_size=8, centroids=centroids, nprobe=1)
    s.add(Episode(ts_ns=1, episode_id="x", embedding=(0.9, 0.1)))  # → c0
    s.add(Episode(ts_ns=2, episode_id="y", embedding=(0.1, 0.9)))  # → c1
    # private state pinned via _buckets — sole tier-private check;
    # exposes the IVF assignment determinism.
    assert s._buckets["x"] == 0
    assert s._buckets["y"] == 1


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


def test_delete_returns_true_for_known_id() -> None:
    s = SemanticMemoryStore(dim=2, max_size=8)
    s.add(Episode(ts_ns=1, episode_id="a", embedding=(1.0, 0.0)))
    assert s.delete("a") is True
    assert "a" not in s


def test_delete_returns_false_for_unknown_id() -> None:
    s = SemanticMemoryStore(dim=2, max_size=8)
    assert s.delete("missing") is False


def test_delete_clears_bucket_in_ivf_mode() -> None:
    centroids = ((1.0, 0.0), (0.0, 1.0))
    s = SemanticMemoryStore(dim=2, max_size=8, centroids=centroids, nprobe=1)
    s.add(Episode(ts_ns=1, episode_id="x", embedding=(0.9, 0.1)))
    assert s.delete("x") is True
    assert "x" not in s._buckets


# ---------------------------------------------------------------------------
# __contains__ / __iter__
# ---------------------------------------------------------------------------


def test_contains_rejects_non_str() -> None:
    s = SemanticMemoryStore(dim=2, max_size=8)
    assert (123 in s) is False
    assert (None in s) is False


def test_iter_is_sorted_by_ts_then_id() -> None:
    s = SemanticMemoryStore(dim=2, max_size=8)
    s.add(Episode(ts_ns=10, episode_id="z", embedding=(1.0, 0.0)))
    s.add(Episode(ts_ns=5, episode_id="b", embedding=(0.0, 1.0)))
    s.add(Episode(ts_ns=10, episode_id="a", embedding=(1.0, 1.0)))
    ids = [ep.episode_id for ep in s]
    assert ids == ["b", "a", "z"]


# ---------------------------------------------------------------------------
# search() — flat (no centroids)
# ---------------------------------------------------------------------------


def test_search_returns_zero_distance_for_exact_match() -> None:
    s = SemanticMemoryStore(dim=3, max_size=8)
    s.add(Episode(ts_ns=1, episode_id="exact", embedding=(0.5, -0.5, 1.0)))
    q = MemoryQuery(ts_ns=2, query_id="q", embedding=(0.5, -0.5, 1.0), k=1)
    res = s.search(q)
    assert isinstance(res, MemoryResult)
    assert len(res.hits) == 1
    assert res.hits[0].episode_id == "exact"
    assert res.hits[0].distance == pytest.approx(0.0)


def test_search_distance_for_orthogonal_pair() -> None:
    s = SemanticMemoryStore(dim=2, max_size=8)
    s.add(Episode(ts_ns=1, episode_id="a", embedding=(1.0, 0.0)))
    q = MemoryQuery(ts_ns=2, query_id="q", embedding=(0.0, 1.0), k=1)
    res = s.search(q)
    assert res.hits[0].distance == pytest.approx(1.0)  # cos = 0


def test_search_distance_for_opposing_pair() -> None:
    s = SemanticMemoryStore(dim=2, max_size=8)
    s.add(Episode(ts_ns=1, episode_id="a", embedding=(1.0, 0.0)))
    q = MemoryQuery(ts_ns=2, query_id="q", embedding=(-1.0, 0.0), k=1)
    res = s.search(q)
    assert res.hits[0].distance == pytest.approx(2.0)  # cos = -1


def test_search_returns_empty_for_empty_store() -> None:
    s = SemanticMemoryStore(dim=2, max_size=8)
    q = MemoryQuery(ts_ns=2, query_id="q", embedding=(1.0, 0.0), k=5)
    res = s.search(q)
    assert res.hits == ()


def test_search_rejects_non_query() -> None:
    s = SemanticMemoryStore(dim=2, max_size=8)
    with pytest.raises(TypeError):
        s.search("not a query")  # type: ignore[arg-type]


def test_search_rejects_dim_mismatch() -> None:
    s = SemanticMemoryStore(dim=3, max_size=8)
    q = MemoryQuery(ts_ns=2, query_id="q", embedding=(1.0, 0.0), k=1)
    with pytest.raises(ValueError):
        s.search(q)


def test_search_truncates_to_k() -> None:
    s = SemanticMemoryStore(dim=2, max_size=8)
    for i in range(5):
        s.add(
            Episode(
                ts_ns=i + 1,
                episode_id=f"e{i}",
                embedding=(math.cos(i * 0.1), math.sin(i * 0.1)),
            )
        )
    q = MemoryQuery(ts_ns=99, query_id="q", embedding=(1.0, 0.0), k=2)
    res = s.search(q)
    assert len(res.hits) == 2


def test_search_hits_sorted_ascending_by_distance() -> None:
    s = SemanticMemoryStore(dim=2, max_size=8)
    s.add(Episode(ts_ns=1, episode_id="a", embedding=(1.0, 0.0)))  # cos=1, d=0
    s.add(Episode(ts_ns=2, episode_id="b", embedding=(-1.0, 0.0)))  # cos=-1, d=2
    s.add(Episode(ts_ns=3, episode_id="c", embedding=(0.0, 1.0)))  # cos=0, d=1
    q = MemoryQuery(ts_ns=99, query_id="q", embedding=(1.0, 0.0), k=3)
    res = s.search(q)
    assert [h.episode_id for h in res.hits] == ["a", "c", "b"]
    assert all(res.hits[i].distance <= res.hits[i + 1].distance for i in range(len(res.hits) - 1))


def test_search_breaks_distance_ties_by_ts_then_episode_id() -> None:
    s = SemanticMemoryStore(dim=2, max_size=8)
    s.add(Episode(ts_ns=30, episode_id="z", embedding=(0.0, 1.0)))
    s.add(Episode(ts_ns=10, episode_id="a", embedding=(0.0, 1.0)))
    s.add(Episode(ts_ns=10, episode_id="m", embedding=(0.0, 1.0)))
    q = MemoryQuery(ts_ns=99, query_id="q", embedding=(1.0, 0.0), k=3)
    res = s.search(q)
    # All tied at distance 1.0; tie order = (ts_ns asc, episode_id asc)
    assert [h.episode_id for h in res.hits] == ["a", "m", "z"]


def test_search_returns_at_most_len_when_k_exceeds_size() -> None:
    s = SemanticMemoryStore(dim=2, max_size=8)
    s.add(Episode(ts_ns=1, episode_id="a", embedding=(1.0, 0.0)))
    s.add(Episode(ts_ns=2, episode_id="b", embedding=(0.0, 1.0)))
    q = MemoryQuery(ts_ns=99, query_id="q", embedding=(1.0, 1.0), k=10)
    res = s.search(q)
    assert len(res.hits) == 2


def test_search_carries_payload_and_ts() -> None:
    s = SemanticMemoryStore(dim=2, max_size=8)
    s.add(
        Episode(
            ts_ns=42,
            episode_id="e",
            embedding=(1.0, 0.0),
            payload={"k": "v"},
        )
    )
    q = MemoryQuery(ts_ns=99, query_id="q", embedding=(1.0, 0.0), k=1)
    res = s.search(q)
    h = res.hits[0]
    assert isinstance(h, MemoryHit)
    assert h.ts_ns == 42
    assert h.payload == {"k": "v"}


def test_search_zero_norm_query_yields_distance_one() -> None:
    s = SemanticMemoryStore(dim=2, max_size=8)
    s.add(Episode(ts_ns=1, episode_id="a", embedding=(1.0, 0.0)))
    q = MemoryQuery(ts_ns=2, query_id="q", embedding=(0.0, 0.0), k=1)
    res = s.search(q)
    # Zero-norm guard: cos undefined → d = 1.0
    assert res.hits[0].distance == pytest.approx(1.0)


def test_search_query_carries_query_id_and_ts() -> None:
    s = SemanticMemoryStore(dim=2, max_size=8)
    s.add(Episode(ts_ns=1, episode_id="a", embedding=(1.0, 0.0)))
    q = MemoryQuery(ts_ns=99, query_id="qid-7", embedding=(1.0, 0.0), k=1)
    res = s.search(q)
    assert res.ts_ns == 99
    assert res.query_id == "qid-7"


# ---------------------------------------------------------------------------
# search() — IVF (with centroids)
# ---------------------------------------------------------------------------


def _build_ivf_store() -> SemanticMemoryStore:
    centroids = ((1.0, 0.0), (0.0, 1.0))
    s = SemanticMemoryStore(dim=2, max_size=16, centroids=centroids, nprobe=1)
    # Cluster 0 (along x-axis)
    s.add(Episode(ts_ns=1, episode_id="x1", embedding=(1.0, 0.05)))
    s.add(Episode(ts_ns=2, episode_id="x2", embedding=(0.95, 0.0)))
    # Cluster 1 (along y-axis)
    s.add(Episode(ts_ns=3, episode_id="y1", embedding=(0.05, 1.0)))
    s.add(Episode(ts_ns=4, episode_id="y2", embedding=(0.0, 0.95)))
    return s


def test_ivf_search_only_visits_top_nprobe_buckets() -> None:
    s = _build_ivf_store()
    q = MemoryQuery(ts_ns=99, query_id="q", embedding=(1.0, 0.0), k=10)
    res = s.search(q)
    ids = {h.episode_id for h in res.hits}
    # nprobe=1, query is closest to centroid 0 → only x* visited.
    assert ids == {"x1", "x2"}


def test_ivf_search_visits_all_when_nprobe_equals_centroid_count() -> None:
    centroids = ((1.0, 0.0), (0.0, 1.0))
    s = SemanticMemoryStore(dim=2, max_size=16, centroids=centroids, nprobe=2)
    s.add(Episode(ts_ns=1, episode_id="x1", embedding=(1.0, 0.0)))
    s.add(Episode(ts_ns=2, episode_id="y1", embedding=(0.0, 1.0)))
    q = MemoryQuery(ts_ns=99, query_id="q", embedding=(1.0, 0.0), k=10)
    res = s.search(q)
    ids = {h.episode_id for h in res.hits}
    assert ids == {"x1", "y1"}


def test_ivf_search_returns_zero_distance_for_exact_match_inside_probe() -> None:
    s = _build_ivf_store()
    q = MemoryQuery(ts_ns=99, query_id="q", embedding=(1.0, 0.05), k=1)
    res = s.search(q)
    assert res.hits[0].episode_id == "x1"
    assert res.hits[0].distance == pytest.approx(0.0)


def test_ivf_search_empty_when_probed_buckets_are_empty() -> None:
    centroids = ((1.0, 0.0), (0.0, 1.0))
    s = SemanticMemoryStore(dim=2, max_size=16, centroids=centroids, nprobe=1)
    s.add(Episode(ts_ns=1, episode_id="y1", embedding=(0.0, 1.0)))  # → c1
    q = MemoryQuery(ts_ns=99, query_id="q", embedding=(1.0, 0.0), k=10)
    # Query closest to c0; c0 has no episodes; nprobe=1 → empty hits.
    res = s.search(q)
    assert res.hits == ()


# ---------------------------------------------------------------------------
# Replay determinism (INV-15)
# ---------------------------------------------------------------------------


def test_search_is_deterministic_across_runs() -> None:
    def build() -> SemanticMemoryStore:
        s = SemanticMemoryStore(dim=2, max_size=8)
        s.add(Episode(ts_ns=1, episode_id="a", embedding=(1.0, 0.0)))
        s.add(Episode(ts_ns=2, episode_id="b", embedding=(0.5, 0.5)))
        s.add(Episode(ts_ns=3, episode_id="c", embedding=(0.0, 1.0)))
        return s

    q = MemoryQuery(ts_ns=99, query_id="q", embedding=(1.0, 0.1), k=3)
    results = [build().search(q) for _ in range(3)]
    assert results[0] == results[1] == results[2]


def test_serialize_is_deterministic_across_runs() -> None:
    def build() -> SemanticMemoryStore:
        s = SemanticMemoryStore(dim=2, max_size=8)
        s.add(Episode(ts_ns=1, episode_id="a", embedding=(1.0, 0.0)))
        s.add(Episode(ts_ns=2, episode_id="b", embedding=(0.5, 0.5)))
        return s

    blobs = {build().serialize() for _ in range(3)}
    assert len(blobs) == 1


def test_ivf_search_is_deterministic_across_runs() -> None:
    results = [
        _build_ivf_store().search(MemoryQuery(ts_ns=99, query_id="q", embedding=(1.0, 0.1), k=2))
        for _ in range(3)
    ]
    assert results[0] == results[1] == results[2]


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


def _build_reference_store(*, ivf: bool = False) -> SemanticMemoryStore:
    if ivf:
        centroids: tuple[tuple[float, ...], ...] | None = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    else:
        centroids = None
    s = SemanticMemoryStore(dim=3, max_size=8, centroids=centroids, nprobe=1)
    s.add(
        Episode(
            ts_ns=10,
            episode_id="alpha",
            embedding=(1.0, 0.0, 0.0),
            payload={"src": "test"},
        )
    )
    s.add(
        Episode(
            ts_ns=20,
            episode_id="beta",
            embedding=(0.0, 1.0, 0.0),
            payload={"src": "test"},
        )
    )
    return s


def test_serialize_returns_bytes() -> None:
    blob = _build_reference_store().serialize()
    assert isinstance(blob, bytes)


def test_round_trip_is_byte_equal_flat() -> None:
    s1 = _build_reference_store()
    blob = s1.serialize()
    s2 = SemanticMemoryStore.deserialize(blob)
    assert s2.serialize() == blob


def test_round_trip_is_byte_equal_ivf() -> None:
    s1 = _build_reference_store(ivf=True)
    blob = s1.serialize()
    s2 = SemanticMemoryStore.deserialize(blob)
    assert s2.serialize() == blob


def test_round_trip_preserves_search_flat() -> None:
    s1 = _build_reference_store()
    s2 = SemanticMemoryStore.deserialize(s1.serialize())
    q = MemoryQuery(ts_ns=99, query_id="q", embedding=(1.0, 0.0, 0.0), k=2)
    assert s1.search(q) == s2.search(q)


def test_round_trip_preserves_search_ivf() -> None:
    s1 = _build_reference_store(ivf=True)
    s2 = SemanticMemoryStore.deserialize(s1.serialize())
    q = MemoryQuery(ts_ns=99, query_id="q", embedding=(1.0, 0.0, 0.0), k=2)
    assert s1.search(q) == s2.search(q)


def test_round_trip_preserves_dim_and_cap() -> None:
    s2 = SemanticMemoryStore.deserialize(_build_reference_store().serialize())
    assert s2.dim == 3
    assert s2.max_size == 8
    assert len(s2) == 2


def test_round_trip_preserves_centroids_and_nprobe() -> None:
    s1 = _build_reference_store(ivf=True)
    s2 = SemanticMemoryStore.deserialize(s1.serialize())
    assert s2.centroids == s1.centroids
    assert s2.nprobe == s1.nprobe
    assert s2.is_ivf is True


def test_round_trip_preserves_payload() -> None:
    s2 = SemanticMemoryStore.deserialize(_build_reference_store().serialize())
    payloads = {ep.episode_id: dict(ep.payload) for ep in s2}
    assert payloads == {
        "alpha": {"src": "test"},
        "beta": {"src": "test"},
    }


# ---------------------------------------------------------------------------
# Serialization defensive cases
# ---------------------------------------------------------------------------


def test_deserialize_rejects_non_bytes() -> None:
    with pytest.raises(TypeError):
        SemanticMemoryStore.deserialize("not bytes")  # type: ignore[arg-type]


def test_deserialize_rejects_invalid_json() -> None:
    with pytest.raises(ValueError):
        SemanticMemoryStore.deserialize(b"this is not json")


def test_deserialize_rejects_non_object_top_level() -> None:
    with pytest.raises(ValueError):
        SemanticMemoryStore.deserialize(b"[]")


def test_deserialize_rejects_unknown_version() -> None:
    blob = b'{"version": 99, "dim": 2, "max_size": 8, "nprobe": 1, "centroids": [], "episodes": []}'
    with pytest.raises(ValueError):
        SemanticMemoryStore.deserialize(blob)


def test_deserialize_rejects_bad_dim() -> None:
    blob = (
        b'{"version": 1, "dim": "x", "max_size": 8, "nprobe": 1, "centroids": [], "episodes": []}'
    )
    with pytest.raises(ValueError):
        SemanticMemoryStore.deserialize(blob)


def test_deserialize_rejects_bad_max_size() -> None:
    blob = (
        b'{"version": 1, "dim": 2, "max_size": "x", "nprobe": 1, "centroids": [], "episodes": []}'
    )
    with pytest.raises(ValueError):
        SemanticMemoryStore.deserialize(blob)


def test_deserialize_rejects_bad_nprobe() -> None:
    blob = (
        b'{"version": 1, "dim": 2, "max_size": 8, "nprobe": "x", "centroids": [], "episodes": []}'
    )
    with pytest.raises(ValueError):
        SemanticMemoryStore.deserialize(blob)


def test_deserialize_rejects_bad_centroids_type() -> None:
    blob = (
        b'{"version": 1, "dim": 2, "max_size": 8, "nprobe": 1, "centroids": "no", "episodes": []}'
    )
    with pytest.raises(ValueError):
        SemanticMemoryStore.deserialize(blob)


def test_deserialize_rejects_bad_episodes_type() -> None:
    blob = (
        b'{"version": 1, "dim": 2, "max_size": 8, "nprobe": 1, "centroids": [], "episodes": "no"}'
    )
    with pytest.raises(ValueError):
        SemanticMemoryStore.deserialize(blob)


def test_deserialize_rejects_bad_episode_row() -> None:
    blob = (
        b'{"version": 1, "dim": 2, "max_size": 8, "nprobe": 1, '
        b'"centroids": [], "episodes": ["bad"]}'
    )
    with pytest.raises(ValueError):
        SemanticMemoryStore.deserialize(blob)


def test_deserialize_rejects_nan_embedding() -> None:
    blob = (
        b'{"version": 1, "dim": 2, "max_size": 8, "nprobe": 1, '
        b'"centroids": [], "episodes": [{"ts_ns": 1, "episode_id": "a", '
        b'"embedding": [NaN, 0.0], "payload": {}}]}'
    )
    with pytest.raises(ValueError):
        SemanticMemoryStore.deserialize(blob)


def test_deserialize_rejects_non_object_payload() -> None:
    blob = (
        b'{"version": 1, "dim": 2, "max_size": 8, "nprobe": 1, '
        b'"centroids": [], "episodes": [{"ts_ns": 1, "episode_id": "a", '
        b'"embedding": [1.0, 0.0], "payload": "no"}]}'
    )
    with pytest.raises(ValueError):
        SemanticMemoryStore.deserialize(blob)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_memory_store_protocol() -> None:
    s = SemanticMemoryStore(dim=2, max_size=8)
    assert isinstance(s, MemoryStoreBase)


def test_protocol_round_trip_works_through_facade() -> None:
    store: MemoryStoreBase = SemanticMemoryStore(dim=2, max_size=8)
    store.add(Episode(ts_ns=1, episode_id="a", embedding=(1.0, 0.0)))
    q = MemoryQuery(ts_ns=2, query_id="q", embedding=(1.0, 0.0), k=1)
    res = store.search(q)
    assert res.hits[0].episode_id == "a"
