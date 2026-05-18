"""Tests for state.memory_tensor.episodic.EpisodicMemoryStore (S-08.2)."""

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
from state.memory_tensor.episodic import (
    NEW_PIP_DEPENDENCIES,
    EpisodicMemoryStore,
)

_EPISODIC_PATH = Path(__file__).resolve().parents[1] / "state" / "memory_tensor" / "episodic.py"

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
        "random",  # PRNG state would break replay determinism
        "secrets",
    }
)


def _module_ast() -> ast.Module:
    return ast.parse(_EPISODIC_PATH.read_text())


def _imported_modules(tree: ast.Module) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module.split(".", 1)[0])
    return out


# ---------------------------------------------------------------------------
# Module-level metadata
# ---------------------------------------------------------------------------


def test_no_new_pip_dependencies() -> None:
    assert NEW_PIP_DEPENDENCIES == ()


def test_episodic_has_adapted_from_header() -> None:
    src = _EPISODIC_PATH.read_text()
    assert src.startswith("# ADAPTED FROM: facebookresearch/faiss"), (
        "S-08 spec requires '# ADAPTED FROM: facebookresearch/faiss Python interface' header"
    )


def test_episodic_does_not_import_forbidden_modules() -> None:
    imports = _imported_modules(_module_ast())
    bad = imports & _FORBIDDEN_TOP_LEVEL_IMPORTS
    assert not bad, f"episodic.py must not import {bad}"


def test_episodic_has_no_clock_calls() -> None:
    src = _EPISODIC_PATH.read_text()
    for needle in (
        "time.time(",
        "time.monotonic(",
        "time.perf_counter(",
        "datetime.now(",
        "datetime.utcnow(",
    ):
        assert needle not in src, f"episodic.py must not call {needle!r} (INV-15)"


# ---------------------------------------------------------------------------
# Constructor / property invariants
# ---------------------------------------------------------------------------


def test_ctor_minimal() -> None:
    s = EpisodicMemoryStore(dim=4, max_size=10)
    assert s.dim == 4
    assert s.max_size == 10
    assert len(s) == 0


def test_ctor_rejects_zero_dim() -> None:
    with pytest.raises(ValueError, match="dim must be positive"):
        EpisodicMemoryStore(dim=0, max_size=10)


def test_ctor_rejects_negative_dim() -> None:
    with pytest.raises(ValueError, match="dim must be positive"):
        EpisodicMemoryStore(dim=-1, max_size=10)


def test_ctor_rejects_non_int_dim() -> None:
    with pytest.raises(TypeError, match="dim must be int"):
        EpisodicMemoryStore(dim=4.0, max_size=10)  # type: ignore[arg-type]


def test_ctor_rejects_zero_max_size() -> None:
    with pytest.raises(ValueError, match="max_size must be positive"):
        EpisodicMemoryStore(dim=4, max_size=0)


def test_ctor_rejects_negative_max_size() -> None:
    with pytest.raises(ValueError, match="max_size must be positive"):
        EpisodicMemoryStore(dim=4, max_size=-5)


def test_ctor_rejects_non_int_max_size() -> None:
    with pytest.raises(TypeError, match="max_size must be int"):
        EpisodicMemoryStore(dim=4, max_size="10")  # type: ignore[arg-type]


def test_store_has_slots() -> None:
    assert hasattr(EpisodicMemoryStore, "__slots__")
    assert "_episodes" in EpisodicMemoryStore.__slots__


# ---------------------------------------------------------------------------
# add() invariants
# ---------------------------------------------------------------------------


def _ep(ts_ns: int, episode_id: str, *, dim: int = 4) -> Episode:
    base = (1.0, 2.0, 3.0, 4.0)[:dim]
    return Episode(ts_ns=ts_ns, episode_id=episode_id, embedding=base)


def test_add_one_episode_increments_len() -> None:
    s = EpisodicMemoryStore(dim=4, max_size=10)
    s.add(_ep(1, "a"))
    assert len(s) == 1
    assert "a" in s


def test_add_rejects_non_episode() -> None:
    s = EpisodicMemoryStore(dim=4, max_size=10)
    with pytest.raises(TypeError, match="expects Episode"):
        s.add("not-an-episode")  # type: ignore[arg-type]


def test_add_rejects_dim_mismatch() -> None:
    s = EpisodicMemoryStore(dim=4, max_size=10)
    bad = Episode(ts_ns=1, episode_id="a", embedding=(1.0, 2.0))
    with pytest.raises(ValueError, match="dim mismatch"):
        s.add(bad)


def test_add_rejects_duplicate_id() -> None:
    s = EpisodicMemoryStore(dim=4, max_size=10)
    s.add(_ep(1, "a"))
    with pytest.raises(ValueError, match="already present"):
        s.add(_ep(2, "a"))


def test_add_evicts_oldest_when_full() -> None:
    s = EpisodicMemoryStore(dim=4, max_size=2)
    s.add(_ep(10, "a"))
    s.add(_ep(20, "b"))
    s.add(_ep(30, "c"))  # evicts "a" (ts_ns=10)
    assert len(s) == 2
    assert "a" not in s
    assert "b" in s
    assert "c" in s


def test_add_eviction_breaks_ts_ns_ties_by_episode_id() -> None:
    s = EpisodicMemoryStore(dim=4, max_size=2)
    s.add(_ep(10, "b"))
    s.add(_ep(10, "a"))  # same ts_ns; "a" < "b"
    s.add(_ep(20, "c"))  # evicts "a" (lower episode_id at same ts_ns)
    assert "a" not in s
    assert "b" in s
    assert "c" in s


def test_add_does_not_evict_when_below_cap() -> None:
    s = EpisodicMemoryStore(dim=4, max_size=5)
    for i, eid in enumerate(("a", "b", "c")):
        s.add(_ep(10 * (i + 1), eid))
    assert len(s) == 3
    for eid in ("a", "b", "c"):
        assert eid in s


def test_add_repeated_eviction_holds_invariant() -> None:
    s = EpisodicMemoryStore(dim=4, max_size=3)
    for i in range(20):
        s.add(_ep(i + 1, f"e{i:02d}"))
    assert len(s) == 3
    # Three latest IDs survive; FIFO is by ts_ns ascending.
    assert "e17" in s
    assert "e18" in s
    assert "e19" in s


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


def test_delete_known_episode_returns_true() -> None:
    s = EpisodicMemoryStore(dim=4, max_size=10)
    s.add(_ep(1, "a"))
    assert s.delete("a") is True
    assert "a" not in s


def test_delete_unknown_episode_returns_false() -> None:
    s = EpisodicMemoryStore(dim=4, max_size=10)
    assert s.delete("missing") is False


# ---------------------------------------------------------------------------
# __contains__ / __iter__
# ---------------------------------------------------------------------------


def test_contains_rejects_non_str() -> None:
    s = EpisodicMemoryStore(dim=4, max_size=10)
    s.add(_ep(1, "a"))
    assert (1 in s) is False
    assert (None in s) is False


def test_iter_is_sorted_by_ts_ns_then_episode_id() -> None:
    s = EpisodicMemoryStore(dim=4, max_size=10)
    s.add(_ep(20, "b"))
    s.add(_ep(10, "z"))
    s.add(_ep(10, "a"))
    s.add(_ep(30, "m"))
    seen = [(ep.ts_ns, ep.episode_id) for ep in s]
    assert seen == [(10, "a"), (10, "z"), (20, "b"), (30, "m")]


# ---------------------------------------------------------------------------
# search() — correctness
# ---------------------------------------------------------------------------


def test_search_returns_memory_result_type() -> None:
    s = EpisodicMemoryStore(dim=4, max_size=10)
    s.add(_ep(1, "a"))
    q = MemoryQuery(ts_ns=2, query_id="q1", embedding=(1.0, 2.0, 3.0, 4.0), k=1)
    res = s.search(q)
    assert isinstance(res, MemoryResult)
    assert res.query_id == "q1"
    assert res.ts_ns == 2


def test_search_empty_store_returns_empty_hits() -> None:
    s = EpisodicMemoryStore(dim=4, max_size=10)
    q = MemoryQuery(ts_ns=1, query_id="q", embedding=(0.0, 0.0, 0.0, 0.0), k=5)
    res = s.search(q)
    assert res.hits == ()


def test_search_rejects_non_query() -> None:
    s = EpisodicMemoryStore(dim=4, max_size=10)
    with pytest.raises(TypeError, match="expects MemoryQuery"):
        s.search("not-a-query")  # type: ignore[arg-type]


def test_search_rejects_dim_mismatch() -> None:
    s = EpisodicMemoryStore(dim=4, max_size=10)
    s.add(_ep(1, "a"))
    q = MemoryQuery(ts_ns=2, query_id="q", embedding=(1.0, 2.0), k=1)
    with pytest.raises(ValueError, match="dim mismatch"):
        s.search(q)


def test_search_returns_zero_distance_for_exact_match() -> None:
    s = EpisodicMemoryStore(dim=3, max_size=10)
    s.add(Episode(ts_ns=1, episode_id="exact", embedding=(0.5, -0.5, 1.5)))
    q = MemoryQuery(ts_ns=2, query_id="q", embedding=(0.5, -0.5, 1.5), k=1)
    res = s.search(q)
    assert len(res.hits) == 1
    assert res.hits[0].episode_id == "exact"
    assert res.hits[0].distance == 0.0


def test_search_distances_are_l2() -> None:
    s = EpisodicMemoryStore(dim=2, max_size=10)
    s.add(Episode(ts_ns=1, episode_id="a", embedding=(0.0, 0.0)))
    s.add(Episode(ts_ns=2, episode_id="b", embedding=(3.0, 4.0)))  # 5.0
    s.add(Episode(ts_ns=3, episode_id="c", embedding=(1.0, 0.0)))  # 1.0
    q = MemoryQuery(ts_ns=4, query_id="q", embedding=(0.0, 0.0), k=3)
    res = s.search(q)
    distances = {h.episode_id: h.distance for h in res.hits}
    assert distances["a"] == pytest.approx(0.0)
    assert distances["c"] == pytest.approx(1.0)
    assert distances["b"] == pytest.approx(5.0)


def test_search_truncates_to_k() -> None:
    s = EpisodicMemoryStore(dim=2, max_size=10)
    for i in range(7):
        s.add(
            Episode(
                ts_ns=i + 1,
                episode_id=f"e{i}",
                embedding=(float(i), float(i)),
            )
        )
    q = MemoryQuery(ts_ns=100, query_id="q", embedding=(0.0, 0.0), k=3)
    res = s.search(q)
    assert len(res.hits) == 3


def test_search_returns_hits_sorted_ascending() -> None:
    s = EpisodicMemoryStore(dim=2, max_size=10)
    s.add(Episode(ts_ns=1, episode_id="far", embedding=(10.0, 10.0)))
    s.add(Episode(ts_ns=2, episode_id="mid", embedding=(2.0, 0.0)))
    s.add(Episode(ts_ns=3, episode_id="near", embedding=(0.1, 0.0)))
    q = MemoryQuery(ts_ns=4, query_id="q", embedding=(0.0, 0.0), k=3)
    res = s.search(q)
    # MemoryResult itself enforces ascending — round-trip through it.
    assert res.hits[0].episode_id == "near"
    assert res.hits[1].episode_id == "mid"
    assert res.hits[2].episode_id == "far"


def test_search_breaks_distance_ties_by_ts_ns_then_episode_id() -> None:
    s = EpisodicMemoryStore(dim=2, max_size=10)
    # All three are at L2 distance 1.0 from the origin
    s.add(Episode(ts_ns=30, episode_id="z", embedding=(1.0, 0.0)))
    s.add(Episode(ts_ns=10, episode_id="a", embedding=(0.0, 1.0)))
    s.add(Episode(ts_ns=10, episode_id="m", embedding=(-1.0, 0.0)))
    q = MemoryQuery(ts_ns=99, query_id="q", embedding=(0.0, 0.0), k=3)
    res = s.search(q)
    ids = [h.episode_id for h in res.hits]
    # ts_ns asc → 10 before 30; within ts_ns=10, "a" before "m".
    assert ids == ["a", "m", "z"]


def test_search_hit_carries_episode_payload_and_ts() -> None:
    s = EpisodicMemoryStore(dim=2, max_size=10)
    payload = {"k": "v", "x": "y"}
    ep = Episode(
        ts_ns=42,
        episode_id="p",
        embedding=(0.0, 0.0),
        payload=payload,
    )
    s.add(ep)
    q = MemoryQuery(ts_ns=99, query_id="q", embedding=(0.0, 0.0), k=1)
    res = s.search(q)
    hit = res.hits[0]
    assert isinstance(hit, MemoryHit)
    assert hit.episode_id == "p"
    assert hit.ts_ns == 42
    assert hit.payload["k"] == "v"
    assert hit.payload["x"] == "y"


def test_search_returns_at_most_k_when_store_smaller_than_k() -> None:
    s = EpisodicMemoryStore(dim=2, max_size=10)
    s.add(Episode(ts_ns=1, episode_id="a", embedding=(1.0, 1.0)))
    s.add(Episode(ts_ns=2, episode_id="b", embedding=(2.0, 2.0)))
    q = MemoryQuery(ts_ns=3, query_id="q", embedding=(0.0, 0.0), k=10)
    res = s.search(q)
    assert len(res.hits) == 2


# ---------------------------------------------------------------------------
# Replay determinism (INV-15)
# ---------------------------------------------------------------------------


def _build_reference_store() -> EpisodicMemoryStore:
    s = EpisodicMemoryStore(dim=3, max_size=5)
    s.add(Episode(ts_ns=10, episode_id="alpha", embedding=(0.1, 0.2, 0.3)))
    s.add(Episode(ts_ns=20, episode_id="beta", embedding=(1.0, 0.0, 0.0)))
    s.add(
        Episode(
            ts_ns=30,
            episode_id="gamma",
            embedding=(0.5, 0.5, 0.5),
            payload={"region": "us", "kind": "rebalance"},
        )
    )
    return s


def test_search_is_deterministic_across_runs() -> None:
    q = MemoryQuery(ts_ns=99, query_id="q", embedding=(0.5, 0.5, 0.5), k=3)
    runs = [_build_reference_store().search(q) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_serialize_is_deterministic_across_runs() -> None:
    blobs = {_build_reference_store().serialize() for _ in range(3)}
    assert len(blobs) == 1


def test_search_hits_field_consistency() -> None:
    # Hit fields must mirror the original episode (ts_ns / payload pinned)
    s = _build_reference_store()
    q = MemoryQuery(ts_ns=99, query_id="q", embedding=(0.5, 0.5, 0.5), k=3)
    res = s.search(q)
    by_id = {h.episode_id: h for h in res.hits}
    assert by_id["gamma"].ts_ns == 30
    assert by_id["gamma"].payload["region"] == "us"
    assert by_id["alpha"].ts_ns == 10


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


def test_serialize_returns_bytes() -> None:
    s = _build_reference_store()
    out = s.serialize()
    assert isinstance(out, bytes)
    assert len(out) > 0


def test_serialize_is_sorted_and_versioned() -> None:
    import json as _json

    s = _build_reference_store()
    obj = _json.loads(s.serialize().decode("utf-8"))
    assert obj["version"] == 1
    assert obj["dim"] == 3
    assert obj["max_size"] == 5
    ids = [e["episode_id"] for e in obj["episodes"]]
    # Already sorted by (ts_ns, episode_id) ascending.
    assert ids == ["alpha", "beta", "gamma"]


def test_deserialize_round_trip_is_byte_equal() -> None:
    s1 = _build_reference_store()
    blob = s1.serialize()
    s2 = EpisodicMemoryStore.deserialize(blob)
    assert s2.serialize() == blob


def test_deserialize_round_trip_preserves_search() -> None:
    s1 = _build_reference_store()
    s2 = EpisodicMemoryStore.deserialize(s1.serialize())
    q = MemoryQuery(ts_ns=99, query_id="q", embedding=(0.5, 0.5, 0.5), k=3)
    assert s1.search(q) == s2.search(q)


def test_deserialize_round_trip_preserves_dim_and_cap() -> None:
    s1 = _build_reference_store()
    s2 = EpisodicMemoryStore.deserialize(s1.serialize())
    assert s2.dim == s1.dim
    assert s2.max_size == s1.max_size
    assert len(s2) == len(s1)


def test_deserialize_round_trip_preserves_payload() -> None:
    s1 = _build_reference_store()
    s2 = EpisodicMemoryStore.deserialize(s1.serialize())
    q = MemoryQuery(ts_ns=99, query_id="q", embedding=(0.5, 0.5, 0.5), k=3)
    res2 = s2.search(q)
    by_id = {h.episode_id: h for h in res2.hits}
    assert by_id["gamma"].payload["region"] == "us"
    assert by_id["gamma"].payload["kind"] == "rebalance"


# ---------------------------------------------------------------------------
# Serialization defensive cases
# ---------------------------------------------------------------------------


def test_deserialize_rejects_non_bytes() -> None:
    with pytest.raises(TypeError, match="expects bytes"):
        EpisodicMemoryStore.deserialize("not-bytes")  # type: ignore[arg-type]


def test_deserialize_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="invalid blob"):
        EpisodicMemoryStore.deserialize(b"\xff\xfe not json")


def test_deserialize_rejects_non_object_top_level() -> None:
    with pytest.raises(ValueError, match="top-level must be object"):
        EpisodicMemoryStore.deserialize(b"[1,2,3]")


def test_deserialize_rejects_unknown_version() -> None:
    blob = b'{"version":99,"dim":1,"max_size":1,"episodes":[]}'
    with pytest.raises(ValueError, match="unsupported version"):
        EpisodicMemoryStore.deserialize(blob)


def test_deserialize_rejects_bad_dim() -> None:
    blob = b'{"version":1,"dim":"x","max_size":1,"episodes":[]}'
    with pytest.raises(ValueError, match="'dim' must be int"):
        EpisodicMemoryStore.deserialize(blob)


def test_deserialize_rejects_bad_max_size() -> None:
    blob = b'{"version":1,"dim":1,"max_size":"x","episodes":[]}'
    with pytest.raises(ValueError, match="'max_size' must be int"):
        EpisodicMemoryStore.deserialize(blob)


def test_deserialize_rejects_non_list_episodes() -> None:
    blob = b'{"version":1,"dim":1,"max_size":1,"episodes":{}}'
    with pytest.raises(ValueError, match="episodes' must be list"):
        EpisodicMemoryStore.deserialize(blob)


def test_deserialize_rejects_non_object_episode_row() -> None:
    blob = b'{"version":1,"dim":1,"max_size":2,"episodes":[1]}'
    with pytest.raises(ValueError, match="episodes\\[0\\] must be object"):
        EpisodicMemoryStore.deserialize(blob)


def test_deserialize_rejects_bad_ts_ns() -> None:
    blob = (
        b'{"version":1,"dim":1,"max_size":2,"episodes":'
        b'[{"ts_ns":"x","episode_id":"a","embedding":[0.0],"payload":{}}]}'
    )
    with pytest.raises(ValueError, match="ts_ns must be int"):
        EpisodicMemoryStore.deserialize(blob)


def test_deserialize_rejects_bad_episode_id() -> None:
    blob = (
        b'{"version":1,"dim":1,"max_size":2,"episodes":'
        b'[{"ts_ns":1,"episode_id":1,"embedding":[0.0],"payload":{}}]}'
    )
    with pytest.raises(ValueError, match="episode_id must be str"):
        EpisodicMemoryStore.deserialize(blob)


def test_deserialize_rejects_non_list_embedding() -> None:
    blob = (
        b'{"version":1,"dim":1,"max_size":2,"episodes":'
        b'[{"ts_ns":1,"episode_id":"a","embedding":"x","payload":{}}]}'
    )
    with pytest.raises(ValueError, match="embedding must be list"):
        EpisodicMemoryStore.deserialize(blob)


def test_deserialize_rejects_nan_embedding() -> None:
    nan = math.nan
    blob = (
        b'{"version":1,"dim":2,"max_size":2,"episodes":'
        b'[{"ts_ns":1,"episode_id":"a","embedding":[NaN,0.0],"payload":{}}]}'
    )
    # Built JSON above isn't strictly valid; do it via dict instead.
    import json as _json

    blob = _json.dumps(
        {
            "version": 1,
            "dim": 2,
            "max_size": 2,
            "episodes": [
                {
                    "ts_ns": 1,
                    "episode_id": "a",
                    "embedding": [nan, 0.0],
                    "payload": {},
                }
            ],
        }
    ).encode("utf-8")
    with pytest.raises(ValueError, match="must be finite"):
        EpisodicMemoryStore.deserialize(blob)


def test_deserialize_rejects_non_object_payload() -> None:
    blob = (
        b'{"version":1,"dim":1,"max_size":2,"episodes":'
        b'[{"ts_ns":1,"episode_id":"a","embedding":[0.0],"payload":1}]}'
    )
    with pytest.raises(ValueError, match="payload must be object"):
        EpisodicMemoryStore.deserialize(blob)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_episodic_satisfies_memory_store_base() -> None:
    s = EpisodicMemoryStore(dim=4, max_size=10)
    assert isinstance(s, MemoryStoreBase)


def test_episodic_round_trips_via_protocol_facade() -> None:
    """Treat the store strictly through the Protocol surface."""

    base: MemoryStoreBase = EpisodicMemoryStore(dim=3, max_size=5)
    base.add(Episode(ts_ns=1, episode_id="a", embedding=(1.0, 2.0, 3.0)))
    q = MemoryQuery(ts_ns=2, query_id="q", embedding=(1.0, 2.0, 3.0), k=1)
    res = base.search(q)
    assert res.hits[0].episode_id == "a"
    assert isinstance(base.serialize(), bytes)
    assert base.dim == 3
    assert base.max_size == 5
    assert len(base) == 1
    assert "a" in base


# ---------------------------------------------------------------------------
# Replay determinism micro-pin
# ---------------------------------------------------------------------------


def test_repeatedly_serialised_store_is_byte_identical() -> None:
    s = _build_reference_store()
    blob1 = s.serialize()
    blob2 = s.serialize()
    blob3 = s.serialize()
    assert blob1 == blob2 == blob3


def test_search_result_is_value_equal_across_three_runs() -> None:
    q = MemoryQuery(ts_ns=99, query_id="q", embedding=(0.0, 0.0, 0.0), k=2)
    a = _build_reference_store().search(q)
    b = _build_reference_store().search(q)
    c = _build_reference_store().search(q)
    assert a == b == c
