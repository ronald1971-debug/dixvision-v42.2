"""Tests for ``state.memory_tensor.contracts`` (S-08.1).

Pure backend Python+pytest. Pins:

* Field validation on every dataclass.
* Embedding shape / type / finite-float guard.
* MemoryResult ascending-distance ordering invariant.
* Frozen + slotted (no setattr) on every dataclass.
* Payload immutability (caller mutation cannot leak in).
* Equality / hashing (replay determinism, INV-15).
* MemoryStoreBase Protocol structural-check.
* AST sweep — no clock, no os, no asyncio, no faiss/numpy/qdrant
  imports in the contracts module (those belong to the backends).
"""

from __future__ import annotations

import ast
import dataclasses
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from state.memory_tensor.contracts import (
    Episode,
    MemoryHit,
    MemoryQuery,
    MemoryResult,
    MemoryStoreBase,
    validate_embedding,
)

# ---------------------------------------------------------------------------
# validate_embedding
# ---------------------------------------------------------------------------


def test_validate_embedding_accepts_finite_tuple() -> None:
    validate_embedding((0.0, 1.0, -1.0), field="x")


def test_validate_embedding_rejects_list() -> None:
    with pytest.raises(TypeError, match="must be a tuple"):
        validate_embedding([0.0, 1.0], field="x")  # type: ignore[arg-type]


def test_validate_embedding_rejects_empty() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        validate_embedding((), field="x")


def test_validate_embedding_rejects_non_float() -> None:
    with pytest.raises(TypeError, match=r"x\[0\] must be float"):
        validate_embedding((1, 2.0), field="x")  # type: ignore[arg-type]


def test_validate_embedding_rejects_nan() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        validate_embedding((float("nan"),), field="x")


def test_validate_embedding_rejects_inf() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        validate_embedding((float("inf"),), field="x")


def test_validate_embedding_rejects_neg_inf() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        validate_embedding((float("-inf"),), field="x")


def test_validate_embedding_field_name_is_in_error() -> None:
    with pytest.raises(ValueError, match="Episode.embedding"):
        validate_embedding((), field="Episode.embedding")


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------


def _ep(
    *,
    ts_ns: int = 1,
    episode_id: str = "ep-1",
    embedding: tuple[float, ...] = (0.0, 1.0, 2.0),
    payload: Mapping[str, str] | None = None,
) -> Episode:
    if payload is None:
        return Episode(
            ts_ns=ts_ns,
            episode_id=episode_id,
            embedding=embedding,
        )
    return Episode(
        ts_ns=ts_ns,
        episode_id=episode_id,
        embedding=embedding,
        payload=payload,
    )


def test_episode_minimal_construction() -> None:
    ep = _ep()
    assert ep.ts_ns == 1
    assert ep.episode_id == "ep-1"
    assert ep.embedding == (0.0, 1.0, 2.0)
    assert ep.dim == 3
    assert dict(ep.payload) == {}


def test_episode_rejects_zero_ts_ns() -> None:
    with pytest.raises(ValueError, match="ts_ns must be positive"):
        _ep(ts_ns=0)


def test_episode_rejects_negative_ts_ns() -> None:
    with pytest.raises(ValueError, match="ts_ns must be positive"):
        _ep(ts_ns=-1)


def test_episode_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="episode_id must be non-empty"):
        _ep(episode_id="")


def test_episode_rejects_empty_embedding() -> None:
    with pytest.raises(ValueError, match="Episode.embedding must not be empty"):
        _ep(embedding=())


def test_episode_rejects_non_tuple_embedding() -> None:
    with pytest.raises(TypeError, match="Episode.embedding must be a tuple"):
        Episode(
            ts_ns=1,
            episode_id="x",
            embedding=[0.0, 1.0],  # type: ignore[arg-type]
        )


def test_episode_rejects_nan_embedding() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        _ep(embedding=(float("nan"),))


def test_episode_rejects_payload_with_non_str_key() -> None:
    with pytest.raises(TypeError, match="payload keys must be str"):
        _ep(payload={1: "v"})  # type: ignore[dict-item]


def test_episode_rejects_payload_with_non_str_value() -> None:
    with pytest.raises(TypeError, match=r"payload\[.*\] must be str"):
        _ep(payload={"k": 1})  # type: ignore[dict-item]


def test_episode_payload_is_frozen_after_construction() -> None:
    src = {"k": "v"}
    ep = _ep(payload=src)
    src["k"] = "MUTATED"
    assert ep.payload["k"] == "v"


def test_episode_payload_is_mappingproxy() -> None:
    ep = _ep(payload={"k": "v"})
    assert isinstance(ep.payload, MappingProxyType)


def test_episode_is_frozen() -> None:
    ep = _ep()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ep.ts_ns = 99  # type: ignore[misc]


def test_episode_has_slots() -> None:
    assert hasattr(Episode, "__slots__")
    assert "ts_ns" in Episode.__slots__
    assert "embedding" in Episode.__slots__


def test_episode_equality_is_structural() -> None:
    a = _ep(payload={"k": "v"})
    b = _ep(payload={"k": "v"})
    assert a == b


def test_episode_inequality_on_embedding() -> None:
    a = _ep(embedding=(1.0, 2.0))
    b = _ep(embedding=(1.0, 2.0001))
    assert a != b


def test_episode_dim_property() -> None:
    assert _ep(embedding=(0.0,)).dim == 1
    assert _ep(embedding=(0.0,) * 64).dim == 64


# ---------------------------------------------------------------------------
# MemoryQuery
# ---------------------------------------------------------------------------


def _q(
    *,
    ts_ns: int = 1,
    query_id: str = "q-1",
    embedding: tuple[float, ...] = (0.0, 1.0, 2.0),
    k: int = 5,
) -> MemoryQuery:
    return MemoryQuery(
        ts_ns=ts_ns,
        query_id=query_id,
        embedding=embedding,
        k=k,
    )


def test_query_minimal_construction() -> None:
    q = _q()
    assert q.ts_ns == 1
    assert q.query_id == "q-1"
    assert q.embedding == (0.0, 1.0, 2.0)
    assert q.k == 5
    assert q.dim == 3


def test_query_rejects_non_positive_ts_ns() -> None:
    with pytest.raises(ValueError, match="ts_ns must be positive"):
        _q(ts_ns=0)


def test_query_rejects_empty_query_id() -> None:
    with pytest.raises(ValueError, match="query_id must be non-empty"):
        _q(query_id="")


def test_query_rejects_zero_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        _q(k=0)


def test_query_rejects_negative_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        _q(k=-1)


def test_query_rejects_nan_embedding() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        _q(embedding=(float("nan"),))


def test_query_is_frozen() -> None:
    q = _q()
    with pytest.raises(dataclasses.FrozenInstanceError):
        q.k = 99  # type: ignore[misc]


def test_query_equality_is_structural() -> None:
    assert _q() == _q()
    assert hash(_q()) == hash(_q())


# ---------------------------------------------------------------------------
# MemoryHit
# ---------------------------------------------------------------------------


def _h(
    *,
    episode_id: str = "ep-1",
    distance: float = 0.0,
    ts_ns: int = 1,
    payload: Mapping[str, str] | None = None,
) -> MemoryHit:
    if payload is None:
        return MemoryHit(
            episode_id=episode_id,
            distance=distance,
            ts_ns=ts_ns,
        )
    return MemoryHit(
        episode_id=episode_id,
        distance=distance,
        ts_ns=ts_ns,
        payload=payload,
    )


def test_hit_minimal_construction() -> None:
    h = _h()
    assert h.episode_id == "ep-1"
    assert h.distance == 0.0
    assert h.ts_ns == 1


def test_hit_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="episode_id must be non-empty"):
        _h(episode_id="")


def test_hit_rejects_negative_distance() -> None:
    with pytest.raises(ValueError, match="distance must be non-negative"):
        _h(distance=-0.1)


def test_hit_rejects_nan_distance() -> None:
    with pytest.raises(ValueError, match="distance must be finite"):
        _h(distance=float("nan"))


def test_hit_rejects_inf_distance() -> None:
    with pytest.raises(ValueError, match="distance must be finite"):
        _h(distance=float("inf"))


def test_hit_rejects_non_float_distance() -> None:
    with pytest.raises(TypeError, match="distance must be float"):
        MemoryHit(  # type: ignore[arg-type]
            episode_id="x",
            distance=1,
            ts_ns=1,
        )


def test_hit_rejects_zero_ts_ns() -> None:
    with pytest.raises(ValueError, match="ts_ns must be positive"):
        _h(ts_ns=0)


def test_hit_payload_frozen_after_construction() -> None:
    src = {"k": "v"}
    h = _h(payload=src)
    src["k"] = "MUTATED"
    assert h.payload["k"] == "v"


def test_hit_is_frozen() -> None:
    h = _h()
    with pytest.raises(dataclasses.FrozenInstanceError):
        h.distance = 99.0  # type: ignore[misc]


def test_hit_payload_rejects_non_str_value() -> None:
    with pytest.raises(TypeError, match=r"payload\[.*\] must be str"):
        _h(payload={"k": 1})  # type: ignore[dict-item]


def test_hit_payload_rejects_non_str_key() -> None:
    with pytest.raises(TypeError, match="payload keys must be str"):
        _h(payload={1: "v"})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# MemoryResult
# ---------------------------------------------------------------------------


def _r(
    *,
    ts_ns: int = 1,
    query_id: str = "q-1",
    hits: tuple[MemoryHit, ...] = (),
) -> MemoryResult:
    return MemoryResult(
        ts_ns=ts_ns,
        query_id=query_id,
        hits=hits,
    )


def test_result_empty_hits_is_valid() -> None:
    r = _r()
    assert r.hits == ()


def test_result_rejects_non_positive_ts_ns() -> None:
    with pytest.raises(ValueError, match="ts_ns must be positive"):
        _r(ts_ns=0)


def test_result_rejects_empty_query_id() -> None:
    with pytest.raises(ValueError, match="query_id must be non-empty"):
        _r(query_id="")


def test_result_rejects_non_tuple_hits() -> None:
    with pytest.raises(TypeError, match="hits must be a tuple"):
        MemoryResult(  # type: ignore[arg-type]
            ts_ns=1,
            query_id="q",
            hits=[],
        )


def test_result_rejects_non_hit_in_tuple() -> None:
    with pytest.raises(TypeError, match=r"hits\[0\] must be MemoryHit"):
        MemoryResult(  # type: ignore[arg-type]
            ts_ns=1,
            query_id="q",
            hits=("not-a-hit",),
        )


def test_result_accepts_ascending_hits() -> None:
    hits = (
        _h(episode_id="a", distance=0.1),
        _h(episode_id="b", distance=0.5),
        _h(episode_id="c", distance=0.5),  # tie OK
        _h(episode_id="d", distance=2.0),
    )
    r = _r(hits=hits)
    assert r.hits == hits


def test_result_rejects_descending_hits() -> None:
    hits = (
        _h(episode_id="a", distance=2.0),
        _h(episode_id="b", distance=0.1),
    )
    with pytest.raises(ValueError, match="sorted by ascending distance"):
        _r(hits=hits)


def test_result_is_frozen() -> None:
    r = _r()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.ts_ns = 99  # type: ignore[misc]


def test_result_equality_is_structural() -> None:
    h = _h(episode_id="a", distance=0.1)
    a = _r(hits=(h,))
    b = _r(hits=(h,))
    assert a == b


# ---------------------------------------------------------------------------
# MemoryStoreBase Protocol
# ---------------------------------------------------------------------------


class _ConformingFakeStore:
    """A minimal Protocol-conforming structural shape (not a backend)."""

    def __init__(self) -> None:
        self._d = 4
        self._cap = 100

    @property
    def dim(self) -> int:
        return self._d

    @property
    def max_size(self) -> int:
        return self._cap

    def __len__(self) -> int:
        return 0

    def __contains__(self, episode_id: str) -> bool:
        return False

    def add(self, episode: Episode) -> None:
        return None

    def search(self, query: MemoryQuery) -> MemoryResult:
        return MemoryResult(ts_ns=query.ts_ns, query_id=query.query_id, hits=())

    def serialize(self) -> bytes:
        return b""


def test_protocol_conforming_class_passes_isinstance() -> None:
    assert isinstance(_ConformingFakeStore(), MemoryStoreBase)


class _NonConformingMissingSearch:
    @property
    def dim(self) -> int:  # pragma: no cover - protocol only
        return 0

    @property
    def max_size(self) -> int:  # pragma: no cover - protocol only
        return 0

    def __len__(self) -> int:  # pragma: no cover - protocol only
        return 0

    def __contains__(self, episode_id: str) -> bool:  # pragma: no cover
        return False

    def add(self, episode: Episode) -> None:  # pragma: no cover
        return None

    def serialize(self) -> bytes:  # pragma: no cover
        return b""


def test_protocol_non_conforming_class_fails_isinstance() -> None:
    assert not isinstance(_NonConformingMissingSearch(), MemoryStoreBase)


# ---------------------------------------------------------------------------
# AST sweep — pure-data invariants on the contracts module
# ---------------------------------------------------------------------------


_CONTRACTS_PATH = (
    Path(__file__).resolve().parent.parent / "state" / "memory_tensor" / "contracts.py"
)


def _module_ast() -> ast.Module:
    return ast.parse(_CONTRACTS_PATH.read_text())


_FORBIDDEN_TOP_LEVEL_IMPORTS = {
    "time",
    "datetime",
    "os",
    "asyncio",
    "threading",
    "subprocess",
    "socket",
    "logging",
    # third-party libs that belong to the backend implementations only
    "numpy",
    "faiss",
    "qdrant_client",
}


def _imported_modules(tree: ast.Module) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                out.add(n.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module.split(".", 1)[0])
    return out


def test_contracts_does_not_import_forbidden_modules() -> None:
    imports = _imported_modules(_module_ast())
    bad = imports & _FORBIDDEN_TOP_LEVEL_IMPORTS
    assert not bad, f"contracts.py must not import {bad}"


def test_contracts_has_no_engine_cross_imports() -> None:
    imports = _imported_modules(_module_ast())
    forbidden_prefixes = {
        "execution_engine",
        "governance_engine",
        "system_engine",
        "intelligence_engine",
        "learning_engine",
        "evolution_engine",
    }
    bad = imports & forbidden_prefixes
    assert not bad, f"contracts.py must not import engines: {bad}"


def test_contracts_has_adapted_from_header() -> None:
    src = _CONTRACTS_PATH.read_text()
    assert src.startswith("# ADAPTED FROM: facebookresearch/faiss"), (
        "S-08 spec requires '# ADAPTED FROM: facebookresearch/faiss Python interface' header"
    )


def test_contracts_has_no_clock_calls() -> None:
    src = _CONTRACTS_PATH.read_text()
    for needle in ("time.time(", "time.monotonic(", "datetime.now(", "time_ns("):
        assert needle not in src, f"contracts.py must not call {needle!r} (INV-15)"


def test_contracts_uses_no_typing_any_for_payloads() -> None:
    """Payload values must be ``str`` per the S-08 ledger-friendly rule."""
    src = _CONTRACTS_PATH.read_text()
    assert "Mapping[str, Any]" not in src, (
        "Payload mappings must be Mapping[str, str] (ledger-friendly), not Mapping[str, Any]"
    )


# ---------------------------------------------------------------------------
# Replay determinism micro-pin
# ---------------------------------------------------------------------------


def test_episode_repr_is_deterministic_across_calls() -> None:
    """Same fields → same ``repr``. Used as a cheap INV-15 smoke test."""
    a = _ep(payload={"k": "v", "a": "b"})
    b = _ep(payload={"k": "v", "a": "b"})
    assert repr(a) == repr(b)


def test_query_hash_stable() -> None:
    """Query has no Mapping field, so it is hashable.

    Episode / Hit / Result carry payload mappings and are NOT hashable,
    matching the existing convention in ``core/contracts/learning.py``.
    """
    assert hash(_q()) == hash(_q())


def test_unused_imports_kept_intentionally() -> None:
    """Drop-guard: ``Any`` is intentionally NOT imported in contracts.py.

    This test exists to keep the contributor from re-adding ``Any`` —
    payloads must stay ``Mapping[str, str]`` per S-08.
    """
    # Reference Any locally so the linter doesn't trip on this test
    # file's own import of typing.Any (we don't actually import it).
    _ = Any
    src = _CONTRACTS_PATH.read_text()
    assert "from typing import" in src
    assert "Any" not in src.split("__all__")[0].split("from typing import")[1].split("\n")[0], (
        "contracts.py must not import typing.Any"
    )
