"""state.memory_tensor — vector-memory tier (S-08, OFFLINE).

The memory tensor stores frozen :class:`Episode` records keyed by an
embedding vector. Search is exposed as a :class:`MemoryQuery` /
:class:`MemoryResult` round-trip that any concrete backend (FAISS,
Qdrant, brute-force) can satisfy through the
:class:`~state.memory_tensor.contracts.MemoryStoreBase` Protocol.

S-08 ships the contracts here; concrete backends land in S-08.2
(:mod:`state.memory_tensor.episodic`) and S-08.3
(:mod:`state.memory_tensor.semantic`).
"""

from __future__ import annotations

from state.memory_tensor.contracts import (
    Episode,
    MemoryHit,
    MemoryQuery,
    MemoryResult,
    MemoryStoreBase,
    validate_embedding,
)

__all__ = [
    "Episode",
    "MemoryHit",
    "MemoryQuery",
    "MemoryResult",
    "MemoryStoreBase",
    "validate_embedding",
]
