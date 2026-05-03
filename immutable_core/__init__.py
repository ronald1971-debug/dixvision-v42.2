"""Immutable core — the v42.2 axioms registry (P0-1d).

The neuromorphic axioms are the foundational invariants the system
rests on. The original spec called for an external Lean 4 proof
artefact (``immutable_core/neuromorphic_axioms.lean``); shipping
that requires the Lean toolchain in CI and is filed as a separate
research spike (Wave-A follow-up).

This package ships the **Python-form registry** of those axioms so
that:

  * every INV-* / SAFE-* identifier scattered across the codebase
    has a single named source of truth (id → kind → label →
    where-it-was-introduced);
  * the next P0 step (P0-2 hazard chain) can refer to axioms by id
    in code/comments and a test will catch typos / orphans;
  * future formal verification work can target the same id space
    without renumbering.

Pure / read-only / no I/O. The registry is a frozen mapping; calling
:func:`get_axiom` with an unknown id raises :class:`KeyError`.
"""

from __future__ import annotations

from immutable_core.axioms import (
    AXIOM_REGISTRY,
    Axiom,
    AxiomKind,
    get_axiom,
    is_axiom,
)

__all__ = (
    "AXIOM_REGISTRY",
    "Axiom",
    "AxiomKind",
    "get_axiom",
    "is_axiom",
)
