"""Authority matrix — single conflict-resolution table.

Closes operator concern **L3** from the v3.5 critique. The matrix is
the single source of truth for "who wins in conflict?" — every
authority on the v42.2 control plane (Governance, Indira, Executor,
Dyon, Operator, Learning, Evolution, Ledger), the precedence ordering
between them, the documented decision points where they collide, and
the legal override edges (always routed through Governance).

Pure / read-only. Engines do not import each other; they all import
this matrix to decide who they defer to.
"""

from system_engine.authority.matrix import (
    AuthorityActor,
    AuthorityMatrix,
    AuthorityOverride,
    ConflictRow,
    load_authority_matrix,
)

__all__ = [
    "AuthorityActor",
    "AuthorityMatrix",
    "AuthorityOverride",
    "ConflictRow",
    "load_authority_matrix",
]
