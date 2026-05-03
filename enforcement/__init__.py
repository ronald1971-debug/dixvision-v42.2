"""Enforcement layer — single import surface for runtime guards (P0-1c).

The v42.2 enforcement layer is the set of runtime defences that turn
policy / authority / safety into hard rejections at execute time.
Before this package the primitives were scattered across six modules
and four packages, so:

  * authority-matrix audit (``registry/authority_matrix.yaml``
    references "enforcement layer") had no single Python target;
  * the next P0 step (P0-2 hazard-throttle chain) would have had to
    import six different paths to compose a complete enforcement
    surface.

This package is a **thin re-export façade** — it does not introduce
new state or new behaviour. Every primitive lives where it always
lived; ``from enforcement import X`` is just a stable alias. The
intent is exactly the same as ``re`` / ``json`` in the standard
library: a single named import for a layer that is conceptually one
thing even though the implementation is several files.

Five enforcement layers are exposed:

  1. **Authority matrix** (``AuthorityMatrix``, ``load_authority_matrix``)
     — declarative source of truth for "who wins in conflict?".
     Loaded from ``registry/authority_matrix.yaml``.

  2. **Authority guard** (``AuthorityGuard``, ``UnauthorizedActorError``)
     — runtime guard at the execution chokepoint
     (HARDEN-02 / INV-68). Verifies every ``ExecutionIntent`` was
     governance-approved, content-hash-matched, and originated from
     an actor declared in the matrix.

  3. **Policy engine** (``PolicyEngine``) — O(1) decision table for
     operator action authorization (PR #55 / I7 reframe). Used by
     the governance bridge before any state mutation.

  4. **Constraint engine** (``RuleGraph``, ``compile_rules``) —
     single rule-graph oracle for INV / SAFE / HAZ / SCVS / GOV /
     PERF. Pure / read-only; compiled from
     ``registry/constraint_rules.yaml``.

  5. **Adaptive freeze** (``LearningEvolutionFreezePolicy``,
     ``assert_unfrozen``) — HARDEN-04 / INV-70. Gates every
     adaptive-mutation emission point so the closed learning loop
     can be paused without code changes.

  6. **Kill switch** (``KillSwitch``, ``KillReason``) — SAFE-01
     primitive (P0-1b). Single chokepoint for system-wide
     ``SystemMode.LOCKED`` engagement.
"""

from __future__ import annotations

from core.constraint_engine import (
    CompiledRule,
    RuleAction,
    RuleGraph,
    RuleKind,
    RuleSeverity,
    compile_rules,
)
from core.contracts.learning_evolution_freeze import (
    LearningEvolutionFreezePolicy,
    LearningEvolutionFrozenError,
    assert_unfrozen,
    is_unfrozen,
)
from execution_engine.execution_gate import (
    AuthorityGuard,
    AuthorityViolation,
    UnauthorizedActorError,
)
from governance_engine.control_plane.policy_engine import (
    PolicyEngine,
    install_policy_table,
    verify_policy_table_hash,
)
from system.kill_switch import (
    KillReason,
    KillRequest,
    KillSwitch,
)
from system_engine.authority import (
    AuthorityActor,
    AuthorityMatrix,
    AuthorityOverride,
    ConflictRow,
    load_authority_matrix,
)

__all__ = (
    # Authority matrix
    "AuthorityActor",
    "AuthorityMatrix",
    "AuthorityOverride",
    "ConflictRow",
    "load_authority_matrix",
    # Authority guard
    "AuthorityGuard",
    "AuthorityViolation",
    "UnauthorizedActorError",
    # Policy engine
    "PolicyEngine",
    "install_policy_table",
    "verify_policy_table_hash",
    # Constraint engine
    "CompiledRule",
    "RuleAction",
    "RuleGraph",
    "RuleKind",
    "RuleSeverity",
    "compile_rules",
    # Adaptive freeze
    "LearningEvolutionFreezePolicy",
    "LearningEvolutionFrozenError",
    "assert_unfrozen",
    "is_unfrozen",
    # Kill switch
    "KillReason",
    "KillRequest",
    "KillSwitch",
)
