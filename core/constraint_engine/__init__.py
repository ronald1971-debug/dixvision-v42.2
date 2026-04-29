"""Constraint engine — single rule-graph oracle for INV/SAFE/HAZ/SCVS/GOV/PERF.

The engine never mutates state. It compiles
``registry/constraint_rules.yaml`` (validated against
``registry/authority_matrix.yaml``) into an immutable :class:`RuleGraph`
that the runtime queries via :meth:`RuleGraph.evaluate`.
"""

from __future__ import annotations

from core.constraint_engine.compiler import (
    CompiledRule,
    RuleAction,
    RuleGraph,
    RuleKind,
    RuleSeverity,
    compile_rules,
)

__all__ = [
    "CompiledRule",
    "RuleAction",
    "RuleGraph",
    "RuleKind",
    "RuleSeverity",
    "compile_rules",
]
