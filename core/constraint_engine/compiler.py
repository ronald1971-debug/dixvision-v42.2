"""Constraint-rule compiler for the v42.2 control plane.

Loads ``registry/constraint_rules.yaml`` into a strictly-validated
``RuleGraph`` of frozen dataclasses. The compiler enforces:

* every rule references a known authority-matrix actor as its ``owner``;
* every dependency edge points at an existing rule (no dangling refs);
* the dependency graph is a DAG (no cycles);
* ``severity`` and ``action`` fall in the closed enumerations defined
  here;
* ``kind`` falls in the closed enumeration defined here;
* the optional ``when`` predicate parses against the small expression
  grammar in :mod:`core.constraint_engine.expr`.

The rule graph is pure data: the only runtime API is
``RuleGraph.evaluate(facts)`` which returns the rules whose ``when``
predicate fires for a given fact mapping, in topological order. There
are no side effects, no clock reads, no PRNG, and no I/O. INV-15.

The constraint engine intentionally does NOT execute hazard escalation
or governance writes itself — those remain owned by the engines named
in ``authority_matrix.yaml``. The graph is the *oracle* the runtime
queries, not the actor that mutates state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from core.constraint_engine import expr as expr_mod
from system_engine.authority import AuthorityMatrix, load_authority_matrix


class RuleKind(StrEnum):
    INV = "INV"
    SAFE = "SAFE"
    HAZ = "HAZ"
    SCVS = "SCVS"
    GOV = "GOV"
    PERF = "PERF"


class RuleSeverity(StrEnum):
    BLOCK = "BLOCK"
    HIGH = "HIGH"
    WARN = "WARN"
    AUDIT = "AUDIT"


class RuleAction(StrEnum):
    REJECT = "REJECT"
    HALT = "HALT"
    HAZARD_EMIT = "HAZARD_EMIT"
    WARN = "WARN"
    AUDIT = "AUDIT"


@dataclass(frozen=True, slots=True)
class CompiledRule:
    """A single rule after compilation."""

    id: str
    kind: RuleKind
    severity: RuleSeverity
    action: RuleAction
    owner: str
    description: str
    depends_on: tuple[str, ...] = ()
    when_src: str | None = None
    when_ast: expr_mod.Expr | None = None
    facts: frozenset[str] = field(default_factory=frozenset)
    notes: str | None = None

    def fires(self, facts: Mapping[str, Any]) -> bool:
        """Return True iff the rule's ``when`` predicate evaluates true.

        Rules without a ``when`` clause never *automatically fire*; the
        runtime evaluates them by direct invariant check elsewhere. Only
        rules with a predicate have an opinion that the constraint
        engine itself can express, so we return False for the rest.
        """

        if self.when_ast is None:
            return False
        return expr_mod.evaluate(self.when_ast, facts)


@dataclass(frozen=True, slots=True)
class RuleGraph:
    """Compiled, immutable rule graph."""

    version: str
    rules: tuple[CompiledRule, ...]
    order: tuple[str, ...]  # topological order of rule ids
    by_id: Mapping[str, CompiledRule]
    matrix: AuthorityMatrix

    def get(self, rule_id: str) -> CompiledRule:
        if rule_id not in self.by_id:
            raise KeyError(f"unknown rule {rule_id!r}")
        return self.by_id[rule_id]

    def evaluate(self, facts: Mapping[str, Any]) -> tuple[CompiledRule, ...]:
        """Return rules whose ``when`` predicate fires, in topo order.

        Pure / deterministic. Same facts → same result.
        """

        out: list[CompiledRule] = []
        for rid in self.order:
            rule = self.by_id[rid]
            if rule.when_ast is not None and rule.fires(facts):
                out.append(rule)
        return tuple(out)

    def rules_owned_by(self, actor_id: str) -> tuple[CompiledRule, ...]:
        return tuple(r for r in self.rules if r.owner == actor_id)

    def rules_of_kind(self, kind: RuleKind) -> tuple[CompiledRule, ...]:
        return tuple(r for r in self.rules if r.kind == kind)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _expect_keys(body: Mapping[str, Any], required: tuple[str, ...]) -> None:
    missing = [k for k in required if k not in body]
    if missing:
        raise ValueError(
            f"constraint_rules: missing required top-level keys: {sorted(missing)!r}"
        )


def _topo_sort(
    rules: Mapping[str, tuple[str, ...]], rule_ids: list[str]
) -> tuple[str, ...]:
    """Kahn's algorithm with deterministic ordering by rule id."""

    indegree: dict[str, int] = {rid: 0 for rid in rule_ids}
    children: dict[str, list[str]] = {rid: [] for rid in rule_ids}
    for rid, deps in rules.items():
        for dep in deps:
            indegree[rid] += 1
            children[dep].append(rid)

    ready = sorted(rid for rid, deg in indegree.items() if deg == 0)
    out: list[str] = []
    while ready:
        rid = ready.pop(0)
        out.append(rid)
        for child in sorted(children[rid]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    if len(out) != len(rule_ids):
        cycle = sorted(rid for rid, deg in indegree.items() if deg > 0)
        raise ValueError(f"constraint_rules: dependency cycle detected: {cycle!r}")
    return tuple(out)


def compile_rules(
    rules_path: str | Path,
    *,
    matrix: AuthorityMatrix | None = None,
    matrix_path: str | Path | None = None,
) -> RuleGraph:
    """Load, validate, and compile ``constraint_rules.yaml``.

    Either ``matrix`` (already-loaded) or ``matrix_path`` may be passed;
    if both are omitted, the canonical
    ``registry/authority_matrix.yaml`` next to the rules file is used.
    """

    path = Path(rules_path)
    body = yaml.safe_load(path.read_text())
    if not isinstance(body, Mapping):
        raise ValueError("constraint_rules: top-level must be a mapping")

    _expect_keys(body, ("version", "rules"))

    if matrix is None:
        if matrix_path is None:
            matrix_path = path.parent / "authority_matrix.yaml"
        matrix = load_authority_matrix(matrix_path)
    actor_ids = matrix.actor_ids

    raw_rules = body["rules"]
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError("constraint_rules: 'rules' must be a non-empty list")

    seen: set[str] = set()
    by_id: dict[str, CompiledRule] = {}
    deps_map: dict[str, tuple[str, ...]] = {}

    for raw in raw_rules:
        if not isinstance(raw, Mapping):
            raise ValueError("constraint_rules: every rule must be a mapping")
        for required in ("id", "kind", "severity", "action", "owner", "description"):
            if required not in raw:
                raise ValueError(
                    f"constraint_rules: rule missing required key {required!r}: {raw!r}"
                )
        rid = str(raw["id"])
        if rid in seen:
            raise ValueError(f"constraint_rules: duplicate rule id {rid!r}")
        seen.add(rid)

        try:
            kind = RuleKind(raw["kind"])
        except ValueError as e:
            raise ValueError(
                f"constraint_rules: rule {rid!r} has unknown kind {raw['kind']!r}"
            ) from e
        try:
            severity = RuleSeverity(raw["severity"])
        except ValueError as e:
            raise ValueError(
                f"constraint_rules: rule {rid!r} has unknown severity {raw['severity']!r}"
            ) from e
        try:
            action = RuleAction(raw["action"])
        except ValueError as e:
            raise ValueError(
                f"constraint_rules: rule {rid!r} has unknown action {raw['action']!r}"
            ) from e

        owner = str(raw["owner"])
        if owner not in actor_ids:
            raise ValueError(
                f"constraint_rules: rule {rid!r} owner {owner!r} is not a "
                f"declared authority-matrix actor"
            )

        deps_raw = raw.get("depends_on")
        if deps_raw is None:
            deps_raw = []
        if not isinstance(deps_raw, list):
            raise ValueError(
                f"constraint_rules: rule {rid!r} 'depends_on' must be a list"
            )
        deps = tuple(str(d) for d in deps_raw)
        if rid in deps:
            raise ValueError(f"constraint_rules: rule {rid!r} self-dependency")

        when_src_raw = raw.get("when")
        when_src: str | None = None
        when_ast: expr_mod.Expr | None = None
        facts: frozenset[str] = frozenset()
        if when_src_raw is not None:
            when_src = str(when_src_raw).strip()
            if not when_src:
                raise ValueError(
                    f"constraint_rules: rule {rid!r} has empty 'when' clause"
                )
            try:
                when_ast = expr_mod.parse(when_src)
            except ValueError as e:
                raise ValueError(
                    f"constraint_rules: rule {rid!r} has invalid 'when' "
                    f"expression: {e}"
                ) from e
            facts = expr_mod.free_idents(when_ast)

        notes_raw = raw.get("notes")
        notes = str(notes_raw).strip() if notes_raw is not None else None

        by_id[rid] = CompiledRule(
            id=rid,
            kind=kind,
            severity=severity,
            action=action,
            owner=owner,
            description=str(raw["description"]),
            depends_on=deps,
            when_src=when_src,
            when_ast=when_ast,
            facts=facts,
            notes=notes,
        )
        deps_map[rid] = deps

    # Validate dependency targets
    for rid, deps in deps_map.items():
        for dep in deps:
            if dep not in by_id:
                raise ValueError(
                    f"constraint_rules: rule {rid!r} depends on unknown rule {dep!r}"
                )

    order = _topo_sort(deps_map, list(by_id.keys()))

    return RuleGraph(
        version=str(body["version"]),
        rules=tuple(by_id[rid] for rid in order),
        order=order,
        by_id=by_id,
        matrix=matrix,
    )


__all__ = [
    "CompiledRule",
    "RuleAction",
    "RuleGraph",
    "RuleKind",
    "RuleSeverity",
    "compile_rules",
]
