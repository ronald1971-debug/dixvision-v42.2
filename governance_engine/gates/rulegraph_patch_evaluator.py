"""P0-B — RuleGraph-backed patch evaluator.

Bridges :class:`core.constraint_engine.RuleGraph` (the constraint
oracle compiled from ``registry/constraint_rules.yaml``) and the
``CANARY → APPROVED`` edge of the patch pipeline FSM.

The evaluator builds a small, fully-typed fact mapping from a
:class:`core.contracts.learning.PatchProposal` + a
:class:`governance_engine.gates.QuantitativeMetrics` snapshot, asks the
:class:`RuleGraph` which rules fire, and converts the result into a
single verdict. The verdict is *pure* w.r.t. inputs — INV-15 — and
never mutates the rule graph or the proposal.

The constraint-engine surface is intentionally narrow: this evaluator
only inspects *which rules fired*; it never reads a clock, never
constructs typed bus events, and never executes a rule's
``action`` itself. Those remain owned by the engines named in
``registry/authority_matrix.yaml``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.constraint_engine import CompiledRule, RuleAction, RuleGraph
from core.contracts.learning import PatchProposal
from governance_engine.gates.quantitative_evaluator import QuantitativeMetrics


class RuleGraphPatchVerdictKind(StrEnum):
    """Three-valued verdict over a patch's RuleGraph evaluation."""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    HALTED = "HALTED"


# Rule actions that block the ``CANARY → APPROVED`` edge.
_BLOCKING_ACTIONS: frozenset[RuleAction] = frozenset({RuleAction.REJECT, RuleAction.HALT})


@dataclass(frozen=True, slots=True)
class PatchEvaluationFacts:
    """Frozen fact mapping derived from a patch + its quantitative metrics.

    The fields chosen here are deliberately the ones referenced by the
    ``GOV-PATCH-*`` rules in ``registry/constraint_rules.yaml`` plus a
    small set of stable identifiers (``patch_id`` / ``source`` /
    ``target_strategy``) so the verdict can be audited deterministically.

    Numeric fields use the same units as
    :class:`QuantitativeMetrics`:

    * ``sharpe_ratio`` / ``max_drawdown`` / ``samples`` — raw metrics.
    * ``is_oos_divergence_sigma`` — ``|is-oos| / is_std`` when
      ``is_std > 0``; otherwise the raw absolute divergence.

    Threshold fields (``sharpe_ratio_min`` / ``max_drawdown_max`` /
    ``samples_min`` / ``is_oos_divergence_max_sigma``) live alongside
    the metrics so the rule predicates can compare *fact_field op
    threshold_field* without external context.
    """

    patch_id: str
    source: str
    target_strategy: str
    sharpe_ratio: float
    max_drawdown: float
    samples: int
    is_oos_divergence_sigma: float
    sharpe_ratio_min: float
    max_drawdown_max: float
    samples_min: int
    is_oos_divergence_max_sigma: float

    def as_mapping(self) -> Mapping[str, Any]:
        """Return a deterministic mapping the constraint engine can read."""

        # Sorted keys keep canonical-JSON projections byte-identical
        # across replays (INV-15). Inserting in alphabetical order is
        # not strictly required by the constraint engine (the AST only
        # reads named fields) but it makes deterministic auditing
        # cheap.
        return {
            "is_oos_divergence_max_sigma": self.is_oos_divergence_max_sigma,
            "is_oos_divergence_sigma": self.is_oos_divergence_sigma,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_max": self.max_drawdown_max,
            "patch_id": self.patch_id,
            "samples": self.samples,
            "samples_min": self.samples_min,
            "sharpe_ratio": self.sharpe_ratio,
            "sharpe_ratio_min": self.sharpe_ratio_min,
            "source": self.source,
            "target_strategy": self.target_strategy,
        }


@dataclass(frozen=True, slots=True)
class RuleGraphPatchVerdict:
    """Frozen verdict over a single patch + its quantitative metrics."""

    kind: RuleGraphPatchVerdictKind
    passed: bool
    fired_rule_ids: tuple[str, ...] = ()
    blocking_rule_ids: tuple[str, ...] = ()
    detail: str = ""
    meta: Mapping[str, str] = field(default_factory=dict)


def build_patch_facts(
    *,
    proposal: PatchProposal,
    metrics: QuantitativeMetrics,
    sharpe_ratio_min: float,
    max_drawdown_max: float,
    samples_min: int,
    is_oos_divergence_max_sigma: float,
) -> PatchEvaluationFacts:
    """Pure builder — :class:`PatchProposal` + metrics → facts.

    Computes ``is_oos_divergence_sigma`` from
    :class:`QuantitativeMetrics` deterministically. When
    ``metrics.is_std == 0`` the absolute divergence is reported so a
    rule comparing ``> threshold`` still has meaningful semantics.
    """

    divergence_abs = abs(metrics.is_score - metrics.oos_score)
    if metrics.is_std > 0.0:
        divergence_sigma = divergence_abs / metrics.is_std
    else:
        divergence_sigma = divergence_abs

    return PatchEvaluationFacts(
        patch_id=proposal.patch_id,
        source=proposal.source,
        target_strategy=proposal.target_strategy,
        sharpe_ratio=metrics.sharpe_ratio,
        max_drawdown=metrics.max_drawdown,
        samples=metrics.samples,
        is_oos_divergence_sigma=divergence_sigma,
        sharpe_ratio_min=sharpe_ratio_min,
        max_drawdown_max=max_drawdown_max,
        samples_min=samples_min,
        is_oos_divergence_max_sigma=is_oos_divergence_max_sigma,
    )


class RuleGraphPatchEvaluator:
    """Constraint-engine-backed evaluator over a single patch + metrics.

    The evaluator wraps an already-compiled :class:`RuleGraph` so the
    same graph instance can be reused across patches with O(1) overhead
    per evaluation.
    """

    __slots__ = ("_rule_graph", "_rule_id_prefix")

    def __init__(
        self,
        *,
        rule_graph: RuleGraph,
        rule_id_prefix: str = "GOV-PATCH-",
    ) -> None:
        """Wire the evaluator against a compiled rule graph.

        ``rule_id_prefix`` filters which fired rules count towards the
        verdict. Only rules whose ``id`` starts with the prefix are
        considered — this prevents unrelated invariants firing on the
        shared fact mapping from blocking patch approval (which would
        be a *different* control-plane concern).
        """

        self._rule_graph = rule_graph
        self._rule_id_prefix = rule_id_prefix

    @property
    def rule_graph(self) -> RuleGraph:
        return self._rule_graph

    @property
    def rule_id_prefix(self) -> str:
        return self._rule_id_prefix

    # ------------------------------------------------------------------
    def evaluate(self, facts: PatchEvaluationFacts | Mapping[str, Any]) -> RuleGraphPatchVerdict:
        """Evaluate the rule graph against ``facts`` and return a verdict."""

        if isinstance(facts, PatchEvaluationFacts):
            mapping = facts.as_mapping()
        else:
            mapping = facts

        # Pre-filter by prefix so unrelated rules whose ``when``
        # predicate references facts outside the patch domain (e.g.
        # ``actor`` / ``hot_path_latency_ns``) never run on this
        # mapping. This keeps the patch-fact contract narrow and
        # avoids spurious ``KeyError`` from the expression engine.
        candidates: tuple[CompiledRule, ...] = tuple(
            r for r in self._rule_graph.rules if r.id.startswith(self._rule_id_prefix)
        )
        scoped = tuple(r for r in candidates if r.when_ast is not None and r.fires(mapping))
        fired_ids = tuple(sorted(r.id for r in scoped))
        blocking = tuple(sorted(r.id for r in scoped if r.action in _BLOCKING_ACTIONS))

        if not blocking:
            return RuleGraphPatchVerdict(
                kind=RuleGraphPatchVerdictKind.APPROVED,
                passed=True,
                fired_rule_ids=fired_ids,
                blocking_rule_ids=(),
                detail="no blocking rules fired",
            )

        # Determine whether any blocking rule asked for HALT (a harder
        # stop than REJECT). HALT escalates the verdict kind so the
        # bridge can wire it to a system-wide halt path if desired.
        any_halt = any(r.action == RuleAction.HALT for r in scoped if r.id in blocking)
        kind = RuleGraphPatchVerdictKind.HALTED if any_halt else RuleGraphPatchVerdictKind.REJECTED
        return RuleGraphPatchVerdict(
            kind=kind,
            passed=False,
            fired_rule_ids=fired_ids,
            blocking_rule_ids=blocking,
            detail="; ".join(blocking),
        )


__all__ = [
    "PatchEvaluationFacts",
    "RuleGraphPatchEvaluator",
    "RuleGraphPatchVerdict",
    "RuleGraphPatchVerdictKind",
    "build_patch_facts",
]
