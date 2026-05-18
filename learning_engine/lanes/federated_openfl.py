"""C-11 openfl — Intel-style federated-learning *plan* lane (OFFLINE_ONLY).

Canonical adaptation of ``intel/openfl`` (``openfl/federated/plan/plan.py``
and ``openfl/interface/{collaborator,aggregator}``) per
``DIX_MASTER_CANONICAL.md`` lines 2444–2452.

Differentiator vs C-09 (flwr) and C-10 (fedml):

* C-09 ships the *atomic* FedAvg primitive (single round, single parameter,
  weighted average).
* C-10 ships *topology orchestration* (hierarchical / ring) on top of that
  primitive — still a single round per call.
* C-11 ships the *multi-round plan*. OpenFL's central artefact is the
  ``plan.yaml`` — a declarative, version-controlled federation specification
  that fixes the number of rounds, the parameter under training, the
  collaborator allow-list, and the aggregation strategy ahead of time. The
  plan is hashed; running the same plan against the same per-round
  contributions yields a byte-identical multi-round report. That is exactly
  the property we need to give the harness a reproducible federation under
  INV-15.

Semantics:

* :class:`FederationPlan` — frozen, slotted, validated. Pins ``plan_id``,
  ``parameter``, the ordered ``collaborator_ids`` (sorted on construction
  for stability), the integer ``n_rounds`` (``>= 1``), and the per-round
  ``min_collaborators_per_round`` floor.
* :func:`plan_digest(plan)` — BLAKE2b-16 over the canonical bytes of the
  plan. Two plans that differ only in collaborator order have the same
  digest by construction.
* :class:`RoundContribution(round_index, collaborator_id, delta,
  num_samples, ts_ns)` — one collaborator's contribution to one round.
  ``round_index`` is zero-based and must be strictly ``< plan.n_rounds``.
* :class:`RoundReport(plan_id, round_index, n_collaborators,
  aggregated_delta, total_samples, parameter, ts_ns, digest)` — per-round
  weighted-average aggregate, structurally identical to the C-09
  :class:`FederatedAggregate` shape but carrying ``round_index`` instead of
  a freeform ``round_id``.
* :class:`MultiRoundReport(plan_id, plan_digest, n_rounds, rounds,
  final_value, initial_value, ts_ns, digest)` — the full multi-round
  report. ``rounds`` is in ascending ``round_index`` order; ``final_value``
  is the result of folding each round's aggregated delta into
  ``initial_value`` in order.
* :func:`execute_plan(plan, contributions, initial_value, ts_ns)` →
  ``(MultiRoundReport, LearningUpdate)`` — runs every round of the plan,
  validates per-round membership against the plan's collaborator list,
  applies FedAvg per round (delegating to :func:`fed_avg_aggregate`),
  folds the per-round delta into the running value, and emits a single
  domain :class:`LearningUpdate` for the *cumulative* multi-round delta.

Authority + safety constraints (mirrors C-09 / C-10):

* **L2 / B1.** OFFLINE-only. No runtime-tier imports, no top-level
  ``openfl`` / ``flwr`` / ``fedml`` / ``time`` / ``datetime`` / ``random`` /
  ``asyncio`` / ``os`` / ``subprocess`` / ``socket`` / ``ssl`` / ``numpy`` /
  ``torch`` / ``polars`` / ``pandas`` / ``requests`` / ``httpx`` /
  ``aiohttp`` / ``tornado`` / ``sqlite3``.
* **INV-15.** Every function is pure: no clock reads, no PRNG, no I/O.
  All ``ts_ns`` values are caller-supplied monotone event-time.
  Within-round aggregation order is canonicalised via the C-09 sort key;
  cross-round order is fixed by ``round_index``.
* **B27 / B28 / INV-71.** This lane never constructs transport-layer typed
  events. It produces a domain :class:`LearningUpdate` only; the existing
  :class:`UpdateEmitter` is the sole transport-layer constructor.
* **Privacy.** Per-round contributions are passed through C-09's
  :func:`verify_privacy` guard. The typed surface never carries a raw-data
  field — only ``delta`` (scalar) and ``num_samples`` (int).
* **Lazy seam.** ``NEW_PIP_DEPENDENCIES = ("openfl",)`` is declared but
  the ``openfl`` package is *never* imported. Plan orchestration is pure
  arithmetic over C-09 primitives.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from core.contracts.learning import LearningUpdate
from learning_engine.lanes.federated import (
    GradientUpdate,
    canonical_sort_updates,
    fed_avg_aggregate,
    verify_privacy,
)

OPENFL_VERSION = "v3.7-C11"
NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("openfl",)


# ---------------------------------------------------------------------------
# FederationPlan
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FederationPlan:
    """Declarative federation plan (analogue of OpenFL's ``plan.yaml``).

    The plan fixes — ahead of time — exactly which collaborators may
    contribute, what parameter is under training, how many rounds the
    federation will run, and the minimum number of collaborators a round
    needs to be valid. The plan is hashed and the hash travels in every
    :class:`RoundReport` / :class:`MultiRoundReport` so two runs against
    the same per-round contributions are byte-identical.
    """

    plan_id: str
    parameter: str
    collaborator_ids: tuple[str, ...]
    n_rounds: int
    min_collaborators_per_round: int
    aggregator_id: str = "aggregator-0"
    strategy: str = "fedavg"
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.plan_id:
            raise ValueError("plan_id must be non-empty")
        if not self.parameter:
            raise ValueError("parameter must be non-empty")
        if not self.aggregator_id:
            raise ValueError("aggregator_id must be non-empty")
        if self.strategy != "fedavg":
            raise ValueError(
                f"strategy {self.strategy!r} unsupported — only 'fedavg' for now",
            )
        if not self.collaborator_ids:
            raise ValueError("collaborator_ids must be non-empty")
        if len(set(self.collaborator_ids)) != len(self.collaborator_ids):
            raise ValueError("collaborator_ids must not contain duplicates")
        for c in self.collaborator_ids:
            if not c:
                raise ValueError("collaborator_id must be non-empty")
        # canonicalise: sort collaborators for stable digest
        object.__setattr__(
            self,
            "collaborator_ids",
            tuple(sorted(self.collaborator_ids)),
        )
        if self.n_rounds < 1:
            raise ValueError("n_rounds must be >= 1")
        if self.min_collaborators_per_round < 1:
            raise ValueError("min_collaborators_per_round must be >= 1")
        if self.min_collaborators_per_round > len(self.collaborator_ids):
            raise ValueError(
                "min_collaborators_per_round must be <= number of collaborators",
            )
        # canonicalise meta
        object.__setattr__(
            self,
            "meta",
            dict(sorted(self.meta.items())),
        )
        for k, v in self.meta.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise TypeError("plan meta must be str→str")


def _plan_canonical_bytes(plan: FederationPlan) -> bytes:
    """Canonical byte form of a plan for digest stability."""
    parts: list[str] = []
    parts.append(f"plan_id={plan.plan_id}\n")
    parts.append(f"parameter={plan.parameter}\n")
    parts.append(f"aggregator_id={plan.aggregator_id}\n")
    parts.append(f"strategy={plan.strategy}\n")
    parts.append(f"n_rounds={plan.n_rounds}\n")
    parts.append(f"min={plan.min_collaborators_per_round}\n")
    parts.append("collaborators:" + ",".join(plan.collaborator_ids) + "\n")
    parts.append("meta:" + ",".join(f"{k}={v}" for k, v in plan.meta.items()) + "\n")
    return "".join(parts).encode("utf-8")


def plan_digest(plan: FederationPlan) -> str:
    """BLAKE2b-16 digest over the plan's canonical bytes."""
    return hashlib.blake2b(_plan_canonical_bytes(plan), digest_size=16).hexdigest()


# ---------------------------------------------------------------------------
# RoundContribution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RoundContribution:
    """One collaborator's contribution to one round of a plan."""

    round_index: int
    collaborator_id: str
    delta: float
    num_samples: int
    ts_ns: int
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.round_index < 0:
            raise ValueError("round_index must be >= 0")
        if not self.collaborator_id:
            raise ValueError("collaborator_id must be non-empty")
        if not math.isfinite(self.delta):
            raise ValueError("delta must be finite (no NaN/Inf)")
        if self.num_samples < 0:
            raise ValueError("num_samples must be >= 0")
        if self.ts_ns < 0:
            raise ValueError("ts_ns must be >= 0 (monotone event-time)")
        object.__setattr__(self, "meta", dict(self.meta))
        for k, v in self.meta.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise TypeError("contribution meta must be str→str")

    def as_gradient_update(self, parameter: str) -> GradientUpdate:
        """Project this contribution onto a C-09 :class:`GradientUpdate`."""
        return GradientUpdate(
            client_id=self.collaborator_id,
            parameter=parameter,
            delta=self.delta,
            num_samples=self.num_samples,
            ts_ns=self.ts_ns,
            meta=dict(self.meta),
        )


# ---------------------------------------------------------------------------
# RoundReport / MultiRoundReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RoundReport:
    """Per-round weighted-average aggregate."""

    plan_id: str
    round_index: int
    parameter: str
    n_collaborators: int
    aggregated_delta: float
    total_samples: int
    ts_ns: int
    digest: str

    def __post_init__(self) -> None:
        if not self.plan_id:
            raise ValueError("plan_id must be non-empty")
        if self.round_index < 0:
            raise ValueError("round_index must be >= 0")
        if not self.parameter:
            raise ValueError("parameter must be non-empty")
        if self.n_collaborators < 0:
            raise ValueError("n_collaborators must be >= 0")
        if not math.isfinite(self.aggregated_delta):
            raise ValueError("aggregated_delta must be finite")
        if self.total_samples < 0:
            raise ValueError("total_samples must be >= 0")
        if self.ts_ns < 0:
            raise ValueError("ts_ns must be >= 0")
        if len(self.digest) != 32:
            raise ValueError("digest must be a 32-char BLAKE2b-16 hex string")


@dataclass(frozen=True, slots=True)
class MultiRoundReport:
    """Full multi-round report — output of :func:`execute_plan`."""

    plan_id: str
    plan_digest: str
    parameter: str
    n_rounds: int
    rounds: tuple[RoundReport, ...]
    initial_value: float
    final_value: float
    ts_ns: int
    digest: str

    def __post_init__(self) -> None:
        if not self.plan_id:
            raise ValueError("plan_id must be non-empty")
        if len(self.plan_digest) != 32:
            raise ValueError("plan_digest must be a 32-char BLAKE2b-16 hex string")
        if not self.parameter:
            raise ValueError("parameter must be non-empty")
        if self.n_rounds < 1:
            raise ValueError("n_rounds must be >= 1")
        if len(self.rounds) != self.n_rounds:
            raise ValueError("rounds length must equal n_rounds")
        for i, r in enumerate(self.rounds):
            if r.round_index != i:
                raise ValueError("rounds must be in ascending round_index order")
            if r.plan_id != self.plan_id:
                raise ValueError("round.plan_id must equal outer plan_id")
            if r.parameter != self.parameter:
                raise ValueError("round.parameter must equal outer parameter")
        if not math.isfinite(self.initial_value):
            raise ValueError("initial_value must be finite")
        if not math.isfinite(self.final_value):
            raise ValueError("final_value must be finite")
        if self.ts_ns < 0:
            raise ValueError("ts_ns must be >= 0")
        if len(self.digest) != 32:
            raise ValueError("digest must be a 32-char BLAKE2b-16 hex string")


# ---------------------------------------------------------------------------
# Per-round canonical bytes and digest
# ---------------------------------------------------------------------------


def _round_canonical_bytes(
    plan: FederationPlan,
    round_index: int,
    sorted_updates: Sequence[GradientUpdate],
) -> bytes:
    """Canonical byte form of one round (plan ref + sorted updates)."""
    parts: list[str] = []
    parts.append(f"plan_id={plan.plan_id}\n")
    parts.append(f"round_index={round_index}\n")
    parts.append(f"parameter={plan.parameter}\n")
    for u in sorted_updates:
        parts.append(
            f"{u.client_id}|{u.delta!r}|{u.num_samples}|{u.ts_ns}\n",
        )
    return "".join(parts).encode("utf-8")


def _round_digest(
    plan: FederationPlan,
    round_index: int,
    sorted_updates: Sequence[GradientUpdate],
) -> str:
    body = _round_canonical_bytes(plan, round_index, sorted_updates)
    return hashlib.blake2b(body, digest_size=16).hexdigest()


def _multi_round_digest(
    plan: FederationPlan,
    rounds: Sequence[RoundReport],
    initial_value: float,
    final_value: float,
) -> str:
    parts: list[str] = []
    parts.append(f"plan={plan_digest(plan)}\n")
    parts.append(f"initial={initial_value!r}\n")
    parts.append(f"final={final_value!r}\n")
    for r in rounds:
        parts.append(
            f"{r.round_index}|{r.aggregated_delta!r}|{r.total_samples}|{r.digest}\n",
        )
    return hashlib.blake2b(
        "".join(parts).encode("utf-8"),
        digest_size=16,
    ).hexdigest()


# ---------------------------------------------------------------------------
# Plan execution
# ---------------------------------------------------------------------------


def _contributions_per_round(
    plan: FederationPlan,
    contributions: Sequence[RoundContribution],
) -> tuple[tuple[GradientUpdate, ...], ...]:
    """Group contributions by round_index, canonical-sort within each round.

    Validates:
      * every collaborator_id appears in plan.collaborator_ids,
      * no round_index >= plan.n_rounds,
      * every round has at least min_collaborators_per_round contributions,
      * within a round, no duplicate collaborator_id,
      * privacy guard fires on every contribution.
    """
    allowed: frozenset[str] = frozenset(plan.collaborator_ids)
    by_round: dict[int, list[GradientUpdate]] = {i: [] for i in range(plan.n_rounds)}
    seen_per_round: dict[int, set[str]] = {i: set() for i in range(plan.n_rounds)}
    for c in contributions:
        if c.round_index >= plan.n_rounds:
            raise ValueError(
                f"contribution round_index {c.round_index} >= plan.n_rounds {plan.n_rounds}",
            )
        if c.collaborator_id not in allowed:
            raise ValueError(
                f"collaborator {c.collaborator_id!r} not in plan.collaborator_ids",
            )
        if c.collaborator_id in seen_per_round[c.round_index]:
            raise ValueError(
                f"duplicate collaborator {c.collaborator_id!r} in round {c.round_index}",
            )
        seen_per_round[c.round_index].add(c.collaborator_id)
        gu = c.as_gradient_update(plan.parameter)
        verify_privacy(gu)
        by_round[c.round_index].append(gu)
    for i in range(plan.n_rounds):
        if len(by_round[i]) < plan.min_collaborators_per_round:
            raise ValueError(
                f"round {i} has {len(by_round[i])} contributors, "
                f"plan.min_collaborators_per_round={plan.min_collaborators_per_round}",
            )
    return tuple(tuple(canonical_sort_updates(by_round[i])) for i in range(plan.n_rounds))


def execute_plan(
    *,
    plan: FederationPlan,
    contributions: Sequence[RoundContribution],
    initial_value: float,
    ts_ns: int,
) -> tuple[MultiRoundReport, LearningUpdate]:
    """Execute every round of the plan deterministically.

    For each ``round_index`` in ``[0, plan.n_rounds)`` the function:

    1. selects the contributions targeting that round,
    2. canonical-sorts them via :func:`canonical_sort_updates`,
    3. aggregates via :func:`fed_avg_aggregate`,
    4. folds the per-round delta into the running parameter value.

    Returns:
        ``(MultiRoundReport, LearningUpdate)`` — the typed multi-round
        report plus a domain :class:`LearningUpdate` whose ``new_value`` is
        the cumulative value after all rounds and whose ``reason``
        records the plan digest + round count.
    """
    if not math.isfinite(initial_value):
        raise ValueError("initial_value must be finite")
    if ts_ns < 0:
        raise ValueError("ts_ns must be >= 0")

    per_round = _contributions_per_round(plan, contributions)
    pdigest = plan_digest(plan)

    running_value = initial_value
    round_reports: list[RoundReport] = []
    for i in range(plan.n_rounds):
        sorted_updates = per_round[i]
        agg_delta, total_samples = fed_avg_aggregate(sorted_updates)
        rdigest = _round_digest(plan, i, sorted_updates)
        round_reports.append(
            RoundReport(
                plan_id=plan.plan_id,
                round_index=i,
                parameter=plan.parameter,
                n_collaborators=len(sorted_updates),
                aggregated_delta=agg_delta,
                total_samples=total_samples,
                ts_ns=ts_ns,
                digest=rdigest,
            ),
        )
        running_value = running_value + agg_delta

    mdigest = _multi_round_digest(plan, round_reports, initial_value, running_value)
    report = MultiRoundReport(
        plan_id=plan.plan_id,
        plan_digest=pdigest,
        parameter=plan.parameter,
        n_rounds=plan.n_rounds,
        rounds=tuple(round_reports),
        initial_value=initial_value,
        final_value=running_value,
        ts_ns=ts_ns,
        digest=mdigest,
    )

    if not math.isfinite(running_value):
        raise ValueError("final_value not finite after plan execution")
    update = LearningUpdate(
        ts_ns=ts_ns,
        strategy_id=plan.aggregator_id,
        parameter=plan.parameter,
        old_value=repr(initial_value),
        new_value=repr(running_value),
        reason=(
            f"openfl_plan plan_id={plan.plan_id} n_rounds={plan.n_rounds} "
            f"plan_digest={pdigest} report_digest={mdigest}"
        ),
        meta={
            "lane": "federated_openfl",
            "version": OPENFL_VERSION,
            "plan_id": plan.plan_id,
            "plan_digest": pdigest,
            "n_rounds": str(plan.n_rounds),
            "report_digest": mdigest,
        },
    )
    return report, update


__all__ = [
    "NEW_PIP_DEPENDENCIES",
    "OPENFL_VERSION",
    "FederationPlan",
    "MultiRoundReport",
    "RoundContribution",
    "RoundReport",
    "execute_plan",
    "plan_digest",
]
