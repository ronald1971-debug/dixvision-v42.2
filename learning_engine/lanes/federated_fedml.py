"""C-10 fedml — distributed-ML orchestration lane (OFFLINE_ONLY).

Canonical adaptation of ``FedML-AI/FedML`` (`fedml/core/distributed/` +
`fedml/ml/trainer/`) into a deterministic, OFFLINE-only federated learning
overlay on top of the FedAvg primitives shipped in C-09 (`federated.py`).

Differentiator vs C-09 flwr:

* FedML's value over Flower is *federation topology*: hierarchical
  (two-tier client → group-aggregator → root), ring (sequential
  all-reduce-style fold), and more complex split-learning patterns. The
  same FedAvg arithmetic underpins each, but the *order of aggregation*
  is structured differently.
* This lane adds :class:`FederationTopology` and two structured
  aggregation entry points — :func:`hierarchical_aggregate` and
  :func:`ring_aggregate` — that both reduce to the same final weighted
  average a flat FedAvg round would produce, but expose per-group /
  per-step structured records for telemetry and audit.

Authority + safety constraints (mirrors C-09):

* **L2 / B1.** Lives in the offline ``learning_engine``. Top-level
  imports are ``__future__`` / ``collections.abc`` / ``dataclasses`` /
  ``enum`` / ``hashlib`` / ``math`` plus C-09 (`.federated`) and
  :mod:`core.contracts.learning`. No runtime-tier imports, no ``fedml``
  / ``flwr`` / ``numpy`` / ``torch`` / ``time`` / ``datetime`` /
  ``random`` / ``asyncio`` / ``os`` / ``polars`` / ``requests`` / etc.
* **INV-15.** Pure / deterministic. Group iteration order is fixed by
  sorting on ``group_id`` (and ``client_id`` within each group); floating
  point summation order is therefore byte-stable across replays.
* **B27 / B28 / INV-71.** This lane never constructs transport-layer
  typed events. It produces a :class:`LearningUpdate` domain record
  only; :class:`UpdateEmitter` is the sole transport-layer constructor.
* **Privacy.** Inherits :func:`verify_privacy` from C-09 — only
  ``delta`` + ``num_samples`` travel.
* **Lazy seam.** ``NEW_PIP_DEPENDENCIES = ("fedml",)`` declared but the
  ``fedml`` package is never imported — topology-only orchestration is
  pure arithmetic over C-09 primitives.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from core.contracts.learning import LearningUpdate
from learning_engine.lanes.federated import (
    FederatedAggregate,
    GradientUpdate,
    canonical_sort_updates,
    fed_avg_aggregate,
    is_valid_round,
    updates_digest,
    verify_privacy,
)

FEDML_VERSION = "v3.7-C10"
NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("fedml",)


# ---------------------------------------------------------------------------
# Topology enum
# ---------------------------------------------------------------------------


class FederationTopology(StrEnum):
    """Federation topology kinds supported by this lane.

    * ``FLAT``: clients aggregate directly into the root (delegates to
      C-09's :func:`aggregate_round`).
    * ``HIERARCHICAL``: two-tier — clients aggregate into named groups,
      groups then aggregate into the root. Same final number; structured
      per-group records.
    * ``RING``: clients folded sequentially in a deterministic order
      (sorted by ``client_id``). Same final number as flat FedAvg;
      structured per-step state for audit.
    """

    FLAT = "flat"
    HIERARCHICAL = "hierarchical"
    RING = "ring"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GroupAssignment:
    """Static mapping from a group identifier to its member clients.

    Used by :func:`hierarchical_aggregate` to partition a round's updates
    into named sub-rounds. Each ``client_id`` listed must correspond to
    exactly one :class:`GradientUpdate` in the round (no orphans, no
    overlaps across groups).

    Attributes:
        group_id: Stable non-empty identifier for the group. Used as a
            primary sort key for canonical iteration.
        client_ids: Tuple of member client identifiers. Non-empty, with
            no duplicates within the group.
    """

    group_id: str
    client_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("group_id must be non-empty")
        if not self.client_ids:
            raise ValueError("client_ids must be non-empty")
        if len(set(self.client_ids)) != len(self.client_ids):
            raise ValueError("client_ids must not contain duplicates")
        for c in self.client_ids:
            if not c:
                raise ValueError("client_id must be non-empty")


@dataclass(frozen=True, slots=True)
class HierarchicalRoundResult:
    """Structured result of a :func:`hierarchical_aggregate` round.

    Attributes:
        round_id: Caller-supplied round identifier.
        parameter: Name of the aggregated parameter.
        n_groups: Number of groups that contributed.
        group_aggregates: Tuple of per-group :class:`FederatedAggregate`,
            ordered by ``group_id`` ascending.
        root_aggregate: The root :class:`FederatedAggregate` produced by
            aggregating the group results (weighted by each group's
            ``total_samples``).
        ts_ns: Caller-supplied monotone event-time.
        digest: BLAKE2b-16 digest over the canonical bytes of the input
            updates AND the group assignments. Pins INV-15.
    """

    round_id: str
    parameter: str
    n_groups: int
    group_aggregates: tuple[FederatedAggregate, ...]
    root_aggregate: FederatedAggregate
    ts_ns: int
    digest: str

    def __post_init__(self) -> None:
        if not self.round_id:
            raise ValueError("round_id must be non-empty")
        if not self.parameter:
            raise ValueError("parameter must be non-empty")
        if self.n_groups < 1:
            raise ValueError("n_groups must be >= 1")
        if len(self.group_aggregates) != self.n_groups:
            raise ValueError("group_aggregates length must equal n_groups")
        if self.ts_ns < 0:
            raise ValueError("ts_ns must be non-negative")
        if len(self.digest) != 32 or not all(c in "0123456789abcdef" for c in self.digest):
            raise ValueError("digest must be a 32-char lowercase hex string")


@dataclass(frozen=True, slots=True)
class RingStep:
    """One step of a ring all-reduce-style fold.

    Attributes:
        step_index: 0-based position of this step in the ring.
        client_id: The client folded in at this step.
        running_weighted_sum: Running ``Σᵢ (nᵢ · δᵢ)`` after this step.
        running_total_samples: Running ``Σᵢ nᵢ`` after this step.
    """

    step_index: int
    client_id: str
    running_weighted_sum: float
    running_total_samples: int

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        if not self.client_id:
            raise ValueError("client_id must be non-empty")
        if not math.isfinite(self.running_weighted_sum):
            raise ValueError("running_weighted_sum must be finite")
        if self.running_total_samples < 0:
            raise ValueError("running_total_samples must be non-negative")


@dataclass(frozen=True, slots=True)
class RingRoundResult:
    """Structured result of a :func:`ring_aggregate` round.

    Same final ``aggregated_delta`` and ``total_samples`` as a flat
    FedAvg round, but with the sequential ring trajectory exposed for
    audit.
    """

    round_id: str
    parameter: str
    ring_order: tuple[str, ...]
    aggregated_delta: float
    total_samples: int
    steps: tuple[RingStep, ...]
    ts_ns: int
    digest: str

    def __post_init__(self) -> None:
        if not self.round_id:
            raise ValueError("round_id must be non-empty")
        if not self.parameter:
            raise ValueError("parameter must be non-empty")
        if len(self.ring_order) != len(self.steps):
            raise ValueError("ring_order length must equal steps length")
        if not math.isfinite(self.aggregated_delta):
            raise ValueError("aggregated_delta must be finite")
        if self.total_samples < 0:
            raise ValueError("total_samples must be non-negative")
        if self.ts_ns < 0:
            raise ValueError("ts_ns must be non-negative")
        if len(self.digest) != 32 or not all(c in "0123456789abcdef" for c in self.digest):
            raise ValueError("digest must be a 32-char lowercase hex string")


# ---------------------------------------------------------------------------
# Partition helpers
# ---------------------------------------------------------------------------


def is_valid_group_partition(
    updates: Sequence[GradientUpdate],
    groups: Sequence[GroupAssignment],
) -> bool:
    """``True`` iff ``groups`` is a valid partition of ``updates``' clients.

    Constraints:

    * Every ``client_id`` that appears in ``updates`` must belong to
      exactly one group.
    * No group lists a ``client_id`` that does not appear in ``updates``.
    * Group identifiers must be distinct across ``groups``.
    """
    if not updates or not groups:
        return False
    group_ids = [g.group_id for g in groups]
    if len(set(group_ids)) != len(group_ids):
        return False
    all_group_clients: list[str] = []
    for g in groups:
        all_group_clients.extend(g.client_ids)
    if len(set(all_group_clients)) != len(all_group_clients):
        return False  # client listed in two groups
    update_clients = {u.client_id for u in updates}
    if set(all_group_clients) != update_clients:
        return False
    return True


def partition_into_groups(
    updates: Sequence[GradientUpdate],
    groups: Sequence[GroupAssignment],
) -> dict[str, tuple[GradientUpdate, ...]]:
    """Bucket ``updates`` by group.

    Returns a dict mapping each ``group_id`` (in lexicographic order via
    sorted iteration at call sites — Python preserves insertion order, so
    we insert in lexicographic order here) to its tuple of updates
    canonical-sorted by ``(client_id, ts_ns)``.

    Raises:
        ValueError: If ``groups`` is not a valid partition of ``updates``.
    """
    if not is_valid_group_partition(updates, groups):
        raise ValueError("groups is not a valid partition of updates' clients")
    client_to_group: dict[str, str] = {}
    for g in groups:
        for c in g.client_ids:
            client_to_group[c] = g.group_id
    buckets: dict[str, list[GradientUpdate]] = {}
    for u in updates:
        gid = client_to_group[u.client_id]
        buckets.setdefault(gid, []).append(u)
    result: dict[str, tuple[GradientUpdate, ...]] = {}
    for gid in sorted(buckets.keys()):
        result[gid] = canonical_sort_updates(buckets[gid])
    return result


# ---------------------------------------------------------------------------
# Canonical bytes / digest over a hierarchical input
# ---------------------------------------------------------------------------


def _hierarchical_canonical_bytes(
    updates: Sequence[GradientUpdate],
    groups: Sequence[GroupAssignment],
) -> bytes:
    """Canonical byte form of (updates, groups) for digest stability."""
    parts: list[str] = []
    parts.append("UPDATES\n")
    for u in canonical_sort_updates(updates):
        parts.append(
            f"{u.client_id}|{u.parameter}|{u.delta!r}|{u.num_samples}|{u.ts_ns}\n",
        )
    parts.append("GROUPS\n")
    for g in sorted(groups, key=lambda x: x.group_id):
        clients = ",".join(sorted(g.client_ids))
        parts.append(f"{g.group_id}|{clients}\n")
    return "".join(parts).encode("utf-8")


def hierarchical_digest(
    updates: Sequence[GradientUpdate],
    groups: Sequence[GroupAssignment],
) -> str:
    """BLAKE2b-16 digest over canonical bytes of (updates, groups)."""
    body = _hierarchical_canonical_bytes(updates, groups)
    return hashlib.blake2b(body, digest_size=16).hexdigest()


# ---------------------------------------------------------------------------
# Hierarchical aggregation
# ---------------------------------------------------------------------------


def hierarchical_aggregate(
    *,
    round_id: str,
    strategy_id: str,
    parameter: str,
    current_value: float,
    updates: Sequence[GradientUpdate],
    groups: Sequence[GroupAssignment],
    ts_ns: int,
) -> tuple[HierarchicalRoundResult, LearningUpdate]:
    """Two-tier FedML-style hierarchical aggregation.

    Math (for a parameter scalar):

    .. code:: text

        group_aggregate_g = Σ_{c ∈ g} (nᵢ · δᵢ) / Σ_{c ∈ g} nᵢ
        root_aggregate    = Σ_g (Nᵍ · group_aggregate_g) / Σ_g Nᵍ
        where Nᵍ = Σ_{c ∈ g} nᵢ

    By construction this reduces to flat FedAvg on the full set —
    sample-weighted averaging is associative under partition. The point
    of the hierarchical path is *audit structure* (per-group records),
    not a different math.
    """
    if not round_id:
        raise ValueError("round_id must be non-empty")
    if not strategy_id:
        raise ValueError("strategy_id must be non-empty")
    if not parameter:
        raise ValueError("parameter must be non-empty")
    if not math.isfinite(current_value):
        raise ValueError("current_value must be finite")
    if ts_ns < 0:
        raise ValueError("ts_ns must be non-negative")
    for u in updates:
        verify_privacy(u)
        if u.parameter != parameter:
            raise ValueError(
                f"GradientUpdate.parameter mismatch: round targets {parameter!r}, "
                f"client {u.client_id!r} sent {u.parameter!r}",
            )
    if not is_valid_round(updates, min_clients=1):
        raise ValueError("invalid round: empty or zero-total-samples")
    buckets = partition_into_groups(updates, groups)
    group_aggregates: list[FederatedAggregate] = []
    group_means: list[tuple[float, int]] = []
    for gid in sorted(buckets.keys()):
        members = buckets[gid]
        agg_delta, agg_samples = fed_avg_aggregate(members)
        group_digest = updates_digest(members)
        group_aggregates.append(
            FederatedAggregate(
                round_id=f"{round_id}::{gid}",
                parameter=parameter,
                n_clients=len({m.client_id for m in members}),
                aggregated_delta=agg_delta,
                total_samples=agg_samples,
                ts_ns=ts_ns,
                digest=group_digest,
            ),
        )
        group_means.append((agg_delta, agg_samples))
    root_weighted_sum = sum(d * n for d, n in group_means)
    root_total_samples = sum(n for _, n in group_means)
    if root_total_samples == 0:
        raise ValueError("hierarchical_aggregate: total samples must be > 0")
    root_delta = root_weighted_sum / root_total_samples
    if not math.isfinite(root_delta):
        raise ValueError("root_delta not finite")
    digest = hierarchical_digest(updates, groups)
    root_aggregate = FederatedAggregate(
        round_id=round_id,
        parameter=parameter,
        n_clients=len({u.client_id for u in updates}),
        aggregated_delta=root_delta,
        total_samples=root_total_samples,
        ts_ns=ts_ns,
        digest=digest,
    )
    new_value = current_value + root_delta
    if not math.isfinite(new_value):
        raise ValueError("new_value not finite")
    result = HierarchicalRoundResult(
        round_id=round_id,
        parameter=parameter,
        n_groups=len(group_aggregates),
        group_aggregates=tuple(group_aggregates),
        root_aggregate=root_aggregate,
        ts_ns=ts_ns,
        digest=digest,
    )
    learning_update = LearningUpdate(
        ts_ns=ts_ns,
        strategy_id=strategy_id,
        parameter=parameter,
        old_value=repr(current_value),
        new_value=repr(new_value),
        reason=(
            f"federated_fedml.hierarchical round={round_id} "
            f"n_groups={len(group_aggregates)} "
            f"total_samples={root_total_samples} "
            f"delta={root_delta!r} digest={digest}"
        ),
        meta={
            "lane": "federated_fedml",
            "topology": FederationTopology.HIERARCHICAL.value,
            "version": FEDML_VERSION,
            "round_id": round_id,
            "digest": digest,
            "n_groups": str(len(group_aggregates)),
            "total_samples": str(root_total_samples),
        },
    )
    return result, learning_update


# ---------------------------------------------------------------------------
# Ring aggregation
# ---------------------------------------------------------------------------


def ring_aggregate(
    *,
    round_id: str,
    strategy_id: str,
    parameter: str,
    current_value: float,
    updates: Sequence[GradientUpdate],
    ts_ns: int,
) -> tuple[RingRoundResult, LearningUpdate]:
    """All-reduce-style ring fold over ``updates``.

    Clients are visited in canonical-sort order (``(client_id, ts_ns)``).
    Per-step running sums are exposed via :class:`RingStep` for audit.
    The final number matches flat FedAvg by construction.
    """
    if not round_id:
        raise ValueError("round_id must be non-empty")
    if not strategy_id:
        raise ValueError("strategy_id must be non-empty")
    if not parameter:
        raise ValueError("parameter must be non-empty")
    if not math.isfinite(current_value):
        raise ValueError("current_value must be finite")
    if ts_ns < 0:
        raise ValueError("ts_ns must be non-negative")
    for u in updates:
        verify_privacy(u)
        if u.parameter != parameter:
            raise ValueError(
                f"GradientUpdate.parameter mismatch: round targets {parameter!r}, "
                f"client {u.client_id!r} sent {u.parameter!r}",
            )
    if not is_valid_round(updates, min_clients=1):
        raise ValueError("invalid round: empty or zero-total-samples")
    sorted_updates = canonical_sort_updates(updates)
    ring_order: list[str] = []
    seen: set[str] = set()
    for u in sorted_updates:
        if u.client_id not in seen:
            ring_order.append(u.client_id)
            seen.add(u.client_id)
    steps: list[RingStep] = []
    running_sum = 0.0
    running_samples = 0
    for idx, u in enumerate(sorted_updates):
        running_sum += u.num_samples * u.delta
        running_samples += u.num_samples
        steps.append(
            RingStep(
                step_index=idx,
                client_id=u.client_id,
                running_weighted_sum=running_sum,
                running_total_samples=running_samples,
            ),
        )
    if running_samples == 0:
        raise ValueError("ring_aggregate: total samples must be > 0")
    aggregated_delta = running_sum / running_samples
    digest = updates_digest(sorted_updates)
    new_value = current_value + aggregated_delta
    if not math.isfinite(new_value):
        raise ValueError("new_value not finite")
    result = RingRoundResult(
        round_id=round_id,
        parameter=parameter,
        ring_order=tuple(ring_order),
        aggregated_delta=aggregated_delta,
        total_samples=running_samples,
        steps=tuple(steps),
        ts_ns=ts_ns,
        digest=digest,
    )
    learning_update = LearningUpdate(
        ts_ns=ts_ns,
        strategy_id=strategy_id,
        parameter=parameter,
        old_value=repr(current_value),
        new_value=repr(new_value),
        reason=(
            f"federated_fedml.ring round={round_id} "
            f"n_steps={len(steps)} total_samples={running_samples} "
            f"delta={aggregated_delta!r} digest={digest}"
        ),
        meta={
            "lane": "federated_fedml",
            "topology": FederationTopology.RING.value,
            "version": FEDML_VERSION,
            "round_id": round_id,
            "digest": digest,
            "n_steps": str(len(steps)),
            "total_samples": str(running_samples),
        },
    )
    return result, learning_update


__all__ = [
    "FEDML_VERSION",
    "FederationTopology",
    "GroupAssignment",
    "HierarchicalRoundResult",
    "NEW_PIP_DEPENDENCIES",
    "RingRoundResult",
    "RingStep",
    "hierarchical_aggregate",
    "hierarchical_digest",
    "is_valid_group_partition",
    "partition_into_groups",
    "ring_aggregate",
]
