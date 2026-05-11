"""C-09 flwr federated learning — FedAvg aggregation lane (OFFLINE_ONLY).

Canonical adaptation of ``adap/flower`` (`src/py/flwr/server/strategy/fedavg.py`)
into a deterministic, OFFLINE-only Federated Averaging primitive.

Semantics (mirrored from Flower's :class:`FedAvg`):

* Each *client* owns its raw training data. The client never ships the data
  itself — only a **delta** (gradient or parameter increment) plus a
  ``num_samples`` weight that represents how much local data backed the
  update.
* The server (this module) aggregates a *round*'s deltas with
  :func:`fed_avg_aggregate`:

  .. code:: text

      aggregated_delta = Σᵢ (nᵢ · δᵢ) / Σᵢ nᵢ

  where ``nᵢ`` is the i-th client's ``num_samples`` and ``δᵢ`` is its
  ``delta``. This is Flower's :class:`FedAvg.aggregate_fit` weighted-average
  rule reduced to a single scalar parameter.
* The aggregated delta is then folded into the running parameter value and
  emitted as a :class:`core.contracts.learning.LearningUpdate` whose ``reason``
  records the round id + client count + sample total. ``UpdateEmitter`` then
  materialises that record as ``SystemEvent(UPDATE_PROPOSED)``.

Authority + safety constraints:

* **L2 / B1.** Lives in the offline ``learning_engine``. Top-level imports are
  ``__future__`` / ``collections.abc`` / ``dataclasses`` / ``hashlib`` /
  ``math`` and the project-local :mod:`core.contracts.learning`. No runtime-tier
  imports, no ``flwr`` / ``numpy`` / ``torch`` / ``time`` / ``datetime`` /
  ``random`` / ``asyncio`` / ``os`` / ``polars`` / ``requests`` / ``httpx`` /
  ``aiohttp`` / ``tornado`` / ``socket`` / ``ssl``.
* **INV-15.** Every function is pure: no clock reads, no PRNG, no I/O. All
  ``ts_ns`` values are caller-supplied monotone event-time. Aggregation order
  is canonicalised by :func:`canonical_sort_updates` so replay is byte-stable.
* **B27 / B28 / INV-71.** This lane never constructs transport-layer typed
  events (:class:`SystemEvent` / :class:`HazardEvent` / :class:`SignalEvent` /
  :class:`ExecutionEvent` / :class:`PatchProposal`). It produces a domain
  :class:`LearningUpdate` only; the :class:`UpdateEmitter` (already in the
  repo) is the sole transport-layer constructor.
* **Privacy.** :func:`verify_privacy` is a structural guard: a
  :class:`GradientUpdate` whose ``meta`` carries forbidden raw-data keys
  is rejected. The module never accepts a "raw data" field of any kind on
  the typed surface — the only payload fields are ``delta`` (a scalar) and
  ``num_samples`` (an int).
* **Lazy seam.** ``NEW_PIP_DEPENDENCIES = ("flwr",)`` is declared but the
  ``flwr`` package is never imported — FedAvg math is pure arithmetic, so
  the optional Flower client/server stack only matters when a future PR
  promotes this lane out of OFFLINE_ONLY.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from core.contracts.learning import LearningUpdate

FEDERATED_VERSION = "v3.7-C09"
NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("flwr",)
MIN_CLIENTS_PER_ROUND = 2
PRIVACY_FORBIDDEN_META_KEYS: frozenset[str] = frozenset(
    {
        "raw_data",
        "training_data",
        "dataset",
        "samples",
        "features",
        "labels",
        "X",
        "y",
    },
)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GradientUpdate:
    """One client's contribution to a federated round.

    A client computes a *delta* (the change it would apply to a parameter
    given its local data) and ships it together with ``num_samples`` — the
    weight the server should use when averaging across clients. The client's
    raw data is never part of this record. Any attempt to smuggle raw data
    through ``meta`` is rejected by :func:`verify_privacy`.

    Attributes:
        client_id: Stable identifier for the client. Non-empty. Used as the
            primary sort key in :func:`canonical_sort_updates`.
        parameter: Name of the parameter this update targets. Non-empty.
        delta: Scalar increment the client would apply to ``parameter``.
            Must be a finite ``float`` (no NaN/Inf).
        num_samples: Number of local samples backing this update. Must be a
            non-negative integer. ``0`` is allowed (a client that observed
            no data this round contributes zero weight) but the canonical
            FedAvg pipeline filters such updates before aggregation.
        ts_ns: Caller-supplied monotone event-time. Used as the secondary
            sort key so two updates with the same ``client_id`` (e.g. a
            retry) order deterministically by event-time.
        meta: Opaque caller-controlled metadata. Validated by
            :func:`verify_privacy` — must not carry raw-data fields.
    """

    client_id: str
    parameter: str
    delta: float
    num_samples: int
    ts_ns: int
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.client_id:
            raise ValueError("client_id must be non-empty")
        if not self.parameter:
            raise ValueError("parameter must be non-empty")
        if not isinstance(self.delta, float):
            raise TypeError("delta must be a float")
        if not math.isfinite(self.delta):
            raise ValueError("delta must be finite (no NaN/Inf)")
        if not isinstance(self.num_samples, int) or isinstance(self.num_samples, bool):
            raise TypeError("num_samples must be a plain int")
        if self.num_samples < 0:
            raise ValueError("num_samples must be non-negative")
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise TypeError("ts_ns must be a plain int")
        if self.ts_ns < 0:
            raise ValueError("ts_ns must be non-negative")


@dataclass(frozen=True, slots=True)
class FederatedAggregate:
    """Result of one FedAvg round.

    Attributes:
        round_id: Caller-supplied identifier for the round. Non-empty.
        parameter: Name of the parameter aggregated. Non-empty.
        n_clients: Number of clients that contributed (after filtering).
        aggregated_delta: The weighted-average delta across clients.
        total_samples: ``Σᵢ nᵢ`` — total weight backing the aggregate.
        ts_ns: Caller-supplied monotone event-time for the aggregate.
        digest: BLAKE2b-16 digest over the canonical bytes of the inputs.
            Pins INV-15 byte-identical replay — equal inputs (after canonical
            sort) produce equal digests.
    """

    round_id: str
    parameter: str
    n_clients: int
    aggregated_delta: float
    total_samples: int
    ts_ns: int
    digest: str

    def __post_init__(self) -> None:
        if not self.round_id:
            raise ValueError("round_id must be non-empty")
        if not self.parameter:
            raise ValueError("parameter must be non-empty")
        if self.n_clients < 0:
            raise ValueError("n_clients must be non-negative")
        if self.total_samples < 0:
            raise ValueError("total_samples must be non-negative")
        if not math.isfinite(self.aggregated_delta):
            raise ValueError("aggregated_delta must be finite")
        if self.ts_ns < 0:
            raise ValueError("ts_ns must be non-negative")
        if len(self.digest) != 32 or not all(c in "0123456789abcdef" for c in self.digest):
            raise ValueError("digest must be a 32-char lowercase hex string")


# ---------------------------------------------------------------------------
# Privacy guard
# ---------------------------------------------------------------------------


def verify_privacy(update: GradientUpdate) -> None:
    """Structural privacy assertion on ``update``.

    Rejects any update whose ``meta`` carries a forbidden raw-data key —
    ``raw_data``, ``training_data``, ``dataset``, ``samples``, ``features``,
    ``labels``, ``X``, ``y``. The typed surface of :class:`GradientUpdate`
    has no field for raw data; this guard catches accidental smuggling
    through the opaque ``meta`` Mapping.

    Raises:
        ValueError: If ``update.meta`` contains any forbidden key.
    """
    for key in update.meta:
        if key in PRIVACY_FORBIDDEN_META_KEYS:
            raise ValueError(
                f"GradientUpdate.meta carries forbidden raw-data key: {key!r}. "
                "Federated clients must ship gradients only — never raw data."
            )


# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------


def canonical_sort_updates(
    updates: Sequence[GradientUpdate],
) -> tuple[GradientUpdate, ...]:
    """Return ``updates`` ordered deterministically by ``(client_id, ts_ns)``.

    The order client updates arrive in is not part of FedAvg's mathematical
    contract, but it *is* part of INV-15's byte-identical-replay contract —
    floating-point summation is order-sensitive. Canonical-sort fixes the
    summation order so two replays of the same set of updates produce
    byte-identical aggregates.
    """
    return tuple(sorted(updates, key=lambda u: (u.client_id, u.ts_ns)))


def _canonical_bytes(updates: Sequence[GradientUpdate]) -> bytes:
    """Canonical byte form of a sequence of (already-sorted) updates.

    The encoding is the UTF-8 form of:

    .. code:: text

        client_id|parameter|delta_repr|num_samples|ts_ns\n
        ...

    where ``delta_repr`` is :func:`repr` of the float (round-trippable).
    No JSON, no third-party serializers — pure stdlib, byte-stable across
    Python builds.
    """
    parts: list[str] = []
    for u in updates:
        parts.append(
            f"{u.client_id}|{u.parameter}|{u.delta!r}|{u.num_samples}|{u.ts_ns}\n",
        )
    return "".join(parts).encode("utf-8")


def updates_digest(updates: Sequence[GradientUpdate]) -> str:
    """BLAKE2b-16 digest over the canonical bytes of ``updates``.

    Equal inputs (after :func:`canonical_sort_updates`) produce equal
    digests — pins INV-15 byte-identical replay over the aggregation input.
    """
    body = _canonical_bytes(canonical_sort_updates(updates))
    return hashlib.blake2b(body, digest_size=16).hexdigest()


# ---------------------------------------------------------------------------
# Round validation
# ---------------------------------------------------------------------------


def is_valid_round(
    updates: Sequence[GradientUpdate],
    *,
    min_clients: int = MIN_CLIENTS_PER_ROUND,
) -> bool:
    """``True`` iff ``updates`` forms a valid FedAvg round.

    A valid round has at least ``min_clients`` distinct ``client_id`` values,
    all updates target the *same* ``parameter``, and the total ``num_samples``
    weight is strictly positive. Duplicate ``client_id`` across two updates
    is *not* a structural failure here — it can happen on retries — but only
    the highest ``ts_ns`` per ``client_id`` is counted toward ``min_clients``.
    """
    if not updates:
        return False
    if min_clients < 1:
        raise ValueError("min_clients must be >= 1")
    parameters = {u.parameter for u in updates}
    if len(parameters) != 1:
        return False
    client_ids = {u.client_id for u in updates}
    if len(client_ids) < min_clients:
        return False
    total_samples = sum(u.num_samples for u in updates)
    return total_samples > 0


# ---------------------------------------------------------------------------
# FedAvg aggregation
# ---------------------------------------------------------------------------


def fed_avg_aggregate(updates: Sequence[GradientUpdate]) -> tuple[float, int]:
    """Weighted-average aggregation, mirroring Flower's FedAvg rule.

    Computes ``Σᵢ (nᵢ · δᵢ) / Σᵢ nᵢ`` over ``updates`` in canonical order
    (sort by ``(client_id, ts_ns)``). Returns the aggregated delta and the
    total sample weight.

    Raises:
        ValueError: If ``updates`` is empty, or if ``Σᵢ nᵢ`` is zero
            (zero-weight rounds are mathematically undefined under FedAvg).
    """
    if not updates:
        raise ValueError("fed_avg_aggregate requires at least one update")
    sorted_updates = canonical_sort_updates(updates)
    weighted_sum = 0.0
    total_samples = 0
    for u in sorted_updates:
        weighted_sum += u.num_samples * u.delta
        total_samples += u.num_samples
    if total_samples == 0:
        raise ValueError("fed_avg_aggregate: total num_samples must be > 0")
    return weighted_sum / total_samples, total_samples


# ---------------------------------------------------------------------------
# One-round orchestration
# ---------------------------------------------------------------------------


def aggregate_round(
    *,
    round_id: str,
    strategy_id: str,
    parameter: str,
    current_value: float,
    updates: Sequence[GradientUpdate],
    ts_ns: int,
    min_clients: int = MIN_CLIENTS_PER_ROUND,
) -> tuple[FederatedAggregate, LearningUpdate]:
    """Run one FedAvg round.

    Validates the round, aggregates the deltas, folds the aggregate into
    ``current_value`` to produce the proposed new value, and returns both
    a typed :class:`FederatedAggregate` summary and a domain
    :class:`LearningUpdate` proposal ready for :class:`UpdateEmitter`.

    This function never constructs a transport-layer typed event
    (:class:`SystemEvent` / :class:`HazardEvent` / :class:`SignalEvent` /
    :class:`ExecutionEvent` / :class:`PatchProposal`). The
    :class:`LearningUpdate` it returns is the only domain record exported.
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
    if not is_valid_round(updates, min_clients=min_clients):
        raise ValueError(
            f"invalid round: need >= {min_clients} distinct clients "
            "and positive total num_samples on a single parameter",
        )
    sorted_updates = canonical_sort_updates(updates)
    aggregated_delta, total_samples = fed_avg_aggregate(sorted_updates)
    distinct_clients = len({u.client_id for u in sorted_updates})
    digest = updates_digest(sorted_updates)
    aggregate = FederatedAggregate(
        round_id=round_id,
        parameter=parameter,
        n_clients=distinct_clients,
        aggregated_delta=aggregated_delta,
        total_samples=total_samples,
        ts_ns=ts_ns,
        digest=digest,
    )
    new_value = current_value + aggregated_delta
    if not math.isfinite(new_value):
        raise ValueError("new_value not finite after aggregation")
    learning_update = LearningUpdate(
        ts_ns=ts_ns,
        strategy_id=strategy_id,
        parameter=parameter,
        old_value=repr(current_value),
        new_value=repr(new_value),
        reason=(
            f"federated_fedavg round={round_id} n_clients={distinct_clients} "
            f"total_samples={total_samples} delta={aggregated_delta!r} "
            f"digest={digest}"
        ),
        meta={
            "lane": "federated",
            "version": FEDERATED_VERSION,
            "round_id": round_id,
            "digest": digest,
            "n_clients": str(distinct_clients),
            "total_samples": str(total_samples),
        },
    )
    return aggregate, learning_update


__all__ = [
    "FEDERATED_VERSION",
    "FederatedAggregate",
    "GradientUpdate",
    "MIN_CLIENTS_PER_ROUND",
    "NEW_PIP_DEPENDENCIES",
    "PRIVACY_FORBIDDEN_META_KEYS",
    "aggregate_round",
    "canonical_sort_updates",
    "fed_avg_aggregate",
    "is_valid_round",
    "updates_digest",
    "verify_privacy",
]
