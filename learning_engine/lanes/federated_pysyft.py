"""C-12 syft (PySyft) — privacy-preserving federated lane (OFFLINE_ONLY).

# ADAPTED FROM:
#   PySyft — https://github.com/OpenMined/PySyft
#     packages/syft/src/syft/service/dataset/ (privacy controls)
#     packages/syft/src/syft/client/ (client / DP request envelope)
#   ARXIV — Dwork & Roth, "The Algorithmic Foundations of Differential
#     Privacy" (2014) — Gaussian + Laplace mechanisms, basic composition.

Differentiator vs C-09 / C-10 / C-11
====================================

C-09 (flwr) ships the atomic FedAvg primitive (weighted average, single
round, single parameter). C-10 (fedml) layers topology orchestration —
hierarchical (two-tier) + ring (sequential fold) — over C-09. C-11 (openfl)
layers a declarative multi-round *plan* over C-09. This lane (C-12) ships
the **privacy-preserving** layer that all three lack: a deterministic
differential-privacy noise mechanism, a per-round privacy accountant,
and a budget-exhausted hard stop, expressed as pure-arithmetic value
objects on top of the existing C-09 primitives.

PySyft's central artefact is a *privacy request* envelope — the data
owner approves an aggregation only if the requester's epsilon / delta
budget covers the computation. This lane mirrors that semantic shape:
every contribution carries its per-client noise scale + sensitivity;
every round consumes the round's slice of the global ``PrivacyBudget``;
the ``PrivacyAccountant`` is a frozen, append-only ledger that walks
forward only — never resets — and refuses to aggregate when the budget
is exhausted.

Semantic model
--------------
* :class:`PrivacyBudget` — (``epsilon``, ``delta``) global allowance.
* :class:`NoiseConfig` — mechanism (``"gaussian"`` / ``"laplace"``),
  sensitivity, noise multiplier.
* :class:`PrivateContribution` — frozen, slotted: an already-noised
  per-client update with its consumed epsilon / delta accounted at
  construction time.
* :class:`PrivateRoundReport` — frozen, slotted: per-round aggregate
  + cumulative ``epsilon_spent`` / ``delta_spent``.
* :class:`PrivacyAccountant` — frozen, slotted: running tally of total
  epsilon / delta spent across rounds.
* :func:`apply_dp_noise` — deterministic noise sample derived from a
  seed (no ``random`` import; uses ``hashlib.blake2b`` Box-Muller
  stream) so 3-run replay is byte-identical.
* :func:`aggregate_private_round` — verifies budget, FedAvg-aggregates
  via C-09's :func:`fed_avg_aggregate`, returns ``(report,
  LearningUpdate)`` and a successor :class:`PrivacyAccountant`.

Authority constraints
---------------------
* **L2 / B1.** OFFLINE-only. No runtime-tier imports. No top-level
  ``syft`` / ``openfl`` / ``flwr`` / ``fedml`` / ``time`` / ``datetime``
  / ``random`` / ``asyncio`` / ``os`` / ``subprocess`` / ``socket`` /
  ``ssl`` / ``numpy`` / ``torch`` / ``polars`` / ``pandas`` /
  ``requests`` / ``httpx`` / ``aiohttp`` / ``tornado`` / ``sqlite3``.
* **INV-15.** Caller-supplied monotone event-time. Within-round order
  is canonicalised via C-09 :func:`canonical_sort_updates`; cross-round
  order is fixed by ``round_index``. Noise derived deterministically
  from BLAKE2b stream — no ``random`` module dependency.
* **B27 / B28 / INV-71.** Never constructs transport-layer typed events
  (``SystemEvent`` / ``HazardEvent`` / ``SignalEvent`` /
  ``ExecutionEvent`` / ``PatchProposal``). Produces a domain
  :class:`LearningUpdate` only.
* **Privacy.** Per-client contributions pass through C-09's
  :func:`verify_privacy`. Typed surface carries only ``delta`` (scalar)
  + ``num_samples`` (int) + ``epsilon_consumed`` (scalar). Raw data
  forbidden by parameter-set + meta-key guard.
* **Lazy seam.** ``NEW_PIP_DEPENDENCIES = ("syft",)`` declared but
  never imported.
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

PYSYFT_VERSION = "v3.7-C12"
NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("syft",)

_SUPPORTED_MECHANISMS: tuple[str, ...] = ("gaussian", "laplace")
_DIGEST_HEX_LEN = 32


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PrivacyBudget:
    """Global differential-privacy budget (``epsilon``, ``delta``).

    Smaller ``epsilon`` means tighter privacy. ``delta`` is the failure
    probability — the probability that the ``epsilon``-bound is violated.
    Standard practice keeps ``delta`` strictly less than ``1 / N`` for a
    dataset of size ``N``; this contract enforces only ``0 <= delta < 1``
    + ``epsilon > 0`` and leaves task-specific tightening to the caller.
    """

    epsilon: float
    delta: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        if not math.isfinite(self.delta) or self.delta < 0.0 or self.delta >= 1.0:
            raise ValueError("delta must be in [0.0, 1.0)")


@dataclass(frozen=True, slots=True)
class NoiseConfig:
    """Noise mechanism specification.

    * ``mechanism`` — either ``"gaussian"`` or ``"laplace"``.
    * ``sensitivity`` — L1 (Laplace) or L2 (Gaussian) sensitivity of the
      function being privatised — i.e. the maximum change in output
      magnitude that any single record can cause. Caller responsibility
      to bound this (e.g. via gradient clipping).
    * ``noise_multiplier`` — multiplier applied to ``sensitivity / epsilon``
      to scale the noise. For Gaussian, ``sigma = noise_multiplier *
      sensitivity``; for Laplace, ``b = noise_multiplier * sensitivity /
      epsilon``.
    """

    mechanism: str
    sensitivity: float
    noise_multiplier: float

    def __post_init__(self) -> None:
        if self.mechanism not in _SUPPORTED_MECHANISMS:
            raise ValueError(
                f"unsupported mechanism: {self.mechanism!r} (supported: {_SUPPORTED_MECHANISMS})"
            )
        if not math.isfinite(self.sensitivity) or self.sensitivity <= 0.0:
            raise ValueError("sensitivity must be finite and positive")
        if not math.isfinite(self.noise_multiplier) or self.noise_multiplier <= 0.0:
            raise ValueError("noise_multiplier must be finite and positive")


@dataclass(frozen=True, slots=True)
class PrivateContribution:
    """An already-noised per-client gradient update.

    Built by :func:`apply_dp_noise`. Carries the post-noise ``delta``
    plus the privacy cost (``epsilon_consumed``, ``delta_consumed``) so
    the accountant can sum across contributions without re-deriving the
    noise.
    """

    client_id: str
    parameter: str
    delta: float
    num_samples: int
    ts_ns: int
    epsilon_consumed: float
    delta_consumed: float = 0.0
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.client_id:
            raise ValueError("client_id must be non-empty")
        if not self.parameter:
            raise ValueError("parameter must be non-empty")
        if not math.isfinite(self.delta):
            raise ValueError("delta must be finite")
        if self.num_samples < 0:
            raise ValueError("num_samples must be >= 0")
        if self.ts_ns < 0:
            raise ValueError("ts_ns must be >= 0")
        if not math.isfinite(self.epsilon_consumed) or self.epsilon_consumed < 0.0:
            raise ValueError("epsilon_consumed must be finite and non-negative")
        if (
            not math.isfinite(self.delta_consumed)
            or self.delta_consumed < 0.0
            or self.delta_consumed >= 1.0
        ):
            raise ValueError("delta_consumed must be in [0.0, 1.0)")
        for key, value in self.meta.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("meta keys and values must be str")

    def as_gradient_update(self) -> GradientUpdate:
        """Project to a C-09 :class:`GradientUpdate` for shared aggregation."""
        return GradientUpdate(
            client_id=self.client_id,
            parameter=self.parameter,
            delta=self.delta,
            num_samples=self.num_samples,
            ts_ns=self.ts_ns,
            meta=self.meta,
        )


@dataclass(frozen=True, slots=True)
class PrivateRoundReport:
    """One round of private aggregation.

    The digest is a BLAKE2b-16 hex over the canonical bytes of the round
    so 3-run replay is byte-identical.
    """

    round_id: str
    parameter: str
    n_clients: int
    aggregated_delta: float
    total_samples: int
    epsilon_spent: float
    delta_spent: float
    ts_ns: int
    digest: str

    def __post_init__(self) -> None:
        if not self.round_id:
            raise ValueError("round_id must be non-empty")
        if not self.parameter:
            raise ValueError("parameter must be non-empty")
        if self.n_clients < 0:
            raise ValueError("n_clients must be >= 0")
        if not math.isfinite(self.aggregated_delta):
            raise ValueError("aggregated_delta must be finite")
        if self.total_samples < 0:
            raise ValueError("total_samples must be >= 0")
        if not math.isfinite(self.epsilon_spent) or self.epsilon_spent < 0.0:
            raise ValueError("epsilon_spent must be finite and non-negative")
        if not math.isfinite(self.delta_spent) or self.delta_spent < 0.0 or self.delta_spent >= 1.0:
            raise ValueError("delta_spent must be in [0.0, 1.0)")
        if self.ts_ns < 0:
            raise ValueError("ts_ns must be >= 0")
        if len(self.digest) != _DIGEST_HEX_LEN:
            raise ValueError(f"digest must be {_DIGEST_HEX_LEN}-char hex")


@dataclass(frozen=True, slots=True)
class PrivacyAccountant:
    """Append-only privacy-budget ledger.

    Frozen + slotted — every call to :meth:`account_round` returns a
    *new* accountant with the cumulative spend, preserving the previous
    value object byte-for-byte. Basic composition: total epsilon =
    sum(per-round epsilon); total delta = sum(per-round delta).
    """

    budget: PrivacyBudget
    epsilon_spent: float = 0.0
    delta_spent: float = 0.0
    n_rounds: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.epsilon_spent) or self.epsilon_spent < 0.0:
            raise ValueError("epsilon_spent must be finite and non-negative")
        if not math.isfinite(self.delta_spent) or self.delta_spent < 0.0:
            raise ValueError("delta_spent must be finite and non-negative")
        if self.n_rounds < 0:
            raise ValueError("n_rounds must be >= 0")

    @property
    def epsilon_remaining(self) -> float:
        return self.budget.epsilon - self.epsilon_spent

    @property
    def delta_remaining(self) -> float:
        return self.budget.delta - self.delta_spent

    def can_afford(self, epsilon: float, delta: float = 0.0) -> bool:
        """Return ``True`` iff (epsilon, delta) fit within remaining budget."""
        if epsilon < 0.0 or delta < 0.0:
            return False
        return (
            self.epsilon_spent + epsilon <= self.budget.epsilon
            and self.delta_spent + delta <= self.budget.delta
        )

    def account_round(self, *, epsilon: float, delta: float = 0.0) -> PrivacyAccountant:
        """Charge (epsilon, delta) to the budget; return a new accountant.

        Raises ``ValueError`` if the charge would exceed the budget — the
        caller must abort the aggregation rather than leak past the bound.
        """
        if not math.isfinite(epsilon) or epsilon < 0.0:
            raise ValueError("epsilon charge must be finite and non-negative")
        if not math.isfinite(delta) or delta < 0.0:
            raise ValueError("delta charge must be finite and non-negative")
        if not self.can_afford(epsilon, delta):
            raise ValueError(
                "privacy budget exhausted: "
                f"remaining=(eps={self.epsilon_remaining:.6g}, "
                f"delta={self.delta_remaining:.6g}), "
                f"charge=(eps={epsilon:.6g}, delta={delta:.6g})"
            )
        return PrivacyAccountant(
            budget=self.budget,
            epsilon_spent=self.epsilon_spent + epsilon,
            delta_spent=self.delta_spent + delta,
            n_rounds=self.n_rounds + 1,
        )


# ---------------------------------------------------------------------------
# Deterministic noise (no ``random`` import — BLAKE2b stream + Box-Muller)
# ---------------------------------------------------------------------------


def _uniform_stream(seed: bytes, count: int) -> tuple[float, ...]:
    """Deterministic stream of ``count`` uniforms in ``(0, 1)``.

    Uses ``hashlib.blake2b`` as the underlying PRNG; each 8-byte block
    of the digest produces one uniform. Caller responsibility to supply
    distinct seeds per draw site.
    """
    out: list[float] = []
    counter = 0
    while len(out) < count:
        block = hashlib.blake2b(
            seed + counter.to_bytes(8, "big"),
            digest_size=8,
        ).digest()
        u_int = int.from_bytes(block, "big")
        # map to (0, 1) — clamp away from 0 to avoid log(0) in Box-Muller
        u = (u_int + 1) / (2**64 + 1)
        out.append(u)
        counter += 1
    return tuple(out)


def _gaussian_noise(seed: bytes, sigma: float) -> float:
    """Single deterministic Gaussian sample with mean 0, stddev ``sigma``.

    Box-Muller transform applied to two BLAKE2b uniforms. Only the first
    of the pair is returned; the second is discarded — accepted overhead
    for byte-stable, single-call determinism.
    """
    u1, u2 = _uniform_stream(seed, 2)
    r = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2
    return sigma * r * math.cos(theta)


def _laplace_noise(seed: bytes, scale: float) -> float:
    """Single deterministic Laplace sample with mean 0, scale ``scale``.

    Inverse-CDF method: ``F^{-1}(u) = -scale * sign(u - 0.5) *
    ln(1 - 2 |u - 0.5|)`` for ``u`` ~ U(0, 1).
    """
    (u,) = _uniform_stream(seed, 1)
    centered = u - 0.5
    if centered >= 0.0:
        sign = 1.0
    else:
        sign = -1.0
    return -scale * sign * math.log(max(1.0 - 2.0 * abs(centered), 1e-300))


def _noise_seed(
    *,
    round_id: str,
    client_id: str,
    parameter: str,
    mechanism: str,
) -> bytes:
    """Stable per-draw seed derived from the round / client / parameter."""
    h = hashlib.blake2b(digest_size=32)
    h.update(round_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(client_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(parameter.encode("utf-8"))
    h.update(b"\x00")
    h.update(mechanism.encode("utf-8"))
    return h.digest()


def apply_dp_noise(
    *,
    update: GradientUpdate,
    noise: NoiseConfig,
    round_id: str,
    epsilon: float,
    delta: float = 0.0,
    meta: Mapping[str, str] | None = None,
) -> PrivateContribution:
    """Apply a deterministic DP noise sample to ``update``.

    Returns a :class:`PrivateContribution` whose ``delta`` is the raw
    update delta plus a noise sample drawn from the configured mechanism.
    The seed is derived from (round_id, client_id, parameter, mechanism)
    so repeated calls with the same inputs return byte-identical noise.
    """
    if not round_id:
        raise ValueError("round_id must be non-empty")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    verify_privacy(update)
    seed = _noise_seed(
        round_id=round_id,
        client_id=update.client_id,
        parameter=update.parameter,
        mechanism=noise.mechanism,
    )
    if noise.mechanism == "gaussian":
        sigma = noise.noise_multiplier * noise.sensitivity
        sample = _gaussian_noise(seed, sigma)
    else:  # laplace
        scale = noise.noise_multiplier * noise.sensitivity / epsilon
        sample = _laplace_noise(seed, scale)
    if not math.isfinite(sample):
        raise ValueError("noise sample not finite")
    noised_delta = update.delta + sample
    if not math.isfinite(noised_delta):
        raise ValueError("noised delta not finite")
    payload_meta: dict[str, str] = dict(update.meta)
    if meta is not None:
        for key, value in meta.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("meta keys and values must be str")
            payload_meta[key] = value
    return PrivateContribution(
        client_id=update.client_id,
        parameter=update.parameter,
        delta=noised_delta,
        num_samples=update.num_samples,
        ts_ns=update.ts_ns,
        epsilon_consumed=epsilon,
        delta_consumed=delta,
        meta=payload_meta,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _round_canonical_bytes(
    *,
    round_id: str,
    parameter: str,
    sorted_updates: Sequence[GradientUpdate],
    epsilon_spent: float,
    delta_spent: float,
) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(b"private_round\x00")
    h.update(round_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(parameter.encode("utf-8"))
    h.update(b"\x00")
    h.update(repr(epsilon_spent).encode("utf-8"))
    h.update(b"\x00")
    h.update(repr(delta_spent).encode("utf-8"))
    h.update(b"\x00")
    for u in sorted_updates:
        h.update(u.client_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(repr(u.delta).encode("utf-8"))
        h.update(b"\x00")
        h.update(repr(u.num_samples).encode("utf-8"))
        h.update(b"\x00")
    return h.digest()


def _validate_round_inputs(
    *,
    round_id: str,
    parameter: str,
    strategy_id: str,
    current_value: float,
    contributions: Sequence[PrivateContribution],
    ts_ns: int,
) -> None:
    if not round_id:
        raise ValueError("round_id must be non-empty")
    if not parameter:
        raise ValueError("parameter must be non-empty")
    if not strategy_id:
        raise ValueError("strategy_id must be non-empty")
    if not math.isfinite(current_value):
        raise ValueError("current_value must be finite")
    if ts_ns < 0:
        raise ValueError("ts_ns must be >= 0")
    if not contributions:
        raise ValueError("contributions must be non-empty")
    seen: set[str] = set()
    for c in contributions:
        if c.parameter != parameter:
            raise ValueError(
                f"contribution parameter {c.parameter!r} does not match "
                f"round parameter {parameter!r}"
            )
        if c.client_id in seen:
            raise ValueError(f"duplicate client_id in round: {c.client_id!r}")
        seen.add(c.client_id)


def aggregate_private_round(
    *,
    round_id: str,
    parameter: str,
    strategy_id: str,
    current_value: float,
    contributions: Sequence[PrivateContribution],
    accountant: PrivacyAccountant,
    ts_ns: int,
) -> tuple[PrivateRoundReport, LearningUpdate, PrivacyAccountant]:
    """Aggregate one round of :class:`PrivateContribution` and update budget.

    Returns ``(report, learning_update, next_accountant)``:

    * ``report`` — :class:`PrivateRoundReport` whose ``digest`` pins
      INV-15 byte-equality across replay.
    * ``learning_update`` — domain :class:`LearningUpdate` carrying the
      post-aggregation parameter value + the per-round epsilon spent.
    * ``next_accountant`` — successor :class:`PrivacyAccountant` whose
      ``epsilon_spent`` / ``delta_spent`` have been charged for this round.

    Aggregation is done via C-09 :func:`fed_avg_aggregate` over the
    canonical-sorted projected :class:`GradientUpdate` sequence so the
    output is permutation-invariant on ``contributions`` and byte-stable
    across replay.
    """
    _validate_round_inputs(
        round_id=round_id,
        parameter=parameter,
        strategy_id=strategy_id,
        current_value=current_value,
        contributions=contributions,
        ts_ns=ts_ns,
    )
    round_epsilon = sum(c.epsilon_consumed for c in contributions)
    round_delta = sum(c.delta_consumed for c in contributions)
    next_accountant = accountant.account_round(
        epsilon=round_epsilon,
        delta=round_delta,
    )
    updates = [c.as_gradient_update() for c in contributions]
    sorted_updates = canonical_sort_updates(updates)
    agg_delta, total_samples = fed_avg_aggregate(sorted_updates)
    n_clients = len({u.client_id for u in sorted_updates})
    digest = _round_canonical_bytes(
        round_id=round_id,
        parameter=parameter,
        sorted_updates=sorted_updates,
        epsilon_spent=next_accountant.epsilon_spent,
        delta_spent=next_accountant.delta_spent,
    ).hex()
    new_value = current_value + agg_delta
    if not math.isfinite(new_value):
        raise ValueError("new_value not finite after aggregation")
    report = PrivateRoundReport(
        round_id=round_id,
        parameter=parameter,
        n_clients=n_clients,
        aggregated_delta=agg_delta,
        total_samples=total_samples,
        epsilon_spent=next_accountant.epsilon_spent,
        delta_spent=next_accountant.delta_spent,
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
            f"pysyft_dp round={round_id} n_clients={n_clients} "
            f"total_samples={total_samples} delta={agg_delta!r} "
            f"epsilon_spent={next_accountant.epsilon_spent!r} "
            f"delta_spent={next_accountant.delta_spent!r} "
            f"digest={digest}"
        ),
        meta={
            "lane": "federated_pysyft",
            "version": PYSYFT_VERSION,
            "round_id": round_id,
            "digest": digest,
            "epsilon_spent": repr(next_accountant.epsilon_spent),
            "delta_spent": repr(next_accountant.delta_spent),
            "n_rounds": str(next_accountant.n_rounds),
        },
    )
    return report, learning_update, next_accountant


__all__ = [
    "NEW_PIP_DEPENDENCIES",
    "PYSYFT_VERSION",
    "NoiseConfig",
    "PrivacyAccountant",
    "PrivacyBudget",
    "PrivateContribution",
    "PrivateRoundReport",
    "aggregate_private_round",
    "apply_dp_noise",
]
