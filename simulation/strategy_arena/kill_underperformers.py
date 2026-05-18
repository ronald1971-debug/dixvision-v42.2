"""Kill underperformers — emit STRATEGY_DEMOTED recommendations from arena.

# ADAPTED FROM: deap/algorithms.py (eaSimple "replace worst" pattern)
# License mitigation: DEAP is LGPL-3.0. We extract the algorithm
# pattern (identify the contestants that did NOT survive a tournament
# round) and re-implement it in pure stdlib Python. No DEAP toolbox
# classes are imported or used.

Authority constraints (manifest §H1, A-03 spec lines 852-892):

* OFFLINE_ONLY tier — never imported by ``execution_engine`` /
  ``governance_engine`` / ``system_engine`` / ``intelligence_engine``
  on the hot path. The B7 / B-CLOCK lints enforce this at AST level.
* No clock, no IO, no PRNG. Caller supplies ``ts_ns``.
* INV-13 / INV-14 governance isolation: this module is a **pure
  function** of :class:`TournamentResult`. It builds typed
  :class:`DemotionRecommendation` advisory records and **never**
  demotes, halts, or deploys anything itself.
* INV-71 / HARDEN-06 / B27-B28 authority symmetry: only
  ``evolution_engine.*`` may construct
  :class:`core.contracts.learning.PatchProposal`. The simulation tier
  emits a richer advisory record that downstream
  ``evolution_engine`` adapters (out-of-scope for A-03.2) translate
  into a typed PatchProposal once they're ready to drive the patch
  pipeline.
* Spec line 880: "Killing underperformers must emit STRATEGY_DEMOTED
  governance event to ledger". Ledger emission is the operator's
  responsibility — this leaf only emits the typed advisory payload.
* Frozen + slotted public surface; deterministic
  ``recommendation_id`` so 3 runs with the same ``(arena_digest,
  strategy_id, ts_ns)`` produce byte-identical records (INV-15).

Refs:
- DIX_MASTER_CANONICAL.md §A-03 (lines 852-892)
- simulation/strategy_arena/arena.py — :class:`TournamentResult`
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping
from typing import Final

from simulation.strategy_arena.arena import (
    Contestant,
    TournamentResult,
)

# ----------------------------------------------------------------------
# Module metadata
# ----------------------------------------------------------------------

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ()
"""Pure stdlib + ``simulation.strategy_arena.arena``."""

PROPOSAL_SOURCE: Final[str] = "simulation.strategy_arena.kill_underperformers"
"""Stable string downstream adapters reference for routing."""

DEMOTION_KIND: Final[str] = "STRATEGY_DEMOTED"
"""Structural-mutation kind. Operator-side wiring projects this onto
the bus as a ``SystemEvent(sub_kind=STRATEGY_DEMOTED)`` row in the
ledger (A-03 spec line 880)."""

DEMOTION_TOUCHPOINT: Final[str] = "strategy_registry.demoted"
"""Touchpoint string identifying *what* the recommendation would mutate."""

# ----------------------------------------------------------------------
# Validation caps
# ----------------------------------------------------------------------

MAX_KILL_BATCH: Final[int] = 1024
"""Hard cap on the number of demotion recommendations one call may
emit. Mirrors :data:`simulation.strategy_arena.arena.MAX_CONTESTANTS`
— an arena run can never produce more eliminated contestants than its
input population."""

MAX_RATIONALE_LEN: Final[int] = 256
"""Cap on the human-readable rationale string."""

# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------


class KillUnderperformersInputError(ValueError):
    """Raised when :func:`build_demotion_recommendations` is called with bad inputs."""


# ----------------------------------------------------------------------
# Public value object
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class DemotionRecommendation:
    """Advisory STRATEGY_DEMOTED record emitted by the arena leaf.

    A pure-data record. The downstream operator-side wiring (NOT this
    module) routes it through ``evolution_engine.patch_pipeline`` /
    governance, which constructs the typed
    :class:`core.contracts.learning.PatchProposal` (B28 authority
    symmetry: only ``evolution_engine.*`` may build PatchProposal).

    Fields:

    * ``ts_ns`` — caller-supplied event timestamp.
    * ``recommendation_id`` — deterministic 24-char id (BLAKE2b-8 over
      ``(arena_digest, strategy_id, ts_ns)``). INV-15 byte-identity.
    * ``source`` — :data:`PROPOSAL_SOURCE`.
    * ``kind`` — :data:`DEMOTION_KIND`.
    * ``arena_id`` / ``arena_digest`` — propagated from the arena run.
    * ``strategy_id`` — eliminated strategy.
    * ``touchpoint`` — :data:`DEMOTION_TOUCHPOINT`.
    * ``pnl_mean_usd`` / ``max_drawdown_usd`` — context for governance.
    * ``rationale`` — human-readable summary, capped at
      :data:`MAX_RATIONALE_LEN` chars.
    * ``meta`` — sorted key/value pairs (str→str) so meta-key-order
      independence is enforced.
    """

    ts_ns: int
    recommendation_id: str
    source: str
    kind: str
    arena_id: str
    arena_digest: str
    strategy_id: str
    touchpoint: str
    pnl_mean_usd: float
    max_drawdown_usd: float
    rationale: str
    meta: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise KillUnderperformersInputError("DemotionRecommendation.ts_ns must be int")
        if self.ts_ns < 0:
            raise KillUnderperformersInputError("DemotionRecommendation.ts_ns must be non-negative")
        for field_name in (
            "recommendation_id",
            "source",
            "kind",
            "arena_id",
            "arena_digest",
            "strategy_id",
            "touchpoint",
            "rationale",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise KillUnderperformersInputError(
                    f"DemotionRecommendation.{field_name} must be non-empty str"
                )
        if not isinstance(self.pnl_mean_usd, float):
            raise KillUnderperformersInputError("DemotionRecommendation.pnl_mean_usd must be float")
        if not isinstance(self.max_drawdown_usd, float):
            raise KillUnderperformersInputError(
                "DemotionRecommendation.max_drawdown_usd must be float"
            )
        if not isinstance(self.meta, tuple):
            raise KillUnderperformersInputError(
                "DemotionRecommendation.meta must be a tuple of (str, str) pairs"
            )
        seen_keys: set[str] = set()
        for pair in self.meta:
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or not isinstance(pair[0], str)
                or not isinstance(pair[1], str)
                or not pair[0]
            ):
                raise KillUnderperformersInputError(
                    "DemotionRecommendation.meta entries must be (non-empty str, str) tuples"
                )
            if pair[0] in seen_keys:
                raise KillUnderperformersInputError(
                    f"DemotionRecommendation.meta has duplicate key {pair[0]!r}"
                )
            seen_keys.add(pair[0])


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def build_demotion_recommendations(
    *,
    result: TournamentResult,
    ts_ns: int,
    extra_meta: Mapping[str, str] | None = None,
) -> tuple[DemotionRecommendation, ...]:
    """Translate a :class:`TournamentResult` into demotion recommendations.

    Eliminated contestants are exactly those whose ``strategy_id`` is
    **not** in :attr:`TournamentResult.survivors`. Order is preserved
    from the original ``result.contestants`` sequence so 3 runs over
    the same arena output produce byte-identical recommendation
    tuples (INV-15).

    INV-13 / INV-14: this function is a pure transform. It never
    demotes, halts, or deploys anything. The operator's wiring layer
    routes the returned tuple through governance.
    """

    if not isinstance(result, TournamentResult):
        raise KillUnderperformersInputError(
            "build_demotion_recommendations.result must be a TournamentResult, "
            f"got {type(result).__name__}"
        )
    if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
        raise KillUnderperformersInputError(
            f"build_demotion_recommendations.ts_ns must be int, got {type(ts_ns).__name__}"
        )
    if ts_ns < 0:
        raise KillUnderperformersInputError(
            f"build_demotion_recommendations.ts_ns must be non-negative, got {ts_ns!r}"
        )
    extra = _validate_extra_meta(extra_meta)

    survivors = frozenset(result.survivors)
    eliminated: list[Contestant] = [c for c in result.contestants if c.strategy_id not in survivors]
    if len(eliminated) > MAX_KILL_BATCH:
        raise KillUnderperformersInputError(
            "build_demotion_recommendations: eliminated count exceeds "
            f"MAX_KILL_BATCH={MAX_KILL_BATCH}, got {len(eliminated)}"
        )

    recommendations: list[DemotionRecommendation] = []
    for contestant in eliminated:
        recommendations.append(
            _build_one_recommendation(
                contestant=contestant,
                result=result,
                ts_ns=ts_ns,
                extra=extra,
            )
        )
    return tuple(recommendations)


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


_RESERVED_META_KEYS: Final[frozenset[str]] = frozenset(
    {"kind", "arena_id", "arena_digest", "pnl_mean_usd", "max_drawdown_usd"}
)


def _validate_extra_meta(
    extra_meta: Mapping[str, str] | None,
) -> tuple[tuple[str, str], ...]:
    """Validate and canonicalise caller-supplied extra meta.

    Returns a sorted tuple-of-pairs so meta-key order does not affect
    digest determinism (INV-15) — mirrors the
    ``meta-key-order independence`` pin in A-02.1
    :class:`StrategyChromosome`.
    """

    if extra_meta is None:
        return ()
    if not isinstance(extra_meta, Mapping):
        raise KillUnderperformersInputError(
            "build_demotion_recommendations.extra_meta must be a Mapping or None, "
            f"got {type(extra_meta).__name__}"
        )
    pairs: list[tuple[str, str]] = []
    for key, value in extra_meta.items():
        if not isinstance(key, str) or not key:
            raise KillUnderperformersInputError(
                f"extra_meta keys must be non-empty str, got {key!r}"
            )
        if not isinstance(value, str):
            raise KillUnderperformersInputError(
                "extra_meta values must be str, "
                f"got value type {type(value).__name__} for key {key!r}"
            )
        if key in _RESERVED_META_KEYS:
            raise KillUnderperformersInputError(
                "extra_meta keys cannot collide with reserved names "
                f"{sorted(_RESERVED_META_KEYS)!r}, got {key!r}"
            )
        pairs.append((key, value))
    pairs.sort(key=lambda kv: kv[0])
    return tuple(pairs)


def _build_one_recommendation(
    *,
    contestant: Contestant,
    result: TournamentResult,
    ts_ns: int,
    extra: tuple[tuple[str, str], ...],
) -> DemotionRecommendation:
    rationale = _build_rationale(contestant, result)
    rec_id = _recommendation_id(
        arena_digest=result.arena_digest,
        strategy_id=contestant.strategy_id,
        ts_ns=ts_ns,
    )
    return DemotionRecommendation(
        ts_ns=ts_ns,
        recommendation_id=rec_id,
        source=PROPOSAL_SOURCE,
        kind=DEMOTION_KIND,
        arena_id=result.arena_id,
        arena_digest=result.arena_digest,
        strategy_id=contestant.strategy_id,
        touchpoint=DEMOTION_TOUCHPOINT,
        pnl_mean_usd=float(contestant.summary.pnl_mean_usd),
        max_drawdown_usd=float(contestant.summary.max_drawdown_usd),
        rationale=rationale,
        meta=extra,
    )


def _build_rationale(
    contestant: Contestant,
    result: TournamentResult,
) -> str:
    """Compose a rationale string within :data:`MAX_RATIONALE_LEN`.

    Truncated deterministically so the same input always yields the
    same string (INV-15).
    """

    text = (
        f"arena={result.arena_id} eliminated strategy={contestant.strategy_id} "
        f"pnl_mean={contestant.summary.pnl_mean_usd!r} "
        f"max_dd={contestant.summary.max_drawdown_usd!r} "
        f"survivors={len(result.survivors)} "
        f"contestants={len(result.contestants)}"
    )
    if len(text) > MAX_RATIONALE_LEN:
        text = text[:MAX_RATIONALE_LEN]
    return text


def _recommendation_id(
    *,
    arena_digest: str,
    strategy_id: str,
    ts_ns: int,
) -> str:
    """Deterministic 23-char id over ``(arena_digest, strategy_id, ts_ns)``.

    Format: ``demote-`` + 16-hex BLAKE2b-8. Two arena outputs that
    differ in any of those fields produce different ids, but the same
    triple always yields the same id — INV-15 byte-identity.
    """

    canonical = (
        f"arena_digest={arena_digest}\nstrategy_id={strategy_id}\nts_ns={ts_ns}\n"
    ).encode()
    return "demote-" + hashlib.blake2b(canonical, digest_size=8).hexdigest()


__all__ = [
    "DEMOTION_KIND",
    "DEMOTION_TOUCHPOINT",
    "DemotionRecommendation",
    "KillUnderperformersInputError",
    "MAX_KILL_BATCH",
    "MAX_RATIONALE_LEN",
    "NEW_PIP_DEPENDENCIES",
    "PROPOSAL_SOURCE",
    "build_demotion_recommendations",
]
