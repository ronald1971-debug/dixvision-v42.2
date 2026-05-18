"""Promotion engine — emit STRATEGY_PROMOTED recommendations from arena.

# ADAPTED FROM: deap/algorithms.py (eaSimple "carry the elites forward"
# pattern + selBest wrapper around fitness ordering)
# License mitigation: DEAP is LGPL-3.0. We extract the algorithm
# pattern (the contestants that DID survive a tournament round, both
# the elites carried forward and the tournament winners) and
# re-implement it in pure stdlib Python. No DEAP toolbox classes are
# imported or used.

Authority constraints (manifest §H1, A-03 spec lines 852-892):

* OFFLINE_ONLY tier — never imported by ``execution_engine`` /
  ``governance_engine`` / ``system_engine`` / ``intelligence_engine``
  on the hot path. The B7 / B-CLOCK lints enforce this at AST level.
* No clock, no IO, no PRNG. Caller supplies ``ts_ns``.
* INV-13 / INV-14 governance isolation: this module is a **pure
  function** of :class:`TournamentResult`. It builds typed
  :class:`PromotionRecommendation` advisory records and **never**
  promotes, deploys, or mutates the strategy registry itself. Spec
  line 881: "Promotion must pass through governance approval gate
  (never auto-promote)".
* INV-71 / HARDEN-06 / B27-B28 authority symmetry: only
  ``evolution_engine.*`` may construct
  :class:`core.contracts.learning.PatchProposal`. The simulation tier
  emits a richer advisory record that downstream
  ``evolution_engine`` adapters (out-of-scope for A-03.3) translate
  into a typed PatchProposal once they're ready to drive the patch
  pipeline.
* Frozen + slotted public surface; deterministic
  ``recommendation_id`` so 3 runs with the same ``(arena_digest,
  strategy_id, ts_ns)`` produce byte-identical records (INV-15).

Refs:
- DIX_MASTER_CANONICAL.md §A-03 (lines 852-892)
- simulation/strategy_arena/arena.py — :class:`TournamentResult`
- simulation/strategy_arena/kill_underperformers.py — symmetric leaf
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

PROPOSAL_SOURCE: Final[str] = "simulation.strategy_arena.promotion_engine"
"""Stable string downstream adapters reference for routing."""

PROMOTION_KIND: Final[str] = "STRATEGY_PROMOTED"
"""Structural-mutation kind. Operator-side wiring projects this onto
the bus as a ``SystemEvent(sub_kind=STRATEGY_PROMOTED)`` row in the
ledger and routes it through the governance approval gate (A-03 spec
line 881)."""

PROMOTION_TOUCHPOINT: Final[str] = "strategy_registry.promoted"
"""Touchpoint string identifying *what* the recommendation would mutate."""

# Survivor-role tags propagated via :attr:`PromotionRecommendation.role`
# so the downstream governance adapter can distinguish "carried forward
# by elitism" from "won a tournament bracket". Both are advisory; both
# require governance approval before any registry mutation.
ROLE_ELITE: Final[str] = "ELITE"
"""Survivor was carried forward by the arena's elitism cap (top-N)."""

ROLE_WINNER: Final[str] = "TOURNAMENT_WINNER"
"""Survivor won at least one tournament bracket."""

ROLE_BOTH: Final[str] = "ELITE_AND_WINNER"
"""Survivor was both carried by elitism and won a bracket."""

# ----------------------------------------------------------------------
# Validation caps
# ----------------------------------------------------------------------

MAX_PROMOTE_BATCH: Final[int] = 1024
"""Hard cap on the number of promotion recommendations one call may
emit. Mirrors :data:`simulation.strategy_arena.arena.MAX_CONTESTANTS`
— an arena run can never produce more survivors than its input
population."""

MAX_RATIONALE_LEN: Final[int] = 256
"""Cap on the human-readable rationale string."""

# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------


class PromotionEngineInputError(ValueError):
    """Raised when :func:`build_promotion_recommendations` is called with bad inputs."""


# ----------------------------------------------------------------------
# Public value object
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class PromotionRecommendation:
    """Advisory STRATEGY_PROMOTED record emitted by the arena leaf.

    A pure-data record. The downstream operator-side wiring (NOT this
    module) routes it through the governance approval gate, which
    eventually constructs the typed
    :class:`core.contracts.learning.PatchProposal` (B28 authority
    symmetry: only ``evolution_engine.*`` may build PatchProposal).

    Fields:

    * ``ts_ns`` — caller-supplied event timestamp.
    * ``recommendation_id`` — deterministic 24-char id (BLAKE2b-8 over
      ``(arena_digest, strategy_id, ts_ns)``). INV-15 byte-identity.
    * ``source`` — :data:`PROPOSAL_SOURCE`.
    * ``kind`` — :data:`PROMOTION_KIND`.
    * ``arena_id`` / ``arena_digest`` — propagated from the arena run.
    * ``strategy_id`` — surviving strategy.
    * ``role`` — one of :data:`ROLE_ELITE`, :data:`ROLE_WINNER`,
      :data:`ROLE_BOTH`. Tells the downstream adapter why the
      strategy survived.
    * ``touchpoint`` — :data:`PROMOTION_TOUCHPOINT`.
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
    role: str
    touchpoint: str
    pnl_mean_usd: float
    max_drawdown_usd: float
    rationale: str
    meta: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise PromotionEngineInputError("PromotionRecommendation.ts_ns must be int")
        if self.ts_ns < 0:
            raise PromotionEngineInputError("PromotionRecommendation.ts_ns must be non-negative")
        for field_name in (
            "recommendation_id",
            "source",
            "kind",
            "arena_id",
            "arena_digest",
            "strategy_id",
            "role",
            "touchpoint",
            "rationale",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise PromotionEngineInputError(
                    f"PromotionRecommendation.{field_name} must be non-empty str"
                )
        if self.role not in {ROLE_ELITE, ROLE_WINNER, ROLE_BOTH}:
            raise PromotionEngineInputError(
                f"PromotionRecommendation.role must be one of "
                f"{{ELITE, TOURNAMENT_WINNER, ELITE_AND_WINNER}}, "
                f"got {self.role!r}"
            )
        if not isinstance(self.pnl_mean_usd, float):
            raise PromotionEngineInputError("PromotionRecommendation.pnl_mean_usd must be float")
        if not isinstance(self.max_drawdown_usd, float):
            raise PromotionEngineInputError(
                "PromotionRecommendation.max_drawdown_usd must be float"
            )
        if not isinstance(self.meta, tuple):
            raise PromotionEngineInputError(
                "PromotionRecommendation.meta must be a tuple of (str, str) pairs"
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
                raise PromotionEngineInputError(
                    "PromotionRecommendation.meta entries must be (non-empty str, str) tuples"
                )
            if pair[0] in seen_keys:
                raise PromotionEngineInputError(
                    f"PromotionRecommendation.meta has duplicate key {pair[0]!r}"
                )
            seen_keys.add(pair[0])


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def build_promotion_recommendations(
    *,
    result: TournamentResult,
    ts_ns: int,
    extra_meta: Mapping[str, str] | None = None,
) -> tuple[PromotionRecommendation, ...]:
    """Translate a :class:`TournamentResult` into promotion recommendations.

    Surviving contestants are exactly those whose ``strategy_id`` is
    in :attr:`TournamentResult.survivors`. Order is preserved from
    the original ``result.survivors`` sequence (which the arena
    already orders deterministically: elites first by composite
    fitness, then tournament winners), so 3 runs over the same arena
    output produce byte-identical recommendation tuples (INV-15).

    Each survivor is tagged with a ``role``: ``ELITE`` (carried by
    elitism), ``TOURNAMENT_WINNER`` (won at least one bracket), or
    ``ELITE_AND_WINNER`` (both).

    INV-13 / INV-14: this function is a pure transform. It never
    promotes, deploys, or mutates the strategy registry. The
    operator's wiring layer routes the returned tuple through the
    governance approval gate.
    """

    if not isinstance(result, TournamentResult):
        raise PromotionEngineInputError(
            "build_promotion_recommendations.result must be a TournamentResult, "
            f"got {type(result).__name__}"
        )
    if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
        raise PromotionEngineInputError(
            f"build_promotion_recommendations.ts_ns must be int, got {type(ts_ns).__name__}"
        )
    if ts_ns < 0:
        raise PromotionEngineInputError(
            f"build_promotion_recommendations.ts_ns must be non-negative, got {ts_ns!r}"
        )
    extra = _validate_extra_meta(extra_meta)

    elites = frozenset(result.elites)
    winners = frozenset(result.tournament_winners)

    contestants_by_id: dict[str, Contestant] = {c.strategy_id: c for c in result.contestants}

    recommendations: list[PromotionRecommendation] = []
    for sid in result.survivors:
        contestant = contestants_by_id.get(sid)
        if contestant is None:  # pragma: no cover — TournamentResult invariants prevent this
            raise PromotionEngineInputError(
                f"build_promotion_recommendations: survivor {sid!r} not in "
                "result.contestants — broken TournamentResult"
            )
        role = _classify_role(sid, elites=elites, winners=winners)
        recommendations.append(
            _build_one_recommendation(
                contestant=contestant,
                role=role,
                result=result,
                ts_ns=ts_ns,
                extra=extra,
            )
        )

    if len(recommendations) > MAX_PROMOTE_BATCH:
        raise PromotionEngineInputError(
            "build_promotion_recommendations: survivor count exceeds "
            f"MAX_PROMOTE_BATCH={MAX_PROMOTE_BATCH}, got {len(recommendations)}"
        )
    return tuple(recommendations)


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


_RESERVED_META_KEYS: Final[frozenset[str]] = frozenset(
    {"kind", "role", "arena_id", "arena_digest", "pnl_mean_usd", "max_drawdown_usd"}
)


def _classify_role(
    strategy_id: str,
    *,
    elites: frozenset[str],
    winners: frozenset[str],
) -> str:
    is_elite = strategy_id in elites
    is_winner = strategy_id in winners
    if is_elite and is_winner:
        return ROLE_BOTH
    if is_elite:
        return ROLE_ELITE
    if is_winner:
        return ROLE_WINNER
    # The arena guarantees survivors == elites ∪ winners, so this
    # branch is unreachable from arena.run output. Defensive.
    raise PromotionEngineInputError(
        f"strategy {strategy_id!r} is in survivors but neither in "
        "elites nor in tournament_winners — broken TournamentResult"
    )


def _validate_extra_meta(
    extra_meta: Mapping[str, str] | None,
) -> tuple[tuple[str, str], ...]:
    """Validate and canonicalise caller-supplied extra meta.

    Returns a sorted tuple-of-pairs so meta-key order does not affect
    digest determinism (INV-15) — mirrors the meta-key-order
    independence pin in A-02.1 :class:`StrategyChromosome` and
    A-03.2 ``kill_underperformers``.
    """

    if extra_meta is None:
        return ()
    if not isinstance(extra_meta, Mapping):
        raise PromotionEngineInputError(
            "build_promotion_recommendations.extra_meta must be a Mapping or None, "
            f"got {type(extra_meta).__name__}"
        )
    pairs: list[tuple[str, str]] = []
    for key, value in extra_meta.items():
        if not isinstance(key, str) or not key:
            raise PromotionEngineInputError(f"extra_meta keys must be non-empty str, got {key!r}")
        if not isinstance(value, str):
            raise PromotionEngineInputError(
                "extra_meta values must be str, "
                f"got value type {type(value).__name__} for key {key!r}"
            )
        if key in _RESERVED_META_KEYS:
            raise PromotionEngineInputError(
                "extra_meta keys cannot collide with reserved names "
                f"{sorted(_RESERVED_META_KEYS)!r}, got {key!r}"
            )
        pairs.append((key, value))
    pairs.sort(key=lambda kv: kv[0])
    return tuple(pairs)


def _build_one_recommendation(
    *,
    contestant: Contestant,
    role: str,
    result: TournamentResult,
    ts_ns: int,
    extra: tuple[tuple[str, str], ...],
) -> PromotionRecommendation:
    rationale = _build_rationale(contestant, role, result)
    rec_id = _recommendation_id(
        arena_digest=result.arena_digest,
        strategy_id=contestant.strategy_id,
        ts_ns=ts_ns,
    )
    return PromotionRecommendation(
        ts_ns=ts_ns,
        recommendation_id=rec_id,
        source=PROPOSAL_SOURCE,
        kind=PROMOTION_KIND,
        arena_id=result.arena_id,
        arena_digest=result.arena_digest,
        strategy_id=contestant.strategy_id,
        role=role,
        touchpoint=PROMOTION_TOUCHPOINT,
        pnl_mean_usd=float(contestant.summary.pnl_mean_usd),
        max_drawdown_usd=float(contestant.summary.max_drawdown_usd),
        rationale=rationale,
        meta=extra,
    )


def _build_rationale(
    contestant: Contestant,
    role: str,
    result: TournamentResult,
) -> str:
    """Compose a rationale string within :data:`MAX_RATIONALE_LEN`.

    Truncated deterministically so the same input always yields the
    same string (INV-15).
    """

    text = (
        f"arena={result.arena_id} promoted strategy={contestant.strategy_id} "
        f"role={role} "
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
    """Deterministic 24-char id over ``(arena_digest, strategy_id, ts_ns)``.

    Format: ``promote-`` + 16-hex BLAKE2b-8. Two arena outputs that
    differ in any of those fields produce different ids, but the same
    triple always yields the same id — INV-15 byte-identity.
    """

    canonical = (
        f"arena_digest={arena_digest}\nstrategy_id={strategy_id}\nts_ns={ts_ns}\n"
    ).encode()
    return "promote-" + hashlib.blake2b(canonical, digest_size=8).hexdigest()


__all__ = [
    "MAX_PROMOTE_BATCH",
    "MAX_RATIONALE_LEN",
    "NEW_PIP_DEPENDENCIES",
    "PROMOTION_KIND",
    "PROMOTION_TOUCHPOINT",
    "PROPOSAL_SOURCE",
    "ROLE_BOTH",
    "ROLE_ELITE",
    "ROLE_WINNER",
    "PromotionEngineInputError",
    "PromotionRecommendation",
    "build_promotion_recommendations",
]
