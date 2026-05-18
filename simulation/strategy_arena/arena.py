"""Strategy arena — DEAP-style tournament selection over RealitySummary.

# ADAPTED FROM: deap/tools/selection.py (selTournament, selRandom)
# ADAPTED FROM: deap/algorithms.py (eaSimple elitism pattern)
# License mitigation: DEAP is LGPL-3.0. We extract the algorithm math
# only (tournament selection + elitism) and re-implement it in pure
# stdlib Python. No DEAP toolbox classes are imported or used.

Authority constraints (manifest §H1, A-03 spec lines 852-892):

* OFFLINE_ONLY tier. The arena is never invoked on the
  ``execution_engine`` / ``governance_engine`` / ``system_engine`` /
  ``intelligence_engine`` hot path. The B7 / B-CLOCK lints enforce
  this at AST level.
* No clock, no IO, no PRNG without caller-supplied seed. The arena's
  PRNG is a stateless splitmix64 unrolling so 3 runs with the same
  ``(seed, ts_ns, contestant set)`` produce byte-identical
  :class:`TournamentResult` (INV-15).
* INV-13 / INV-14 governance isolation: the arena ranks contestants
  and reports survivors. It never demotes, kills, or deploys anything
  itself. The :mod:`simulation.strategy_arena.kill_underperformers`
  (A-03.2) and :mod:`simulation.strategy_arena.promotion_engine`
  (A-03.3) leaves consume this output and emit typed proposals onto
  governance.
* Frozen + slotted dataclasses (no ``__dict__``) — every value object
  is immutable.
* Fitness is composite: ``pnl_mean_usd - drawdown_weight *
  max_drawdown_usd``. Spec: "Tournament selection uses DIX
  RealitySummary.pnl_mean as primary fitness". The drawdown penalty
  is configurable and defaults to 0.0 so the primary-fitness
  contract is preserved.

Refs:
- DIX_MASTER_CANONICAL.md §A-03 (lines 852-892)
- core/contracts/simulation.py — :class:`RealitySummary`
- docs/promotion_gates.yaml — downstream criteria for the survivors
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Iterable, Sequence
from typing import Final

from core.contracts.simulation import RealitySummary

# ----------------------------------------------------------------------
# Module metadata
# ----------------------------------------------------------------------

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ()
"""Pure stdlib. DEAP is *not* a runtime dependency (LGPL-3.0 mitigation).

The A-03 spec (line 856) requires that DEAP toolbox classes never
appear in production code. We extract the math (tournament selection
+ elitism) and re-implement it in pure Python. Replay determinism
(INV-15) is achieved via a stateless splitmix64 PRNG seeded from the
caller's ``seed`` and ``ts_ns``.
"""

PROPOSAL_SOURCE: Final[str] = "simulation.strategy_arena.arena"
"""Stable string downstream demotion / promotion proposals reference."""

# ----------------------------------------------------------------------
# Validation caps
# ----------------------------------------------------------------------

MIN_CONTESTANTS: Final[int] = 2
"""A tournament needs at least two contestants to be meaningful."""

MAX_CONTESTANTS: Final[int] = 1024
"""Hard cap on arena fan-in. Keeps the bracket array bounded."""

MIN_TOURNAMENT_SIZE: Final[int] = 2
"""Smallest non-degenerate tournament size."""

MAX_TOURNAMENT_SIZE: Final[int] = 64
"""Cap on tournament size to prevent pathological brackets."""

MAX_N_WINNERS: Final[int] = 1024
"""Cap on tournament-selection round count."""

MAX_ELITISM_COUNT: Final[int] = 256
"""Cap on elitism preservation."""

MAX_ARENA_ID_LEN: Final[int] = 128
"""Cap on the human-readable arena identifier."""

DEFAULT_DRAWDOWN_WEIGHT: Final[float] = 0.0
"""Default fitness penalty per unit max-drawdown.

Set to 0.0 so the primary fitness reduces to ``pnl_mean_usd``
(matching the A-03 spec's "primary fitness = pnl_mean" rule).
Operators can pass a positive value to penalise drawdown-heavy
strategies during selection.
"""


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------


class ArenaConfigError(ValueError):
    """Raised when :class:`ArenaConfig` is constructed with bad inputs."""


class ArenaInputError(ValueError):
    """Raised when :meth:`Arena.run` is called with bad arguments."""


# ----------------------------------------------------------------------
# Splitmix64 PRNG (stateless, deterministic; mirrors A-02.2 cmaes_optimizer)
# ----------------------------------------------------------------------


_MASK_64: Final[int] = (1 << 64) - 1


def _splitmix64(seed: int, counter: int) -> int:
    """Stateless splitmix64 step.

    Used as a deterministic uniform source for tournament-bracket
    sampling. Same ``(seed, counter)`` always yields the same 64-bit
    integer, so brackets reproduce byte-identically across runs.
    """

    x = (seed + counter * 0x9E3779B97F4A7C15) & _MASK_64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK_64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK_64
    return x ^ (x >> 31)


def _uniform_index(seed: int, counter: int, n: int) -> int:
    """Return a uniform integer in ``[0, n)`` using splitmix64.

    The modulo bias is negligible for ``n <= MAX_CONTESTANTS`` and
    keeps the encoding stable across machines.
    """

    if n <= 0:
        raise ArenaInputError(f"_uniform_index requires n > 0, got {n!r}")
    return _splitmix64(seed, counter) % n


# ----------------------------------------------------------------------
# Frozen value objects
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Contestant:
    """One arena contestant: a strategy id and its reality summary.

    The ``summary`` is the distributional output of an upstream
    parallel-realities batch (SIM-07). The arena treats it as opaque
    aside from ``pnl_mean_usd`` (primary fitness) and
    ``max_drawdown_usd`` (penalty).
    """

    strategy_id: str
    summary: RealitySummary

    def __post_init__(self) -> None:
        if not isinstance(self.strategy_id, str):
            raise ArenaInputError(
                f"Contestant.strategy_id must be str, got {type(self.strategy_id).__name__}"
            )
        if not self.strategy_id:
            raise ArenaInputError("Contestant.strategy_id must be non-empty")
        if len(self.strategy_id) > 128:
            raise ArenaInputError(
                f"Contestant.strategy_id must be <= 128 chars, got len={len(self.strategy_id)}"
            )
        if not isinstance(self.summary, RealitySummary):
            raise ArenaInputError(
                f"Contestant.summary must be a RealitySummary, got {type(self.summary).__name__}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class TournamentBracket:
    """One tournament round: candidates considered, winner chosen.

    Stored on :class:`TournamentResult` for replay/audit. Equal-fitness
    ties are broken by ``strategy_id`` ascending — deterministic
    across runs.
    """

    bracket_id: int
    candidate_ids: tuple[str, ...]
    winner_id: str
    winner_fitness: float

    def __post_init__(self) -> None:
        if not isinstance(self.bracket_id, int) or isinstance(self.bracket_id, bool):
            raise ArenaInputError(
                f"TournamentBracket.bracket_id must be int, got {type(self.bracket_id).__name__}"
            )
        if self.bracket_id < 0:
            raise ArenaInputError(
                f"TournamentBracket.bracket_id must be non-negative, got {self.bracket_id!r}"
            )
        if not self.candidate_ids:
            raise ArenaInputError("TournamentBracket.candidate_ids must be non-empty")
        if not all(isinstance(c, str) and c for c in self.candidate_ids):
            raise ArenaInputError("TournamentBracket.candidate_ids entries must be non-empty str")
        if not isinstance(self.winner_id, str) or not self.winner_id:
            raise ArenaInputError("TournamentBracket.winner_id must be non-empty str")
        if self.winner_id not in self.candidate_ids:
            raise ArenaInputError(
                "TournamentBracket.winner_id must appear in candidate_ids, "
                f"got winner={self.winner_id!r}, "
                f"candidates={self.candidate_ids!r}"
            )
        if not _is_finite_float(self.winner_fitness):
            raise ArenaInputError(
                f"TournamentBracket.winner_fitness must be finite, got {self.winner_fitness!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class ArenaConfig:
    """Tournament-selection configuration.

    * ``tournament_size`` (k) — number of contestants drawn into each
      bracket.
    * ``n_winners`` — number of tournament-selection rounds. Each
      round picks one bracket winner.
    * ``elitism_count`` — top-N contestants by fitness preserved
      unconditionally (DEAP eaSimple elitism pattern).
    * ``drawdown_weight`` — composite fitness penalty per unit max
      drawdown. Default 0.0 keeps primary fitness == ``pnl_mean_usd``.
    """

    arena_id: str
    tournament_size: int
    n_winners: int
    elitism_count: int = 0
    drawdown_weight: float = DEFAULT_DRAWDOWN_WEIGHT

    def __post_init__(self) -> None:
        if not isinstance(self.arena_id, str):
            raise ArenaConfigError(
                f"ArenaConfig.arena_id must be str, got {type(self.arena_id).__name__}"
            )
        if not self.arena_id:
            raise ArenaConfigError("ArenaConfig.arena_id must be non-empty")
        if len(self.arena_id) > MAX_ARENA_ID_LEN:
            raise ArenaConfigError(
                "ArenaConfig.arena_id must be <= "
                f"{MAX_ARENA_ID_LEN} chars, got len={len(self.arena_id)}"
            )
        _require_int(self.tournament_size, "tournament_size")
        if not (MIN_TOURNAMENT_SIZE <= self.tournament_size <= MAX_TOURNAMENT_SIZE):
            raise ArenaConfigError(
                "ArenaConfig.tournament_size must be in "
                f"[{MIN_TOURNAMENT_SIZE}, {MAX_TOURNAMENT_SIZE}], "
                f"got {self.tournament_size!r}"
            )
        _require_int(self.n_winners, "n_winners")
        if not (1 <= self.n_winners <= MAX_N_WINNERS):
            raise ArenaConfigError(
                f"ArenaConfig.n_winners must be in [1, {MAX_N_WINNERS}], got {self.n_winners!r}"
            )
        _require_int(self.elitism_count, "elitism_count")
        if not (0 <= self.elitism_count <= MAX_ELITISM_COUNT):
            raise ArenaConfigError(
                "ArenaConfig.elitism_count must be in "
                f"[0, {MAX_ELITISM_COUNT}], got {self.elitism_count!r}"
            )
        if not _is_finite_float(self.drawdown_weight):
            raise ArenaConfigError(
                f"ArenaConfig.drawdown_weight must be finite, got {self.drawdown_weight!r}"
            )
        if self.drawdown_weight < 0.0:
            raise ArenaConfigError(
                f"ArenaConfig.drawdown_weight must be non-negative, got {self.drawdown_weight!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class TournamentResult:
    """Frozen audit record of one arena run.

    * ``elites`` — top-N strategies by fitness, in fitness-descending
      order (tie-break: ``strategy_id`` ascending). Always preserved.
    * ``tournament_winners`` — one entry per bracket, in bracket
      order. Duplicates are allowed (tournament selection samples
      with replacement, mirroring DEAP's selRandom).
    * ``survivors`` — deduplicated union of ``elites`` then
      ``tournament_winners`` (first-occurrence ordering preserved).
      Downstream A-03.2 / A-03.3 leaves consume this list.
    * ``arena_digest`` — 16-hex BLAKE2b-8 content hash over the
      canonical text projection of the result. Drives INV-15 replay
      checks.
    """

    arena_id: str
    ts_ns: int
    seed: int
    contestants: tuple[Contestant, ...]
    elites: tuple[str, ...]
    tournament_winners: tuple[str, ...]
    survivors: tuple[str, ...]
    brackets: tuple[TournamentBracket, ...]
    arena_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.arena_id, str) or not self.arena_id:
            raise ArenaInputError("TournamentResult.arena_id must be non-empty str")
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise ArenaInputError("TournamentResult.ts_ns must be int")
        if self.ts_ns < 0:
            raise ArenaInputError(
                f"TournamentResult.ts_ns must be non-negative, got {self.ts_ns!r}"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ArenaInputError("TournamentResult.seed must be int")
        if self.seed < 0:
            raise ArenaInputError(f"TournamentResult.seed must be non-negative, got {self.seed!r}")
        if not self.contestants:
            raise ArenaInputError("TournamentResult.contestants must be non-empty")
        ids = {c.strategy_id for c in self.contestants}
        for elite in self.elites:
            if elite not in ids:
                raise ArenaInputError(
                    f"TournamentResult.elites entries must be among contestants, got {elite!r}"
                )
        for winner in self.tournament_winners:
            if winner not in ids:
                raise ArenaInputError(
                    "TournamentResult.tournament_winners entries "
                    f"must be among contestants, got {winner!r}"
                )
        for survivor in self.survivors:
            if survivor not in ids:
                raise ArenaInputError(
                    "TournamentResult.survivors entries must be among "
                    f"contestants, got {survivor!r}"
                )
        if not isinstance(self.arena_digest, str):
            raise ArenaInputError("TournamentResult.arena_digest must be str")
        if len(self.arena_digest) != 16:
            raise ArenaInputError(
                "TournamentResult.arena_digest must be 16 hex chars, "
                f"got len={len(self.arena_digest)}"
            )
        try:
            int(self.arena_digest, 16)
        except ValueError as exc:
            raise ArenaInputError(
                f"TournamentResult.arena_digest must be hex, got {self.arena_digest!r}"
            ) from exc


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _is_finite_float(value: object) -> bool:
    if not isinstance(value, float) and not isinstance(value, int):
        return False
    if isinstance(value, bool):
        return False
    f = float(value)
    return f == f and f not in (float("inf"), float("-inf"))


def _require_int(value: object, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ArenaConfigError(f"ArenaConfig.{label} must be int, got {type(value).__name__}")


def _composite_fitness(summary: RealitySummary, drawdown_weight: float) -> float:
    """Composite fitness used for tournament + elitism ranking.

    Primary fitness is ``pnl_mean_usd`` (per A-03 spec line 879).
    ``drawdown_weight`` adds an optional penalty:

        fitness = pnl_mean_usd - drawdown_weight * max_drawdown_usd

    With the default ``drawdown_weight = 0.0`` the formula reduces
    exactly to ``pnl_mean_usd``.
    """

    return summary.pnl_mean_usd - drawdown_weight * summary.max_drawdown_usd


def _ranked_by_fitness(
    contestants: Sequence[Contestant],
    drawdown_weight: float,
) -> list[Contestant]:
    """Return contestants sorted by composite fitness DESC, tie-break by id ASC.

    Deterministic across runs: Python's ``sorted`` is stable and the
    tie-break key is ``strategy_id`` (ascending), which makes the
    ordering canonical regardless of input shuffle order.
    """

    return sorted(
        contestants,
        key=lambda c: (
            -_composite_fitness(c.summary, drawdown_weight),
            c.strategy_id,
        ),
    )


# ----------------------------------------------------------------------
# Tournament selection (ADAPTED FROM deap/tools/selection.py:selTournament)
# ----------------------------------------------------------------------


def _select_tournament_winner(
    contestants: Sequence[Contestant],
    bracket: tuple[int, ...],
    drawdown_weight: float,
) -> tuple[Contestant, float]:
    """Pick the highest-fitness contestant from ``bracket``.

    DEAP ``selTournament`` uses ``max(aspirants, key=fitness)``. We
    do the same but make tie-breaks deterministic by sorting on
    ``(-fitness, strategy_id)`` — stable across runs/machines.
    """

    aspirants = [contestants[i] for i in bracket]
    ranked = _ranked_by_fitness(aspirants, drawdown_weight)
    winner = ranked[0]
    return winner, _composite_fitness(winner.summary, drawdown_weight)


def _build_brackets(
    n_contestants: int,
    tournament_size: int,
    n_winners: int,
    seed: int,
    ts_ns: int,
) -> list[tuple[int, ...]]:
    """Build ``n_winners`` bracket index tuples deterministically.

    Mirrors DEAP's ``selRandom`` (sample with replacement). Same
    ``(seed, ts_ns)`` always yields the same bracket sequence so
    INV-15 replay holds.
    """

    brackets: list[tuple[int, ...]] = []
    counter = (ts_ns & _MASK_64) + 1
    derived_seed = (seed ^ ((ts_ns << 1) & _MASK_64)) & _MASK_64
    for bracket_id in range(n_winners):
        indices: list[int] = []
        for slot in range(tournament_size):
            counter += 1
            indices.append(
                _uniform_index(
                    derived_seed,
                    counter * (1 + slot) + bracket_id,
                    n_contestants,
                )
            )
        brackets.append(tuple(indices))
    return brackets


# ----------------------------------------------------------------------
# Digest
# ----------------------------------------------------------------------


def _arena_digest(
    arena_id: str,
    ts_ns: int,
    seed: int,
    drawdown_weight: float,
    elites: Sequence[str],
    winners: Sequence[str],
    survivors: Sequence[str],
    brackets: Sequence[TournamentBracket],
) -> str:
    """Stable 16-hex content hash of the arena outcome.

    Drives the 3-run INV-15 byte-identity test pin in the test
    suite. Caller-supplied ``arena_id`` / ``ts_ns`` / ``seed`` are
    embedded so different runs of the same contestant set with
    different seeds produce different digests.
    """

    parts: list[str] = []
    parts.append(f"arena_id={arena_id}")
    parts.append(f"ts_ns={ts_ns}")
    parts.append(f"seed={seed}")
    parts.append(f"drawdown_weight={drawdown_weight!r}")
    parts.append("elites=" + "|".join(elites))
    parts.append("winners=" + "|".join(winners))
    parts.append("survivors=" + "|".join(survivors))
    parts.append("brackets=")
    for b in brackets:
        parts.append(
            f"  bracket_id={b.bracket_id}"
            f" candidates={'|'.join(b.candidate_ids)}"
            f" winner={b.winner_id}"
            f" fitness={b.winner_fitness!r}"
        )
    canonical = "\n".join(parts).encode("utf-8")
    return hashlib.blake2b(canonical, digest_size=8).hexdigest()


# ----------------------------------------------------------------------
# Arena
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Arena:
    """Stateless tournament-selection arena.

    Construct once and call :meth:`run` per generation. Results are
    advisory: the arena never demotes a strategy or deploys a winner
    itself. INV-13 / INV-14 governance isolation: downstream consumers
    (A-03.2 kill_underperformers, A-03.3 promotion_engine) translate
    arena outcomes into typed governance proposals.
    """

    def run(
        self,
        *,
        contestants: Sequence[Contestant],
        config: ArenaConfig,
        seed: int,
        ts_ns: int,
    ) -> TournamentResult:
        """Run tournament selection + elitism over ``contestants``."""

        contestants_tuple = _validate_contestants(contestants)
        if not isinstance(config, ArenaConfig):
            raise ArenaInputError(f"Arena.run requires an ArenaConfig, got {type(config).__name__}")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ArenaInputError(f"Arena.run.seed must be int, got {type(seed).__name__}")
        if seed < 0:
            raise ArenaInputError(f"Arena.run.seed must be non-negative, got {seed!r}")
        if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
            raise ArenaInputError(f"Arena.run.ts_ns must be int, got {type(ts_ns).__name__}")
        if ts_ns < 0:
            raise ArenaInputError(f"Arena.run.ts_ns must be non-negative, got {ts_ns!r}")
        if config.elitism_count > len(contestants_tuple):
            raise ArenaInputError(
                "ArenaConfig.elitism_count exceeds contestants count, "
                f"elitism={config.elitism_count}, "
                f"contestants={len(contestants_tuple)}"
            )

        ranked = _ranked_by_fitness(contestants_tuple, config.drawdown_weight)
        elites = tuple(c.strategy_id for c in ranked[: config.elitism_count])

        brackets_idx = _build_brackets(
            n_contestants=len(contestants_tuple),
            tournament_size=config.tournament_size,
            n_winners=config.n_winners,
            seed=seed,
            ts_ns=ts_ns,
        )
        brackets: list[TournamentBracket] = []
        winners: list[str] = []
        for bracket_id, idx_tuple in enumerate(brackets_idx):
            winner, fitness = _select_tournament_winner(
                contestants_tuple, idx_tuple, config.drawdown_weight
            )
            candidate_ids = tuple(contestants_tuple[i].strategy_id for i in idx_tuple)
            brackets.append(
                TournamentBracket(
                    bracket_id=bracket_id,
                    candidate_ids=candidate_ids,
                    winner_id=winner.strategy_id,
                    winner_fitness=fitness,
                )
            )
            winners.append(winner.strategy_id)

        survivors = _dedupe_preserving_order(list(elites) + winners)

        digest = _arena_digest(
            arena_id=config.arena_id,
            ts_ns=ts_ns,
            seed=seed,
            drawdown_weight=config.drawdown_weight,
            elites=elites,
            winners=winners,
            survivors=survivors,
            brackets=brackets,
        )

        return TournamentResult(
            arena_id=config.arena_id,
            ts_ns=ts_ns,
            seed=seed,
            contestants=contestants_tuple,
            elites=elites,
            tournament_winners=tuple(winners),
            survivors=tuple(survivors),
            brackets=tuple(brackets),
            arena_digest=digest,
        )


def _validate_contestants(
    contestants: Iterable[Contestant],
) -> tuple[Contestant, ...]:
    """Validate the contestant set and freeze it as a tuple.

    Enforces unique ``strategy_id`` (no duplicates), bounds count, and
    confirms every entry is a :class:`Contestant`.
    """

    if not isinstance(contestants, (list, tuple)):
        raise ArenaInputError(
            f"Arena.run.contestants must be a list or tuple, got {type(contestants).__name__}"
        )
    seen: set[str] = set()
    out: list[Contestant] = []
    for entry in contestants:
        if not isinstance(entry, Contestant):
            raise ArenaInputError(
                f"Arena.run.contestants entries must be Contestant, got {type(entry).__name__}"
            )
        if entry.strategy_id in seen:
            raise ArenaInputError(
                f"Arena.run.contestants strategy_id must be unique, duplicate {entry.strategy_id!r}"
            )
        seen.add(entry.strategy_id)
        out.append(entry)
    if len(out) < MIN_CONTESTANTS:
        raise ArenaInputError(
            f"Arena.run.contestants must have >= {MIN_CONTESTANTS} entries, got {len(out)}"
        )
    if len(out) > MAX_CONTESTANTS:
        raise ArenaInputError(
            f"Arena.run.contestants must have <= {MAX_CONTESTANTS} entries, got {len(out)}"
        )
    return tuple(out)


def _dedupe_preserving_order(items: Sequence[str]) -> list[str]:
    """First-occurrence-wins dedup preserving order."""

    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


__all__ = [
    "Arena",
    "ArenaConfig",
    "ArenaConfigError",
    "ArenaInputError",
    "Contestant",
    "DEFAULT_DRAWDOWN_WEIGHT",
    "MAX_ARENA_ID_LEN",
    "MAX_CONTESTANTS",
    "MAX_ELITISM_COUNT",
    "MAX_N_WINNERS",
    "MAX_TOURNAMENT_SIZE",
    "MIN_CONTESTANTS",
    "MIN_TOURNAMENT_SIZE",
    "NEW_PIP_DEPENDENCIES",
    "PROPOSAL_SOURCE",
    "TournamentBracket",
    "TournamentResult",
]
