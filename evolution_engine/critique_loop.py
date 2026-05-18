# ADAPTED FROM: https://github.com/geekan/MetaGPT (MIT)
#
# Tier-C C-14 — iterative critique loop wrapping the C-14 strategy
# council.
#
# MetaGPT's ``Team.run_project()`` orchestrates Roles across **many**
# rounds — each round's :class:`Message` log feeds the next round's
# context, and the Team terminates when a budget is exhausted or a
# Role signals completion. C-14 adapts that shape:
#
# * One :func:`run_critique_loop` call runs N
#   :func:`~intelligence_engine.agents.strategy_council.\
# run_critique_round` invocations.
# * Between rounds the proposal payload is mutated by the
#   caller-supplied :class:`ProposalMutator` Protocol — production
#   wires this to a LiteLLMRouter-backed mutator that consumes the
#   arbiter's :attr:`CritiqueConsensus.refinements` lines and emits
#   a refined payload.
# * The loop terminates as soon as the arbiter returns
#   :attr:`CritiqueRecommendation.APPROVE` /
#   :attr:`CritiqueRecommendation.REJECT` /
#   :attr:`CritiqueRecommendation.ESCALATE`, or when ``max_rounds``
#   is reached.
#
# Authority constraints (pinned by tests):
#
#   * **ADVISORY only** (INV-12) — the loop emits
#     :class:`CritiqueLoopResult` value objects. Never typed bus
#     events (:class:`PatchProposal` /
#     :class:`GovernanceDecision` / :class:`ExecutionIntent` /
#     :class:`SignalEvent`). Promotion to a :class:`PatchProposal`
#     happens inside :mod:`evolution_engine.patch_pipeline` only.
#   * **RUNTIME_SAFE** — pure dispatcher. No clock, no I/O, no
#     PRNG. Three independent runs with identical inputs produce
#     byte-identical :class:`CritiqueLoopResult` instances
#     (INV-15).
#   * **B1 / L2** — no execution_engine / governance_engine /
#     system_engine / learning_engine / intelligence_engine
#     submodule cross-imports. The council types live in
#     :mod:`core.contracts.critique` so both the runtime-side
#     facade (``intelligence_engine.agents.strategy_council``) and
#     this offline-side loop can depend on a single neutral
#     contract without crossing the runtime ↔ offline boundary.
#   * No top-level imports of :mod:`metagpt`, :mod:`openai`,
#     :mod:`anthropic`, :mod:`litellm`, :mod:`requests`,
#     :mod:`asyncio`, :mod:`time`, :mod:`datetime`,
#     :mod:`random`.
"""C-14 critique loop — iterative MetaGPT-shape refinement."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Final, Protocol, runtime_checkable

from core.contracts.critique import (
    CANONICAL_CRITIC_ROLES,
    CriticRole,
    CritiqueConsensus,
    CritiqueLog,
    CritiqueRecommendation,
    RoleSpec,
    StructuredSpeaker,
    run_critique_round,
)

__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "CritiqueLoopError",
    "MutationError",
    "ProposalMutator",
    "CritiqueRoundRecord",
    "CritiqueLoopResult",
    "run_critique_loop",
)


NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("metagpt",)


MIN_ROUNDS: Final[int] = 1
MAX_ROUNDS: Final[int] = 16


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CritiqueLoopError(ValueError):
    """Raised when the critique loop receives a malformed argument
    or its mutator violates the deterministic-string-payload
    contract."""


class MutationError(CritiqueLoopError):
    """Raised when a :class:`ProposalMutator` implementation
    returns a non-Mapping / non-string payload, or omits keys
    from the original proposal."""


# ---------------------------------------------------------------------------
# ProposalMutator Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ProposalMutator(Protocol):
    """Callable that applies arbiter refinements to a proposal.

    Production wires this to a LiteLLMRouter-backed mutator that
    rewrites the proposal payload in response to the arbiter's
    :attr:`CritiqueConsensus.refinements`. Tests pass deterministic
    stubs. The mutator must:

    * Return a :class:`Mapping` with the same key set as the
      original ``proposal`` (no schema drift mid-loop).
    * Return ``Mapping[str, str]`` — no nested objects, no
      non-string values.
    * Be pure: no clock, no I/O, no PRNG. Given identical inputs
      it must produce a byte-identical payload (INV-15).
    """

    def mutate(
        self,
        proposal: Mapping[str, str],
        consensus: CritiqueConsensus,
    ) -> Mapping[str, str]:  # pragma: no cover - Protocol
        ...


# ---------------------------------------------------------------------------
# Round record + loop result
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class CritiqueRoundRecord:
    """One row in :attr:`CritiqueLoopResult.rounds`.

    Captures the proposal that was critiqued in this round, the
    full critique log (4 :class:`CriticMessage` rows), and the
    arbiter's consensus."""

    round_index: int
    proposal: Mapping[str, str]
    log: CritiqueLog
    consensus: CritiqueConsensus

    def __post_init__(self) -> None:
        if not isinstance(self.round_index, int) or isinstance(self.round_index, bool):
            raise CritiqueLoopError(
                "CritiqueRoundRecord.round_index must be int, "
                f"got {type(self.round_index).__name__}"
            )
        if self.round_index < 0:
            raise CritiqueLoopError(
                f"CritiqueRoundRecord.round_index must be >= 0, got {self.round_index!r}"
            )
        if not isinstance(self.proposal, Mapping):
            raise CritiqueLoopError(
                "CritiqueRoundRecord.proposal must be a Mapping, "
                f"got {type(self.proposal).__name__}"
            )
        if not isinstance(self.log, CritiqueLog):
            raise CritiqueLoopError(
                f"CritiqueRoundRecord.log must be CritiqueLog, got {type(self.log).__name__}"
            )
        if not isinstance(self.consensus, CritiqueConsensus):
            raise CritiqueLoopError(
                "CritiqueRoundRecord.consensus must be "
                "CritiqueConsensus, got "
                f"{type(self.consensus).__name__}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class CritiqueLoopResult:
    """Final advisory output of :func:`run_critique_loop`.

    * ``proposal_id`` — echoed from input.
    * ``initial_proposal`` — round 0 proposal payload.
    * ``final_proposal`` — proposal that was critiqued in the
      terminating round.
    * ``rounds`` — per-round records, in temporal order.
    * ``terminal_recommendation`` — convenience projection of
      ``rounds[-1].consensus.recommendation``.
    * ``converged`` — True iff the loop terminated on a terminal
      recommendation (APPROVE / REJECT / ESCALATE) rather than on
      ``max_rounds`` exhaustion with a REFINE verdict still
      pending.
    """

    proposal_id: str
    initial_proposal: Mapping[str, str]
    final_proposal: Mapping[str, str]
    rounds: tuple[CritiqueRoundRecord, ...]
    terminal_recommendation: CritiqueRecommendation
    converged: bool

    def __post_init__(self) -> None:
        if not isinstance(self.proposal_id, str) or not self.proposal_id:
            raise CritiqueLoopError(
                f"CritiqueLoopResult.proposal_id must be a non-empty str, got {self.proposal_id!r}"
            )
        if not isinstance(self.initial_proposal, Mapping):
            raise CritiqueLoopError(
                "CritiqueLoopResult.initial_proposal must be a "
                f"Mapping, got {type(self.initial_proposal).__name__}"
            )
        if not isinstance(self.final_proposal, Mapping):
            raise CritiqueLoopError(
                "CritiqueLoopResult.final_proposal must be a "
                f"Mapping, got {type(self.final_proposal).__name__}"
            )
        if not isinstance(self.rounds, tuple):
            raise CritiqueLoopError(
                f"CritiqueLoopResult.rounds must be a tuple, got {type(self.rounds).__name__}"
            )
        if not self.rounds:
            raise CritiqueLoopError("CritiqueLoopResult.rounds must be non-empty")
        for i, record in enumerate(self.rounds):
            if not isinstance(record, CritiqueRoundRecord):
                raise CritiqueLoopError(
                    f"CritiqueLoopResult.rounds[{i}] must be "
                    f"CritiqueRoundRecord, got {type(record).__name__}"
                )
            if record.round_index != i:
                raise CritiqueLoopError(
                    f"CritiqueLoopResult.rounds[{i}].round_index "
                    f"must equal {i}, got {record.round_index!r}"
                )
        if not isinstance(self.terminal_recommendation, CritiqueRecommendation):
            raise CritiqueLoopError(
                "CritiqueLoopResult.terminal_recommendation must "
                "be CritiqueRecommendation, got "
                f"{type(self.terminal_recommendation).__name__}"
            )
        if not isinstance(self.converged, bool):
            raise CritiqueLoopError(
                f"CritiqueLoopResult.converged must be bool, got {type(self.converged).__name__}"
            )
        if self.rounds[-1].consensus.recommendation is not self.terminal_recommendation:
            raise CritiqueLoopError(
                "CritiqueLoopResult.terminal_recommendation must mirror the last round's consensus"
            )


# ---------------------------------------------------------------------------
# Loop runner
# ---------------------------------------------------------------------------


_TERMINAL_RECOMMENDATIONS: Final[frozenset[CritiqueRecommendation]] = frozenset(
    (
        CritiqueRecommendation.APPROVE,
        CritiqueRecommendation.REJECT,
        CritiqueRecommendation.ESCALATE,
    )
)


def _validate_role_bundle(
    bundle: Mapping[CriticRole, RoleSpec],
) -> None:
    if not isinstance(bundle, Mapping):
        raise CritiqueLoopError(
            f"run_critique_loop role_bundle must be a Mapping, got {type(bundle).__name__}"
        )
    for role in CANONICAL_CRITIC_ROLES:
        if role not in bundle:
            raise CritiqueLoopError(
                f"run_critique_loop role_bundle missing canonical role {role.value!r}"
            )


def _apply_mutator(
    mutator: ProposalMutator,
    proposal: Mapping[str, str],
    consensus: CritiqueConsensus,
) -> Mapping[str, str]:
    out = mutator.mutate(proposal, consensus)
    if not isinstance(out, Mapping):
        raise MutationError(
            f"ProposalMutator.mutate must return a Mapping, got {type(out).__name__}"
        )
    expected = set(proposal.keys())
    got = set(out.keys())
    if expected != got:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise MutationError(
            "ProposalMutator.mutate must preserve the proposal "
            f"key set; missing={missing!r} extra={extra!r}"
        )
    # Reorder to match the input declaration order so the round
    # log is byte-identical regardless of dict iteration order in
    # the mutator implementation (INV-15).
    ordered: dict[str, str] = {}
    for k in proposal.keys():
        v = out[k]
        if not isinstance(v, str):
            raise MutationError(
                f"ProposalMutator.mutate output [{k!r}] must be str, got {type(v).__name__}"
            )
        ordered[k] = v
    return ordered


def run_critique_loop(
    *,
    proposal_id: str,
    proposal: Mapping[str, str],
    role_bundle: Mapping[CriticRole, RoleSpec],
    speaker: StructuredSpeaker,
    mutator: ProposalMutator,
    max_rounds: int,
) -> CritiqueLoopResult:
    """Run an iterative MetaGPT-shape critique loop.

    Each round:

    1. :func:`run_critique_round` runs the 4-role council against
       the current proposal.
    2. If the arbiter returns a terminal recommendation (APPROVE /
       REJECT / ESCALATE), the loop stops.
    3. Otherwise (REFINE) the :class:`ProposalMutator` is invoked
       to produce the next round's proposal.

    The loop also stops when ``max_rounds`` is exhausted; in that
    case :attr:`CritiqueLoopResult.converged` is ``False`` and the
    last round's REFINE recommendation is the terminal
    recommendation.
    """

    if not isinstance(proposal_id, str) or not proposal_id:
        raise CritiqueLoopError(
            f"run_critique_loop proposal_id must be a non-empty str, got {proposal_id!r}"
        )
    if not isinstance(proposal, Mapping):
        raise CritiqueLoopError(
            f"run_critique_loop proposal must be a Mapping, got {type(proposal).__name__}"
        )
    if not proposal:
        raise CritiqueLoopError("run_critique_loop proposal must be non-empty")
    for k, v in proposal.items():
        if not isinstance(k, str) or not k:
            raise CritiqueLoopError(
                f"run_critique_loop proposal keys must be non-empty str, got {k!r}"
            )
        if not isinstance(v, str):
            raise CritiqueLoopError(
                f"run_critique_loop proposal[{k!r}] must be str, got {type(v).__name__}"
            )
    _validate_role_bundle(role_bundle)
    if not isinstance(speaker, StructuredSpeaker):
        raise CritiqueLoopError(
            "run_critique_loop speaker must implement "
            "StructuredSpeaker, got "
            f"{type(speaker).__name__}"
        )
    if not isinstance(mutator, ProposalMutator):
        raise CritiqueLoopError(
            "run_critique_loop mutator must implement "
            "ProposalMutator, got "
            f"{type(mutator).__name__}"
        )
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, int):
        raise CritiqueLoopError(
            f"run_critique_loop max_rounds must be int, got {type(max_rounds).__name__}"
        )
    if not (MIN_ROUNDS <= max_rounds <= MAX_ROUNDS):
        raise CritiqueLoopError(
            "run_critique_loop max_rounds must be in "
            f"[{MIN_ROUNDS}, {MAX_ROUNDS}], got {max_rounds!r}"
        )

    # Snapshot in declaration order so the initial_proposal field
    # is byte-stable on replay.
    initial: dict[str, str] = {k: proposal[k] for k in proposal.keys()}
    current: Mapping[str, str] = initial
    rounds: list[CritiqueRoundRecord] = []
    terminal: CritiqueRecommendation = CritiqueRecommendation.REFINE
    converged = False

    for i in range(max_rounds):
        log, consensus = run_critique_round(
            proposal_id=proposal_id,
            proposal=current,
            role_bundle=role_bundle,
            speaker=speaker,
        )
        rounds.append(
            CritiqueRoundRecord(
                round_index=i,
                proposal=dict(current),
                log=log,
                consensus=consensus,
            )
        )
        terminal = consensus.recommendation
        if terminal in _TERMINAL_RECOMMENDATIONS:
            converged = True
            break
        if i == max_rounds - 1:
            break
        current = _apply_mutator(mutator, current, consensus)

    final = dict(current)
    return CritiqueLoopResult(
        proposal_id=proposal_id,
        initial_proposal=initial,
        final_proposal=final,
        rounds=tuple(rounds),
        terminal_recommendation=terminal,
        converged=converged,
    )
