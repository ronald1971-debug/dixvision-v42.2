# ADAPTED FROM: https://github.com/geekan/MetaGPT (MIT)
#
# Tier-C C-14 — MetaGPT-shaped multi-role strategy critique council.
#
# MetaGPT's distinguishing surface is a **typed structured Message
# bus** (``metagpt.schema.Message``) over a **``Team`` of Roles**
# that ``act()`` in declaration order under a shared ``Memory`` log.
# Each role's :class:`Action` consumes the visible messages and emits
# new typed messages — never free-form strings — which the next role
# in the rotation can then critique.
#
# C-14 adapts that shape behind DIX contracts. The crewai-shape
# council already exists at
# :mod:`intelligence_engine.agents.crew_strategy_council` (B-04),
# which runs **one** sequential pass and stops. C-14 contributes the
# **iterative critique round**: every Role sees every prior message
# in the round, structured critiques accrete on a shared
# :class:`CritiqueLog`, and the Arbiter synthesises a final
# :class:`CritiqueConsensus` at the end of the round. The
# :mod:`evolution_engine.critique_loop` wraps multiple rounds
# together with mutation between rounds.
#
# The 4 canonical roles (per master directive C-14):
#
#   1. ``SIGNAL_ANALYST`` — critiques signal alpha / entry edge.
#   2. ``RISK_OFFICER`` — critiques position size / drawdown /
#      exposure.
#   3. ``REGIME_EXPERT`` — critiques fit to current macro / micro
#      regime.
#   4. ``ARBITER`` — synthesises a single
#      :class:`CritiqueConsensus` from the three critic messages.
#
# Authority constraints (pinned by tests):
#
#   * **ADVISORY only** (INV-12) — council emits
#     :class:`CritiqueConsensus` value objects only. Never
#     :class:`SignalEvent` / :class:`ExecutionIntent` /
#     :class:`PatchProposal` / :class:`GovernanceDecision`. Promotion
#     to a typed bus event is the caller's job inside
#     :mod:`governance_engine`.
#   * **RUNTIME_SAFE** — pure dispatcher. No clock, no I/O, no PRNG.
#     Three independent runs with identical inputs produce
#     byte-identical :class:`CritiqueLog` instances (INV-15).
#   * **B1** — no execution_engine / governance_engine /
#     system_engine / learning_engine / evolution_engine /
#     core.contracts.events imports.
#   * **B24** — ``metagpt`` is permitted under
#     :mod:`intelligence_engine.agents` only.
#   * No top-level imports of :mod:`metagpt`, :mod:`openai`,
#     :mod:`anthropic`, :mod:`litellm`, :mod:`requests`,
#     :mod:`asyncio`, :mod:`time`, :mod:`datetime`, :mod:`random`.
#     All LLM dispatch flows through the caller-supplied
#     :class:`StructuredSpeaker` Protocol (in production a
#     :class:`~intelligence_engine.cognitive.litellm_router.\
# LiteLLMRouter`-backed implementation).
#   * MetaGPT's ``WebUI`` / ``actions/web_browse_and_summarize`` /
#     ``DataInterpreter`` / shell-execution actions are **not**
#     re-exported. The MetaGPT UI is disabled by construction —
#     this module never imports anything from ``metagpt`` at module
#     scope.
#
# NEW_PIP_DEPENDENCIES = ("metagpt",) — declared as the lazy seam
# even though the production wiring routes through LiteLLMRouter and
# never needs ``metagpt`` installed; the constant is the contract
# advertised to ``tools/cli.py install-c-tier``.
"""C-14 strategy council — MetaGPT-shape iterative critique."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable

__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "StrategyCouncilError",
    "RoleSpecError",
    "MessageError",
    "ConsensusError",
    "CritiqueRecommendation",
    "CriticRole",
    "CANONICAL_CRITIC_ROLES",
    "RoleSpec",
    "CriticMessage",
    "CritiqueLog",
    "CritiqueConsensus",
    "StructuredSpeaker",
    "run_critique_round",
)


NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("metagpt",)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class StrategyCouncilError(ValueError):
    """Base class for C-14 strategy-council errors."""


class RoleSpecError(StrategyCouncilError):
    """Raised when a :class:`RoleSpec` is malformed or the council
    bundle is missing / duplicates a canonical role."""


class MessageError(StrategyCouncilError):
    """Raised when a :class:`CriticMessage` violates the typed
    payload contract (forbidden key, non-string value, schema
    mismatch)."""


class ConsensusError(StrategyCouncilError):
    """Raised when the arbiter's reply cannot be projected into a
    :class:`CritiqueConsensus` (missing recommendation /
    rationale)."""


# ---------------------------------------------------------------------------
# Roles + recommendations
# ---------------------------------------------------------------------------


class CriticRole(StrEnum):
    """The four canonical roles of the C-14 council.

    Order matters: :func:`run_critique_round` always iterates the
    critic trio in :data:`CANONICAL_CRITIC_ROLES` order; the
    ``ARBITER`` runs last and consumes the trio's critiques."""

    SIGNAL_ANALYST = "signal_analyst"
    RISK_OFFICER = "risk_officer"
    REGIME_EXPERT = "regime_expert"
    ARBITER = "arbiter"


CANONICAL_CRITIC_ROLES: Final[tuple[CriticRole, ...]] = (
    CriticRole.SIGNAL_ANALYST,
    CriticRole.RISK_OFFICER,
    CriticRole.REGIME_EXPERT,
    CriticRole.ARBITER,
)


class CritiqueRecommendation(StrEnum):
    """Arbiter's terminal verdict.

    * ``APPROVE`` — proposal passes; caller may forward.
    * ``REJECT`` — proposal is unsalvageable; caller drops.
    * ``REFINE`` — proposal needs another round; caller invokes
      :func:`~evolution_engine.critique_loop.run_critique_loop`
      again with the arbiter's :attr:`CritiqueConsensus.refinements`
      applied.
    * ``ESCALATE`` — proposal needs an out-of-band human review.
    """

    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REFINE = "REFINE"
    ESCALATE = "ESCALATE"


_ARBITER_RECOMMENDATION_KEY: Final[str] = "recommendation"
_ARBITER_RATIONALE_KEY: Final[str] = "rationale"
_ARBITER_REFINEMENTS_KEY: Final[str] = "refinements"

_RESERVED_ARBITER_KEYS: Final[frozenset[str]] = frozenset(
    (
        _ARBITER_RECOMMENDATION_KEY,
        _ARBITER_RATIONALE_KEY,
        _ARBITER_REFINEMENTS_KEY,
    )
)


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


MAX_GOAL_LEN: Final[int] = 1024
MAX_PROFILE_LEN: Final[int] = 4096
MAX_PAYLOAD_KEYS: Final[int] = 16
MAX_PAYLOAD_KEY_LEN: Final[int] = 64
MAX_PAYLOAD_VALUE_LEN: Final[int] = 8192
MAX_RATIONALE_LEN: Final[int] = 4096
MAX_REFINEMENT_LEN: Final[int] = 1024
MAX_REFINEMENTS: Final[int] = 16


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*"
)


def _validate_identifier(label: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise StrategyCouncilError(
            f"{label} must be a non-empty str, got {value!r}"
        )
    if not _IDENTIFIER_RE.fullmatch(value):
        raise StrategyCouncilError(
            f"{label} must match [A-Za-z_][A-Za-z0-9_]*, got "
            f"{value!r}"
        )


# ---------------------------------------------------------------------------
# RoleSpec
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class RoleSpec:
    """MetaGPT-shape Role declaration.

    Mirrors ``metagpt.roles.role.Role`` ``(name, profile, goal)``
    triple but drops the ``actions: list[Action]`` field — DIX
    couples a role to **one** structured action (its critique
    function), invoked by :func:`run_critique_round`. Multi-action
    rotation is intentionally out-of-tier (it relies on MetaGPT's
    internal LLM-driven action picker, which breaks INV-15).

    * ``role`` — canonical :class:`CriticRole` slot.
    * ``goal`` — short critique-target description (MetaGPT
      ``Role.goal``).
    * ``profile`` — long-form persona (MetaGPT ``Role.profile``).
    * ``payload_keys`` — declared structured output schema. The
      role's reply :class:`CriticMessage` must declare exactly
      these keys; extra or missing keys fail
      :class:`MessageError`. Arbiter rows MUST include
      ``recommendation`` + ``rationale``.
    """

    role: CriticRole
    goal: str
    profile: str
    payload_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.role, CriticRole):
            raise RoleSpecError(
                "RoleSpec.role must be CriticRole, got "
                f"{type(self.role).__name__}"
            )
        if not isinstance(self.goal, str) or not self.goal.strip():
            raise RoleSpecError(
                "RoleSpec.goal must be a non-empty str"
            )
        if len(self.goal) > MAX_GOAL_LEN:
            raise RoleSpecError(
                f"RoleSpec.goal length > {MAX_GOAL_LEN}"
            )
        if not isinstance(self.profile, str) or not self.profile.strip():
            raise RoleSpecError(
                "RoleSpec.profile must be a non-empty str"
            )
        if len(self.profile) > MAX_PROFILE_LEN:
            raise RoleSpecError(
                f"RoleSpec.profile length > {MAX_PROFILE_LEN}"
            )
        if not isinstance(self.payload_keys, tuple):
            raise RoleSpecError(
                "RoleSpec.payload_keys must be a tuple, got "
                f"{type(self.payload_keys).__name__}"
            )
        if not self.payload_keys:
            raise RoleSpecError(
                "RoleSpec.payload_keys must be non-empty"
            )
        if len(self.payload_keys) > MAX_PAYLOAD_KEYS:
            raise RoleSpecError(
                "RoleSpec.payload_keys count > "
                f"{MAX_PAYLOAD_KEYS}"
            )
        seen: set[str] = set()
        for i, k in enumerate(self.payload_keys):
            try:
                _validate_identifier(
                    f"RoleSpec.payload_keys[{i}]", k
                )
            except StrategyCouncilError as exc:
                raise RoleSpecError(str(exc)) from exc
            if len(k) > MAX_PAYLOAD_KEY_LEN:
                raise RoleSpecError(
                    "RoleSpec.payload_keys entry length > "
                    f"{MAX_PAYLOAD_KEY_LEN}"
                )
            if k in seen:
                raise RoleSpecError(
                    "RoleSpec.payload_keys contains duplicate "
                    f"{k!r}"
                )
            seen.add(k)
        if self.role is CriticRole.ARBITER:
            missing = _RESERVED_ARBITER_KEYS - {
                _ARBITER_REFINEMENTS_KEY
            } - set(self.payload_keys)
            if missing:
                raise RoleSpecError(
                    "ARBITER RoleSpec.payload_keys must include "
                    f"{sorted(missing)!r}"
                )


# ---------------------------------------------------------------------------
# CriticMessage
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class CriticMessage:
    """One typed structured message emitted by a Role.

    Mirrors ``metagpt.schema.Message`` ``(role, content)`` shape
    but tightens ``content`` from free-form str to a typed
    ``Mapping[str, str]`` payload validated against the
    :class:`RoleSpec.payload_keys` schema. The payload values
    themselves are strings (MetaGPT carries Pydantic models in
    ``Message.instruct_content``; DIX collapses to a flat
    string-keyed string-valued map so JSON-canonical encoding
    is byte-stable).

    * ``sender_role`` — which canonical role authored the message.
    * ``payload`` — keyed structured fields. Keys MUST exactly
      match the role's :attr:`RoleSpec.payload_keys`.
    """

    sender_role: CriticRole
    payload: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.sender_role, CriticRole):
            raise MessageError(
                "CriticMessage.sender_role must be CriticRole, "
                f"got {type(self.sender_role).__name__}"
            )
        if not isinstance(self.payload, Mapping):
            raise MessageError(
                "CriticMessage.payload must be a Mapping, got "
                f"{type(self.payload).__name__}"
            )
        if not self.payload:
            raise MessageError(
                "CriticMessage.payload must be non-empty"
            )
        for k, v in self.payload.items():
            if not isinstance(k, str) or not k:
                raise MessageError(
                    "CriticMessage.payload keys must be "
                    f"non-empty str, got {k!r}"
                )
            if len(k) > MAX_PAYLOAD_KEY_LEN:
                raise MessageError(
                    "CriticMessage.payload key length > "
                    f"{MAX_PAYLOAD_KEY_LEN}: {k!r}"
                )
            if not isinstance(v, str):
                raise MessageError(
                    f"CriticMessage.payload[{k!r}] must be str, "
                    f"got {type(v).__name__}"
                )
            if len(v) > MAX_PAYLOAD_VALUE_LEN:
                raise MessageError(
                    f"CriticMessage.payload[{k!r}] length > "
                    f"{MAX_PAYLOAD_VALUE_LEN}"
                )


# ---------------------------------------------------------------------------
# CritiqueLog
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class CritiqueLog:
    """All :class:`CriticMessage` instances emitted in one round.

    Mirrors MetaGPT's per-Team ``Memory`` log: every Role in the
    round can read every prior message; the log preserves
    declaration order so replay is byte-identical (INV-15).
    """

    proposal_id: str
    messages: tuple[CriticMessage, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.proposal_id, str) or not self.proposal_id:
            raise StrategyCouncilError(
                "CritiqueLog.proposal_id must be a non-empty str, "
                f"got {self.proposal_id!r}"
            )
        if not isinstance(self.messages, tuple):
            raise StrategyCouncilError(
                "CritiqueLog.messages must be a tuple, got "
                f"{type(self.messages).__name__}"
            )
        if len(self.messages) != len(CANONICAL_CRITIC_ROLES):
            raise StrategyCouncilError(
                "CritiqueLog.messages must have exactly "
                f"{len(CANONICAL_CRITIC_ROLES)} entries, got "
                f"{len(self.messages)}"
            )
        for i, msg in enumerate(self.messages):
            if not isinstance(msg, CriticMessage):
                raise StrategyCouncilError(
                    f"CritiqueLog.messages[{i}] must be "
                    f"CriticMessage, got {type(msg).__name__}"
                )
            if msg.sender_role is not CANONICAL_CRITIC_ROLES[i]:
                raise StrategyCouncilError(
                    f"CritiqueLog.messages[{i}].sender_role must "
                    f"be {CANONICAL_CRITIC_ROLES[i]!r}, got "
                    f"{msg.sender_role!r}"
                )

    def by_role(self, role: CriticRole) -> CriticMessage:
        for msg in self.messages:
            if msg.sender_role is role:
                return msg
        raise StrategyCouncilError(
            f"CritiqueLog.by_role: role {role!r} not in log"
        )


# ---------------------------------------------------------------------------
# CritiqueConsensus
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class CritiqueConsensus:
    """Arbiter's terminal verdict for the round.

    Projected from the ``ARBITER`` :class:`CriticMessage`. The
    arbiter's payload MUST carry:

    * ``recommendation`` — a :class:`CritiqueRecommendation` label.
    * ``rationale`` — free-form text rationale, length-bounded.
    * ``refinements`` (optional) — newline-separated structured
      mutation directives to apply before the next round.
    """

    proposal_id: str
    recommendation: CritiqueRecommendation
    rationale: str
    refinements: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.proposal_id, str) or not self.proposal_id:
            raise ConsensusError(
                "CritiqueConsensus.proposal_id must be a "
                f"non-empty str, got {self.proposal_id!r}"
            )
        if not isinstance(
            self.recommendation, CritiqueRecommendation
        ):
            raise ConsensusError(
                "CritiqueConsensus.recommendation must be a "
                "CritiqueRecommendation, got "
                f"{type(self.recommendation).__name__}"
            )
        if not isinstance(self.rationale, str):
            raise ConsensusError(
                "CritiqueConsensus.rationale must be str, got "
                f"{type(self.rationale).__name__}"
            )
        if len(self.rationale) > MAX_RATIONALE_LEN:
            raise ConsensusError(
                "CritiqueConsensus.rationale length > "
                f"{MAX_RATIONALE_LEN}"
            )
        if not isinstance(self.refinements, tuple):
            raise ConsensusError(
                "CritiqueConsensus.refinements must be a tuple, "
                f"got {type(self.refinements).__name__}"
            )
        if len(self.refinements) > MAX_REFINEMENTS:
            raise ConsensusError(
                "CritiqueConsensus.refinements count > "
                f"{MAX_REFINEMENTS}"
            )
        for i, r in enumerate(self.refinements):
            if not isinstance(r, str):
                raise ConsensusError(
                    f"CritiqueConsensus.refinements[{i}] must be "
                    f"str, got {type(r).__name__}"
                )
            if len(r) > MAX_REFINEMENT_LEN:
                raise ConsensusError(
                    f"CritiqueConsensus.refinements[{i}] length > "
                    f"{MAX_REFINEMENT_LEN}"
                )


# ---------------------------------------------------------------------------
# StructuredSpeaker Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class StructuredSpeaker(Protocol):
    """DIX projection of MetaGPT's role-LLM dispatch.

    The caller supplies an implementation that maps
    ``(RoleSpec, proposal_payload, prior_messages) ->
    Mapping[str, str]``. Production wires this to a
    :class:`~intelligence_engine.cognitive.litellm_router.LiteLLMRouter`-backed
    callable; tests pass deterministic stubs. The Protocol is
    intentionally narrow — the council never exposes provider
    SDK objects across this boundary.
    """

    def speak(
        self,
        spec: RoleSpec,
        proposal: Mapping[str, str],
        prior: tuple[CriticMessage, ...],
    ) -> Mapping[str, str]:  # pragma: no cover - Protocol
        ...


# ---------------------------------------------------------------------------
# Round runner
# ---------------------------------------------------------------------------


def _validate_role_bundle(
    bundle: Mapping[CriticRole, RoleSpec],
) -> None:
    if not isinstance(bundle, Mapping):
        raise RoleSpecError(
            "run_critique_round role_bundle must be a Mapping, "
            f"got {type(bundle).__name__}"
        )
    missing = sorted(
        r.value for r in CANONICAL_CRITIC_ROLES if r not in bundle
    )
    if missing:
        raise RoleSpecError(
            "run_critique_round role_bundle missing canonical "
            f"roles: {missing!r}"
        )
    for role, spec in bundle.items():
        if not isinstance(role, CriticRole):
            raise RoleSpecError(
                "run_critique_round role_bundle keys must be "
                f"CriticRole, got {role!r}"
            )
        if not isinstance(spec, RoleSpec):
            raise RoleSpecError(
                "run_critique_round role_bundle values must be "
                f"RoleSpec, got {type(spec).__name__}"
            )
        if spec.role is not role:
            raise RoleSpecError(
                f"run_critique_round role_bundle key {role!r} "
                f"does not match spec.role {spec.role!r}"
            )


def _coerce_reply(
    spec: RoleSpec, reply: Mapping[str, str]
) -> CriticMessage:
    if not isinstance(reply, Mapping):
        raise MessageError(
            "StructuredSpeaker.speak must return a Mapping, got "
            f"{type(reply).__name__}"
        )
    payload = dict(reply)
    expected = set(spec.payload_keys)
    got = set(payload.keys())
    if expected != got:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise MessageError(
            f"StructuredSpeaker.speak reply for role "
            f"{spec.role.value!r} does not match payload_keys; "
            f"missing={missing!r} extra={extra!r}"
        )
    # Re-emit in declaration order so identical inputs produce
    # byte-identical CriticMessage (INV-15).
    ordered: dict[str, str] = {}
    for k in spec.payload_keys:
        ordered[k] = payload[k]
    return CriticMessage(sender_role=spec.role, payload=ordered)


def _project_consensus(
    proposal_id: str, arbiter: CriticMessage
) -> CritiqueConsensus:
    if arbiter.sender_role is not CriticRole.ARBITER:
        raise ConsensusError(
            "consensus may only be projected from an ARBITER "
            f"message, got {arbiter.sender_role!r}"
        )
    rec_raw = arbiter.payload.get(_ARBITER_RECOMMENDATION_KEY)
    if rec_raw is None:
        raise ConsensusError(
            "ARBITER message missing "
            f"{_ARBITER_RECOMMENDATION_KEY!r} key"
        )
    try:
        recommendation = CritiqueRecommendation(rec_raw)
    except ValueError as exc:
        raise ConsensusError(
            f"ARBITER {_ARBITER_RECOMMENDATION_KEY!r} value "
            f"{rec_raw!r} is not a CritiqueRecommendation label"
        ) from exc
    rationale = arbiter.payload.get(_ARBITER_RATIONALE_KEY, "")
    refinements_raw = arbiter.payload.get(
        _ARBITER_REFINEMENTS_KEY, ""
    )
    refinements = tuple(
        line for line in refinements_raw.split("\n") if line
    )
    return CritiqueConsensus(
        proposal_id=proposal_id,
        recommendation=recommendation,
        rationale=rationale,
        refinements=refinements,
    )


def run_critique_round(
    *,
    proposal_id: str,
    proposal: Mapping[str, str],
    role_bundle: Mapping[CriticRole, RoleSpec],
    speaker: StructuredSpeaker,
) -> tuple[CritiqueLog, CritiqueConsensus]:
    """Run one critique round.

    Each canonical role acts in :data:`CANONICAL_CRITIC_ROLES`
    declaration order; the speaker receives the role's
    :class:`RoleSpec`, the original ``proposal`` payload, and the
    tuple of all prior :class:`CriticMessage` instances from this
    round. The Arbiter's reply is projected into a
    :class:`CritiqueConsensus`.

    The function is **pure**: no clock, no I/O, no PRNG. Given a
    deterministic :class:`StructuredSpeaker` the (CritiqueLog,
    CritiqueConsensus) tuple is byte-identical replayable.
    """

    if not isinstance(proposal_id, str) or not proposal_id:
        raise StrategyCouncilError(
            "run_critique_round proposal_id must be a non-empty "
            f"str, got {proposal_id!r}"
        )
    if not isinstance(proposal, Mapping):
        raise StrategyCouncilError(
            "run_critique_round proposal must be a Mapping, got "
            f"{type(proposal).__name__}"
        )
    for k, v in proposal.items():
        if not isinstance(k, str) or not k:
            raise StrategyCouncilError(
                "run_critique_round proposal keys must be "
                f"non-empty str, got {k!r}"
            )
        if not isinstance(v, str):
            raise StrategyCouncilError(
                f"run_critique_round proposal[{k!r}] must be "
                f"str, got {type(v).__name__}"
            )
    _validate_role_bundle(role_bundle)
    if not isinstance(speaker, StructuredSpeaker):
        raise StrategyCouncilError(
            "run_critique_round speaker must implement "
            "StructuredSpeaker, got "
            f"{type(speaker).__name__}"
        )

    prior: list[CriticMessage] = []
    for role in CANONICAL_CRITIC_ROLES:
        spec = role_bundle[role]
        reply = speaker.speak(spec, proposal, tuple(prior))
        msg = _coerce_reply(spec, reply)
        prior.append(msg)

    log = CritiqueLog(
        proposal_id=proposal_id, messages=tuple(prior)
    )
    consensus = _project_consensus(
        proposal_id, log.by_role(CriticRole.ARBITER)
    )
    return log, consensus


# Re-export hook kept on the public surface so callers can build
# their own bundle helpers without re-importing CriticRole twice.
_: Callable[[CriticRole], CriticRole] = lambda r: r  # noqa: E731
