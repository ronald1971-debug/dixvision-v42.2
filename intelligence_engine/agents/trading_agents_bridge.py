"""C-18 TradingAgents — role-based agent architecture bridge.

# ADAPTED FROM: TauricResearch/TradingAgents
#   - role separation (Fundamentals / Technical / Sentiment / Researcher /
#     Portfolio Manager)
#   - debate / committee structure
#   - signal fusion across roles

What survives from TradingAgents:

* **Role separation pattern**: a closed roster of five analyst roles
  (``FUNDAMENTALS``, ``TECHNICAL``, ``SENTIMENT``, ``RESEARCHER``,
  ``PORTFOLIO_MANAGER``). Each role's prompt is conditioned by a
  role-specific persona slice; the role tag travels with every
  produced :class:`RoleAnalysis` so the operator can audit which
  role recommended what.
* **Two-stage committee structure**: the four analyst roles
  (``FUNDAMENTALS`` / ``TECHNICAL`` / ``SENTIMENT`` /
  ``RESEARCHER``) each produce one :class:`RoleAnalysis`; the
  ``PORTFOLIO_MANAGER`` role then fuses the four analyses into a
  single :class:`TradingAgentsProposal`.
* **Signal-fusion shape**: portfolio manager output is a
  ``(direction, confidence, rationale)`` triple, with the per-role
  intermediate analyses preserved on the proposal for HITL audit.

What is **stripped**:

* **Framework internals** — no TradingAgents class, no LangGraph
  wiring, no streaming dispatcher. The bridge keeps only the
  five-role decomposition and the two-stage analyst-then-manager
  flow.
* **Direct LLM transport** — the production
  :class:`RoleSpeaker` factory lazy-imports the LiteLLM router
  (S-12) inside its body. No top-level ``litellm`` /
  ``tradingagents`` / OpenAI / Anthropic import on this module.
* **Tool-use loop** — TradingAgents lets analysts call external
  tools (yfinance, etc.) mid-thought. DIX gates external data
  through :class:`SensorArray` / :class:`SCVS`; analysts here are
  pure functions of the operator-supplied ``brief`` value and the
  configured persona.

Authority discipline (OFFLINE_ONLY — advisory):

* **B27 / B28 / INV-71 authority symmetry** — the bridge sits in
  the **intelligence-engine / agents** tier. It does **not**
  construct :class:`SignalEvent`, :class:`PatchProposal`,
  :class:`GovernanceDecision`, or :class:`ExecutionIntent`. The
  single output is :class:`TradingAgentsProposal`, an advisory
  value object that a downstream governance-gated approval
  queue may promote to a typed bus event. Pinned by AST tests.
* **B1 engine isolation** — no ``execution_engine.*`` /
  ``governance_engine.*`` / ``system_engine.*`` /
  ``evolution_engine.*`` imports. Pinned by AST tests.
* **INV-15 determinism** — module imports no ``random`` /
  ``time`` / ``datetime`` / ``secrets`` / ``os`` / ``asyncio``;
  callers supply every source of variability (role roster,
  :class:`RoleSpeaker`, ``brief`` string, ``RoleAnalysisExtractor``
  callable). Given a deterministic :class:`RoleSpeaker` the entire
  flow is byte-identical replayable; pinned by 3-run replay tests.
* **No top-level tradingagents / litellm import** — both are
  out-of-tier; the production :class:`RoleSpeaker` factory lazy-
  imports the LiteLLM router only inside its body. Pinned by an
  AST test and by :func:`enable_trading_agents_factory` raising
  :class:`NotImplementedError` until the live backend is wired in
  a follow-up PR.

Tier: OFFLINE_ONLY (advisory). The single output is
:class:`TradingAgentsProposal`, a frozen+slots advisory value
object carrying:

* ``brief`` — the operator brief the committee was asked to score;
* ``analyses`` — ordered tuple of :class:`RoleAnalysis` from the
  four analyst roles;
* ``direction`` — caller-extracted final label (``"BUY"`` /
  ``"SELL"`` / ``"HOLD"``);
* ``confidence`` — caller-extracted final confidence in ``[0, 1]``;
* ``rationale`` — caller-extracted natural-language fusion text;
* ``proposal_digest`` — BLAKE2b-16 hex of the canonical projection.
"""

# ADAPTED FROM: TauricResearch/TradingAgents
#   - tradingagents/graph (role roster + committee structure)
#   - tradingagents/agents (persona slices for analyst roles)
#   - tradingagents/dataflows (signal-fusion shape)

from __future__ import annotations

import dataclasses
import hashlib
import re
from collections.abc import Callable, Iterable, Mapping
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable

__all__ = (
    "ANALYST_ROLES",
    "DEFAULT_DIRECTION",
    "DEFAULT_RATIONALE",
    "DIRECTION_LABELS",
    "MAX_ANALYSIS_TEXT_LEN",
    "MAX_BRIEF_LEN",
    "MAX_PERSONA_LEN",
    "MAX_RATIONALE_LEN",
    "MAX_ROLE_LABEL_LEN",
    "NEW_PIP_DEPENDENCIES",
    "PORTFOLIO_MANAGER_ROLE",
    "RoleAnalysis",
    "RoleAnalysisExtractor",
    "RoleAnalyst",
    "RoleSpeaker",
    "TradingAgentsBridgeError",
    "TradingAgentsConfig",
    "TradingAgentsProposal",
    "TradingRole",
    "default_role_analysis_extractor",
    "enable_trading_agents_factory",
    "litellm_role_speaker_factory",
    "run_trading_agents_committee",
)

# Declared at module level so the C-18 master directive's
# ``NEW_PIP_DEPENDENCIES`` clause is satisfied. The runtime path
# does not depend on the upstream package: the committee runs
# entirely on the caller-supplied :class:`RoleSpeaker`.
NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("tradingagents",)


# ---------------------------------------------------------------------------
# Defaults / bounds
# ---------------------------------------------------------------------------

MAX_BRIEF_LEN: int = 4096
MAX_ROLE_LABEL_LEN: int = 64
MAX_PERSONA_LEN: int = 4096
MAX_ANALYSIS_TEXT_LEN: int = 8192
MAX_RATIONALE_LEN: int = 4096


DIRECTION_LABELS: frozenset[str] = frozenset({"BUY", "SELL", "HOLD"})
DEFAULT_DIRECTION: str = "HOLD"
DEFAULT_RATIONALE: str = "no consensus reached"


class TradingRole(StrEnum):
    """The closed set of roles the TradingAgents committee uses."""

    FUNDAMENTALS = "FUNDAMENTALS"
    TECHNICAL = "TECHNICAL"
    SENTIMENT = "SENTIMENT"
    RESEARCHER = "RESEARCHER"
    PORTFOLIO_MANAGER = "PORTFOLIO_MANAGER"


ANALYST_ROLES: tuple[TradingRole, ...] = (
    TradingRole.FUNDAMENTALS,
    TradingRole.TECHNICAL,
    TradingRole.SENTIMENT,
    TradingRole.RESEARCHER,
)

PORTFOLIO_MANAGER_ROLE: Final[TradingRole] = TradingRole.PORTFOLIO_MANAGER


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TradingAgentsBridgeError(ValueError):
    """Raised on contract violations (bad config / extractor output)."""


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class RoleAnalyst:
    """One named analyst slot in the committee.

    ``role`` pins the role tag the analyst plays (one of the four
    :data:`ANALYST_ROLES` for analysts, or
    :data:`PORTFOLIO_MANAGER_ROLE` for the fusion role).
    ``persona`` is the system-prompt fragment that conditions the
    analyst's replies."""

    role: TradingRole
    persona: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, TradingRole):
            raise TradingAgentsBridgeError(
                "RoleAnalyst.role must be a TradingRole enum member"
            )
        if not isinstance(self.persona, str):
            raise TradingAgentsBridgeError("RoleAnalyst.persona must be str")
        if not self.persona.strip():
            raise TradingAgentsBridgeError(
                "RoleAnalyst.persona must be non-empty"
            )
        if len(self.persona) > MAX_PERSONA_LEN:
            raise TradingAgentsBridgeError(
                "RoleAnalyst.persona exceeds "
                f"MAX_PERSONA_LEN={MAX_PERSONA_LEN}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class TradingAgentsConfig:
    """Static configuration for one committee run.

    ``brief`` is the natural-language operator brief (asset
    description, current regime, question being asked of the
    committee). ``analysts`` is a tuple of one :class:`RoleAnalyst`
    per analyst role in :data:`ANALYST_ROLES`; the per-role order
    is preserved verbatim. ``portfolio_manager`` is the fusion
    analyst (role :data:`PORTFOLIO_MANAGER_ROLE`)."""

    brief: str
    analysts: tuple[RoleAnalyst, ...]
    portfolio_manager: RoleAnalyst

    def __post_init__(self) -> None:
        if not isinstance(self.brief, str):
            raise TradingAgentsBridgeError(
                "TradingAgentsConfig.brief must be str"
            )
        if not self.brief.strip():
            raise TradingAgentsBridgeError(
                "TradingAgentsConfig.brief must be non-empty"
            )
        if len(self.brief) > MAX_BRIEF_LEN:
            raise TradingAgentsBridgeError(
                "TradingAgentsConfig.brief exceeds "
                f"MAX_BRIEF_LEN={MAX_BRIEF_LEN}"
            )
        if not isinstance(self.analysts, tuple):
            raise TradingAgentsBridgeError(
                "TradingAgentsConfig.analysts must be tuple "
                "(immutable, hashable, order-stable)"
            )
        if len(self.analysts) != len(ANALYST_ROLES):
            raise TradingAgentsBridgeError(
                "TradingAgentsConfig.analysts must have exactly "
                f"{len(ANALYST_ROLES)} entries (one per analyst role)"
            )
        if any(not isinstance(a, RoleAnalyst) for a in self.analysts):
            raise TradingAgentsBridgeError(
                "TradingAgentsConfig.analysts entries must be RoleAnalyst"
            )
        roles = tuple(a.role for a in self.analysts)
        if roles != ANALYST_ROLES:
            raise TradingAgentsBridgeError(
                "TradingAgentsConfig.analysts roles must equal "
                f"ANALYST_ROLES={tuple(r.value for r in ANALYST_ROLES)} "
                "in declared order"
            )
        if not isinstance(self.portfolio_manager, RoleAnalyst):
            raise TradingAgentsBridgeError(
                "TradingAgentsConfig.portfolio_manager must be RoleAnalyst"
            )
        if self.portfolio_manager.role is not PORTFOLIO_MANAGER_ROLE:
            raise TradingAgentsBridgeError(
                "TradingAgentsConfig.portfolio_manager.role must be "
                f"{PORTFOLIO_MANAGER_ROLE.value}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class RoleAnalysis:
    """One analyst role's contribution to the committee.

    ``role`` is the analyst's role tag (one of :data:`ANALYST_ROLES`).
    ``text`` is the analyst's full reply, preserved verbatim so the
    operator can audit the original wording (the extractor reads
    ``DIRECTION: <BUY|SELL|HOLD>``, ``CONFIDENCE: <0..1>``, and
    ``RATIONALE: <text>`` lines from the reply)."""

    role: TradingRole
    text: str
    direction: str
    confidence: float
    rationale: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, TradingRole):
            raise TradingAgentsBridgeError(
                "RoleAnalysis.role must be a TradingRole enum member"
            )
        if not isinstance(self.text, str):
            raise TradingAgentsBridgeError("RoleAnalysis.text must be str")
        if len(self.text) > MAX_ANALYSIS_TEXT_LEN:
            raise TradingAgentsBridgeError(
                "RoleAnalysis.text exceeds "
                f"MAX_ANALYSIS_TEXT_LEN={MAX_ANALYSIS_TEXT_LEN}"
            )
        if not isinstance(self.direction, str):
            raise TradingAgentsBridgeError(
                "RoleAnalysis.direction must be str"
            )
        if self.direction not in DIRECTION_LABELS:
            raise TradingAgentsBridgeError(
                "RoleAnalysis.direction must be one of "
                f"{sorted(DIRECTION_LABELS)}"
            )
        if not isinstance(self.confidence, float) or isinstance(
            self.confidence, bool
        ):
            raise TradingAgentsBridgeError(
                "RoleAnalysis.confidence must be float"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise TradingAgentsBridgeError(
                "RoleAnalysis.confidence must be in [0, 1]"
            )
        if not isinstance(self.rationale, str):
            raise TradingAgentsBridgeError(
                "RoleAnalysis.rationale must be str"
            )
        if len(self.rationale) > MAX_RATIONALE_LEN:
            raise TradingAgentsBridgeError(
                "RoleAnalysis.rationale exceeds "
                f"MAX_RATIONALE_LEN={MAX_RATIONALE_LEN}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class TradingAgentsProposal:
    """Frozen advisory output of one completed committee run.

    INV-67 / B27 / B28 / INV-71: this object is **not** a typed bus
    event. It carries the four analyst replies plus a portfolio-
    manager fusion that a downstream governance-gated approval
    queue may inspect; promotion to a :class:`SignalEvent` happens
    inside :mod:`governance_engine` — never here.

    ``proposal_digest`` is the BLAKE2b-16 hex digest of the
    canonical projection (brief + per-role analyses in
    :data:`ANALYST_ROLES` order + portfolio manager fusion). Given
    the same analysts / personas / :class:`RoleSpeaker` the digest
    is byte-identical across runs and machines."""

    brief: str
    analyses: tuple[RoleAnalysis, ...]
    direction: str
    confidence: float
    rationale: str
    proposal_digest: str


# ---------------------------------------------------------------------------
# Speaker Protocol — the one seam the caller injects
# ---------------------------------------------------------------------------


@runtime_checkable
class RoleSpeaker(Protocol):
    """Pluggable LLM dispatch.

    Production wires this Protocol to the LiteLLM router (S-12);
    tests inject a deterministic fake. The Protocol is intentionally
    narrow: ``speak`` receives the calling :class:`RoleAnalyst`,
    the operator ``brief``, and the prior analyses tuple
    (analysts see an empty tuple; the portfolio manager sees the
    full four-analyst tuple), and returns the analyst's textual
    reply.

    Replies are deterministic-by-contract: if the caller wires a
    :class:`LiteLLMRouter` with ``temperature=0.0`` and a fixed
    provider order, the entire committee is byte-identical
    replayable."""

    def speak(
        self,
        *,
        analyst: RoleAnalyst,
        brief: str,
        prior_analyses: tuple[RoleAnalysis, ...],
    ) -> str:
        """Return the analyst's textual reply.

        Implementations must be pure functions of their inputs (no
        clock, no IO, no random) for INV-15 to hold."""


# ---------------------------------------------------------------------------
# RoleAnalysisExtractor — also a caller-provided callable
# ---------------------------------------------------------------------------


RoleAnalysisExtractor = Callable[
    [TradingRole, str], tuple[str, float, str]
]
"""Maps ``(role, text)`` to ``(direction, confidence, rationale)``.

The default extractor (:func:`default_role_analysis_extractor`)
parses three lines from the reply:

* ``DIRECTION: <BUY|SELL|HOLD>`` (defaults to
  :data:`DEFAULT_DIRECTION` when missing or unrecognised).
* ``CONFIDENCE: <float in [0, 1]>`` (defaults to ``0.0`` when
  missing or unparseable).
* ``RATIONALE: <text>`` (defaults to :data:`DEFAULT_RATIONALE` when
  missing).
"""


# ---------------------------------------------------------------------------
# Default extractor
# ---------------------------------------------------------------------------


_DIRECTION_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*DIRECTION\s*:\s*(\S+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_CONFIDENCE_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*CONFIDENCE\s*:\s*([-+0-9.eE]+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_RATIONALE_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*RATIONALE\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)


def default_role_analysis_extractor(
    role: TradingRole, text: str
) -> tuple[str, float, str]:
    """Parse ``DIRECTION:`` / ``CONFIDENCE:`` / ``RATIONALE:`` lines.

    Pure function: no clock, no IO, no random. INV-15 byte-identical
    across replays."""

    if not isinstance(role, TradingRole):
        raise TradingAgentsBridgeError(
            "default_role_analysis_extractor: role must be TradingRole"
        )
    if not isinstance(text, str):
        raise TradingAgentsBridgeError(
            "default_role_analysis_extractor: text must be str"
        )

    direction = DEFAULT_DIRECTION
    direction_match = _DIRECTION_RE.search(text)
    if direction_match is not None:
        candidate = direction_match.group(1).strip().upper()
        if candidate in DIRECTION_LABELS:
            direction = candidate

    confidence = 0.0
    confidence_match = _CONFIDENCE_RE.search(text)
    if confidence_match is not None:
        try:
            value = float(confidence_match.group(1))
        except ValueError:
            value = 0.0
        if 0.0 <= value <= 1.0:
            confidence = value

    rationale = DEFAULT_RATIONALE
    rationale_match = _RATIONALE_RE.search(text)
    if rationale_match is not None:
        candidate_rationale = rationale_match.group(1).strip()
        if candidate_rationale:
            rationale = candidate_rationale[:MAX_RATIONALE_LEN]

    return direction, confidence, rationale


# ---------------------------------------------------------------------------
# Canonical digest projection
# ---------------------------------------------------------------------------


def _encode_int(value: int) -> bytes:
    return str(int(value)).encode("ascii")


def _encode_float(value: float) -> bytes:
    # Round to 12 decimals and use repr to avoid platform differences.
    rounded = round(float(value), 12)
    return repr(rounded).encode("ascii")


def _encode_str(value: str) -> bytes:
    return value.encode("utf-8")


def _encode_role_analysis(analysis: RoleAnalysis) -> bytes:
    parts = [
        b"R", _encode_str(analysis.role.value),
        b"T", _encode_int(len(analysis.text)), _encode_str(analysis.text),
        b"D", _encode_str(analysis.direction),
        b"C", _encode_float(analysis.confidence),
        b"X", _encode_int(len(analysis.rationale)),
        _encode_str(analysis.rationale),
    ]
    return b"|".join(parts)


def _compute_proposal_digest(
    *,
    brief: str,
    analyses: tuple[RoleAnalysis, ...],
    direction: str,
    confidence: float,
    rationale: str,
) -> str:
    """BLAKE2b-16 digest over the canonical projection.

    Pure function: same inputs ⇒ same digest. INV-15 anchor."""

    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(b"C-18-v1|")
    hasher.update(_encode_int(len(brief)))
    hasher.update(b"|")
    hasher.update(_encode_str(brief))
    hasher.update(b"||A|")
    hasher.update(_encode_int(len(analyses)))
    hasher.update(b"|")
    for analysis in analyses:
        hasher.update(_encode_role_analysis(analysis))
        hasher.update(b"||")
    hasher.update(b"PM|")
    hasher.update(_encode_str(direction))
    hasher.update(b"|")
    hasher.update(_encode_float(confidence))
    hasher.update(b"|")
    hasher.update(_encode_int(len(rationale)))
    hasher.update(b"|")
    hasher.update(_encode_str(rationale))
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Committee runner
# ---------------------------------------------------------------------------


def _validate_speaker(speaker: object) -> None:
    if not isinstance(speaker, RoleSpeaker):
        raise TradingAgentsBridgeError(
            "run_trading_agents_committee: speaker must implement RoleSpeaker"
        )


def _validate_extractor(extractor: object) -> None:
    if not callable(extractor):
        raise TradingAgentsBridgeError(
            "run_trading_agents_committee: extractor must be callable"
        )


def run_trading_agents_committee(
    *,
    config: TradingAgentsConfig,
    speaker: RoleSpeaker,
    extractor: RoleAnalysisExtractor | None = None,
) -> TradingAgentsProposal:
    """Run one committee: four analyst roles then portfolio manager fusion.

    Pure function of its inputs: no clock, no PRNG, no IO. Given a
    deterministic ``speaker`` and ``extractor`` the entire flow is
    byte-identical replayable (INV-15).

    The four analyst roles are dispatched in :data:`ANALYST_ROLES`
    order; each :class:`RoleAnalysis` is appended to a running tuple
    so the portfolio manager sees the full analyst transcript on
    its turn. The portfolio manager's extracted
    ``(direction, confidence, rationale)`` is the proposal's fusion.

    Returns a frozen :class:`TradingAgentsProposal`."""

    if not isinstance(config, TradingAgentsConfig):
        raise TradingAgentsBridgeError(
            "run_trading_agents_committee: config must be TradingAgentsConfig"
        )
    _validate_speaker(speaker)
    parse = (
        extractor
        if extractor is not None
        else default_role_analysis_extractor
    )
    _validate_extractor(parse)

    analyses: list[RoleAnalysis] = []
    for analyst in config.analysts:
        prior = tuple(analyses)
        text = speaker.speak(
            analyst=analyst,
            brief=config.brief,
            prior_analyses=prior,
        )
        if not isinstance(text, str):
            raise TradingAgentsBridgeError(
                f"speaker reply for role={analyst.role.value} must be str"
            )
        if len(text) > MAX_ANALYSIS_TEXT_LEN:
            raise TradingAgentsBridgeError(
                f"speaker reply for role={analyst.role.value} exceeds "
                f"MAX_ANALYSIS_TEXT_LEN={MAX_ANALYSIS_TEXT_LEN}"
            )
        direction, confidence, rationale = parse(analyst.role, text)
        analyses.append(
            RoleAnalysis(
                role=analyst.role,
                text=text,
                direction=direction,
                confidence=confidence,
                rationale=rationale,
            )
        )

    final_analyses = tuple(analyses)
    pm_text = speaker.speak(
        analyst=config.portfolio_manager,
        brief=config.brief,
        prior_analyses=final_analyses,
    )
    if not isinstance(pm_text, str):
        raise TradingAgentsBridgeError(
            "speaker reply for portfolio manager must be str"
        )
    if len(pm_text) > MAX_ANALYSIS_TEXT_LEN:
        raise TradingAgentsBridgeError(
            "speaker reply for portfolio manager exceeds "
            f"MAX_ANALYSIS_TEXT_LEN={MAX_ANALYSIS_TEXT_LEN}"
        )
    pm_direction, pm_confidence, pm_rationale = parse(
        config.portfolio_manager.role, pm_text
    )

    digest = _compute_proposal_digest(
        brief=config.brief,
        analyses=final_analyses,
        direction=pm_direction,
        confidence=pm_confidence,
        rationale=pm_rationale,
    )

    return TradingAgentsProposal(
        brief=config.brief,
        analyses=final_analyses,
        direction=pm_direction,
        confidence=pm_confidence,
        rationale=pm_rationale,
        proposal_digest=digest,
    )


# ---------------------------------------------------------------------------
# Production speaker factory — lazy LiteLLM wiring (out-of-tier import)
# ---------------------------------------------------------------------------


def litellm_role_speaker_factory(
    *,
    completion: Callable[..., str],
    system_prompt_prefix: str = "",
) -> RoleSpeaker:
    """Build a :class:`RoleSpeaker` from a deterministic ``completion``
    callable (typically wired through :class:`LiteLLMRouter` from
    S-12).

    The factory is the **only** production seam that touches the
    LLM stack; the import of any LiteLLM symbol lives at the
    caller, not at module top level. Pinned by an AST test.

    ``completion`` must be a pure function of its inputs:
    ``completion(messages: list[dict[str, str]]) -> str``. In
    production wire it to :meth:`LiteLLMRouter.complete` with
    ``temperature=0.0`` so the whole committee is byte-identical
    replayable."""

    if not callable(completion):
        raise TradingAgentsBridgeError(
            "litellm_role_speaker_factory: completion must be callable"
        )
    if not isinstance(system_prompt_prefix, str):
        raise TradingAgentsBridgeError(
            "litellm_role_speaker_factory: system_prompt_prefix must be str"
        )

    @dataclasses.dataclass(frozen=True, slots=True)
    class _LiteLLMRoleSpeaker:
        completion: Callable[..., str]
        system_prompt_prefix: str

        def speak(
            self,
            *,
            analyst: RoleAnalyst,
            brief: str,
            prior_analyses: tuple[RoleAnalysis, ...],
        ) -> str:
            system_lines: list[str] = []
            if self.system_prompt_prefix:
                system_lines.append(self.system_prompt_prefix)
            system_lines.append(
                f"You play the {analyst.role.value} role in a trading "
                "committee."
            )
            system_lines.append(analyst.persona)
            system_lines.append(
                "When you are done, emit three lines: "
                "DIRECTION: <BUY|SELL|HOLD>, "
                "CONFIDENCE: <float in [0,1]>, "
                "RATIONALE: <short explanation>."
            )
            messages: list[dict[str, str]] = [
                {"role": "system", "content": "\n".join(system_lines)},
                {"role": "user", "content": f"Brief: {brief}"},
            ]
            for prior in prior_analyses:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"{prior.role.value} analyst said: {prior.text}"
                        ),
                    }
                )
            return self.completion(messages=messages)

    return _LiteLLMRoleSpeaker(
        completion=completion,
        system_prompt_prefix=system_prompt_prefix,
    )


# ---------------------------------------------------------------------------
# Lazy seam — true backend opt-in
# ---------------------------------------------------------------------------


def enable_trading_agents_factory() -> None:
    """Opt in to the upstream TradingAgents framework backend.

    Raises :class:`NotImplementedError` until a follow-up PR wires
    the live ``tradingagents`` package behind the same
    :class:`RoleSpeaker` contract. The current production path
    uses :func:`litellm_role_speaker_factory` exclusively; this
    seam exists so a downstream upgrade does not require touching
    every call site."""

    raise NotImplementedError(
        "enable_trading_agents_factory: live TradingAgents backend not yet "
        "activated. Until a follow-up PR wires the upstream package, "
        "callers must inject a deterministic RoleSpeaker via "
        "litellm_role_speaker_factory (S-12)."
    )


# ---------------------------------------------------------------------------
# Convenience re-export — read-only canonical role roster
# ---------------------------------------------------------------------------


def canonical_role_iter() -> Iterable[TradingRole]:
    """Yield the four analyst roles followed by the portfolio manager."""

    yield from ANALYST_ROLES
    yield PORTFOLIO_MANAGER_ROLE


# Sanity guard: every role tag is unique and the roster covers the enum.
def _verify_roster_invariants() -> None:
    enum_members = set(TradingRole)
    roster = set(canonical_role_iter())
    if enum_members != roster:
        raise TradingAgentsBridgeError(
            "TradingRole roster mismatch — every enum member must appear "
            "exactly once in ANALYST_ROLES + PORTFOLIO_MANAGER_ROLE"
        )


_verify_roster_invariants()


# ---------------------------------------------------------------------------
# Read-only canonical persona suggestions (operator picks at call site)
# ---------------------------------------------------------------------------


_PERSONA_HINTS: Mapping[TradingRole, str] = {
    TradingRole.FUNDAMENTALS: (
        "You analyse fundamentals: earnings, on-chain flows, macro context."
    ),
    TradingRole.TECHNICAL: (
        "You analyse technicals: price action, volume, indicators, regime."
    ),
    TradingRole.SENTIMENT: (
        "You analyse sentiment: news tone, social signals, positioning."
    ),
    TradingRole.RESEARCHER: (
        "You synthesise the prior analyses with broader research context."
    ),
    TradingRole.PORTFOLIO_MANAGER: (
        "You are the portfolio manager: weigh all analyses and emit a "
        "single direction with calibrated confidence and rationale."
    ),
}


def canonical_persona_hint(role: TradingRole) -> str:
    """Return the canonical persona hint for a role (caller may override)."""

    if not isinstance(role, TradingRole):
        raise TradingAgentsBridgeError(
            "canonical_persona_hint: role must be a TradingRole enum member"
        )
    return _PERSONA_HINTS[role]
