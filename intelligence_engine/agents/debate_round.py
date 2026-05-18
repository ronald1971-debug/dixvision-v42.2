"""Multi-agent debate round (B-03 — autogen-agentchat canonical adaptation).

# ADAPTED FROM: microsoft/autogen
#   - autogen/agentchat/conversable_agent.py (ConversableAgent base)
#   - autogen/agentchat/groupchat.py         (GroupChat + GroupChatManager
#                                             round-robin speaker selection)
#   - autogen/agentchat/user_proxy_agent.py  (human-in-the-loop terminator
#                                             pattern)

What survives from AutoGen:
* The **GroupChat round-robin debate pattern**: a fixed roster of
  named agents takes turns producing one message each, the manager
  picks the next speaker by stable round-robin order, and the
  conversation terminates either at an explicit terminator marker
  or when ``max_rounds * len(agents)`` turns have been produced.
* The **role + persona** shape of each speaker. AutoGen's
  ``ConversableAgent(name, system_message=...)`` collapses to our
  :class:`DebateAgent` ``(name, role, persona)`` triple — same
  signal, no inheritance, no class-level mutable state.
* The **terminator marker** convention from
  ``UserProxyAgent.is_termination_msg``: if the most recent turn's
  text contains the configured marker the round ends early. We
  default the marker to ``"TERMINATE"`` to match AutoGen, but it
  is fully configurable.

What is **stripped** (LGPL-3.0 mitigation is not the issue — autogen
is MIT — but the surface is still trimmed to fit the DIX contract):

* **Code-execution sandbox** — AutoGen's
  ``ConversableAgent.execute_code_blocks`` and the docker-backed
  runner are disabled. DIX already has its own simulation tier
  (S-03 nautilus, B-08 backtrader) under a totally different
  authority boundary; we never let an LLM reply execute arbitrary
  Python.
* **Azure / cloud wiring** — ``llm_config`` shape (``api_base``,
  ``api_key``, ``api_type``, etc.) is dropped wholesale. The
  caller injects a deterministic :class:`Speaker` Protocol; in
  production that Protocol is implemented by the
  :class:`intelligence_engine.cognitive.litellm_router.LiteLLMRouter`
  (S-12), which is the **only** path the runtime tier may take to
  reach an LLM provider. AutoGen's direct OpenAI / Azure clients
  are never touched.
* **Human-in-the-loop** — AutoGen's ``input()``-based blocking
  prompt is gone. The terminator-marker check is purely textual
  and runs on the LLM-produced transcript.
* **Selection-LLM** — AutoGen optionally asks a second LLM "who
  should speak next?". DIX uses pure stable round-robin keyed
  off ``len(transcript) % len(agents)``. This keeps the debate
  byte-identical replayable (INV-15) and frees the design from
  needing a second LLM-shaped seam.

Authority discipline (OFFLINE_ONLY — advisory, never executes):

* **B27 / B28 / INV-71 authority symmetry** — this module is on
  the **intelligence-engine / agents** side of the boundary. It
  does **not** construct any typed bus event
  (:class:`SignalEvent` / :class:`PatchProposal` /
  :class:`GovernanceDecision` / :class:`ExecutionIntent`). The
  debate output is :class:`DebateRoundProposal`, an advisory
  value object the operator (or a downstream governance-gated
  approval queue) may inspect; promotion to a
  :class:`GovernanceDecision` happens inside
  :mod:`governance_engine` — never here. Pinned by AST tests.
* **B1 engine isolation** — no ``execution_engine.*`` /
  ``governance_engine.*`` / ``system_engine.*`` /
  ``evolution_engine.*`` imports. Pinned by AST tests.
* **INV-15 determinism** — module imports no ``random`` /
  ``time`` / ``datetime`` / ``secrets`` / ``os`` / ``asyncio``;
  callers supply every source of variability (``seed``, agent
  order, transcript). Given a deterministic
  :class:`Speaker` the entire debate is byte-identical
  replayable; pinned by 3-run replay tests.
* **No top-level autogen / litellm import** — both are
  out-of-tier; the production :class:`Speaker` factory lazy-
  imports the LiteLLM router only inside its body. Pinned by
  AST tests.

Tier: OFFLINE_ONLY (advisory). The single output is
:class:`DebateRoundProposal`, a frozen+slots advisory value
object carrying:

* ``topic`` — the question the agents were asked to debate;
* ``turns`` — the ordered tuple of :class:`DebateTurn` records;
* ``recommendation`` — caller-extracted short label
  (e.g. ``"APPROVE"`` / ``"REJECT"`` / ``"ESCALATE"`` /
  ``"ABSTAIN"``);
* ``rationale`` — caller-extracted natural-language explanation;
* ``votes`` — sorted-key per-agent recommendation projection;
* ``converged`` — ``True`` iff the round terminated via marker;
* ``n_rounds`` / ``n_turns`` — counters;
* ``proposal_digest`` — BLAKE2b-16 of the canonical text projection
  (topic + every turn's ``(name, text)`` + recommendation +
  rationale + votes), so the same agents / personas /
  :class:`Speaker` always produce the same ``proposal_digest``.
"""

# ADAPTED FROM: microsoft/autogen (autogen/agentchat/conversable_agent.py,
#                                  autogen/agentchat/groupchat.py,
#                                  autogen/agentchat/user_proxy_agent.py)

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Pip surface — declared as an empty tuple to advertise that this module
# pulls in **no** new third-party dependency. Both ``autogen-agentchat``
# and ``litellm`` are out-of-tier; production wires through the
# :class:`Speaker` Protocol, which is implemented by the
# :class:`LiteLLMRouter` from S-12 (which already declares its own pip
# surface). Pinned by an AST test.
# ---------------------------------------------------------------------------

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ()

# ---------------------------------------------------------------------------
# Defaults / bounds
# ---------------------------------------------------------------------------

MIN_AGENTS: int = 2
MAX_AGENTS: int = 12

MIN_ROUNDS: int = 1
MAX_ROUNDS: int = 32

MAX_TOPIC_LEN: int = 2048
MAX_AGENT_NAME_LEN: int = 64
MAX_AGENT_ROLE_LEN: int = 64
MAX_PERSONA_LEN: int = 4096
MAX_TURN_TEXT_LEN: int = 8192
MAX_RATIONALE_LEN: int = 4096
MAX_RECOMMENDATION_LEN: int = 64

DEFAULT_TERMINATOR: str = "TERMINATE"
DEFAULT_RECOMMENDATION: str = "ABSTAIN"
DEFAULT_RATIONALE: str = "no consensus reached"

RECOMMENDATION_LABELS: frozenset[str] = frozenset(
    {
        "APPROVE",
        "REJECT",
        "ESCALATE",
        "ABSTAIN",
    }
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DebateRoundError(ValueError):
    """Raised on contract violations (bad agents / config / extractor output).

    Subclasses :class:`ValueError` so callers may catch it as either a
    typed :class:`DebateRoundError` or the generic
    :class:`ValueError`; both shapes appear in existing DIX call sites."""


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DebateAgent:
    """One named voice in the debate.

    ``name`` is the stable identifier used to tag every turn (and to
    key the per-agent vote projection). ``role`` is a short
    machine-friendly label (e.g. ``"bull"`` / ``"bear"`` /
    ``"risk-officer"``); ``persona`` is the system-prompt fragment
    that conditions the agent's replies.

    Frozen + slots so the entire shape is hashable and threadable
    through the round runner without surprise mutation."""

    name: str
    role: str
    persona: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise DebateRoundError("DebateAgent.name must be str")
        if not self.name.strip():
            raise DebateRoundError("DebateAgent.name must be non-empty")
        if len(self.name) > MAX_AGENT_NAME_LEN:
            raise DebateRoundError(
                f"DebateAgent.name exceeds MAX_AGENT_NAME_LEN={MAX_AGENT_NAME_LEN}"
            )
        if not isinstance(self.role, str):
            raise DebateRoundError("DebateAgent.role must be str")
        if not self.role.strip():
            raise DebateRoundError("DebateAgent.role must be non-empty")
        if len(self.role) > MAX_AGENT_ROLE_LEN:
            raise DebateRoundError(
                f"DebateAgent.role exceeds MAX_AGENT_ROLE_LEN={MAX_AGENT_ROLE_LEN}"
            )
        if not isinstance(self.persona, str):
            raise DebateRoundError("DebateAgent.persona must be str")
        if not self.persona.strip():
            raise DebateRoundError("DebateAgent.persona must be non-empty")
        if len(self.persona) > MAX_PERSONA_LEN:
            raise DebateRoundError(f"DebateAgent.persona exceeds MAX_PERSONA_LEN={MAX_PERSONA_LEN}")


@dataclass(frozen=True, slots=True)
class DebateRoundConfig:
    """Static configuration for one debate round.

    Mirrors :class:`autogen.agentchat.GroupChat` but trimmed to the
    fields the DIX tier actually consumes. ``agents`` is a tuple
    (immutable / hashable / order-stable). ``max_rounds`` caps the
    number of full round-robin sweeps; the absolute turn budget is
    ``max_rounds * len(agents)``."""

    topic: str
    agents: tuple[DebateAgent, ...]
    max_rounds: int
    terminator_marker: str = DEFAULT_TERMINATOR

    def __post_init__(self) -> None:
        if not isinstance(self.topic, str):
            raise DebateRoundError("DebateRoundConfig.topic must be str")
        if not self.topic.strip():
            raise DebateRoundError("DebateRoundConfig.topic must be non-empty")
        if len(self.topic) > MAX_TOPIC_LEN:
            raise DebateRoundError(f"DebateRoundConfig.topic exceeds MAX_TOPIC_LEN={MAX_TOPIC_LEN}")
        if not isinstance(self.agents, tuple):
            raise DebateRoundError(
                "DebateRoundConfig.agents must be tuple (immutable, hashable, order-stable)"
            )
        if len(self.agents) < MIN_AGENTS:
            raise DebateRoundError(
                f"DebateRoundConfig.agents must have at least {MIN_AGENTS} entries"
            )
        if len(self.agents) > MAX_AGENTS:
            raise DebateRoundError(
                f"DebateRoundConfig.agents must have at most {MAX_AGENTS} entries"
            )
        if any(not isinstance(a, DebateAgent) for a in self.agents):
            raise DebateRoundError("DebateRoundConfig.agents entries must be DebateAgent")
        names = [a.name for a in self.agents]
        if len(set(names)) != len(names):
            raise DebateRoundError("DebateRoundConfig.agents must have unique names")
        if not isinstance(self.max_rounds, int) or isinstance(self.max_rounds, bool):
            raise DebateRoundError("DebateRoundConfig.max_rounds must be int")
        if self.max_rounds < MIN_ROUNDS:
            raise DebateRoundError(f"DebateRoundConfig.max_rounds must be >= {MIN_ROUNDS}")
        if self.max_rounds > MAX_ROUNDS:
            raise DebateRoundError(f"DebateRoundConfig.max_rounds must be <= {MAX_ROUNDS}")
        if not isinstance(self.terminator_marker, str):
            raise DebateRoundError("DebateRoundConfig.terminator_marker must be str")
        if not self.terminator_marker:
            raise DebateRoundError("DebateRoundConfig.terminator_marker must be non-empty")

    @property
    def n_agents(self) -> int:
        return len(self.agents)


@dataclass(frozen=True, slots=True)
class DebateTurn:
    """One agent's contribution to the debate.

    ``round_idx`` is 0-based and counts only *completed* round-robin
    sweeps; the first turn of the first sweep is ``round_idx=0,
    speaker_idx=0``. ``agent_name`` is the canonical key used by
    the per-agent vote projection."""

    round_idx: int
    speaker_idx: int
    agent_name: str
    text: str


@dataclass(frozen=True, slots=True)
class DebateRoundProposal:
    """Frozen advisory output of one completed debate round.

    INV-67 / B27 / B28 / INV-71: this object is **not** a typed bus
    event. It carries the natural-language transcript and a
    caller-extracted recommendation/rationale that a downstream
    governance-gated approval queue may inspect; promotion to a
    :class:`GovernanceDecision` happens inside
    :mod:`governance_engine` — never here.

    ``proposal_digest`` is the BLAKE2b-16 hex digest of the
    canonical text projection (topic + every ``(agent_name,
    text)`` turn + recommendation + rationale + sorted-key votes).
    Given the same agents / personas / :class:`Speaker` the
    digest is byte-identical across runs and machines."""

    topic: str
    turns: tuple[DebateTurn, ...]
    recommendation: str
    rationale: str
    votes: Mapping[str, str]
    converged: bool
    n_rounds: int
    n_turns: int
    proposal_digest: str


# ---------------------------------------------------------------------------
# Speaker Protocol — the one seam the caller injects
# ---------------------------------------------------------------------------


@runtime_checkable
class Speaker(Protocol):
    """Pluggable LLM dispatch.

    Production wires this Protocol to the LiteLLM router (S-12);
    tests inject a deterministic fake. The Protocol is intentionally
    narrow: ``speak`` receives the calling :class:`DebateAgent`, the
    full transcript up to (but not including) the current turn, the
    debate ``topic``, and the 0-based ``round_idx``; it returns the
    agent's textual reply.

    Replies are deterministic-by-contract: if the caller wires a
    :class:`LiteLLMRouter` with ``temperature=0.0`` and a fixed
    provider order, the debate is byte-identical replayable."""

    def speak(
        self,
        *,
        agent: DebateAgent,
        transcript: tuple[DebateTurn, ...],
        topic: str,
        round_idx: int,
    ) -> str:
        """Return the agent's textual reply.

        Implementations must be pure functions of their inputs (no
        clock, no IO, no random) for INV-15 to hold."""


# ---------------------------------------------------------------------------
# Recommendation extractor — also a caller-provided callable
# ---------------------------------------------------------------------------


RecommendationExtractor = Callable[
    [tuple[DebateTurn, ...], DebateRoundConfig], tuple[str, str, Mapping[str, str]]
]
"""Maps a completed transcript to ``(recommendation, rationale, votes)``.

``votes`` is a per-agent-name mapping of the agent's individual
recommendation label (drawn from
:data:`RECOMMENDATION_LABELS`). The default extractor
(:func:`default_recommendation_extractor`) parses every turn for a
``RECOMMENDATION: <label>`` line and falls back to
:data:`DEFAULT_RECOMMENDATION` when none is found."""


# ---------------------------------------------------------------------------
# Default recommendation extractor
# ---------------------------------------------------------------------------


def _extract_label(text: str) -> str | None:
    """Scan ``text`` line-by-line for ``RECOMMENDATION: <label>``.

    Returns the uppercased label iff it is in
    :data:`RECOMMENDATION_LABELS`; otherwise ``None``. Comparison
    is case-insensitive on the marker prefix but exact on the
    label set."""

    for raw in text.splitlines():
        line = raw.strip()
        upper = line.upper()
        if upper.startswith("RECOMMENDATION:"):
            label = line[len("RECOMMENDATION:") :].strip().upper()
            if label in RECOMMENDATION_LABELS:
                return label
    return None


def _extract_rationale(text: str) -> str | None:
    """Scan ``text`` for a ``RATIONALE:`` block.

    Returns the (trimmed) rationale iff found; otherwise ``None``.
    Block continues until the next blank line or end-of-text."""

    lines = text.splitlines()
    out: list[str] = []
    in_block = False
    for raw in lines:
        line = raw.strip()
        if not in_block:
            if line.upper().startswith("RATIONALE:"):
                tail = line[len("RATIONALE:") :].strip()
                if tail:
                    out.append(tail)
                in_block = True
            continue
        if not line:
            break
        out.append(line)
    if not out:
        return None
    return " ".join(out)


def default_recommendation_extractor(
    transcript: tuple[DebateTurn, ...],
    config: DebateRoundConfig,
) -> tuple[str, str, Mapping[str, str]]:
    """Pure default extractor — parses ``RECOMMENDATION:`` /
    ``RATIONALE:`` markers in agent replies.

    Behaviour:

    * For each agent (in :attr:`DebateRoundConfig.agents` order) the
      extractor takes the *last* turn that agent produced and parses
      it for ``RECOMMENDATION: <label>``. If no label is found the
      agent's vote is :data:`DEFAULT_RECOMMENDATION`.
    * The round-level recommendation is the **simple plurality** of
      the votes; ties break by :data:`RECOMMENDATION_LABELS` lex
      order ascending so a tie is byte-deterministic.
    * The rationale is the last non-empty ``RATIONALE:`` block in
      the transcript; falls back to :data:`DEFAULT_RATIONALE`.

    The extractor is deterministic and pure — no random / clock / IO.
    Callers wanting different semantics inject their own
    :data:`RecommendationExtractor`."""

    if not isinstance(transcript, tuple):
        raise DebateRoundError("default_recommendation_extractor: transcript must be tuple")
    if not isinstance(config, DebateRoundConfig):
        raise DebateRoundError("default_recommendation_extractor: config must be DebateRoundConfig")

    last_turn_by_agent: dict[str, DebateTurn] = {}
    for turn in transcript:
        last_turn_by_agent[turn.agent_name] = turn

    votes: dict[str, str] = {}
    for agent in config.agents:
        last = last_turn_by_agent.get(agent.name)
        if last is None:
            votes[agent.name] = DEFAULT_RECOMMENDATION
            continue
        label = _extract_label(last.text)
        votes[agent.name] = label if label is not None else DEFAULT_RECOMMENDATION

    # Plurality vote; tie-break by sorted label order so the result
    # is deterministic across runs / machines.
    counts: dict[str, int] = {}
    for label in votes.values():
        counts[label] = counts.get(label, 0) + 1
    winner_count = max(counts.values()) if counts else 0
    winners = sorted(label for label, c in counts.items() if c == winner_count)
    recommendation = winners[0] if winners else DEFAULT_RECOMMENDATION

    rationale = DEFAULT_RATIONALE
    for turn in reversed(transcript):
        candidate = _extract_rationale(turn.text)
        if candidate:
            rationale = candidate[:MAX_RATIONALE_LEN]
            break

    return recommendation, rationale, MappingProxyType(dict(sorted(votes.items())))


# ---------------------------------------------------------------------------
# Validators for caller-supplied extractor output
# ---------------------------------------------------------------------------


def _validate_recommendation(value: object) -> str:
    if not isinstance(value, str):
        raise DebateRoundError("recommendation_extractor must return str recommendation")
    label = value.strip()
    if not label:
        raise DebateRoundError("recommendation must be non-empty")
    if len(label) > MAX_RECOMMENDATION_LEN:
        raise DebateRoundError(f"recommendation length {len(label)} > {MAX_RECOMMENDATION_LEN}")
    return label


def _validate_rationale(value: object) -> str:
    if not isinstance(value, str):
        raise DebateRoundError("recommendation_extractor must return str rationale")
    if len(value) > MAX_RATIONALE_LEN:
        raise DebateRoundError(f"rationale length {len(value)} > {MAX_RATIONALE_LEN}")
    return value


def _validate_votes(value: object, config: DebateRoundConfig) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise DebateRoundError("recommendation_extractor must return Mapping[str, str] votes")
    expected = {a.name for a in config.agents}
    got = set(value.keys())
    if got != expected:
        raise DebateRoundError(f"votes keys {sorted(got)} != agent names {sorted(expected)}")
    cleaned: dict[str, str] = {}
    for k, v in value.items():
        if not isinstance(k, str):
            raise DebateRoundError("votes keys must be str")
        if not isinstance(v, str):
            raise DebateRoundError(f"votes[{k!r}] must be str")
        if len(v) > MAX_RECOMMENDATION_LEN:
            raise DebateRoundError(f"votes[{k!r}] length {len(v)} > {MAX_RECOMMENDATION_LEN}")
        cleaned[k] = v
    return MappingProxyType(dict(sorted(cleaned.items())))


def _validate_turn_text(value: object, agent_name: str) -> str:
    if not isinstance(value, str):
        raise DebateRoundError(
            f"Speaker.speak({agent_name!r}) must return str, got {type(value).__name__}"
        )
    if len(value) > MAX_TURN_TEXT_LEN:
        raise DebateRoundError(
            f"Speaker.speak({agent_name!r}) returned {len(value)} chars > {MAX_TURN_TEXT_LEN}"
        )
    return value


# ---------------------------------------------------------------------------
# Digest helper
# ---------------------------------------------------------------------------


def _compute_proposal_digest(
    *,
    topic: str,
    turns: tuple[DebateTurn, ...],
    recommendation: str,
    rationale: str,
    votes: Mapping[str, str],
) -> str:
    """Compute the canonical BLAKE2b-16 hex digest over the debate
    transcript + outcome. Stable across runs / Python builds /
    machines so callers may use it as a forensic key."""

    h = hashlib.blake2b(digest_size=16)
    h.update(b"topic\x00")
    h.update(topic.encode("utf-8"))
    h.update(b"\x01")
    for turn in turns:
        h.update(b"turn\x00")
        h.update(str(turn.round_idx).encode("ascii"))
        h.update(b"\x00")
        h.update(str(turn.speaker_idx).encode("ascii"))
        h.update(b"\x00")
        h.update(turn.agent_name.encode("utf-8"))
        h.update(b"\x00")
        h.update(turn.text.encode("utf-8"))
        h.update(b"\x01")
    h.update(b"rec\x00")
    h.update(recommendation.encode("utf-8"))
    h.update(b"\x01")
    h.update(b"rat\x00")
    h.update(rationale.encode("utf-8"))
    h.update(b"\x01")
    for name in sorted(votes.keys()):
        h.update(b"vote\x00")
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        h.update(votes[name].encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Orchestrator — pure function, deterministic
# ---------------------------------------------------------------------------


def run_debate_round(
    *,
    config: DebateRoundConfig,
    speaker: Speaker,
    recommendation_extractor: RecommendationExtractor | None = None,
) -> DebateRoundProposal:
    """Run one full debate round under stable round-robin selection.

    The debate proceeds in sweeps over :attr:`DebateRoundConfig.agents`:
    speaker ``i`` is ``agents[turn_index % n_agents]``. After every
    turn the loop checks for the terminator marker in the just-
    produced text — if present, the round ends early and
    :attr:`DebateRoundProposal.converged` is ``True``. Otherwise
    the loop runs until ``max_rounds * n_agents`` turns are
    produced and :attr:`DebateRoundProposal.converged` is ``False``.

    The function is pure: given the same ``config`` and a
    deterministic ``speaker`` the returned
    :class:`DebateRoundProposal` is byte-identical across runs."""

    if not isinstance(config, DebateRoundConfig):
        raise DebateRoundError("run_debate_round: config must be DebateRoundConfig")
    if not isinstance(speaker, Speaker):
        raise DebateRoundError("run_debate_round: speaker must implement the Speaker Protocol")

    extractor: RecommendationExtractor = (
        recommendation_extractor
        if recommendation_extractor is not None
        else default_recommendation_extractor
    )

    transcript: list[DebateTurn] = []
    n_agents = config.n_agents
    turn_budget = config.max_rounds * n_agents
    converged = False
    turn_index = 0
    while turn_index < turn_budget:
        speaker_idx = turn_index % n_agents
        round_idx = turn_index // n_agents
        agent = config.agents[speaker_idx]
        reply = speaker.speak(
            agent=agent,
            transcript=tuple(transcript),
            topic=config.topic,
            round_idx=round_idx,
        )
        text = _validate_turn_text(reply, agent.name)
        transcript.append(
            DebateTurn(
                round_idx=round_idx,
                speaker_idx=speaker_idx,
                agent_name=agent.name,
                text=text,
            )
        )
        turn_index += 1
        if config.terminator_marker in text:
            converged = True
            break

    final_transcript = tuple(transcript)
    extracted = extractor(final_transcript, config)
    if not isinstance(extracted, tuple) or len(extracted) != 3:
        raise DebateRoundError(
            "recommendation_extractor must return (recommendation, rationale, votes) tuple"
        )
    rec_raw, rat_raw, votes_raw = extracted
    recommendation = _validate_recommendation(rec_raw)
    rationale = _validate_rationale(rat_raw)
    votes = _validate_votes(votes_raw, config)

    digest = _compute_proposal_digest(
        topic=config.topic,
        turns=final_transcript,
        recommendation=recommendation,
        rationale=rationale,
        votes=votes,
    )

    if final_transcript:
        n_rounds = final_transcript[-1].round_idx + 1
    else:  # pragma: no cover - guarded by MIN_ROUNDS / MIN_AGENTS
        n_rounds = 0

    return DebateRoundProposal(
        topic=config.topic,
        turns=final_transcript,
        recommendation=recommendation,
        rationale=rationale,
        votes=votes,
        converged=converged,
        n_rounds=n_rounds,
        n_turns=len(final_transcript),
        proposal_digest=digest,
    )


# ---------------------------------------------------------------------------
# Production speaker factory — lazy LiteLLM wiring (out-of-tier import)
# ---------------------------------------------------------------------------


def litellm_speaker_factory(
    *,
    completion: Callable[..., str],
    system_prompt_prefix: str = "",
) -> Speaker:
    """Build a :class:`Speaker` from a deterministic ``completion``
    callable (typically wired through :class:`LiteLLMRouter` from
    S-12).

    The factory is the **only** production seam that touches the
    LLM stack; the import of any LiteLLM symbol lives at the
    caller, not at module top level. Pinned by an AST test.

    ``completion`` must be a pure function of its inputs:
    ``completion(messages: list[dict[str, str]]) -> str``. In
    production wire it to
    :meth:`LiteLLMRouter.complete` with ``temperature=0.0`` so the
    whole debate is byte-identical replayable."""

    if not callable(completion):
        raise DebateRoundError("litellm_speaker_factory: completion must be callable")
    if not isinstance(system_prompt_prefix, str):
        raise DebateRoundError("litellm_speaker_factory: system_prompt_prefix must be str")

    @dataclass(frozen=True, slots=True)
    class _LiteLLMSpeaker:
        completion: Callable[..., str]
        system_prompt_prefix: str

        def speak(
            self,
            *,
            agent: DebateAgent,
            transcript: tuple[DebateTurn, ...],
            topic: str,
            round_idx: int,
        ) -> str:
            system_lines: list[str] = []
            if self.system_prompt_prefix:
                system_lines.append(self.system_prompt_prefix)
            system_lines.append(f"You are {agent.name} ({agent.role}).")
            system_lines.append(agent.persona)
            system_lines.append(
                f"Topic: {topic}. Round {round_idx}. When you are done, "
                f"emit a line beginning RECOMMENDATION: <APPROVE|REJECT|"
                f"ESCALATE|ABSTAIN> followed by a RATIONALE: block."
            )
            messages: list[dict[str, str]] = [
                {"role": "system", "content": "\n".join(system_lines)},
            ]
            for turn in transcript:
                messages.append(
                    {
                        "role": "user",
                        "content": f"{turn.agent_name}: {turn.text}",
                    }
                )
            messages.append(
                {
                    "role": "user",
                    "content": f"It is now {agent.name}'s turn.",
                }
            )
            return self.completion(messages=messages)

    return _LiteLLMSpeaker(
        completion=completion,
        system_prompt_prefix=system_prompt_prefix,
    )


__all__ = [
    "DEFAULT_RATIONALE",
    "DEFAULT_RECOMMENDATION",
    "DEFAULT_TERMINATOR",
    "DebateAgent",
    "DebateRoundConfig",
    "DebateRoundError",
    "DebateRoundProposal",
    "DebateTurn",
    "MAX_AGENTS",
    "MAX_AGENT_NAME_LEN",
    "MAX_AGENT_ROLE_LEN",
    "MAX_PERSONA_LEN",
    "MAX_RATIONALE_LEN",
    "MAX_RECOMMENDATION_LEN",
    "MAX_ROUNDS",
    "MAX_TOPIC_LEN",
    "MAX_TURN_TEXT_LEN",
    "MIN_AGENTS",
    "MIN_ROUNDS",
    "NEW_PIP_DEPENDENCIES",
    "RECOMMENDATION_LABELS",
    "RecommendationExtractor",
    "Speaker",
    "default_recommendation_extractor",
    "litellm_speaker_factory",
    "run_debate_round",
]
