"""A-05 langgraph — Multi-agent cognitive debate StateGraph.

# ADAPTED FROM: langgraph/graph/state.py
# ADAPTED FROM: langgraph/prebuilt/tool_node.py
# ADAPTED FROM: examples/multi_agent/agent_supervisor.py

Wave-A canonical adaptation, A-05 (langgraph). Extends the cognitive
subsystem with a multi-agent debate state graph that consumes the
existing :class:`RegistryDrivenChatModel` (Wave-03 PR-1, INV-67 / B24)
and persists conversation state to the audit ledger via
:class:`AuditLedgerCheckpointSaver` (Wave-03 PR-2, GOV-CP-05).

Design pillars (DIX MASTER CANONICAL A-05 §STEP 3 + INV-67):

* **Chat orchestration only.** LangGraph never controls execution
  authority. The debate graph never produces typed bus events; its
  output is :class:`DebateOutcome`, a frozen advisory value object
  containing natural-language turns and a moderator synthesis.
  Promotion to :class:`SignalEvent` happens — if at all — through
  the existing approval queue (Wave-03 PR-5) outside this module.

* **No tool calls in this surface.** This module deliberately does
  not import :mod:`langgraph.prebuilt.tool_node` at runtime; any
  future tool-using debate step must route every tool call through
  the DIX governance approval gate per the canonical rule
  ("All tool calls must pass through DIX governance approval gate").
  An AST test in ``tests/test_debate_graph.py`` pins this — the
  module string ``ToolNode`` must not appear in the file.

* **Registry-driven dispatch (B23 / B24).** The graph never names a
  vendor. The chat model is constructed from a
  :class:`ProviderResolver` callable injected at assembly time —
  the same dependency-inversion pattern PR #82 uses to keep
  ``intelligence_engine.cognitive.*`` free of any direct
  ``system_engine`` import (B1).

* **Audit-ledger checkpoints, never LangSmith / SqliteSaver.** The
  graph's checkpointer is always
  :class:`AuditLedgerCheckpointSaver`. The DIX_MASTER_CANONICAL
  rule "Checkpoint state to DIX ledger — not LangSmith" is pinned
  by an AST test asserting the module never imports
  ``langsmith`` and never imports
  ``langgraph.checkpoint.sqlite``.

* **Off by default.** Construction is gated by
  :class:`DebateFeatureFlag`, which reads
  ``DIX_COGNITIVE_DEBATE_ENABLED`` (default: enabled, mirrors the
  cognitive-chat flag). :func:`assemble_debate_graph` raises
  :class:`DebateGraphDisabledError` when the flag is explicitly
  off, so an accidental import in production cannot bring up the
  graph.

The debate is a pure round-robin orchestration:

* ``N`` debater nodes (one per :class:`DebateParticipant`) take
  turns in fixed order. Each debater calls the model with
  ``[topic-system, persona-system, *transcript, turn-prompt]`` and
  appends a single :class:`AIMessage` tagged with
  ``additional_kwargs["debate_participant"]``.
* After every full round (each participant has spoken once), the
  graph checks for convergence: if all (or any, depending on
  :attr:`DebateConfig.require_unanimous_convergence`) of the last
  ``N`` messages contain :attr:`DebateConfig.convergence_marker`
  case-insensitively, the debate stops early.
* If the graph reaches :attr:`DebateConfig.max_rounds` without
  early-stopping, routing falls through to the moderator.
* The moderator node synthesises the transcript into a single
  consensus message and the graph terminates.

The graph is fully deterministic given a deterministic
:class:`ModelLike`, by construction: no clocks, no IO, no random
sampling, no model-side temperature control on this surface. Real
:class:`RegistryDrivenChatModel` calls are non-deterministic by
nature — that non-determinism is quarantined per INV-67.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Final, Protocol, TypedDict, runtime_checkable

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from core.cognitive_router import TaskClass
from intelligence_engine.cognitive.chat.registry_driven_chat_model import (
    ChatTransport,
    ProviderResolver,
    RegistryDrivenChatModel,
)
from intelligence_engine.cognitive.checkpointing import (
    AuditLedgerCheckpointSaver,
    LedgerAppend,
)

__all__ = [
    "DEBATER_NODE_PREFIX",
    "DEFAULT_CONVERGENCE_MARKER",
    "DEFAULT_MODERATOR_PERSONA",
    "DebateConfig",
    "DebateError",
    "DebateFeatureFlag",
    "DebateGraphBundle",
    "DebateGraphDisabledError",
    "DebateGraphState",
    "DebateOutcome",
    "DebateParticipant",
    "DebateTurn",
    "FEATURE_FLAG_ENV_VAR",
    "MAX_PARTICIPANTS",
    "MAX_PARTICIPANT_NAME_LEN",
    "MAX_PERSONA_LEN",
    "MAX_ROUNDS",
    "MAX_SYNTHESIS_LEN",
    "MAX_TOPIC_LEN",
    "MIN_PARTICIPANTS",
    "MIN_ROUNDS",
    "MODERATOR_NODE_ID",
    "ModelLike",
    "PARTICIPANT_KWARG",
    "assemble_debate_graph",
    "build_debate_graph",
    "extract_outcome",
    "run_debate",
]


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


MIN_PARTICIPANTS: Final[int] = 2
"""Smallest legal debate. A 1-participant "debate" is a monologue;
the moderator synthesis loses meaning without a contrasting voice."""

MAX_PARTICIPANTS: Final[int] = 8
"""Largest legal debate. Bounds the StateGraph fan-out and keeps the
per-turn prompt length within reasonable LLM context windows."""

MIN_ROUNDS: Final[int] = 1
"""At least one full round of turns must run before synthesis."""

MAX_ROUNDS: Final[int] = 16
"""Hard ceiling on rounds — bounds total model calls to
``MAX_PARTICIPANTS * MAX_ROUNDS + 1`` (moderator)."""

MAX_TOPIC_LEN: Final[int] = 4096
"""Cap on the topic prompt length (characters). Bounds the
per-turn prompt length and keeps the audit ledger payload small."""

MAX_PERSONA_LEN: Final[int] = 2048
"""Cap on a participant's persona / moderator persona text length."""

MAX_PARTICIPANT_NAME_LEN: Final[int] = 64
"""Cap on participant names. Names are used as graph node id
suffixes; bounding the length avoids surprises in any future
ledger projection."""

MAX_SYNTHESIS_LEN: Final[int] = 8192
"""Cap on moderator synthesis text. Truncated rather than rejected
because a non-deterministic model call may exceed any bound; the
debate is advisory and the truncated synthesis is still useful."""

DEFAULT_MODERATOR_PERSONA: Final[str] = (
    "You are a neutral debate moderator. Read the transcript above and "
    "produce a single concise consensus or disagreement summary. Do not "
    "advocate for any participant; do not propose actions. Output plain "
    "text only."
)

DEFAULT_CONVERGENCE_MARKER: Final[str] = "AGREED"
"""Marker substring that, when present in every (or any, per
:attr:`DebateConfig.require_unanimous_convergence`) message of the
last completed round, ends the debate early. Compared
case-insensitively. Set to the empty string to disable
early-stopping."""

MODERATOR_NODE_ID: Final[str] = "moderator"
"""Stable node id for the synthesis node. Also used as the
participant-name tag on the moderator's :class:`AIMessage`."""

DEBATER_NODE_PREFIX: Final[str] = "debater__"
"""Prefix used to derive node ids from participant names. Keeps
debater node ids disjoint from :data:`MODERATOR_NODE_ID` even if
a participant happens to be named ``"moderator"``."""

PARTICIPANT_KWARG: Final[str] = "debate_participant"
"""Key under which each :class:`AIMessage`'s ``additional_kwargs``
carries the producing participant's name (or
:data:`MODERATOR_NODE_ID`). The :func:`extract_outcome` projection
uses this tag to recover :class:`DebateTurn` rows."""

FEATURE_FLAG_ENV_VAR: Final[str] = "DIX_COGNITIVE_DEBATE_ENABLED"
"""Environment variable that gates :func:`assemble_debate_graph`.

The debate graph is **on by default** (mirrors the cognitive-chat
flag). The flag is read whenever
:attr:`DebateFeatureFlag.enabled` is evaluated; only the explicit
falsy set (``"0"``, ``"false"``, ``"no"``, ``"off"`` —
case-insensitive) flips it off. Unset / empty / unknown / truthy
values keep it enabled."""


_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_FALSY: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DebateError(ValueError):
    """Raised by :class:`DebateParticipant` / :class:`DebateConfig` for
    invalid construction. Distinguished from :class:`ValueError` only
    by name; consumers may catch the parent."""


class DebateGraphDisabledError(RuntimeError):
    """Raised by :func:`assemble_debate_graph` when the feature flag
    is explicitly off. Distinct from a configuration error —
    production processes should never see this; the import path
    stays cheap and the graph never wires up."""


# ---------------------------------------------------------------------------
# Pluggable model surface
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelLike(Protocol):
    """Duck-typed :class:`RegistryDrivenChatModel`-shaped surface.

    The debate graph only needs a single method: invoke the model
    with a list of :class:`BaseMessage` and receive one
    :class:`BaseMessage` back. Production wiring passes a
    :class:`RegistryDrivenChatModel`; tests pass a fake.

    Inverting the model dependency keeps :func:`build_debate_graph`
    free of any ``provider_resolver`` / ``transport`` boilerplate
    and lets tests run offline with no registry."""

    def invoke(self, input: Any, /) -> BaseMessage:
        """Return the model's reply to ``input``.

        ``input`` is typed :class:`Any` to match the LangChain
        :class:`Runnable` protocol surface; in practice this module
        always passes a ``list[BaseMessage]``."""


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DebateParticipant:
    """One named voice in the debate.

    The ``name`` becomes part of the graph's node id and tags every
    :class:`AIMessage` the participant produces. The ``persona`` is
    the system-prompt fragment that conditions that participant's
    replies; it is the only thing distinguishing participants from
    each other on the wire."""

    name: str
    persona: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise DebateError("DebateParticipant.name must be a string")
        if not isinstance(self.persona, str):
            raise DebateError("DebateParticipant.persona must be a string")
        stripped_name = self.name.strip()
        if not stripped_name:
            raise DebateError("DebateParticipant.name must be non-empty")
        if len(self.name) > MAX_PARTICIPANT_NAME_LEN:
            raise DebateError(
                "DebateParticipant.name exceeds "
                f"MAX_PARTICIPANT_NAME_LEN={MAX_PARTICIPANT_NAME_LEN}"
            )
        if not self.persona.strip():
            raise DebateError("DebateParticipant.persona must be non-empty")
        if len(self.persona) > MAX_PERSONA_LEN:
            raise DebateError(
                f"DebateParticipant.persona exceeds MAX_PERSONA_LEN={MAX_PERSONA_LEN}"
            )


@dataclass(frozen=True, slots=True)
class DebateConfig:
    """Static debate configuration.

    Frozen + slots so the entire shape is hashable and threadable
    through the StateGraph builder without surprise mutation."""

    topic: str
    participants: tuple[DebateParticipant, ...]
    max_rounds: int
    moderator_persona: str = DEFAULT_MODERATOR_PERSONA
    convergence_marker: str = DEFAULT_CONVERGENCE_MARKER
    require_unanimous_convergence: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.topic, str):
            raise DebateError("DebateConfig.topic must be a string")
        if not self.topic.strip():
            raise DebateError("DebateConfig.topic must be non-empty")
        if len(self.topic) > MAX_TOPIC_LEN:
            raise DebateError(f"DebateConfig.topic exceeds MAX_TOPIC_LEN={MAX_TOPIC_LEN}")
        if not isinstance(self.participants, tuple):
            raise DebateError("DebateConfig.participants must be a tuple (immutable, hashable)")
        if len(self.participants) < MIN_PARTICIPANTS:
            raise DebateError(
                f"DebateConfig.participants must have at least {MIN_PARTICIPANTS} entries"
            )
        if len(self.participants) > MAX_PARTICIPANTS:
            raise DebateError(
                f"DebateConfig.participants must have at most {MAX_PARTICIPANTS} entries"
            )
        if any(not isinstance(p, DebateParticipant) for p in self.participants):
            raise DebateError("DebateConfig.participants entries must be DebateParticipant")
        names = [p.name for p in self.participants]
        if len(set(names)) != len(names):
            raise DebateError("DebateConfig.participants names must be unique")
        if MODERATOR_NODE_ID in names:
            raise DebateError(
                f"DebateConfig.participants must not use the reserved name {MODERATOR_NODE_ID!r}"
            )
        if not isinstance(self.max_rounds, int) or isinstance(self.max_rounds, bool):
            raise DebateError("DebateConfig.max_rounds must be an int")
        if self.max_rounds < MIN_ROUNDS:
            raise DebateError(f"DebateConfig.max_rounds must be >= {MIN_ROUNDS}")
        if self.max_rounds > MAX_ROUNDS:
            raise DebateError(f"DebateConfig.max_rounds must be <= {MAX_ROUNDS}")
        if not isinstance(self.moderator_persona, str):
            raise DebateError("DebateConfig.moderator_persona must be a string")
        if not self.moderator_persona.strip():
            raise DebateError("DebateConfig.moderator_persona must be non-empty")
        if len(self.moderator_persona) > MAX_PERSONA_LEN:
            raise DebateError(
                f"DebateConfig.moderator_persona exceeds MAX_PERSONA_LEN={MAX_PERSONA_LEN}"
            )
        if not isinstance(self.convergence_marker, str):
            raise DebateError("DebateConfig.convergence_marker must be a string")
        if not isinstance(self.require_unanimous_convergence, bool):
            raise DebateError("DebateConfig.require_unanimous_convergence must be bool")

    @property
    def n_participants(self) -> int:
        return len(self.participants)


@dataclass(frozen=True, slots=True)
class DebateTurn:
    """One participant's contribution to the debate.

    ``round_idx`` is 0-based and counts only completed rounds; the
    first turn of the first round is ``round_idx=0, speaker_idx=0``.
    The synthesizer's output is **not** a :class:`DebateTurn`; it
    is exposed as :attr:`DebateOutcome.synthesis` instead."""

    round_idx: int
    speaker_idx: int
    participant_name: str
    text: str


@dataclass(frozen=True, slots=True)
class DebateOutcome:
    """Frozen advisory result of a completed debate.

    INV-67: this object is **not** a typed bus event. It carries
    natural-language text the operator (or a downstream
    governance-gated approval queue) may inspect; promotion to a
    :class:`SignalEvent` is out of scope for this module."""

    topic: str
    turns: tuple[DebateTurn, ...]
    synthesis: str
    converged: bool
    n_rounds: int
    n_turns: int


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DebateFeatureFlag:
    """Stateless reader for the cognitive-debate feature flag.

    Frozen so multiple call sites observe the same semantics; an
    explicit ``getter`` parameter makes the flag testable without
    environment manipulation. Production wiring leaves ``getter``
    at the default (:func:`os.getenv`)."""

    getter: Callable[[str, str], str] = os.getenv  # type: ignore[assignment]

    @property
    def enabled(self) -> bool:
        raw = self.getter(FEATURE_FLAG_ENV_VAR, "").strip().lower()
        if raw in _FALSY:
            return False
        return True


# ---------------------------------------------------------------------------
# StateGraph state
# ---------------------------------------------------------------------------


class DebateGraphState(TypedDict):
    """Conversation state flowing through the debate StateGraph.

    ``transcript`` accumulates with the LangGraph ``add_messages``
    reducer so each node returns *just the new* messages it
    produced, not the full history. All other fields use the
    default last-writer-wins reducer."""

    topic: str
    transcript: Annotated[list[BaseMessage], add_messages]
    round_idx: int
    speaker_idx: int
    n_participants: int
    max_rounds: int
    converged: bool
    synthesis: str


# ---------------------------------------------------------------------------
# Bundle returned by assemble_debate_graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DebateGraphBundle:
    """Bundle returned by :func:`assemble_debate_graph`.

    Carries the compiled graph alongside the model, saver, and
    config it was built from so callers can hold references for
    inspection (e.g. an operator dashboard wanting to surface the
    active provider list, or a forensic auditor cross-referencing
    the saver's ledger rows against the graph's run)."""

    graph: Any
    model: ModelLike
    saver: BaseCheckpointSaver[Any]
    config: DebateConfig


# ---------------------------------------------------------------------------
# Internal node-id helpers
# ---------------------------------------------------------------------------


def _debater_node_id(participant_name: str) -> str:
    """Map a participant name to its StateGraph node id.

    The :data:`DEBATER_NODE_PREFIX` keeps debater node ids disjoint
    from :data:`MODERATOR_NODE_ID` even if a participant happens to
    be named ``"moderator"`` — though :class:`DebateConfig` rejects
    that name explicitly."""

    return f"{DEBATER_NODE_PREFIX}{participant_name}"


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


def _check_convergence(
    transcript: Sequence[BaseMessage],
    config: DebateConfig,
) -> bool:
    """Decide whether the debate has converged after a full round.

    Inspects the last ``n_participants`` messages in ``transcript``
    (i.e. one message per participant in the just-finished round)
    and looks for :attr:`DebateConfig.convergence_marker` in each.
    With :attr:`DebateConfig.require_unanimous_convergence` set,
    every message must contain the marker; otherwise any one will
    do.

    The empty marker disables convergence — useful for tests and
    for adversarial debates where early agreement is itself
    suspicious."""

    if not config.convergence_marker:
        return False
    n = config.n_participants
    if len(transcript) < n:
        return False
    last_round = transcript[-n:]
    marker = config.convergence_marker.upper()
    contents = [str(m.content).upper() for m in last_round]
    if config.require_unanimous_convergence:
        return all(marker in c for c in contents)
    return any(marker in c for c in contents)


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------


def _coerce_reply_to_text(reply: BaseMessage | str | object) -> str:
    """Defensive projection of the model reply onto a plain string.

    :class:`RegistryDrivenChatModel` returns an :class:`AIMessage`
    whose ``content`` is already a string; the protocol surface is
    looser, so a fake test model returning a raw string is also
    accepted. Anything else is coerced via :func:`str`."""

    if isinstance(reply, BaseMessage):
        content = reply.content
        return content if isinstance(content, str) else str(content)
    if isinstance(reply, str):
        return reply
    return str(reply)


def _debater_node_factory(
    model: ModelLike,
    participant: DebateParticipant,
    config: DebateConfig,
) -> Callable[[DebateGraphState], dict[str, Any]]:
    """Build the StateGraph node for one debater.

    Each invocation:

    1. Assembles the prompt as
       ``[topic-system, persona-system, *transcript, turn-prompt]``.
    2. Calls the model and wraps the reply as an
       :class:`AIMessage` tagged with the participant name.
    3. Advances ``speaker_idx`` round-robin and bumps
       ``round_idx`` when the round wraps.
    4. On round-wrap, recomputes ``converged`` over the just-closed
       round."""

    persona_msg = SystemMessage(content=participant.persona)
    turn_prompt = (
        f"As {participant.name}, contribute one focused turn to the "
        "debate above. Stay in character; be concise; do not address "
        "the moderator directly."
    )

    def _debate(state: DebateGraphState) -> dict[str, Any]:
        prompt: list[BaseMessage] = [
            SystemMessage(content=f"Topic: {state['topic']}"),
            persona_msg,
            *state["transcript"],
            HumanMessage(content=turn_prompt),
        ]
        reply = model.invoke(prompt)
        text = _coerce_reply_to_text(reply)
        tagged = AIMessage(
            content=text,
            additional_kwargs={PARTICIPANT_KWARG: participant.name},
        )

        n = state.get("n_participants") or config.n_participants
        cur_speaker = state.get("speaker_idx", 0)
        new_speaker = (cur_speaker + 1) % n
        round_advanced = new_speaker == 0
        new_round_idx = state.get("round_idx", 0) + (1 if round_advanced else 0)

        converged = bool(state.get("converged", False))
        if round_advanced and not converged:
            converged = _check_convergence(
                tuple(state["transcript"]) + (tagged,),
                config,
            )

        return {
            "transcript": [tagged],
            "speaker_idx": new_speaker,
            "round_idx": new_round_idx,
            "converged": converged,
        }

    return _debate


def _moderator_node_factory(
    model: ModelLike,
    config: DebateConfig,
) -> Callable[[DebateGraphState], dict[str, Any]]:
    """Build the synthesis node.

    The moderator is invoked exactly once per debate, after either
    early convergence or :attr:`DebateConfig.max_rounds`. It calls
    the model with the full transcript plus
    :attr:`DebateConfig.moderator_persona` and returns a single
    :class:`AIMessage` tagged with :data:`MODERATOR_NODE_ID`."""

    moderator_msg = SystemMessage(content=config.moderator_persona)
    synth_prompt = (
        "Synthesize the debate above into one concise summary. "
        "Identify points of consensus and points of disagreement. "
        "Do not advocate; do not propose actions."
    )

    def _moderate(state: DebateGraphState) -> dict[str, Any]:
        prompt: list[BaseMessage] = [
            SystemMessage(content=f"Topic: {state['topic']}"),
            moderator_msg,
            *state["transcript"],
            HumanMessage(content=synth_prompt),
        ]
        reply = model.invoke(prompt)
        text = _coerce_reply_to_text(reply)
        if len(text) > MAX_SYNTHESIS_LEN:
            text = text[:MAX_SYNTHESIS_LEN]
        tagged = AIMessage(
            content=text,
            additional_kwargs={PARTICIPANT_KWARG: MODERATOR_NODE_ID},
        )
        return {"transcript": [tagged], "synthesis": text}

    return _moderate


def _route_after_debater_factory(
    config: DebateConfig,
) -> Callable[[DebateGraphState], str]:
    """Build the :func:`add_conditional_edges` routing function.

    Returns either :data:`MODERATOR_NODE_ID` (terminate the
    debate) or the next debater's node id (continue round-robin)."""

    debater_node_ids = tuple(_debater_node_id(p.name) for p in config.participants)

    def _route(state: DebateGraphState) -> str:
        if state.get("converged", False):
            return MODERATOR_NODE_ID
        if state.get("round_idx", 0) >= state.get("max_rounds", config.max_rounds):
            return MODERATOR_NODE_ID
        idx = int(state.get("speaker_idx", 0))
        if idx < 0 or idx >= len(debater_node_ids):
            # Defensive: route to moderator rather than crash if state
            # is corrupted; the AST tests pin that this branch is
            # otherwise unreachable.
            return MODERATOR_NODE_ID
        return debater_node_ids[idx]

    return _route


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def build_debate_graph(
    *,
    model: ModelLike,
    saver: BaseCheckpointSaver[Any],
    config: DebateConfig,
) -> Any:
    """Compile the LangGraph debate graph wired to ``model`` / ``saver``.

    ``saver`` is typed as :class:`BaseCheckpointSaver` rather than
    :class:`AuditLedgerCheckpointSaver` so tests can sub in a fake.
    Production wiring (via :func:`assemble_debate_graph`) still
    pins it to the audit-ledger saver — the type relaxation is
    unit-test scaffolding, not a public contract."""

    if not isinstance(config, DebateConfig):
        raise DebateError("config must be a DebateConfig")

    builder: StateGraph = StateGraph(DebateGraphState)
    for participant in config.participants:
        builder.add_node(
            _debater_node_id(participant.name),
            _debater_node_factory(model, participant, config),
        )
    builder.add_node(
        MODERATOR_NODE_ID,
        _moderator_node_factory(model, config),
    )

    first_node = _debater_node_id(config.participants[0].name)
    builder.add_edge(START, first_node)

    debater_node_ids = tuple(_debater_node_id(p.name) for p in config.participants)
    edges_map: dict[str, str] = {nid: nid for nid in debater_node_ids}
    edges_map[MODERATOR_NODE_ID] = MODERATOR_NODE_ID

    router = _route_after_debater_factory(config)
    for participant in config.participants:
        builder.add_conditional_edges(
            _debater_node_id(participant.name),
            router,
            edges_map,
        )

    builder.add_edge(MODERATOR_NODE_ID, END)
    return builder.compile(checkpointer=saver)


def assemble_debate_graph(
    *,
    task: TaskClass,
    provider_resolver: ProviderResolver,
    transport: ChatTransport,
    ledger_append: LedgerAppend,
    config: DebateConfig,
    feature_flag: DebateFeatureFlag | None = None,
) -> DebateGraphBundle:
    """Bring up the debate graph end-to-end.

    Wires :class:`RegistryDrivenChatModel` against ``provider_resolver``
    + ``transport``, an :class:`AuditLedgerCheckpointSaver` against
    ``ledger_append``, and compiles them into the LangGraph debate
    graph.

    Raises :class:`DebateGraphDisabledError` unless the cognitive
    debate feature flag is on. The check happens *before* any
    object is constructed so an accidentally-imported call site is
    cheap."""

    flag = feature_flag if feature_flag is not None else DebateFeatureFlag()
    if not flag.enabled:
        raise DebateGraphDisabledError(
            f"cognitive debate is disabled — set {FEATURE_FLAG_ENV_VAR}=true to enable"
        )

    model = RegistryDrivenChatModel(
        task=task,
        provider_resolver=provider_resolver,
        transport=transport,
    )
    saver = AuditLedgerCheckpointSaver(ledger_append=ledger_append)
    graph = build_debate_graph(model=model, saver=saver, config=config)
    return DebateGraphBundle(graph=graph, model=model, saver=saver, config=config)


# ---------------------------------------------------------------------------
# Outcome projection + driver
# ---------------------------------------------------------------------------


def extract_outcome(
    state: Mapping[str, Any],
    config: DebateConfig,
) -> DebateOutcome:
    """Project a final :class:`DebateGraphState` onto a frozen outcome.

    Walks ``state["transcript"]``, separating debater turns from
    the moderator synthesis by inspecting
    :attr:`additional_kwargs[PARTICIPANT_KWARG]`. The synthesis is
    taken from the moderator-tagged message if present, else from
    ``state["synthesis"]`` (a string), else the empty string."""

    transcript: Sequence[BaseMessage] = state.get("transcript", []) or []
    moderator_text: str | None = None
    debate_msgs: list[BaseMessage] = []
    for msg in transcript:
        kwargs: Mapping[str, Any] = getattr(msg, "additional_kwargs", None) or {}
        if kwargs.get(PARTICIPANT_KWARG) == MODERATOR_NODE_ID:
            content = msg.content
            moderator_text = content if isinstance(content, str) else str(content)
        else:
            debate_msgs.append(msg)

    n = config.n_participants
    participant_names = tuple(p.name for p in config.participants)
    turns: list[DebateTurn] = []
    for i, msg in enumerate(debate_msgs):
        round_idx = i // n
        speaker_idx = i % n
        kwargs = getattr(msg, "additional_kwargs", None) or {}
        tag = kwargs.get(PARTICIPANT_KWARG)
        participant_name = (
            str(tag) if isinstance(tag, str) and tag else participant_names[speaker_idx]
        )
        content = msg.content
        text = content if isinstance(content, str) else str(content)
        turns.append(
            DebateTurn(
                round_idx=round_idx,
                speaker_idx=speaker_idx,
                participant_name=participant_name,
                text=text,
            )
        )

    if moderator_text is not None:
        synthesis = moderator_text
    else:
        raw = state.get("synthesis", "")
        synthesis = raw if isinstance(raw, str) else str(raw)

    converged = bool(state.get("converged", False))
    n_rounds = int(state.get("round_idx", 0))
    return DebateOutcome(
        topic=str(state.get("topic", config.topic)),
        turns=tuple(turns),
        synthesis=synthesis,
        converged=converged,
        n_rounds=n_rounds,
        n_turns=len(turns),
    )


def run_debate(
    bundle: DebateGraphBundle,
    *,
    thread_id: str,
) -> DebateOutcome:
    """Run one full debate against ``bundle`` on ``thread_id``.

    Builds an initial :class:`DebateGraphState` from
    ``bundle.config``, invokes the compiled graph synchronously,
    and projects the final state into :class:`DebateOutcome`.

    ``thread_id`` is the LangGraph saver key; reusing the same
    thread id resumes from the previous checkpoint, so callers
    that want a fresh debate must vary it (e.g. by uuid4)."""

    if not isinstance(thread_id, str) or not thread_id.strip():
        raise DebateError("thread_id must be a non-empty string")

    initial: DebateGraphState = {
        "topic": bundle.config.topic,
        "transcript": [],
        "round_idx": 0,
        "speaker_idx": 0,
        "n_participants": bundle.config.n_participants,
        "max_rounds": bundle.config.max_rounds,
        "converged": False,
        "synthesis": "",
    }
    invoke_config = {"configurable": {"thread_id": thread_id}}
    final_state = bundle.graph.invoke(initial, config=invoke_config)
    return extract_outcome(final_state, bundle.config)
