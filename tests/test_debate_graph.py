"""A-05 langgraph — multi-agent cognitive debate StateGraph tests.

These tests exercise the debate graph end-to-end against a fake
:class:`ModelLike` and a list-backed :data:`LedgerAppend`, so they
do not require any real LLM provider, governance engine, or
filesystem.

Coverage targets:

* feature flag: default-on, truthy / falsy / injected getter forms
* :class:`DebateParticipant` validation (name + persona bounds)
* :class:`DebateConfig` validation (participants, rounds, topic,
  moderator persona, convergence flags, reserved-name rejection)
* :func:`build_debate_graph` compiles and registers expected nodes
* :func:`assemble_debate_graph` raises on disabled flag, returns
  bundle on enabled flag, threads the registry-driven model
* end-to-end: multi-round round-robin produces ordered turns +
  one moderator synthesis tagged with :data:`MODERATOR_NODE_ID`
* convergence: unanimous vs. any, marker case-insensitivity,
  empty marker disables, partial agreement does not converge
* :class:`AuditLedgerCheckpointSaver` receives
  ``COGNITIVE_CHECKPOINT`` rows during the debate
* :func:`extract_outcome` projects state onto a frozen advisory
  :class:`DebateOutcome` (no typed bus events emitted)
* tier discipline (AST scan):
  - no governance / system / execution / evolution-engine imports
  - no construction of typed bus events (``PatchProposal``,
    ``SignalEvent``, ``GovernanceDecision``)
  - no ``langsmith`` / ``langgraph.checkpoint.sqlite`` import
  - no ``ToolNode`` reference (tool calls are out of scope here)
  - ``# ADAPTED FROM:`` headers cite langgraph
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)

from core.cognitive_router import (
    AIProvider,
    TaskClass,
    select_providers,
)
from intelligence_engine.cognitive import debate_graph as dg_mod
from intelligence_engine.cognitive.checkpointing.audit_ledger_checkpoint_saver import (
    CHECKPOINT_KIND,
)
from intelligence_engine.cognitive.debate_graph import (
    DEBATER_NODE_PREFIX,
    DEFAULT_CONVERGENCE_MARKER,
    DEFAULT_MODERATOR_PERSONA,
    FEATURE_FLAG_ENV_VAR,
    MAX_PARTICIPANT_NAME_LEN,
    MAX_PARTICIPANTS,
    MAX_PERSONA_LEN,
    MAX_ROUNDS,
    MAX_SYNTHESIS_LEN,
    MAX_TOPIC_LEN,
    MIN_PARTICIPANTS,
    MIN_ROUNDS,
    MODERATOR_NODE_ID,
    PARTICIPANT_KWARG,
    DebateConfig,
    DebateError,
    DebateFeatureFlag,
    DebateGraphBundle,
    DebateGraphDisabledError,
    DebateOutcome,
    DebateParticipant,
    DebateTurn,
    ModelLike,
    assemble_debate_graph,
    build_debate_graph,
    extract_outcome,
    run_debate,
)
from system_engine.scvs.source_registry import (
    SourceCategory,
    SourceDeclaration,
    SourceRegistry,
)

# ---------------------------------------------------------------------------
# Fakes — scriptable model, list-backed ledger, captive saver
# ---------------------------------------------------------------------------


class _ScriptedModel:
    """ModelLike that returns scripted text per call.

    If ``replies`` is a list, replies cycle through it; if a callable,
    it is called with the prompt; if a string, the same string is
    returned for every call. Records every prompt for assertions."""

    def __init__(
        self,
        replies: Sequence[str] | str | None = None,
        *,
        per_caller: Mapping[str, str] | None = None,
    ) -> None:
        if isinstance(replies, str) or replies is None:
            self._replies: list[str] | None = None
            self._fixed = replies if isinstance(replies, str) else "ok"
        else:
            self._replies = list(replies)
            self._fixed = None
        self._per_caller = dict(per_caller) if per_caller else None
        self.calls: list[list[BaseMessage]] = []

    def invoke(self, input: Any, /) -> BaseMessage:
        messages = list(input) if isinstance(input, Sequence) else [input]
        self.calls.append(messages)
        if self._per_caller is not None:
            tag = _detect_tag(messages)
            text = self._per_caller.get(tag, self._fixed or "ok")
            return AIMessage(content=text)
        if self._replies is not None:
            idx = (len(self.calls) - 1) % len(self._replies)
            return AIMessage(content=self._replies[idx])
        return AIMessage(content=self._fixed or "ok")


def _detect_tag(messages: Sequence[BaseMessage]) -> str:
    """Recover the speaker name from a debater prompt, by parsing the
    last :class:`HumanMessage` of the form ``"As {name}, ..."``."""

    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            text = str(msg.content)
            if text.startswith("As "):
                rest = text[3:]
                comma = rest.find(",")
                if comma > 0:
                    return rest[:comma]
            if text.startswith("Synthesize"):
                return MODERATOR_NODE_ID
            break
    return ""


class _RecordingLedger:
    """List-backed :data:`LedgerAppend` for assertion-friendly tests."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, dict[str, str]]] = []

    def __call__(self, kind: str, payload: Mapping[str, str]) -> None:
        self.rows.append((kind, dict(payload)))


def _registry_with_one_provider() -> SourceRegistry:
    decl = SourceDeclaration(
        id="provider-A",
        name="provider-A",
        category=SourceCategory.AI,
        provider="provider-A",
        endpoint="https://example.invalid/provider-A",
        schema="generic_chat",
        auth="bearer",
        enabled=True,
        critical=False,
        liveness_threshold_ms=0,
        capabilities=("reasoning",),
    )
    return SourceRegistry(version="test", sources=(decl,))


def _resolver(registry: SourceRegistry, task: TaskClass):
    def _inner() -> tuple[AIProvider, ...]:
        return select_providers(registry, task)

    return _inner


def _enable_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(FEATURE_FLAG_ENV_VAR, "true")


def _participants(n: int = 2) -> tuple[DebateParticipant, ...]:
    return tuple(DebateParticipant(name=f"p{i}", persona=f"persona-{i}") for i in range(n))


def _config(
    *,
    n: int = 2,
    rounds: int = 2,
    marker: str = DEFAULT_CONVERGENCE_MARKER,
    unanimous: bool = True,
) -> DebateConfig:
    return DebateConfig(
        topic="should we ship?",
        participants=_participants(n),
        max_rounds=rounds,
        convergence_marker=marker,
        require_unanimous_convergence=unanimous,
    )


# ---------------------------------------------------------------------------
# Constants surface
# ---------------------------------------------------------------------------


def test_constants_have_expected_relations() -> None:
    assert MIN_PARTICIPANTS == 2
    assert MAX_PARTICIPANTS == 8
    assert MIN_ROUNDS == 1
    assert MAX_ROUNDS == 16
    assert MAX_TOPIC_LEN > 0
    assert MAX_PERSONA_LEN > 0
    assert MAX_PARTICIPANT_NAME_LEN > 0
    assert MAX_SYNTHESIS_LEN > 0
    assert FEATURE_FLAG_ENV_VAR.startswith("DIX_")
    assert MODERATOR_NODE_ID == "moderator"
    assert DEBATER_NODE_PREFIX.endswith("__")
    assert PARTICIPANT_KWARG == "debate_participant"


def test_default_moderator_persona_is_neutral_text() -> None:
    assert isinstance(DEFAULT_MODERATOR_PERSONA, str)
    assert "moderator" in DEFAULT_MODERATOR_PERSONA.lower()
    assert "do not advocate" in DEFAULT_MODERATOR_PERSONA.lower()


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------


def test_feature_flag_default_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(FEATURE_FLAG_ENV_VAR, raising=False)
    assert DebateFeatureFlag().enabled is True


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "", "maybe"])
def test_feature_flag_truthy_or_unknown_values_enable(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(FEATURE_FLAG_ENV_VAR, value)
    assert DebateFeatureFlag().enabled is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE", "Off"])
def test_feature_flag_falsy_values_disable(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(FEATURE_FLAG_ENV_VAR, value)
    assert DebateFeatureFlag().enabled is False


def test_feature_flag_accepts_injected_getter() -> None:
    on_flag = DebateFeatureFlag(getter=lambda _n, _d: "true")
    off_flag = DebateFeatureFlag(getter=lambda _n, _d: "false")
    assert on_flag.enabled is True
    assert off_flag.enabled is False


# ---------------------------------------------------------------------------
# DebateParticipant validation
# ---------------------------------------------------------------------------


def test_participant_accepts_valid_input() -> None:
    p = DebateParticipant(name="alice", persona="Alice is a contrarian.")
    assert p.name == "alice"
    assert p.persona == "Alice is a contrarian."


def test_participant_is_frozen() -> None:
    p = DebateParticipant(name="alice", persona="x")
    with pytest.raises((AttributeError, DebateError)):
        p.name = "bob"  # type: ignore[misc]


@pytest.mark.parametrize("name", ["", " ", "\t", "\n  \t"])
def test_participant_rejects_blank_name(name: str) -> None:
    with pytest.raises(DebateError):
        DebateParticipant(name=name, persona="x")


def test_participant_rejects_oversized_name() -> None:
    with pytest.raises(DebateError):
        DebateParticipant(name="a" * (MAX_PARTICIPANT_NAME_LEN + 1), persona="x")


@pytest.mark.parametrize("persona", ["", " ", "\t"])
def test_participant_rejects_blank_persona(persona: str) -> None:
    with pytest.raises(DebateError):
        DebateParticipant(name="alice", persona=persona)


def test_participant_rejects_oversized_persona() -> None:
    with pytest.raises(DebateError):
        DebateParticipant(name="alice", persona="x" * (MAX_PERSONA_LEN + 1))


def test_participant_rejects_non_string_fields() -> None:
    with pytest.raises(DebateError):
        DebateParticipant(name=123, persona="x")  # type: ignore[arg-type]
    with pytest.raises(DebateError):
        DebateParticipant(
            name="alice",
            persona=42,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# DebateConfig validation
# ---------------------------------------------------------------------------


def test_config_accepts_valid_input() -> None:
    cfg = _config()
    assert cfg.topic == "should we ship?"
    assert cfg.n_participants == 2
    assert cfg.max_rounds == 2
    assert cfg.moderator_persona == DEFAULT_MODERATOR_PERSONA


def test_config_is_frozen() -> None:
    cfg = _config()
    with pytest.raises((AttributeError, DebateError)):
        cfg.max_rounds = 5  # type: ignore[misc]


@pytest.mark.parametrize("topic", ["", "  \t\n"])
def test_config_rejects_blank_topic(topic: str) -> None:
    with pytest.raises(DebateError):
        DebateConfig(
            topic=topic,
            participants=_participants(),
            max_rounds=1,
        )


def test_config_rejects_oversized_topic() -> None:
    with pytest.raises(DebateError):
        DebateConfig(
            topic="x" * (MAX_TOPIC_LEN + 1),
            participants=_participants(),
            max_rounds=1,
        )


def test_config_rejects_too_few_participants() -> None:
    with pytest.raises(DebateError):
        DebateConfig(
            topic="t",
            participants=_participants(MIN_PARTICIPANTS - 1),
            max_rounds=1,
        )


def test_config_rejects_too_many_participants() -> None:
    with pytest.raises(DebateError):
        DebateConfig(
            topic="t",
            participants=_participants(MAX_PARTICIPANTS + 1),
            max_rounds=1,
        )


def test_config_rejects_duplicate_participant_names() -> None:
    p = DebateParticipant(name="dup", persona="x")
    q = DebateParticipant(name="dup", persona="y")
    with pytest.raises(DebateError):
        DebateConfig(topic="t", participants=(p, q), max_rounds=1)


def test_config_rejects_reserved_moderator_name() -> None:
    p = DebateParticipant(name=MODERATOR_NODE_ID, persona="x")
    q = DebateParticipant(name="other", persona="y")
    with pytest.raises(DebateError):
        DebateConfig(topic="t", participants=(p, q), max_rounds=1)


def test_config_rejects_non_tuple_participants() -> None:
    with pytest.raises(DebateError):
        DebateConfig(
            topic="t",
            participants=list(_participants()),  # type: ignore[arg-type]
            max_rounds=1,
        )


def test_config_rejects_non_participant_entries() -> None:
    with pytest.raises(DebateError):
        DebateConfig(
            topic="t",
            participants=("not-a-participant", "neither"),  # type: ignore[arg-type]
            max_rounds=1,
        )


@pytest.mark.parametrize("rounds", [0, -1, MIN_ROUNDS - 1, MAX_ROUNDS + 1])
def test_config_rejects_out_of_range_rounds(rounds: int) -> None:
    with pytest.raises(DebateError):
        DebateConfig(
            topic="t",
            participants=_participants(),
            max_rounds=rounds,
        )


def test_config_rejects_bool_max_rounds() -> None:
    with pytest.raises(DebateError):
        DebateConfig(
            topic="t",
            participants=_participants(),
            max_rounds=True,  # type: ignore[arg-type]
        )


def test_config_rejects_blank_moderator_persona() -> None:
    with pytest.raises(DebateError):
        DebateConfig(
            topic="t",
            participants=_participants(),
            max_rounds=1,
            moderator_persona="   ",
        )


def test_config_rejects_oversized_moderator_persona() -> None:
    with pytest.raises(DebateError):
        DebateConfig(
            topic="t",
            participants=_participants(),
            max_rounds=1,
            moderator_persona="x" * (MAX_PERSONA_LEN + 1),
        )


def test_config_rejects_non_bool_unanimous_flag() -> None:
    with pytest.raises(DebateError):
        DebateConfig(
            topic="t",
            participants=_participants(),
            max_rounds=1,
            require_unanimous_convergence="yes",  # type: ignore[arg-type]
        )


def test_config_rejects_non_string_marker() -> None:
    with pytest.raises(DebateError):
        DebateConfig(
            topic="t",
            participants=_participants(),
            max_rounds=1,
            convergence_marker=123,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# build_debate_graph compile-shape
# ---------------------------------------------------------------------------


def test_build_debate_graph_returns_compiled_graph() -> None:
    saver = MagicMock()
    saver.put = MagicMock()
    saver.put_writes = MagicMock()
    saver.get_tuple = MagicMock(return_value=None)
    saver.list = MagicMock(return_value=iter([]))
    cfg = _config()
    graph = build_debate_graph(model=_ScriptedModel("ok"), saver=saver, config=cfg)
    assert graph is not None
    assert hasattr(graph, "invoke")


def test_build_debate_graph_rejects_non_config() -> None:
    saver = MagicMock()
    with pytest.raises(DebateError):
        build_debate_graph(
            model=_ScriptedModel("ok"),
            saver=saver,
            config="not-a-config",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# assemble_debate_graph gating + wiring
# ---------------------------------------------------------------------------


class _StubChatTransport:
    """Tiny ChatTransport-shaped stub (registry-driven path is wired
    but never actually invoked in these unit tests)."""

    def invoke(
        self,
        provider: AIProvider,
        messages: Sequence[BaseMessage],
        /,
        **kwargs: Any,
    ) -> str:
        return "stub"


def test_assemble_raises_when_flag_explicitly_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(FEATURE_FLAG_ENV_VAR, "false")
    transport = _StubChatTransport()
    ledger = _RecordingLedger()
    registry = _registry_with_one_provider()
    with pytest.raises(DebateGraphDisabledError):
        assemble_debate_graph(
            task=TaskClass.INDIRA_REASONING,
            provider_resolver=_resolver(registry, TaskClass.INDIRA_REASONING),
            transport=transport,
            ledger_append=ledger,
            config=_config(),
        )


def test_assemble_raises_with_explicit_disabled_flag() -> None:
    transport = _StubChatTransport()
    ledger = _RecordingLedger()
    registry = _registry_with_one_provider()
    disabled = DebateFeatureFlag(getter=lambda _n, _d: "false")
    with pytest.raises(DebateGraphDisabledError):
        assemble_debate_graph(
            task=TaskClass.INDIRA_REASONING,
            provider_resolver=_resolver(registry, TaskClass.INDIRA_REASONING),
            transport=transport,
            ledger_append=ledger,
            config=_config(),
            feature_flag=disabled,
        )


def test_assemble_returns_bundle_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_flag(monkeypatch)
    transport = _StubChatTransport()
    ledger = _RecordingLedger()
    registry = _registry_with_one_provider()
    bundle = assemble_debate_graph(
        task=TaskClass.INDIRA_REASONING,
        provider_resolver=_resolver(registry, TaskClass.INDIRA_REASONING),
        transport=transport,
        ledger_append=ledger,
        config=_config(),
    )
    assert isinstance(bundle, DebateGraphBundle)
    assert bundle.config == _config()
    assert hasattr(bundle.graph, "invoke")
    assert hasattr(bundle.saver, "put")
    assert hasattr(bundle.model, "invoke")


# ---------------------------------------------------------------------------
# End-to-end debate graph
# ---------------------------------------------------------------------------


def _build_bundle(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cfg: DebateConfig | None = None,
    model: ModelLike | None = None,
) -> tuple[DebateGraphBundle, _RecordingLedger, _ScriptedModel]:
    """Drop-in helper to build a bundle around an injected fake model
    rather than going through the registry path."""

    from intelligence_engine.cognitive.checkpointing import (
        AuditLedgerCheckpointSaver,
    )

    _enable_flag(monkeypatch)
    ledger = _RecordingLedger()
    saver = AuditLedgerCheckpointSaver(ledger_append=ledger)
    used_cfg = cfg or _config()
    used_model = model or _ScriptedModel("ok")
    graph = build_debate_graph(model=used_model, saver=saver, config=used_cfg)
    bundle = DebateGraphBundle(graph=graph, model=used_model, saver=saver, config=used_cfg)
    return bundle, ledger, used_model  # type: ignore[return-value]


def test_graph_runs_full_round_robin_then_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _ledger, model = _build_bundle(
        monkeypatch,
        cfg=_config(n=2, rounds=2, marker=""),  # disable convergence
    )
    outcome = run_debate(bundle, thread_id="t1")
    # 2 participants * 2 rounds = 4 debate turns + 1 moderator
    assert isinstance(outcome, DebateOutcome)
    assert outcome.n_turns == 4
    assert len(outcome.turns) == 4
    assert outcome.synthesis  # moderator produced text
    assert outcome.converged is False
    assert outcome.n_rounds == 2
    # 4 debate calls + 1 moderator call = 5
    assert len(model.calls) == 5


def test_graph_round_robin_order(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle, _ledger, _model = _build_bundle(
        monkeypatch,
        cfg=_config(n=3, rounds=2, marker=""),
    )
    outcome = run_debate(bundle, thread_id="t1")
    expected_order = ["p0", "p1", "p2", "p0", "p1", "p2"]
    actual = [t.participant_name for t in outcome.turns]
    assert actual == expected_order


def test_graph_assigns_round_and_speaker_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _ledger, _model = _build_bundle(
        monkeypatch,
        cfg=_config(n=2, rounds=3, marker=""),
    )
    outcome = run_debate(bundle, thread_id="t1")
    expected = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    ]
    actual = [(t.round_idx, t.speaker_idx) for t in outcome.turns]
    assert actual == expected


def test_graph_tags_each_turn_with_participant_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(n=2, rounds=1, marker="")
    bundle, _ledger, _model = _build_bundle(monkeypatch, cfg=cfg)
    config = {"configurable": {"thread_id": "t1"}}
    initial = {
        "topic": cfg.topic,
        "transcript": [],
        "round_idx": 0,
        "speaker_idx": 0,
        "n_participants": cfg.n_participants,
        "max_rounds": cfg.max_rounds,
        "converged": False,
        "synthesis": "",
    }
    final = bundle.graph.invoke(initial, config=config)
    tags = [msg.additional_kwargs.get(PARTICIPANT_KWARG) for msg in final["transcript"]]
    assert tags == ["p0", "p1", MODERATOR_NODE_ID]


def test_moderator_synthesis_is_tagged_moderator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _ledger, _model = _build_bundle(
        monkeypatch,
        cfg=_config(n=2, rounds=1, marker=""),
        model=_ScriptedModel(
            per_caller={"p0": "p0-text", "p1": "p1-text", MODERATOR_NODE_ID: "synth!"}
        ),
    )
    outcome = run_debate(bundle, thread_id="t1")
    assert outcome.synthesis == "synth!"


def test_graph_terminates_at_moderator_after_max_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _ledger, model = _build_bundle(
        monkeypatch,
        cfg=_config(n=2, rounds=1, marker=""),
    )
    outcome = run_debate(bundle, thread_id="t1")
    # 2 debate calls + 1 moderator
    assert len(model.calls) == 3
    assert outcome.n_rounds == 1
    assert outcome.n_turns == 2


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


def test_convergence_unanimous_stops_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _ledger, model = _build_bundle(
        monkeypatch,
        cfg=_config(n=2, rounds=4, marker="AGREED", unanimous=True),
        model=_ScriptedModel(["AGREED yes", "AGREED also"]),
    )
    outcome = run_debate(bundle, thread_id="t1")
    assert outcome.converged is True
    assert outcome.n_rounds == 1
    assert outcome.n_turns == 2
    # 2 debaters + 1 moderator
    assert len(model.calls) == 3


def test_convergence_unanimous_partial_does_not_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _ledger, model = _build_bundle(
        monkeypatch,
        cfg=_config(n=2, rounds=2, marker="AGREED", unanimous=True),
        model=_ScriptedModel(["AGREED", "I dissent", "AGREED 2", "AGREED 3"]),
    )
    outcome = run_debate(bundle, thread_id="t1")
    assert outcome.converged is True  # converges only at end of round 2
    assert outcome.n_rounds == 2
    assert outcome.n_turns == 4


def test_convergence_any_stops_on_partial_agreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _ledger, _model = _build_bundle(
        monkeypatch,
        cfg=_config(n=2, rounds=4, marker="AGREED", unanimous=False),
        model=_ScriptedModel(["AGREED", "I dissent"]),
    )
    outcome = run_debate(bundle, thread_id="t1")
    assert outcome.converged is True
    assert outcome.n_rounds == 1


def test_convergence_marker_is_case_insensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _ledger, _model = _build_bundle(
        monkeypatch,
        cfg=_config(n=2, rounds=4, marker="agreed", unanimous=True),
        model=_ScriptedModel(["AGREED!", "agreed."]),
    )
    outcome = run_debate(bundle, thread_id="t1")
    assert outcome.converged is True


def test_empty_marker_disables_convergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _ledger, _model = _build_bundle(
        monkeypatch,
        cfg=_config(n=2, rounds=2, marker=""),
        model=_ScriptedModel("AGREED"),  # would converge if marker were on
    )
    outcome = run_debate(bundle, thread_id="t1")
    assert outcome.converged is False
    assert outcome.n_rounds == 2


def test_no_agreement_runs_all_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _ledger, _model = _build_bundle(
        monkeypatch,
        cfg=_config(n=2, rounds=3, marker="AGREED", unanimous=True),
        model=_ScriptedModel("dissent only"),
    )
    outcome = run_debate(bundle, thread_id="t1")
    assert outcome.converged is False
    assert outcome.n_rounds == 3
    assert outcome.n_turns == 6


# ---------------------------------------------------------------------------
# AuditLedger checkpoint integration
# ---------------------------------------------------------------------------


def test_saver_receives_checkpoint_rows_during_debate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, ledger, _model = _build_bundle(
        monkeypatch,
        cfg=_config(n=2, rounds=1, marker=""),
    )
    run_debate(bundle, thread_id="t-saver")
    rows = [p for kind, p in ledger.rows if kind == CHECKPOINT_KIND]
    assert rows, "expected ≥1 COGNITIVE_CHECKPOINT row"
    assert all(p["thread_id"] == "t-saver" for p in rows)


def test_saver_isolation_per_thread_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, ledger, _model = _build_bundle(
        monkeypatch,
        cfg=_config(n=2, rounds=1, marker=""),
    )
    run_debate(bundle, thread_id="t-A")
    run_debate(bundle, thread_id="t-B")
    rows_a = [p for k, p in ledger.rows if k == CHECKPOINT_KIND and p["thread_id"] == "t-A"]
    rows_b = [p for k, p in ledger.rows if k == CHECKPOINT_KIND and p["thread_id"] == "t-B"]
    assert rows_a and rows_b


# ---------------------------------------------------------------------------
# extract_outcome — projection
# ---------------------------------------------------------------------------


def test_extract_outcome_with_explicit_synthesis_field() -> None:
    cfg = _config(n=2, rounds=1, marker="")
    state = {
        "topic": "explicit topic",
        "transcript": [
            AIMessage(content="hi", additional_kwargs={PARTICIPANT_KWARG: "p0"}),
            AIMessage(content="ho", additional_kwargs={PARTICIPANT_KWARG: "p1"}),
        ],
        "round_idx": 1,
        "speaker_idx": 0,
        "n_participants": 2,
        "max_rounds": 1,
        "converged": False,
        "synthesis": "fallback synth",
    }
    outcome = extract_outcome(state, cfg)
    assert outcome.topic == "explicit topic"
    assert outcome.synthesis == "fallback synth"
    assert outcome.n_turns == 2
    assert outcome.turns[0].participant_name == "p0"
    assert outcome.turns[1].participant_name == "p1"


def test_extract_outcome_prefers_moderator_message_over_synthesis_field() -> None:
    cfg = _config(n=2, rounds=1, marker="")
    state = {
        "topic": "topic",
        "transcript": [
            AIMessage(content="hi", additional_kwargs={PARTICIPANT_KWARG: "p0"}),
            AIMessage(content="ho", additional_kwargs={PARTICIPANT_KWARG: "p1"}),
            AIMessage(
                content="real synth",
                additional_kwargs={PARTICIPANT_KWARG: MODERATOR_NODE_ID},
            ),
        ],
        "round_idx": 1,
        "speaker_idx": 0,
        "n_participants": 2,
        "max_rounds": 1,
        "converged": False,
        "synthesis": "ignored",
    }
    outcome = extract_outcome(state, cfg)
    assert outcome.synthesis == "real synth"
    assert outcome.n_turns == 2  # moderator excluded


def test_extract_outcome_handles_empty_transcript() -> None:
    cfg = _config(n=2, rounds=1, marker="")
    state = {
        "topic": "topic",
        "transcript": [],
        "round_idx": 0,
        "speaker_idx": 0,
        "n_participants": 2,
        "max_rounds": 1,
        "converged": False,
        "synthesis": "",
    }
    outcome = extract_outcome(state, cfg)
    assert outcome.n_turns == 0
    assert outcome.synthesis == ""


def test_extract_outcome_falls_back_to_speaker_idx_when_tag_missing() -> None:
    cfg = _config(n=2, rounds=1, marker="")
    state = {
        "topic": "topic",
        "transcript": [
            AIMessage(content="a", additional_kwargs={}),
            AIMessage(content="b", additional_kwargs={}),
        ],
        "round_idx": 1,
        "speaker_idx": 0,
        "n_participants": 2,
        "max_rounds": 1,
        "converged": False,
        "synthesis": "",
    }
    outcome = extract_outcome(state, cfg)
    assert [t.participant_name for t in outcome.turns] == ["p0", "p1"]


# ---------------------------------------------------------------------------
# DebateOutcome / DebateTurn frozen contract
# ---------------------------------------------------------------------------


def test_debate_turn_is_frozen() -> None:
    t = DebateTurn(round_idx=0, speaker_idx=0, participant_name="p0", text="x")
    with pytest.raises((AttributeError, DebateError)):
        t.text = "y"  # type: ignore[misc]


def test_debate_outcome_is_frozen() -> None:
    o = DebateOutcome(
        topic="t",
        turns=(),
        synthesis="",
        converged=False,
        n_rounds=0,
        n_turns=0,
    )
    with pytest.raises((AttributeError, DebateError)):
        o.synthesis = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# run_debate — driver
# ---------------------------------------------------------------------------


def test_run_debate_rejects_blank_thread_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _ledger, _model = _build_bundle(monkeypatch)
    with pytest.raises(DebateError):
        run_debate(bundle, thread_id="")
    with pytest.raises(DebateError):
        run_debate(bundle, thread_id="   ")


def test_run_debate_rejects_non_string_thread_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _ledger, _model = _build_bundle(monkeypatch)
    with pytest.raises(DebateError):
        run_debate(bundle, thread_id=42)  # type: ignore[arg-type]


def test_run_debate_passes_topic_through_to_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(n=2, rounds=1, marker="")
    bundle, _ledger, _model = _build_bundle(monkeypatch, cfg=cfg)
    outcome = run_debate(bundle, thread_id="t1")
    assert outcome.topic == cfg.topic


def test_run_debate_synthesis_is_truncated_to_max(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    huge = "x" * (MAX_SYNTHESIS_LEN + 1024)
    bundle, _ledger, _model = _build_bundle(
        monkeypatch,
        cfg=_config(n=2, rounds=1, marker=""),
        model=_ScriptedModel(per_caller={MODERATOR_NODE_ID: huge, "p0": "x", "p1": "x"}),
    )
    outcome = run_debate(bundle, thread_id="t1")
    assert len(outcome.synthesis) == MAX_SYNTHESIS_LEN


# ---------------------------------------------------------------------------
# Prompt-shape sanity (debater + moderator)
# ---------------------------------------------------------------------------


def test_debater_prompt_includes_topic_persona_and_turn_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _ledger, model = _build_bundle(
        monkeypatch,
        cfg=_config(n=2, rounds=1, marker=""),
    )
    run_debate(bundle, thread_id="t1")
    first_call = model.calls[0]
    types = [type(m).__name__ for m in first_call]
    # First call: SystemMessage(topic), SystemMessage(persona), HumanMessage(turn)
    assert types[0] == "SystemMessage"
    assert types[1] == "SystemMessage"
    assert "Topic: should we ship?" in str(first_call[0].content)
    assert "persona-0" in str(first_call[1].content)
    assert isinstance(first_call[-1], HumanMessage)
    assert "As p0" in str(first_call[-1].content)


def test_moderator_prompt_includes_full_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _ledger, model = _build_bundle(
        monkeypatch,
        cfg=_config(n=2, rounds=1, marker=""),
        model=_ScriptedModel(
            per_caller={"p0": "p0-said", "p1": "p1-said", MODERATOR_NODE_ID: "synth"}
        ),
    )
    run_debate(bundle, thread_id="t1")
    moderator_call = model.calls[-1]
    contents = " ".join(str(m.content) for m in moderator_call)
    assert "p0-said" in contents
    assert "p1-said" in contents
    assert "Synthesize" in contents


# ---------------------------------------------------------------------------
# Tier discipline (AST scan of debate_graph.py)
# ---------------------------------------------------------------------------


_MODULE_PATH = Path(dg_mod.__file__)


def _imports(module_path: Path) -> list[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            out.append(node.module)
    return out


def test_module_has_adapted_from_langgraph_header() -> None:
    src = _MODULE_PATH.read_text(encoding="utf-8")
    assert "# ADAPTED FROM: langgraph/graph/state.py" in src


def test_module_does_not_import_governance_engine() -> None:
    for mod in _imports(_MODULE_PATH):
        assert not mod.startswith("governance_engine"), mod


def test_module_does_not_import_system_engine_runtime() -> None:
    # system_engine.scvs is fine in tests; the impl module must not import any
    # part of system_engine at all (B1 cross-engine isolation).
    for mod in _imports(_MODULE_PATH):
        assert not mod.startswith("system_engine"), mod


def test_module_does_not_import_execution_engine() -> None:
    for mod in _imports(_MODULE_PATH):
        assert not mod.startswith("execution_engine"), mod


def test_module_does_not_import_evolution_engine() -> None:
    for mod in _imports(_MODULE_PATH):
        assert not mod.startswith("evolution_engine"), mod


def test_module_does_not_import_langsmith_or_sqlite_saver() -> None:
    for mod in _imports(_MODULE_PATH):
        assert not mod.startswith("langsmith"), mod
        assert mod != "langgraph.checkpoint.sqlite", mod


def test_module_does_not_reference_tool_node() -> None:
    """Tool calls are out of scope; any future use must route through
    the DIX governance approval gate."""
    src = _MODULE_PATH.read_text(encoding="utf-8")
    code = src.split('"""', 2)[-1]  # strip docstring
    assert "ToolNode" not in code


def test_module_does_not_construct_typed_bus_events() -> None:
    """INV-67 / INV-71 / B27 / B28: the cognitive surface is advisory.

    Promotion to typed events must happen elsewhere through a
    governance-gated path. We assert via AST — call expressions, not
    text — so docstring mentions are fine."""

    forbidden = {"PatchProposal", "SignalEvent", "GovernanceDecision"}
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in forbidden:
                raise AssertionError(f"forbidden constructor: {func.id}")
            if isinstance(func, ast.Attribute) and func.attr in forbidden:
                raise AssertionError(f"forbidden constructor: {func.attr}")


def test_module_does_not_import_random_or_clock() -> None:
    src = _MODULE_PATH.read_text(encoding="utf-8")
    code = src.split('"""', 2)[-1]
    forbidden = (
        "import random",
        "from random",
        "import time",
        "from time ",
        "import datetime",
        "from datetime",
    )
    for token in forbidden:
        assert token not in code, token


def test_module_imports_state_graph_primitives() -> None:
    """RUNTIME_SAFE chat surface is allowed to use ``StateGraph`` /
    ``add_node`` / ``add_edge`` / ``add_conditional_edges`` (B24 +
    INV-67). Pin that the impl actually uses these primitives so a
    later refactor does not silently drop the LangGraph dependency."""

    src = _MODULE_PATH.read_text(encoding="utf-8")
    assert "StateGraph" in src
    assert "add_node" in src
    assert "add_edge" in src
    assert "add_conditional_edges" in src


# ---------------------------------------------------------------------------
# ModelLike protocol smoke
# ---------------------------------------------------------------------------


def test_model_like_protocol_smoke() -> None:
    assert isinstance(_ScriptedModel("ok"), ModelLike)


def test_model_like_rejects_object_without_invoke() -> None:
    class _NoInvoke:
        pass

    assert not isinstance(_NoInvoke(), ModelLike)
