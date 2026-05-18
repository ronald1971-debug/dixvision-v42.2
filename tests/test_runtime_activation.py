"""Tests for ``tools/runtime_activation.py`` (PR-RT-2).

These tests pin the **active** half of the runtime topology authority
chain. They cover:

* module identity (constants, exports, no banned top-level imports)
* :class:`RegisteredTransition` value object validation
* :class:`ActivationSnapshot` value object validation
* :class:`RuntimeActivationRegistry` registration semantics
* lifecycle FSM enforcement + RUNTIME_ACTIVATION_VIOLATION audit rows
* audit-sink contract (synchronous, exception-safe)
* topology-aware registration (rejects unknown node_ids)
* :class:`ActivationSnapshot` digest determinism (INV-15 three-run)
* :func:`replay_transitions` rebuilds equivalent registry from history
* AST guards forbidding top-level vendor / network / engine imports
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path

import pytest

from tools.runtime_activation import (
    ACTIVATION_VERSION,
    AUDIT_KIND_TRANSITION,
    AUDIT_KIND_VIOLATION,
    MAX_NODE_ID_LEN,
    MAX_REASON_LEN,
    NEW_PIP_DEPENDENCIES,
    ActivationError,
    ActivationSnapshot,
    ActivationViolation,
    RegisteredTransition,
    RuntimeActivationRegistry,
    build_registry,
    replay_transitions,
)
from tools.runtime_topology import (
    EdgeRelation,
    LifecycleState,
    NodeKind,
    NodeTier,
    RuntimeEdge,
    RuntimeNode,
    RuntimeTopology,
)

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "runtime_activation.py"


# ---------------------------------------------------------------------------
# Audit sink helper
# ---------------------------------------------------------------------------


class _RecordingSink:
    """Audit sink that captures every (kind, payload) pair."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def __call__(self, kind: str, payload: Mapping[str, object]) -> None:
        self.events.append((kind, dict(payload)))


# ---------------------------------------------------------------------------
# Module identity
# ---------------------------------------------------------------------------


def test_activation_version_constant() -> None:
    assert ACTIVATION_VERSION == "v1.0-RT2"


def test_new_pip_dependencies_is_empty() -> None:
    assert NEW_PIP_DEPENDENCIES == ()


def test_audit_kind_constants() -> None:
    assert AUDIT_KIND_TRANSITION == "RUNTIME_ACTIVATION_TRANSITION"
    assert AUDIT_KIND_VIOLATION == "RUNTIME_ACTIVATION_VIOLATION"


def test_caps() -> None:
    assert MAX_NODE_ID_LEN == 128
    assert MAX_REASON_LEN == 512


# ---------------------------------------------------------------------------
# RegisteredTransition
# ---------------------------------------------------------------------------


def test_registered_transition_is_frozen() -> None:
    record = RegisteredTransition(
        node_id="a",
        src=LifecycleState.DECLARED,
        dst=LifecycleState.WIRED,
        reason="boot",
        version_after=1,
    )
    with pytest.raises((AttributeError, TypeError)):
        record.reason = "other"  # type: ignore[misc]


def test_registered_transition_is_slotted() -> None:
    record = RegisteredTransition(
        node_id="a",
        src=LifecycleState.DECLARED,
        dst=LifecycleState.WIRED,
        reason="boot",
        version_after=1,
    )
    assert not hasattr(record, "__dict__")


def test_registered_transition_rejects_empty_node_id() -> None:
    with pytest.raises(ActivationError):
        RegisteredTransition(
            node_id="",
            src=LifecycleState.DECLARED,
            dst=LifecycleState.WIRED,
            reason="boot",
            version_after=1,
        )


def test_registered_transition_rejects_oversize_reason() -> None:
    with pytest.raises(ActivationError):
        RegisteredTransition(
            node_id="a",
            src=LifecycleState.DECLARED,
            dst=LifecycleState.WIRED,
            reason="x" * (MAX_REASON_LEN + 1),
            version_after=1,
        )


def test_registered_transition_rejects_zero_version() -> None:
    with pytest.raises(ActivationError):
        RegisteredTransition(
            node_id="a",
            src=LifecycleState.DECLARED,
            dst=LifecycleState.WIRED,
            reason="boot",
            version_after=0,
        )


def test_registered_transition_rejects_non_enum_states() -> None:
    with pytest.raises(ActivationError):
        RegisteredTransition(
            node_id="a",
            src="DECLARED",  # type: ignore[arg-type]
            dst=LifecycleState.WIRED,
            reason="boot",
            version_after=1,
        )


def test_registered_transition_canonical() -> None:
    record = RegisteredTransition(
        node_id="a",
        src=LifecycleState.DECLARED,
        dst=LifecycleState.WIRED,
        reason="boot",
        version_after=1,
    )
    canonical = record.canonical()
    assert canonical == {
        "dst": "WIRED",
        "node_id": "a",
        "reason": "boot",
        "src": "DECLARED",
        "version_after": 1,
    }


# ---------------------------------------------------------------------------
# ActivationSnapshot
# ---------------------------------------------------------------------------


def test_activation_snapshot_is_frozen() -> None:
    snapshot = ActivationSnapshot(states=(), transitions=(), version=0)
    with pytest.raises((AttributeError, TypeError)):
        snapshot.version = 1  # type: ignore[misc]


def test_activation_snapshot_rejects_negative_version() -> None:
    with pytest.raises(ActivationError):
        ActivationSnapshot(states=(), transitions=(), version=-1)


def test_activation_snapshot_rejects_duplicate_node_id() -> None:
    with pytest.raises(ActivationError):
        ActivationSnapshot(
            states=(
                ("a", LifecycleState.DECLARED),
                ("a", LifecycleState.WIRED),
            ),
            transitions=(),
            version=2,
        )


def test_activation_snapshot_sorts_states() -> None:
    snapshot = ActivationSnapshot(
        states=(
            ("z", LifecycleState.DECLARED),
            ("a", LifecycleState.WIRED),
        ),
        transitions=(),
        version=2,
    )
    assert [node_id for node_id, _ in snapshot.states] == ["a", "z"]


def test_activation_snapshot_active_dormant_wired_buckets() -> None:
    snapshot = ActivationSnapshot(
        states=(
            ("declared_one", LifecycleState.DECLARED),
            ("dormant_one", LifecycleState.DORMANT),
            ("wired_one", LifecycleState.WIRED),
            ("started_one", LifecycleState.STARTED),
            ("healthy_one", LifecycleState.HEALTHY),
            ("degraded_one", LifecycleState.DEGRADED),
            ("stopped_one", LifecycleState.STOPPED),
        ),
        transitions=(),
        version=7,
    )
    assert snapshot.active_node_ids() == {
        "started_one",
        "healthy_one",
        "degraded_one",
    }
    assert snapshot.dormant_node_ids() == {
        "declared_one",
        "dormant_one",
        "stopped_one",
    }
    assert snapshot.wired_node_ids() == {"wired_one"}


def test_activation_snapshot_state_of() -> None:
    snapshot = ActivationSnapshot(
        states=(("a", LifecycleState.STARTED),),
        transitions=(),
        version=1,
    )
    assert snapshot.state_of("a") is LifecycleState.STARTED
    assert snapshot.state_of("missing") is None


def test_activation_snapshot_counts() -> None:
    snapshot = ActivationSnapshot(
        states=(
            ("a", LifecycleState.STARTED),
            ("b", LifecycleState.HEALTHY),
        ),
        transitions=(
            RegisteredTransition(
                node_id="a",
                src=LifecycleState.DECLARED,
                dst=LifecycleState.DECLARED,
                reason="r",
                version_after=1,
            ),
        ),
        version=4,
    )
    assert snapshot.node_count() == 2
    assert snapshot.transition_count() == 1


# ---------------------------------------------------------------------------
# Registry construction
# ---------------------------------------------------------------------------


def test_registry_constructs_with_defaults() -> None:
    registry = RuntimeActivationRegistry()
    assert registry.current_version() == 0
    assert registry.registered_node_ids() == frozenset()


def test_registry_rejects_non_callable_sink() -> None:
    with pytest.raises(ActivationError):
        RuntimeActivationRegistry(audit_sink=42)  # type: ignore[arg-type]


def test_registry_rejects_non_topology() -> None:
    with pytest.raises(ActivationError):
        RuntimeActivationRegistry(topology="topology")  # type: ignore[arg-type]


def test_build_registry_returns_registry() -> None:
    sink = _RecordingSink()
    registry = build_registry(audit_sink=sink)
    assert isinstance(registry, RuntimeActivationRegistry)


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


def test_register_creates_declared_node() -> None:
    sink = _RecordingSink()
    registry = build_registry(audit_sink=sink)
    registry.register("execution_engine")
    assert registry.state_of("execution_engine") is LifecycleState.DECLARED
    assert registry.current_version() == 1
    assert sink.events[-1][0] == AUDIT_KIND_TRANSITION


def test_register_accepts_alternative_initial_state() -> None:
    registry = build_registry()
    registry.register("execution_engine.gate", initial_state=LifecycleState.WIRED)
    assert registry.state_of("execution_engine.gate") is LifecycleState.WIRED


def test_register_is_idempotent_for_same_state() -> None:
    sink = _RecordingSink()
    registry = build_registry(audit_sink=sink)
    registry.register("a")
    registry.register("a")
    assert registry.current_version() == 1
    transition_events = [e for e in sink.events if e[0] == AUDIT_KIND_TRANSITION]
    assert len(transition_events) == 1


def test_register_with_different_state_raises_violation() -> None:
    sink = _RecordingSink()
    registry = build_registry(audit_sink=sink)
    registry.register("a", initial_state=LifecycleState.DECLARED)
    with pytest.raises(ActivationViolation):
        registry.register("a", initial_state=LifecycleState.WIRED)
    violations = [e for e in sink.events if e[0] == AUDIT_KIND_VIOLATION]
    assert len(violations) == 1
    assert violations[0][1]["violation_kind"] == "register_state_mismatch"


def test_register_rejects_non_string_node_id() -> None:
    registry = build_registry()
    with pytest.raises(ActivationError):
        registry.register(42)  # type: ignore[arg-type]


def test_register_rejects_empty_node_id() -> None:
    registry = build_registry()
    with pytest.raises(ActivationError):
        registry.register("")


def test_register_rejects_non_enum_state() -> None:
    registry = build_registry()
    with pytest.raises(ActivationError):
        registry.register("a", initial_state="DECLARED")  # type: ignore[arg-type]


def test_register_rejects_oversize_reason() -> None:
    registry = build_registry()
    with pytest.raises(ActivationError):
        registry.register("a", reason="x" * (MAX_REASON_LEN + 1))


def test_register_rejects_unknown_node_when_topology_attached() -> None:
    node = RuntimeNode(
        node_id="known",
        kind=NodeKind.GATE,
        tier=NodeTier.T0,
        declared_version="v1",
        capabilities=frozenset({"x"}),
    )
    topology = RuntimeTopology(nodes=(node,), edges=())
    registry = build_registry(topology=topology)
    with pytest.raises(ActivationError):
        registry.register("unknown")


def test_register_accepts_declared_node_when_topology_attached() -> None:
    node = RuntimeNode(
        node_id="known",
        kind=NodeKind.GATE,
        tier=NodeTier.T0,
        declared_version="v1",
        capabilities=frozenset({"x"}),
    )
    topology = RuntimeTopology(nodes=(node,), edges=())
    registry = build_registry(topology=topology)
    registry.register("known")
    assert registry.state_of("known") is LifecycleState.DECLARED


# ---------------------------------------------------------------------------
# transition
# ---------------------------------------------------------------------------


def test_transition_advances_state() -> None:
    sink = _RecordingSink()
    registry = build_registry(audit_sink=sink)
    registry.register("a")
    registry.transition("a", LifecycleState.WIRED, reason="boot")
    assert registry.state_of("a") is LifecycleState.WIRED
    transitions = [e for e in sink.events if e[0] == AUDIT_KIND_TRANSITION]
    assert len(transitions) == 2


def test_transition_walks_full_lifecycle() -> None:
    registry = build_registry()
    registry.register("a")
    registry.transition("a", LifecycleState.WIRED)
    registry.transition("a", LifecycleState.STARTED)
    registry.transition("a", LifecycleState.HEALTHY)
    registry.transition("a", LifecycleState.DEGRADED)
    registry.transition("a", LifecycleState.HEALTHY)
    registry.transition("a", LifecycleState.STOPPED)
    assert registry.state_of("a") is LifecycleState.STOPPED


def test_transition_rejects_illegal_jump() -> None:
    sink = _RecordingSink()
    registry = build_registry(audit_sink=sink)
    registry.register("a")
    with pytest.raises(ActivationViolation):
        registry.transition("a", LifecycleState.HEALTHY)
    violations = [e for e in sink.events if e[0] == AUDIT_KIND_VIOLATION]
    assert len(violations) == 1
    assert violations[0][1]["violation_kind"] == "illegal_transition"


def test_transition_rejects_unregistered_node() -> None:
    sink = _RecordingSink()
    registry = build_registry(audit_sink=sink)
    with pytest.raises(ActivationViolation):
        registry.transition("ghost", LifecycleState.WIRED)
    violations = [e for e in sink.events if e[0] == AUDIT_KIND_VIOLATION]
    assert len(violations) == 1
    assert violations[0][1]["violation_kind"] == "transition_unregistered"


def test_transition_rejects_non_enum_state() -> None:
    registry = build_registry()
    registry.register("a")
    with pytest.raises(ActivationError):
        registry.transition("a", "WIRED")  # type: ignore[arg-type]


def test_transition_rejects_oversize_reason() -> None:
    registry = build_registry()
    registry.register("a")
    with pytest.raises(ActivationError):
        registry.transition("a", LifecycleState.WIRED, reason="x" * (MAX_REASON_LEN + 1))


def test_dormant_to_wired_is_legal() -> None:
    registry = build_registry()
    registry.register("a", initial_state=LifecycleState.DECLARED)
    registry.transition("a", LifecycleState.DORMANT, reason="paused")
    registry.transition("a", LifecycleState.WIRED, reason="resumed")
    assert registry.state_of("a") is LifecycleState.WIRED


def test_dormant_to_healthy_is_illegal() -> None:
    registry = build_registry()
    registry.register("a", initial_state=LifecycleState.DECLARED)
    registry.transition("a", LifecycleState.DORMANT)
    with pytest.raises(ActivationViolation):
        registry.transition("a", LifecycleState.HEALTHY)


def test_stopped_to_wired_is_legal() -> None:
    registry = build_registry()
    registry.register("a")
    registry.transition("a", LifecycleState.WIRED)
    registry.transition("a", LifecycleState.STARTED)
    registry.transition("a", LifecycleState.STOPPED)
    registry.transition("a", LifecycleState.WIRED, reason="restart")
    assert registry.state_of("a") is LifecycleState.WIRED


# ---------------------------------------------------------------------------
# Audit sink contract
# ---------------------------------------------------------------------------


def test_audit_sink_receives_full_envelope() -> None:
    sink = _RecordingSink()
    registry = build_registry(audit_sink=sink)
    registry.register("a", reason="boot")
    registry.transition("a", LifecycleState.WIRED, reason="started")
    assert len(sink.events) == 2
    assert sink.events[-1][0] == AUDIT_KIND_TRANSITION
    assert sink.events[-1][1]["dst"] == "WIRED"
    assert sink.events[-1][1]["src"] == "DECLARED"
    assert sink.events[-1][1]["reason"] == "started"


def test_audit_sink_failure_does_not_block_transitions() -> None:
    def angry_sink(_kind: str, _payload: Mapping[str, object]) -> None:
        raise RuntimeError("audit sink down")

    registry = build_registry(audit_sink=angry_sink)
    registry.register("a")
    registry.transition("a", LifecycleState.WIRED)
    assert registry.state_of("a") is LifecycleState.WIRED


# ---------------------------------------------------------------------------
# Read surface
# ---------------------------------------------------------------------------


def test_snapshot_returns_frozen_projection() -> None:
    registry = build_registry()
    registry.register("a")
    registry.register("b", initial_state=LifecycleState.WIRED)
    snapshot = registry.snapshot()
    assert isinstance(snapshot, ActivationSnapshot)
    assert snapshot.state_of("a") is LifecycleState.DECLARED
    assert snapshot.state_of("b") is LifecycleState.WIRED


def test_active_node_ids_excludes_declared_and_dormant() -> None:
    registry = build_registry()
    registry.register("a")
    registry.register("b")
    registry.transition("b", LifecycleState.WIRED)
    registry.transition("b", LifecycleState.STARTED)
    registry.transition("b", LifecycleState.HEALTHY)
    assert registry.active_node_ids() == {"b"}
    assert registry.dormant_node_ids() == {"a"}


def test_transition_history_appends_in_order() -> None:
    registry = build_registry()
    registry.register("a")
    registry.transition("a", LifecycleState.WIRED, reason="boot")
    registry.transition("a", LifecycleState.STARTED, reason="run")
    history = registry.transition_history()
    assert [r.dst for r in history] == [
        LifecycleState.DECLARED,
        LifecycleState.WIRED,
        LifecycleState.STARTED,
    ]


def test_state_of_unregistered_returns_none() -> None:
    registry = build_registry()
    assert registry.state_of("ghost") is None


def test_current_version_monotone() -> None:
    registry = build_registry()
    registry.register("a")
    v1 = registry.current_version()
    registry.transition("a", LifecycleState.WIRED)
    v2 = registry.current_version()
    assert v2 > v1


# ---------------------------------------------------------------------------
# INV-15 determinism
# ---------------------------------------------------------------------------


def _scripted_registry() -> RuntimeActivationRegistry:
    registry = build_registry()
    registry.register("execution_engine.gate", reason="boot")
    registry.register("intelligence_engine.closed_learning_loop", reason="boot")
    registry.transition(
        "execution_engine.gate",
        LifecycleState.WIRED,
        reason="state.build_execution",
    )
    registry.transition(
        "intelligence_engine.closed_learning_loop",
        LifecycleState.WIRED,
        reason="state.build_intelligence",
    )
    registry.transition(
        "execution_engine.gate",
        LifecycleState.STARTED,
        reason="harness.start",
    )
    registry.transition(
        "intelligence_engine.closed_learning_loop",
        LifecycleState.STARTED,
        reason="harness.start",
    )
    registry.transition(
        "execution_engine.gate",
        LifecycleState.HEALTHY,
        reason="first.tick",
    )
    registry.transition(
        "intelligence_engine.closed_learning_loop",
        LifecycleState.HEALTHY,
        reason="first.tick",
    )
    return registry


def test_inv_15_three_runs_produce_identical_snapshot_digest() -> None:
    d1 = _scripted_registry().snapshot().digest()
    d2 = _scripted_registry().snapshot().digest()
    d3 = _scripted_registry().snapshot().digest()
    assert d1 == d2 == d3


def test_inv_15_snapshot_digest_is_blake2b_128_hex() -> None:
    snapshot = _scripted_registry().snapshot()
    digest = snapshot.digest()
    assert len(digest) == 32
    int(digest, 16)


def test_inv_15_digest_changes_on_extra_transition() -> None:
    base = _scripted_registry().snapshot().digest()
    extra = _scripted_registry()
    extra.transition(
        "execution_engine.gate",
        LifecycleState.DEGRADED,
        reason="hazard.observed",
    )
    assert base != extra.snapshot().digest()


# ---------------------------------------------------------------------------
# replay_transitions
# ---------------------------------------------------------------------------


def test_replay_rebuilds_equivalent_registry() -> None:
    original = _scripted_registry()
    history = original.transition_history()
    replayed = replay_transitions(history)
    assert original.snapshot().digest() == replayed.snapshot().digest()


def test_replay_rejects_non_sequence() -> None:
    with pytest.raises(ActivationError):
        replay_transitions("not a sequence")  # type: ignore[arg-type]


def test_replay_rejects_wrong_record_type() -> None:
    with pytest.raises(ActivationError):
        replay_transitions(["not a RegisteredTransition"])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Topology-aware registration
# ---------------------------------------------------------------------------


def _toy_topology() -> RuntimeTopology:
    a = RuntimeNode(
        node_id="execution_engine",
        kind=NodeKind.ENGINE,
        tier=NodeTier.T0,
        declared_version="v1",
        capabilities=frozenset({"execution.dispatch"}),
    )
    b = RuntimeNode(
        node_id="execution_engine.gate",
        kind=NodeKind.GATE,
        tier=NodeTier.T0,
        declared_version="v1",
        capabilities=frozenset({"execution.gate"}),
    )
    return RuntimeTopology(
        nodes=(a, b),
        edges=(
            RuntimeEdge(
                source_id="execution_engine.gate",
                target_id="execution_engine",
                relation=EdgeRelation.GATES,
            ),
        ),
    )


def test_topology_attached_registry_round_trip() -> None:
    topology = _toy_topology()
    registry = build_registry(topology=topology)
    registry.register("execution_engine")
    registry.register("execution_engine.gate")
    registry.transition("execution_engine.gate", LifecycleState.WIRED)
    registry.transition("execution_engine.gate", LifecycleState.STARTED)
    assert registry.state_of("execution_engine.gate") is LifecycleState.STARTED
    assert registry.state_of("execution_engine") is LifecycleState.DECLARED


def test_topology_attached_registry_rejects_undeclared() -> None:
    topology = _toy_topology()
    registry = build_registry(topology=topology)
    with pytest.raises(ActivationError):
        registry.register("ghost.engine")


# ---------------------------------------------------------------------------
# AST guards
# ---------------------------------------------------------------------------


_BANNED_TOP_LEVEL_MODULES: frozenset[str] = frozenset(
    {
        "subprocess",
        "asyncio",
        "socket",
        "random",
        "time",
        "numpy",
        "torch",
        "requests",
        "core_engine",
        "execution_engine",
        "governance_engine",
        "intelligence_engine",
        "system_engine",
        "ui",
    }
)


def _top_level_imports(source: str) -> list[str]:
    tree = ast.parse(source)
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                out.append(node.module)
    return out


def test_no_banned_top_level_imports() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    imports = _top_level_imports(source)
    for imp in imports:
        root = imp.split(".")[0]
        assert root not in _BANNED_TOP_LEVEL_MODULES, (
            f"banned top-level import {imp!r} (root {root!r}) found in tools/runtime_activation.py"
        )


def test_only_allowed_tools_imports() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    imports = _top_level_imports(source)
    tools_imports = [imp for imp in imports if imp.startswith("tools.")]
    assert tools_imports == ["tools.runtime_topology"]
