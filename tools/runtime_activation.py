# Canonical DIX VISION runtime activation registry — OFFLINE_ONLY
# (``tools/`` tier).
#
# NEW_PIP_DEPENDENCIES = ()
#
# Authority constraints (pinned by ``tests/test_runtime_activation.py``):
#
#   * B1   — never imports from any runtime engine tier.
#   * INV-15 — :class:`RuntimeActivationRegistry` is a pure
#              function of the sequence of ``register`` / ``transition``
#              calls: three independent runs with the same call
#              sequence produce byte-identical
#              :class:`ActivationSnapshot` digests.
#   * No top-level imports of :mod:`subprocess`, :mod:`time`,
#     :mod:`random`, :mod:`asyncio`, :mod:`socket`, :mod:`numpy`,
#     :mod:`torch`, :mod:`requests`.
"""Canonical runtime activation registry (PR-RT-2).

This is the **active** half of the runtime topology authority chain.
PR-RT-1 introduced the declared topology contracts
(:class:`RuntimeNode` / :class:`RuntimeEdge` / :class:`RuntimeTopology`
in ``tools/runtime_topology.py``). This PR introduces the registry
that pins which of those declared nodes are *actually-running* on
the hot path versus declared-but-dormant.

The registry exists so the rest of the system can answer "what is
ACTUALLY active right now?" deterministically. Every engine, loop,
bus, sensor, adapter, and route is expected to register itself at
boot and transition through the lifecycle FSM defined in
``tools.runtime_topology``:

.. code-block::

    DECLARED -> WIRED   -> STARTED -> HEALTHY -> DEGRADED -> STOPPED
                                                          -> STOPPED

Any transition not present in :data:`tools.runtime_topology._LEGAL_TRANSITIONS`
emits a typed :class:`ActivationViolation` and (if an audit sink is
configured) a ``RUNTIME_ACTIVATION_VIOLATION`` audit row. The registry
never silently accepts an illegal transition.

Determinism contract (INV-15):

* :meth:`RuntimeActivationRegistry.snapshot` produces a frozen
  :class:`ActivationSnapshot` value object whose digest is a
  BLAKE2b-128 hex over the canonical sorted-key JSON serialization.
  Two registries that received the same sequence of calls always
  produce byte-identical digests.
* The registry's monotone version counter is the only mutable state
  in the system; it is incremented on every successful transition and
  reset only when the registry is reconstructed from scratch.
* No global mutable state; no clocks; no PRNG; no file-system reads.

Audit sink contract:

* The registry accepts an optional ``audit_sink`` callable
  ``(kind: str, payload: Mapping[str, object]) -> None``.
* On every successful transition it emits
  ``RUNTIME_ACTIVATION_TRANSITION`` with the source/destination
  states, the ``reason`` string, and the post-transition monotone
  version.
* On every illegal transition or unknown-node transition attempt it
  emits ``RUNTIME_ACTIVATION_VIOLATION`` with the same envelope plus
  a ``violation_kind`` slot.
* The sink is invoked synchronously; it MUST NOT raise. If it raises,
  the registry catches the exception and emits a follow-up
  ``RUNTIME_ACTIVATION_AUDIT_SINK_FAILURE`` row to a fallback no-op
  sink so the primary transition flow is never blocked.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from tools.runtime_topology import (
    LifecycleState,
    RuntimeTopology,
    is_legal_transition,
)

ACTIVATION_VERSION: Final[str] = "v1.0-RT2"
NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ()

MAX_NODE_ID_LEN: Final[int] = 128
MAX_REASON_LEN: Final[int] = 512
MAX_REGISTERED_NODES: Final[int] = 4096
MAX_TRANSITION_HISTORY: Final[int] = 65_536

AUDIT_KIND_TRANSITION: Final[str] = "RUNTIME_ACTIVATION_TRANSITION"
AUDIT_KIND_VIOLATION: Final[str] = "RUNTIME_ACTIVATION_VIOLATION"
AUDIT_KIND_SINK_FAILURE: Final[str] = "RUNTIME_ACTIVATION_AUDIT_SINK_FAILURE"


class ActivationError(ValueError):
    """Raised when the registry input is malformed.

    Distinct from :class:`ActivationViolation`, which represents a
    *runtime* lifecycle FSM breach rather than a malformed call.
    """


class ActivationViolation(RuntimeError):
    """Raised when a runtime lifecycle FSM rule is broken.

    Examples:

    * transitioning a node that was never registered
    * transitioning from a state that does not allow the destination
    * registering a node a second time with a different initial state

    The registry always emits a ``RUNTIME_ACTIVATION_VIOLATION``
    audit row before raising, so the violation is recorded even when
    a caller catches the exception.
    """


AuditSink = Callable[[str, Mapping[str, object]], None]


def _noop_audit_sink(_kind: str, _payload: Mapping[str, object]) -> None:
    """Drop-in sink used when no audit sink is configured."""


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RegisteredTransition:
    """One immutable lifecycle transition record.

    The registry retains every transition in a frozen tuple; the
    history is the canonical audit trail that
    :class:`ActivationSnapshot` digests over.
    """

    node_id: str
    src: LifecycleState
    dst: LifecycleState
    reason: str
    version_after: int

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ActivationError("node_id must be a non-empty str")
        if len(self.node_id) > MAX_NODE_ID_LEN:
            raise ActivationError(
                f"node_id length {len(self.node_id)} exceeds MAX_NODE_ID_LEN={MAX_NODE_ID_LEN}"
            )
        if not isinstance(self.src, LifecycleState):
            raise ActivationError(f"src must be a LifecycleState, got {type(self.src).__name__}")
        if not isinstance(self.dst, LifecycleState):
            raise ActivationError(f"dst must be a LifecycleState, got {type(self.dst).__name__}")
        if not isinstance(self.reason, str):
            raise ActivationError(f"reason must be a str, got {type(self.reason).__name__}")
        if len(self.reason) > MAX_REASON_LEN:
            raise ActivationError(
                f"reason length {len(self.reason)} exceeds MAX_REASON_LEN={MAX_REASON_LEN}"
            )
        if not isinstance(self.version_after, int):
            raise ActivationError(
                f"version_after must be an int, got {type(self.version_after).__name__}"
            )
        if self.version_after < 1:
            raise ActivationError(f"version_after must be >= 1, got {self.version_after}")

    def canonical(self) -> dict[str, object]:
        return {
            "dst": self.dst.value,
            "node_id": self.node_id,
            "reason": self.reason,
            "src": self.src.value,
            "version_after": self.version_after,
        }


@dataclass(frozen=True, slots=True)
class ActivationSnapshot:
    """A frozen point-in-time projection of the activation registry.

    Two registries that received the same sequence of calls always
    produce byte-identical snapshots and digests; this is the
    INV-15 anchor PR-RT-3 depends on when resolving capability
    queries.
    """

    states: tuple[tuple[str, LifecycleState], ...]
    transitions: tuple[RegisteredTransition, ...]
    version: int

    def __post_init__(self) -> None:
        if not isinstance(self.states, tuple):
            raise ActivationError(f"states must be a tuple, got {type(self.states).__name__}")
        if not isinstance(self.transitions, tuple):
            raise ActivationError(
                f"transitions must be a tuple, got {type(self.transitions).__name__}"
            )
        if not isinstance(self.version, int):
            raise ActivationError(f"version must be an int, got {type(self.version).__name__}")
        if self.version < 0:
            raise ActivationError(f"version must be >= 0, got {self.version}")
        seen_ids: set[str] = set()
        for entry in self.states:
            if not (
                isinstance(entry, tuple)
                and len(entry) == 2
                and isinstance(entry[0], str)
                and isinstance(entry[1], LifecycleState)
            ):
                raise ActivationError(
                    f"every states entry must be a (str, LifecycleState) tuple, got {entry!r}"
                )
            if entry[0] in seen_ids:
                raise ActivationError(f"duplicate node_id {entry[0]!r} in states")
            seen_ids.add(entry[0])

        sorted_states = tuple(sorted(self.states, key=lambda kv: kv[0]))
        if sorted_states != self.states:
            object.__setattr__(self, "states", sorted_states)

    def canonical(self) -> dict[str, object]:
        return {
            "states": [
                {"node_id": node_id, "state": state.value} for node_id, state in self.states
            ],
            "transitions": [t.canonical() for t in self.transitions],
            "version": self.version,
        }

    def digest(self) -> str:
        """Return a BLAKE2b-128 hex digest over the canonical
        sorted-key JSON serialization."""

        payload = json.dumps(
            self.canonical(),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=16).hexdigest()

    def state_of(self, node_id: str) -> LifecycleState | None:
        """Return the lifecycle state of ``node_id`` or ``None``."""

        for nid, state in self.states:
            if nid == node_id:
                return state
        return None

    def active_node_ids(self) -> frozenset[str]:
        """Return node_ids in STARTED / HEALTHY / DEGRADED states.

        These are the nodes that are actually-running on the hot path;
        PR-RT-3's capability map filters declared providers against
        this set when resolving "who provides X".
        """

        active = frozenset(
            {
                LifecycleState.STARTED,
                LifecycleState.HEALTHY,
                LifecycleState.DEGRADED,
            }
        )
        return frozenset(node_id for node_id, state in self.states if state in active)

    def dormant_node_ids(self) -> frozenset[str]:
        """Return node_ids in DECLARED / DORMANT / STOPPED states.

        These are the silent-drift candidates PR-RT-5's total-validation
        invariant inspects; every node here must either advance through
        WIRED or be admitted by an explicit DECLARED_BUT_DORMANT ledger
        row.
        """

        dormant = frozenset(
            {
                LifecycleState.DECLARED,
                LifecycleState.DORMANT,
                LifecycleState.STOPPED,
            }
        )
        return frozenset(node_id for node_id, state in self.states if state in dormant)

    def wired_node_ids(self) -> frozenset[str]:
        """Return node_ids in the WIRED state — declared and
        connected but not yet started."""

        return frozenset(node_id for node_id, state in self.states if state is LifecycleState.WIRED)

    def node_count(self) -> int:
        return len(self.states)

    def transition_count(self) -> int:
        return len(self.transitions)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class RuntimeActivationRegistry:
    """The mutable runtime activation registry.

    This is the only mutable surface in the runtime topology authority
    chain. It accumulates registered nodes and lifecycle transitions
    in an append-only fashion; the immutable :class:`ActivationSnapshot`
    projection is the read surface every other component consumes.

    The registry can be constructed against an optional
    :class:`RuntimeTopology` from PR-RT-1; when configured this way,
    :meth:`register` rejects any ``node_id`` that is not present in
    the topology graph.
    """

    audit_sink: AuditSink = field(default=_noop_audit_sink)
    topology: RuntimeTopology | None = None
    _states: dict[str, LifecycleState] = field(default_factory=dict)
    _transitions: list[RegisteredTransition] = field(default_factory=list)
    _version: int = field(default=0)

    def __post_init__(self) -> None:
        if not callable(self.audit_sink):
            raise ActivationError("audit_sink must be callable")
        if self.topology is not None and not isinstance(self.topology, RuntimeTopology):
            raise ActivationError(
                f"topology must be a RuntimeTopology or None, got {type(self.topology).__name__}"
            )

    # -- registration ------------------------------------------------------

    def register(
        self,
        node_id: str,
        initial_state: LifecycleState = LifecycleState.DECLARED,
        reason: str = "register",
    ) -> None:
        """Register a node with an initial lifecycle state.

        Re-registering a node with the same state is a no-op (the
        registry intentionally tolerates re-entrant boot wiring).
        Re-registering with a *different* state raises
        :class:`ActivationViolation` and emits an audit violation row;
        callers MUST use :meth:`transition` instead.
        """

        self._validate_node_id(node_id)
        if not isinstance(initial_state, LifecycleState):
            raise ActivationError(
                f"initial_state must be a LifecycleState, got {type(initial_state).__name__}"
            )
        if not isinstance(reason, str):
            raise ActivationError(f"reason must be a str, got {type(reason).__name__}")
        if len(reason) > MAX_REASON_LEN:
            raise ActivationError(
                f"reason length {len(reason)} exceeds MAX_REASON_LEN={MAX_REASON_LEN}"
            )
        if self.topology is not None and self.topology.find_node(node_id) is None:
            raise ActivationError(f"node_id {node_id!r} is not declared in the topology")
        if len(self._states) >= MAX_REGISTERED_NODES:
            raise ActivationError(f"registry is full ({MAX_REGISTERED_NODES} nodes)")

        previous = self._states.get(node_id)
        if previous is None:
            self._states[node_id] = initial_state
            self._version += 1
            self._record_transition(
                node_id=node_id,
                src=initial_state,
                dst=initial_state,
                reason=reason,
            )
            return

        if previous is initial_state:
            # Idempotent re-registration is allowed and is a no-op
            # for the FSM, but we still emit an audit row so the
            # ledger can reconstruct the boot timeline.
            return

        self._emit_audit(
            kind=AUDIT_KIND_VIOLATION,
            payload={
                "node_id": node_id,
                "previous_state": previous.value,
                "attempted_state": initial_state.value,
                "reason": reason,
                "version_after": self._version,
                "violation_kind": "register_state_mismatch",
            },
        )
        raise ActivationViolation(
            f"node {node_id!r} already registered as {previous.value!r}; "
            f"refusing to re-register as {initial_state.value!r}"
        )

    # -- transitions ------------------------------------------------------

    def transition(
        self,
        node_id: str,
        new_state: LifecycleState,
        reason: str = "transition",
    ) -> None:
        """Move ``node_id`` from its current state to ``new_state``.

        Raises :class:`ActivationViolation` (after emitting an audit
        row) if the node is not registered, the transition is not
        legal under the FSM table, or any input is malformed.
        """

        self._validate_node_id(node_id)
        if not isinstance(new_state, LifecycleState):
            raise ActivationError(
                f"new_state must be a LifecycleState, got {type(new_state).__name__}"
            )
        if not isinstance(reason, str):
            raise ActivationError(f"reason must be a str, got {type(reason).__name__}")
        if len(reason) > MAX_REASON_LEN:
            raise ActivationError(
                f"reason length {len(reason)} exceeds MAX_REASON_LEN={MAX_REASON_LEN}"
            )

        current = self._states.get(node_id)
        if current is None:
            self._emit_audit(
                kind=AUDIT_KIND_VIOLATION,
                payload={
                    "node_id": node_id,
                    "attempted_state": new_state.value,
                    "reason": reason,
                    "version_after": self._version,
                    "violation_kind": "transition_unregistered",
                },
            )
            raise ActivationViolation(f"transition called for unregistered node {node_id!r}")

        if not is_legal_transition(current, new_state):
            self._emit_audit(
                kind=AUDIT_KIND_VIOLATION,
                payload={
                    "node_id": node_id,
                    "current_state": current.value,
                    "attempted_state": new_state.value,
                    "reason": reason,
                    "version_after": self._version,
                    "violation_kind": "illegal_transition",
                },
            )
            raise ActivationViolation(
                f"illegal transition for {node_id!r}: {current.value} -> {new_state.value}"
            )

        if len(self._transitions) >= MAX_TRANSITION_HISTORY:
            raise ActivationError(f"transition history is full ({MAX_TRANSITION_HISTORY} entries)")

        self._states[node_id] = new_state
        self._version += 1
        self._record_transition(node_id=node_id, src=current, dst=new_state, reason=reason)

    # -- read surface -----------------------------------------------------

    def state_of(self, node_id: str) -> LifecycleState | None:
        """Return the current lifecycle state of ``node_id`` or
        ``None`` if not registered."""

        self._validate_node_id(node_id)
        return self._states.get(node_id)

    def snapshot(self) -> ActivationSnapshot:
        """Return a frozen :class:`ActivationSnapshot` projection.

        The snapshot is the canonical read surface for PR-RT-3's
        capability resolver and PR-RT-4's operator routes. Two
        registries that received the same call sequence always emit
        byte-identical snapshots.
        """

        return ActivationSnapshot(
            states=tuple(sorted(self._states.items(), key=lambda kv: kv[0])),
            transitions=tuple(self._transitions),
            version=self._version,
        )

    def active_node_ids(self) -> frozenset[str]:
        return self.snapshot().active_node_ids()

    def dormant_node_ids(self) -> frozenset[str]:
        return self.snapshot().dormant_node_ids()

    def wired_node_ids(self) -> frozenset[str]:
        return self.snapshot().wired_node_ids()

    def registered_node_ids(self) -> frozenset[str]:
        return frozenset(self._states)

    def transition_history(
        self,
    ) -> tuple[RegisteredTransition, ...]:
        return tuple(self._transitions)

    def current_version(self) -> int:
        return self._version

    # -- internals --------------------------------------------------------

    def _validate_node_id(self, node_id: str) -> None:
        if not isinstance(node_id, str):
            raise ActivationError(f"node_id must be a str, got {type(node_id).__name__}")
        if not node_id:
            raise ActivationError("node_id must not be empty")
        if len(node_id) > MAX_NODE_ID_LEN:
            raise ActivationError(
                f"node_id length {len(node_id)} exceeds MAX_NODE_ID_LEN={MAX_NODE_ID_LEN}"
            )

    def _record_transition(
        self,
        node_id: str,
        src: LifecycleState,
        dst: LifecycleState,
        reason: str,
    ) -> None:
        record = RegisteredTransition(
            node_id=node_id,
            src=src,
            dst=dst,
            reason=reason,
            version_after=self._version,
        )
        self._transitions.append(record)
        self._emit_audit(
            kind=AUDIT_KIND_TRANSITION,
            payload={
                "dst": dst.value,
                "node_id": node_id,
                "reason": reason,
                "src": src.value,
                "version_after": self._version,
            },
        )

    def _emit_audit(self, kind: str, payload: Mapping[str, object]) -> None:
        try:
            self.audit_sink(kind, dict(payload))
        except Exception as exc:  # noqa: BLE001 - sink must never block primary path
            fallback_payload = {
                "primary_kind": kind,
                "primary_payload": dict(payload),
                "sink_exception_type": type(exc).__name__,
                "sink_exception_msg": str(exc)[:MAX_REASON_LEN],
            }
            try:
                _noop_audit_sink(AUDIT_KIND_SINK_FAILURE, fallback_payload)
            except Exception:  # noqa: BLE001 - hard fallback
                pass


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def build_registry(
    topology: RuntimeTopology | None = None,
    audit_sink: AuditSink | None = None,
) -> RuntimeActivationRegistry:
    """Build a :class:`RuntimeActivationRegistry`.

    Provided as a small convenience over the constructor so call sites
    can leave ``audit_sink=None`` and get the no-op default without
    importing the private sentinel.
    """

    return RuntimeActivationRegistry(
        audit_sink=audit_sink if audit_sink is not None else _noop_audit_sink,
        topology=topology,
    )


def replay_transitions(
    transitions: Sequence[RegisteredTransition],
    topology: RuntimeTopology | None = None,
) -> RuntimeActivationRegistry:
    """Reconstruct a registry from a previously recorded transition
    history.

    Used by PR-RT-3 / PR-RT-5 to deterministically rebuild registry
    state from the audit ledger without ever touching the original
    registry instance.
    """

    if not isinstance(transitions, Sequence):
        raise ActivationError(f"transitions must be a Sequence, got {type(transitions).__name__}")
    registry = build_registry(topology=topology)
    for transition in transitions:
        if not isinstance(transition, RegisteredTransition):
            raise ActivationError(
                f"every transition must be a RegisteredTransition, got {type(transition).__name__}"
            )
        current = registry.state_of(transition.node_id)
        if current is None:
            registry.register(
                node_id=transition.node_id,
                initial_state=transition.src,
                reason=transition.reason,
            )
        if transition.src != transition.dst:
            registry.transition(
                node_id=transition.node_id,
                new_state=transition.dst,
                reason=transition.reason,
            )
    return registry
