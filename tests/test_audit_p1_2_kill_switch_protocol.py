"""AUDIT-P1.2 — ``StateTransitionProtocol`` extraction regression tests.

Pins the ``system/`` leaf invariant the audit was protecting:

* ``system.kill_switch`` must not transitively import
  ``governance_engine.*``. This is the bug the audit flagged: the
  previous direct dependency on
  ``governance_engine.control_plane.state_transition_manager``
  silently dragged the entire control plane into any module that
  imported ``system.kill_switch`` (which means every hazard sensor,
  every adapter that calls ``KillSwitch``, etc.).
* ``KillSwitch`` accepts any object satisfying
  :class:`StateTransitionProtocol`; the production
  :class:`StateTransitionManager` still works unchanged because it
  exposes the required ``current_mode`` and ``propose`` methods.
* ``StateTransitionProtocol`` is ``@runtime_checkable`` so duck-typed
  ``isinstance`` checks succeed for the real manager and fail for
  partial fakes that miss either method.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

from core.contracts.governance import (
    DecisionKind,
    ModeTransitionDecision,
    ModeTransitionRequest,
    StateTransitionProtocol,
    SystemMode,
)
from system.kill_switch import KillReason, KillRequest, KillSwitch


class _FakeStateTransitions:
    """Minimal protocol-satisfying double used as the unit-of-isolation.

    Captures every ``propose`` call so the test can assert the kill
    request is forwarded unchanged. Returns a deterministic
    :class:`ModeTransitionDecision` so the kill verdict is testable
    without booting the full :class:`StateTransitionManager`.
    """

    def __init__(self, mode: SystemMode = SystemMode.LIVE) -> None:
        self._mode = mode
        self.calls: list[ModeTransitionRequest] = []

    def current_mode(self) -> SystemMode:
        return self._mode

    def propose(
        self, request: ModeTransitionRequest
    ) -> ModeTransitionDecision:
        self.calls.append(request)
        return ModeTransitionDecision(
            ts_ns=request.ts_ns,
            approved=True,
            prev_mode=self._mode,
            new_mode=request.target_mode,
            reason=request.reason,
            rejection_code="",
            ledger_seq=42,
        )


def test_kill_switch_does_not_import_governance_engine() -> None:
    """Importing ``system.kill_switch`` must keep ``system/`` a leaf.

    Runs the import in a fresh Python subprocess so the module-cache
    pollution from the rest of the test session cannot mask a
    transitive ``governance_engine`` dependency. The subprocess is
    also the only correct way to assert this invariant: re-importing
    ``system.kill_switch`` in-process would replace the cached
    class object and break ``enforcement.KillSwitch is
    system.kill_switch.KillSwitch`` identity assertions in other
    test modules.
    """

    probe = textwrap.dedent(
        """
        import sys
        import system.kill_switch  # noqa: F401
        leaked = sorted(
            name for name in sys.modules
            if name.startswith("governance_engine")
        )
        if leaked:
            print("LEAKED=" + ",".join(leaked))
            raise SystemExit(1)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "system.kill_switch must not transitively import "
        f"governance_engine. stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )


def test_protocol_accepts_real_state_transition_manager() -> None:
    """The production :class:`StateTransitionManager` satisfies the protocol.

    Imported lazily so this test file itself does not pull
    ``governance_engine`` into the import graph at module load.
    """

    from governance_engine.control_plane.ledger_authority_writer import (
        LedgerAuthorityWriter,
    )
    from governance_engine.control_plane.policy_engine import PolicyEngine
    from governance_engine.control_plane.state_transition_manager import (
        StateTransitionManager,
    )

    stm = StateTransitionManager(
        policy=PolicyEngine(),
        ledger=LedgerAuthorityWriter(),
    )
    assert isinstance(stm, StateTransitionProtocol)


def test_protocol_rejects_partial_fake_missing_propose() -> None:
    """Runtime-checkable Protocol enforces both methods.

    A fake exposing ``current_mode`` but not ``propose`` must fail
    the ``isinstance`` check.
    """

    class _MissingPropose:
        def current_mode(self) -> SystemMode:
            return SystemMode.SAFE

    assert not isinstance(_MissingPropose(), StateTransitionProtocol)


def test_protocol_rejects_partial_fake_missing_current_mode() -> None:
    """The other half: missing ``current_mode`` is also rejected."""

    class _MissingCurrentMode:
        def propose(
            self, request: ModeTransitionRequest
        ) -> ModeTransitionDecision:
            raise AssertionError("unreachable")

    assert not isinstance(_MissingCurrentMode(), StateTransitionProtocol)


def test_kill_switch_engage_forwards_to_protocol() -> None:
    """``KillSwitch.engage`` calls ``propose`` on the injected protocol.

    Asserts the request shape is the documented LIVE→LOCKED
    transition tagged ``operator_authorized=True`` and tagged with
    the composed reason layout.
    """

    fake = _FakeStateTransitions(mode=SystemMode.LIVE)
    switch = KillSwitch(state_transitions=fake)
    decision = switch.engage(
        KillRequest(
            requestor="op-1",
            reason="manual halt",
            origin=KillReason.OPERATOR,
            ts_ns=1_234,
        )
    )
    assert len(fake.calls) == 1
    forwarded = fake.calls[0]
    assert forwarded.target_mode == SystemMode.LOCKED
    assert forwarded.current_mode == SystemMode.LIVE
    assert forwarded.requestor == "op-1"
    assert forwarded.operator_authorized is True
    assert "[OPERATOR]" in forwarded.reason
    assert "manual halt" in forwarded.reason
    assert decision.kind == DecisionKind.KILL
    assert decision.approved is True


def test_kill_switch_works_with_minimal_duck_type() -> None:
    """Anything satisfying the structural shape is accepted.

    Documents that ``KillSwitch`` does not accidentally rely on
    private attributes or methods of the concrete
    :class:`StateTransitionManager`. The fake here is the absolute
    minimum: two methods and no class hierarchy.
    """

    fake = _FakeStateTransitions(mode=SystemMode.CANARY)
    switch = KillSwitch(state_transitions=fake)
    decision = switch.engage_hazard(
        sensor="HAZ-CLOCK-DRIFT",
        reason="clock drift exceeded threshold",
        ts_ns=2_000,
    )
    assert decision.approved is True
    assert decision.kind == DecisionKind.KILL
    assert fake.calls[0].target_mode == SystemMode.LOCKED
