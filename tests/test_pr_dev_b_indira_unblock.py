"""PR-DEV-B — Indira full-potential pin (Operator Master Development Mode).

PR-DEV-A inverted the default safety stance: ``development_enabled``
defaults to ``True`` (Indira + Dyon run full-bore at boot regardless
of :class:`SystemMode`) and ``trading_allowed`` defaults to ``False``
(the Execution Gate refuses to dispatch until the operator flips it).

PR-DEV-B pins the **Indira side** of that stance:

1. :class:`~intelligence_engine.engine.IntelligenceEngine` consults a
   single :class:`~intelligence_engine.learning_gate.LearningGate`
   reference. The gate's :meth:`is_open` re-reads the live
   :class:`~core.contracts.development_mode.DevelopmentModePolicy`
   on every consultation.
2. At boot defaults (``development_enabled=True``), the gate is open
   regardless of the current :class:`SystemMode` (SAFE / PAPER /
   CANARY / LIVE all unblock). This is the verified invariant: the
   :func:`MicrostructureV1.on_tick` plugin emits a
   :class:`SignalEvent` from each test mode.
3. When the operator flips the policy to
   ``development_enabled=False`` (via the audited operator route),
   the next :meth:`IntelligenceEngine.on_market` short-circuits and
   returns an empty signal tuple. No plugins are invoked.
4. The migration sentinel (``LearningGate()`` constructed with the
   default ``policy_supplier`` returning ``None``) resolves to
   open-by-default so pre-PR-DEV-B offline tests that build a bare
   :class:`IntelligenceEngine` retain their previous behaviour.
5. :class:`IntelligenceEngine` carries no direct
   :class:`SystemMode` dependence — B31 already pins this, and
   B-DEV-INDIRA (PR-DEV-B) pins the only remaining back-door (calls
   to ``effect_for(mode).<flag>`` from inside Indira's tier).

Audit trail:

* The gate's :meth:`audit_payload` mirrors the canonical
  ``POLICY_STATE`` projection so a closed gate row correlates with
  the ``OPERATOR_DEVELOPMENT_MODE_CHANGED`` ledger row that produced
  the state change.
"""

from __future__ import annotations

from collections.abc import Callable

from core.contracts.development_mode import (
    POLICY_VERSION,
    DevelopmentModePolicy,
)
from core.contracts.events import Side, SignalEvent
from core.contracts.governance import SystemMode
from core.contracts.market import MarketTick
from intelligence_engine.engine import IntelligenceEngine
from intelligence_engine.learning_gate import (
    LEARNING_GATE_CLOSED_REASON,
    LearningGate,
)
from intelligence_engine.plugins import MicrostructureV1

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signal_tick(ts: int = 1) -> MarketTick:
    """A tick whose last clears MicrostructureV1's tolerance band and
    must produce exactly one BUY :class:`SignalEvent`."""

    return MarketTick(
        ts_ns=ts,
        symbol="EURUSD",
        bid=99.99,
        ask=100.01,
        last=100.10,
    )


def _engine_with_gate(
    supplier: Callable[[], DevelopmentModePolicy | None],
) -> IntelligenceEngine:
    return IntelligenceEngine(
        microstructure_plugins=(MicrostructureV1(),),
        learning_gate=LearningGate(policy_supplier=supplier),
    )


# ---------------------------------------------------------------------------
# Contract layer — LearningGate
# ---------------------------------------------------------------------------


def test_learning_gate_default_supplier_resolves_open():
    """The migration sentinel (``None`` policy) resolves fail-open."""

    gate = LearningGate()

    assert gate.current_policy() is None
    assert gate.is_open() is True
    assert gate.is_closed() is False


def test_learning_gate_live_policy_open_at_boot_default():
    """Boot default ``development_enabled=True`` keeps the gate open."""

    policy = DevelopmentModePolicy()
    gate = LearningGate(policy_supplier=lambda: policy)

    assert policy.development_enabled is True
    assert policy.trading_allowed is False
    assert gate.is_open() is True


def test_learning_gate_closes_when_operator_pauses_learning():
    """Operator flip of ``development_enabled=False`` closes the gate."""

    policy = DevelopmentModePolicy(
        development_enabled=False,
        trading_allowed=False,
    )
    gate = LearningGate(policy_supplier=lambda: policy)

    assert gate.is_open() is False
    assert gate.is_closed() is True


def test_learning_gate_audit_payload_sentinel_shape():
    """Sentinel audit payload reports ``supplier=sentinel`` and
    open-state strings."""

    gate = LearningGate()
    payload = gate.audit_payload()

    assert payload["policy"] == "DevelopmentModePolicy"
    assert payload["version"] == POLICY_VERSION
    assert payload["reason"] == LEARNING_GATE_CLOSED_REASON
    assert payload["supplier"] == "sentinel"
    assert payload["development_enabled"] == "true"
    assert payload["trading_allowed"] == "true"
    assert payload["mode"] == ""


def test_learning_gate_audit_payload_live_closed_shape():
    """Closed-gate audit payload projects the live flags + mode."""

    policy = DevelopmentModePolicy(
        development_enabled=False,
        trading_allowed=False,
        mode=SystemMode.CANARY,
    )
    gate = LearningGate(policy_supplier=lambda: policy)
    payload = gate.audit_payload()

    assert payload["supplier"] == "live"
    assert payload["development_enabled"] == "false"
    assert payload["trading_allowed"] == "false"
    assert payload["mode"] == SystemMode.CANARY.name


def test_learning_gate_audit_payload_live_open_shape():
    """Open-gate audit payload still uses ``learning_gate_closed_by_operator``
    as the canonical row reason — it is the only kind of row this
    payload shape is used for; the open/closed state lives on the
    ``development_enabled`` field."""

    policy = DevelopmentModePolicy(
        development_enabled=True,
        trading_allowed=False,
        mode=SystemMode.SAFE,
    )
    gate = LearningGate(policy_supplier=lambda: policy)
    payload = gate.audit_payload()

    assert payload["supplier"] == "live"
    assert payload["development_enabled"] == "true"
    assert payload["trading_allowed"] == "false"
    assert payload["mode"] == SystemMode.SAFE.name


# ---------------------------------------------------------------------------
# IntelligenceEngine — sentinel fail-open
# ---------------------------------------------------------------------------


def test_intelligence_engine_without_gate_emits_signals():
    """The migration sentinel: an engine constructed without a
    learning_gate retains pre-PR-DEV-B unconditional emission."""

    engine = IntelligenceEngine(
        microstructure_plugins=(MicrostructureV1(),),
    )

    assert engine.learning_gate is None
    emitted = engine.on_market(_signal_tick())

    assert len(emitted) == 1
    assert isinstance(emitted[0], SignalEvent)
    assert emitted[0].side is Side.BUY


# ---------------------------------------------------------------------------
# IntelligenceEngine — mode independence (the actual PR-DEV-B invariant)
# ---------------------------------------------------------------------------


def test_intelligence_engine_emits_signals_at_all_system_modes():
    """Boot default ``development_enabled=True`` unblocks Indira at
    every :class:`SystemMode`. This is the operator-vision invariant
    PR-DEV-B pins: Indira's signal-emission surface runs full-bore
    regardless of mode; only ``development_enabled=False`` pauses it.
    """

    for mode in (
        SystemMode.SAFE,
        SystemMode.PAPER,
        SystemMode.CANARY,
        SystemMode.LIVE,
    ):
        policy = DevelopmentModePolicy(
            development_enabled=True,
            trading_allowed=False,
            mode=mode,
        )
        engine = _engine_with_gate(lambda p=policy: p)

        emitted = engine.on_market(_signal_tick())

        assert len(emitted) == 1, (
            f"Indira should emit at mode={mode.name} with "
            f"development_enabled=True; got {len(emitted)} signals"
        )
        assert emitted[0].side is Side.BUY


def test_intelligence_engine_pauses_when_operator_disables_development():
    """Operator flip of ``development_enabled=False`` short-circuits
    the next ``on_market`` call: no plugins invoked, empty tuple
    returned, signal window unchanged."""

    policy = DevelopmentModePolicy(
        development_enabled=False,
        trading_allowed=False,
        mode=SystemMode.LIVE,
    )
    engine = _engine_with_gate(lambda: policy)

    emitted = engine.on_market(_signal_tick())

    assert emitted == ()
    assert engine.signal_window == ()


def test_intelligence_engine_resumes_when_operator_re_enables_development():
    """Re-flipping ``development_enabled=True`` resumes signal
    emission on the **next** ``on_market`` call — the supplier
    re-reads the policy each tick; there is no caching."""

    # The supplier captures a mutable container so the test can flip
    # the policy under it without rebuilding the engine.
    holder: dict[str, DevelopmentModePolicy] = {
        "policy": DevelopmentModePolicy(
            development_enabled=False,
            trading_allowed=False,
            mode=SystemMode.LIVE,
        )
    }
    engine = _engine_with_gate(lambda: holder["policy"])

    paused = engine.on_market(_signal_tick(ts=1))
    assert paused == ()

    holder["policy"] = DevelopmentModePolicy(
        development_enabled=True,
        trading_allowed=False,
        mode=SystemMode.LIVE,
    )
    resumed = engine.on_market(_signal_tick(ts=2))

    assert len(resumed) == 1
    assert resumed[0].side is Side.BUY
    assert resumed[0].ts_ns == 2


def test_intelligence_engine_set_learning_gate_swaps_atomically():
    """``set_learning_gate`` swaps the active gate; the next
    ``on_market`` call observes the new gate's policy."""

    open_policy = DevelopmentModePolicy(development_enabled=True)
    closed_policy = DevelopmentModePolicy(development_enabled=False)

    engine = IntelligenceEngine(
        microstructure_plugins=(MicrostructureV1(),),
    )
    engine.set_learning_gate(LearningGate(policy_supplier=lambda: open_policy))
    emitted_open = engine.on_market(_signal_tick(ts=1))
    assert len(emitted_open) == 1

    engine.set_learning_gate(LearningGate(policy_supplier=lambda: closed_policy))
    emitted_closed = engine.on_market(_signal_tick(ts=2))
    assert emitted_closed == ()


def test_intelligence_engine_trading_blocked_does_not_pause_learning():
    """Boot default: ``trading_allowed=False`` (Execution Gate
    closed) must **not** pause learning. Indira still emits; the
    Execution Gate is the layer that refuses dispatch downstream."""

    policy = DevelopmentModePolicy(
        development_enabled=True,
        trading_allowed=False,
        mode=SystemMode.SAFE,
    )
    engine = _engine_with_gate(lambda: policy)

    emitted = engine.on_market(_signal_tick())

    assert len(emitted) == 1
    assert emitted[0].side is Side.BUY
