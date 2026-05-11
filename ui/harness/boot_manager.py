"""HarnessBootManager — domain-organised harness construction (P1.2).

Extracted from ``ui.server._State.__init__`` as part of the P1
harness god-object refactor. The historic ~480-line constructor
is now split into clearly-named ``_build_*`` sections on
:class:`ui.server._State` — one per concern (intelligence /
execution / system / governance / dashboard / cognitive chat /
plugin registry / approval edge / live feeds / learning-evolution
loops) — invoked in the same byte-stable order by
:meth:`HarnessBootManager.populate`.

This is a pure code-organisation refactor: zero behaviour change.
Every attribute the previous inline ``__init__`` set on ``_State``
is still set, in the same order, with the same value, by the
corresponding ``_build_*`` method on ``_State``. Construction
order is preserved bit-for-bit; the section methods live on
``_State`` itself so the literal ``self.<attr>`` references from
the original constructor are reused verbatim.

INV-15 byte-identical replay, B27 / B28 / INV-71 authority
symmetry, B32 single-mutator FSM, HARDEN-04 / INV-70 freeze
policy, and B7 dashboard-prefix lint are all preserved by
construction (no new typed-event kinds, no new ledger rows, no
new env vars). The existing ``test_ui_server_*`` suite is the
behavioural pin.

The manager holds no per-instance state: it is a coordinator that
calls back into the target ``_State`` so any closure that captured
``self`` in the original constructor (e.g. ``ApprovalEdge``
capturing ``_emit_cognitive_signal_locked``; the feed runners
capturing ``_ingest_*_locked`` sinks; the learning / evolution
loops capturing ``_live_freeze_policy``) keeps pointing at the
same canonical bound method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ui.server import _State


class HarnessBootManager:
    """Coordinator that walks the named ``_build_*`` sections.

    The previous inline ``_State.__init__`` ran ~480 LOC of
    construction code in one body. This manager preserves the
    construction order bit-for-bit but splits it into named
    sections so a reader can navigate one concern at a time:

    1. ``_build_intelligence_tier`` — SCVS source registry,
       meta-controller config, MetaControllerHotPath, IntelligenceEngine.
    2. ``_build_execution_tier`` — hazard throttle, decision signer,
       signal-trust caps + overlay, authority guard, risk baseline,
       learning sinks, ExecutionEngine.
    3. ``_build_system_tier`` — SensorArray with all 12 HAZ sensors,
       SystemEngine, RuntimeContext builder.
    4. ``_build_governance_tier`` — authority ledger writer,
       StrategyRegistry, ExposureStore, replay of source-trust
       promotions, GovernanceEngine, PolicyHashAnchor, learning /
       evolution engines, learning-override flag.
    5. ``_build_event_buffers`` — in-memory ring buffer + seq.
    6. ``_build_dashboard_widgets`` — Phase-6 control-plane widgets.
    7. ``_build_cognitive_chat`` — plugin toggle state, chat runtime.
    8. ``_build_plugin_registry`` — plugin manager registry.
    9. ``_build_approval_edge`` — cognitive approval edge.
    10. ``_build_live_feeds`` — Binance / CoinDesk / Pump.fun /
        Raydium runners and bounded ring buffers; binds the
        plugin registry's feed-runners table.
    11. ``_build_learning_evolution_loops`` — P0-A closed learning
        loop + structural evolution loop under the live HARDEN-04
        freeze policy.

    Each method mutates the ``state`` argument in place. The class
    holds no per-instance state of its own.
    """

    __slots__ = ()

    def populate(self, state: _State) -> None:
        """Run every ``_build_*`` section in canonical order."""

        state._build_intelligence_tier()
        state._build_execution_tier()
        state._build_system_tier()
        state._build_governance_tier()
        state._build_event_buffers()
        state._build_dashboard_widgets()
        state._build_cognitive_chat()
        state._build_plugin_registry()
        state._build_approval_edge()
        state._build_live_feeds()
        state._build_learning_evolution_loops()


__all__ = ("HarnessBootManager",)
