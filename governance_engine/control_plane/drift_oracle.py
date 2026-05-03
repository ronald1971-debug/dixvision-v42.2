"""GOV-CP-08 — Drift composite oracle (P0-7).

Closes the AUTO-mode safety loop reviewer #5 demanded.

PR #125 shipped a four-component drift specification on the dashboard
governance route (``model`` / ``exec`` / ``latency`` / ``causal``) but
nothing computed a composite or downgraded mode on breach. This
module fills both gaps:

* :class:`DriftCompositeOracle` aggregates an ordered tuple of
  per-component deviation readings into a max-component composite.
* :meth:`DriftCompositeOracle.evaluate_and_downgrade` proposes a
  one-step *backward* transition through the system-mode chain via
  :meth:`StateTransitionManager.propose` whenever the composite
  exceeds ``downgrade_threshold`` and the current mode is at or above
  ``CANARY``. Lower modes (``SHADOW`` and below) are already
  signals-on-execution-off or fully read-only, so a drift-driven
  downgrade has nothing to act on.

Determinism (INV-15): every input is caller-supplied; the oracle
reads no clock and no PRNG.

Mode FSM single-mutator (P0-6 / B32): the oracle never flips the
mode bit itself. It always routes the request through
:class:`StateTransitionManager`, which keeps the FSM legality check,
policy gate, promotion-gate hash anchor, and authority-ledger row in
the loop.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.contracts.governance import (
    ModeTransitionDecision,
    ModeTransitionRequest,
    SystemMode,
)
from governance_engine.control_plane.state_transition_manager import (
    StateTransitionManager,
)

__all__ = [
    "DEFAULT_DOWNGRADE_THRESHOLD",
    "DriftComponentReading",
    "DriftCompositeOracle",
    "DriftCompositeReading",
]

#: Default drift threshold used by both the dashboard panel
#: (``ui/governance_routes.py`` ``downgrade_threshold``) and the
#: oracle's per-component breach gate. Aligned with PR #125 so the
#: panel and the runtime agree on a single number.
DEFAULT_DOWNGRADE_THRESHOLD: float = 0.25


# ---------------------------------------------------------------------------
# Downgrade chain — reverse of the forward ratchet in StateTransitionManager.
# Entries are ``(from_mode, to_mode)`` pairs. Modes not in the table are
# ignored (no drift-driven downgrade applies — already at the safe floor).
# ---------------------------------------------------------------------------
_DOWNGRADE_CHAIN: tuple[tuple[SystemMode, SystemMode], ...] = (
    (SystemMode.AUTO, SystemMode.LIVE),
    (SystemMode.LIVE, SystemMode.CANARY),
    (SystemMode.CANARY, SystemMode.SHADOW),
)


def _downgrade_target(current: SystemMode) -> SystemMode | None:
    for src, dst in _DOWNGRADE_CHAIN:
        if src is current:
            return dst
    return None


@dataclass(frozen=True, slots=True)
class DriftComponentReading:
    """One axis of the four-component drift composite.

    ``deviation`` is the dimensionless ratio surfaced by
    :class:`system_engine.state.drift_monitor.DriftReading.deviation`;
    callers may also pass a precomputed scalar from a non-EWMA source.
    """

    component_id: str
    deviation: float
    threshold: float = DEFAULT_DOWNGRADE_THRESHOLD


@dataclass(frozen=True, slots=True)
class DriftCompositeReading:
    """Aggregated read of the drift oracle for a single tick."""

    ts_ns: int
    components: tuple[DriftComponentReading, ...]
    composite: float
    breached_components: tuple[str, ...]
    is_breaching: bool


class DriftCompositeOracle:
    """GOV-CP-08."""

    name: str = "drift_composite_oracle"
    spec_id: str = "GOV-CP-08"

    __slots__ = (
        "_component_ids",
        "_downgrade_threshold",
        "_requestor",
    )

    def __init__(
        self,
        *,
        component_ids: tuple[str, ...] = (
            "model",
            "exec",
            "latency",
            "causal",
        ),
        downgrade_threshold: float = DEFAULT_DOWNGRADE_THRESHOLD,
        requestor: str = "drift_oracle",
    ) -> None:
        if not component_ids:
            raise ValueError("component_ids must be non-empty")
        if downgrade_threshold <= 0.0:
            raise ValueError("downgrade_threshold must be positive")
        self._component_ids = component_ids
        self._downgrade_threshold = downgrade_threshold
        self._requestor = requestor

    @property
    def expected_components(self) -> tuple[str, ...]:
        """The canonical set of component ids the oracle expects.

        Surfaced for the dashboard so it can render a "missing axis"
        cell when a runtime caller forgets to plumb one of the four
        expected drift signals.
        """

        return self._component_ids

    @property
    def downgrade_threshold(self) -> float:
        """Composite-level breach gate (max-component deviation)."""

        return self._downgrade_threshold

    # ------------------------------------------------------------------
    # Pure projection
    # ------------------------------------------------------------------

    def observe(
        self,
        *,
        ts_ns: int,
        readings: tuple[DriftComponentReading, ...],
    ) -> DriftCompositeReading:
        """Aggregate per-component deviations into a composite reading.

        The composite is the max component deviation. A reading is
        ``is_breaching`` when **either**

        * any per-component deviation exceeds its
          :attr:`DriftComponentReading.threshold` (so a single noisy
          axis triggers a downgrade even if the others are calm), or
        * the composite (max-component) deviation is at or above the
          oracle's :attr:`downgrade_threshold` — the secondary gate
          documented at the module level.

        ``breached_components`` lists only the per-component breaches;
        a composite-only breach surfaces in ``is_breaching`` with an
        empty breach list.
        """

        if not readings:
            return DriftCompositeReading(
                ts_ns=ts_ns,
                components=(),
                composite=0.0,
                breached_components=(),
                is_breaching=False,
            )

        composite = max(r.deviation for r in readings)
        breached = tuple(
            r.component_id for r in readings if r.deviation > r.threshold
        )
        is_breaching = bool(breached) or composite >= self._downgrade_threshold
        return DriftCompositeReading(
            ts_ns=ts_ns,
            components=readings,
            composite=composite,
            breached_components=breached,
            is_breaching=is_breaching,
        )

    # ------------------------------------------------------------------
    # Auto-downgrade — routes through StateTransitionManager (B32)
    # ------------------------------------------------------------------

    def evaluate_and_downgrade(
        self,
        *,
        ts_ns: int,
        readings: tuple[DriftComponentReading, ...],
        stm: StateTransitionManager,
    ) -> tuple[DriftCompositeReading, ModeTransitionDecision | None]:
        """Compute the composite and propose a downgrade on breach.

        Returns the reading plus the ``ModeTransitionDecision`` from
        :meth:`StateTransitionManager.propose`, or ``None`` when no
        downgrade applies (no breach, or current mode already at or
        below ``SHADOW``).
        """

        reading = self.observe(ts_ns=ts_ns, readings=readings)
        if not reading.is_breaching:
            return reading, None

        current = stm.current_mode()
        target = _downgrade_target(current)
        if target is None:
            return reading, None

        if reading.breached_components:
            reason = (
                "drift_composite_breach: "
                + ",".join(reading.breached_components)
            )
        else:
            reason = (
                f"drift_composite_breach: composite={reading.composite:.4f}"
                f">=threshold={self._downgrade_threshold:.4f}"
            )
        request = ModeTransitionRequest(
            ts_ns=ts_ns,
            requestor=self._requestor,
            current_mode=current,
            target_mode=target,
            reason=reason,
            operator_authorized=False,
        )
        decision = stm.propose(request)
        return reading, decision
