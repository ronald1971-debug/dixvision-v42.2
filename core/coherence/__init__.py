"""Coherence layer — read-only system projections (Phase 6.T1a).

The coherence package holds **read-only projections** of the live system:
:class:`BeliefState` (regime + system view) and :class:`PressureVector`
(5-D constraint summary). Both are pure-function derivations from
ledgered inputs — they NEVER mutate state, NEVER write to governance,
and NEVER reach the hot path.

References:

- ``docs/manifest_v3.1_delta.md`` §B1 (Belief State as projection),
  H2 (continuous safety modifier)
- ``docs/manifest_v3.2_delta.md`` §1.3 / INV-50 (cross-signal entropy
  in ``PressureVector.uncertainty``)
- ``docs/manifest_v3.3_delta.md`` §1.2 / INV-53 (calibration hook —
  ``BELIEF_STATE_SNAPSHOT`` + ``PRESSURE_VECTOR_SNAPSHOT`` ledger rows
  feed the offline ``coherence_calibrator``)

Authority lint:

- The package is whitelisted via the ``core.*`` allow-list.
- It has zero engine dependencies — only ``core.contracts``.
"""

from core.coherence.belief_state import (
    BeliefState,
    Regime,
    derive_belief_state,
)
from core.coherence.performance_pressure import (
    PressureConfig,
    PressureVector,
    derive_pressure_vector,
    load_pressure_config,
)

__all__ = [
    "BeliefState",
    "PressureConfig",
    "PressureVector",
    "Regime",
    "derive_belief_state",
    "derive_pressure_vector",
    "load_pressure_config",
]
