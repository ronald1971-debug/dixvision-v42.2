"""Meta-Controller — H1 conceptual split (Phase 6.T1b/T1e).

Internal layout (manifest_v3.1_delta.md §H1):

* ``perception/`` — regime classification & hysteresis (T1e).
* ``evaluation/`` — confidence engine + debate round (T1b).
* ``allocation/`` — position sizer (T1b).
* ``policy/`` — execution policy + INV-48 fallback + INV-52 shadow (T1b).
* ``orchestrator`` — composes the four sub-packages into one pure
  per-tick function.

Authority lint:

* B1 — no cross-runtime-engine direct imports (intelligence_engine
  cannot import execution_engine / system_engine / governance_engine).
* L3 — no learning / evolution imports.
* The package depends only on ``core.contracts`` and ``core.coherence``.
"""

from intelligence_engine.meta_controller.orchestrator import (
    META_CONTROLLER_VERSION,
    MetaControllerConfig,
    MetaControllerOutput,
    MetaControllerState,
    initial_meta_controller_state,
    run_meta_controller_tick,
)

__all__ = [
    "META_CONTROLLER_VERSION",
    "MetaControllerConfig",
    "MetaControllerOutput",
    "MetaControllerState",
    "initial_meta_controller_state",
    "run_meta_controller_tick",
]
