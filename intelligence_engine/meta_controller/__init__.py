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

from intelligence_engine.meta_controller.config import (
    DEFAULT_CONFIDENCE_PATH,
    DEFAULT_LATENCY_BUDGET_NS,
    DEFAULT_REGIME_PATH,
    DEFAULT_SIZER_PATH,
    load_meta_controller_config,
)
from intelligence_engine.meta_controller.hot_path import MetaControllerHotPath
from intelligence_engine.meta_controller.orchestrator import (
    META_CONTROLLER_VERSION,
    MetaControllerConfig,
    MetaControllerOutput,
    MetaControllerState,
    initial_meta_controller_state,
    run_meta_controller_tick,
)
from intelligence_engine.meta_controller.runtime_adapter import (
    RUNTIME_ADAPTER_SOURCE,
    build_meta_audit_event,
    step_meta_controller_hot_path,
)

__all__ = [
    "DEFAULT_CONFIDENCE_PATH",
    "DEFAULT_LATENCY_BUDGET_NS",
    "DEFAULT_REGIME_PATH",
    "DEFAULT_SIZER_PATH",
    "META_CONTROLLER_VERSION",
    "MetaControllerConfig",
    "MetaControllerHotPath",
    "MetaControllerOutput",
    "MetaControllerState",
    "RUNTIME_ADAPTER_SOURCE",
    "build_meta_audit_event",
    "initial_meta_controller_state",
    "load_meta_controller_config",
    "run_meta_controller_tick",
    "step_meta_controller_hot_path",
]
