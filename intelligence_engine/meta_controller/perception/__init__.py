"""Meta-Controller / perception — regime classification (Phase 6.T1e).

Contains :mod:`intelligence_engine.meta_controller.perception.regime_router`,
the INV-49 hysteresis-bounded regime transition gate.
"""

from intelligence_engine.meta_controller.perception.regime_router import (
    RegimeRouterConfig,
    RegimeRouterState,
    initial_router_state,
    load_regime_router_config,
    step_regime_router,
)

__all__ = [
    "RegimeRouterConfig",
    "RegimeRouterState",
    "initial_router_state",
    "load_regime_router_config",
    "step_regime_router",
]
