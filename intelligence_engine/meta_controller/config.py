"""Meta-Controller config loader — Phase 6.T1c.

Bundles the three sub-package YAML configs (regime / confidence /
sizer) into a single :class:`MetaControllerConfig` for the runtime
adapter. The latency budget (INV-48) is supplied by the caller — it
is a runtime tuning knob, not a sub-package coefficient.

Authority:

* Pure I/O on the registry filesystem; no clock, no PRNG.
* No cross-runtime-engine imports (B1).
* Each sub-config is delegated to its own ``load_*_config`` function
  so this module never duplicates validation logic.
"""

from __future__ import annotations

from pathlib import Path

from intelligence_engine.meta_controller.allocation import (
    load_position_sizer_config,
)
from intelligence_engine.meta_controller.evaluation import (
    load_confidence_engine_config,
)
from intelligence_engine.meta_controller.orchestrator import (
    META_CONTROLLER_VERSION,
    MetaControllerConfig,
)
from intelligence_engine.meta_controller.perception.regime_router import (
    load_regime_router_config,
)

DEFAULT_REGIME_PATH = "registry/regime.yaml"
DEFAULT_CONFIDENCE_PATH = "registry/confidence.yaml"
DEFAULT_SIZER_PATH = "registry/position_sizer.yaml"

DEFAULT_LATENCY_BUDGET_NS: int = 500_000  # 500 µs (manifest §6 T1)


def load_meta_controller_config(
    *,
    regime_path: str | Path = DEFAULT_REGIME_PATH,
    confidence_path: str | Path = DEFAULT_CONFIDENCE_PATH,
    sizer_path: str | Path = DEFAULT_SIZER_PATH,
    latency_budget_ns: int = DEFAULT_LATENCY_BUDGET_NS,
    version: str = META_CONTROLLER_VERSION,
) -> MetaControllerConfig:
    """Build a :class:`MetaControllerConfig` from the on-disk registry.

    Each sub-config validates its own YAML; ``MetaControllerConfig``
    validates ``latency_budget_ns > 0``.
    """
    return MetaControllerConfig(
        router_config=load_regime_router_config(regime_path),
        confidence_config=load_confidence_engine_config(confidence_path),
        sizer_config=load_position_sizer_config(sizer_path),
        latency_budget_ns=latency_budget_ns,
        version=version,
    )


__all__ = [
    "DEFAULT_CONFIDENCE_PATH",
    "DEFAULT_LATENCY_BUDGET_NS",
    "DEFAULT_REGIME_PATH",
    "DEFAULT_SIZER_PATH",
    "load_meta_controller_config",
]
