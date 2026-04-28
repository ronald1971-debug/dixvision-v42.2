"""Meta-Controller / allocation — position sizing (Phase 6.T1b).

Pure deterministic sizing function consumed by the policy layer.
The output ``final_size`` plugs into ``decide_execution_policy``'s
``proposed_size`` parameter.
"""

from intelligence_engine.meta_controller.allocation.position_sizer import (
    POSITION_SIZER_VERSION,
    PositionSizerConfig,
    SizingComponents,
    compute_position_size,
    load_position_sizer_config,
)

__all__ = [
    "POSITION_SIZER_VERSION",
    "PositionSizerConfig",
    "SizingComponents",
    "compute_position_size",
    "load_position_sizer_config",
]
