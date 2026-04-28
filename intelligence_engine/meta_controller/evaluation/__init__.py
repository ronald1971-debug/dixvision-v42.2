"""Meta-Controller / evaluation — confidence + debate (Phase 6.T1b/c).

Phase 6.T1b ships :mod:`confidence_engine` only. The debate-round
module (manifest §H1) lands in a follow-up branch alongside the rest
of the agent-coordination work.
"""

from intelligence_engine.meta_controller.evaluation.confidence_engine import (
    CONFIDENCE_ENGINE_VERSION,
    ConfidenceComponents,
    ConfidenceEngineConfig,
    compute_confidence,
    load_confidence_engine_config,
    resolve_proposed_side,
)

__all__ = [
    "CONFIDENCE_ENGINE_VERSION",
    "ConfidenceComponents",
    "ConfidenceEngineConfig",
    "compute_confidence",
    "load_confidence_engine_config",
    "resolve_proposed_side",
]
