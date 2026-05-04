"""Strategy runtime (Phase 3 / v2-B).

The strategy runtime turns *many independent plugin outputs* into *one
coordinated portfolio decision* by sequencing four pure components:

* :mod:`state_machine`     — strategy lifecycle FSM
  (``PROPOSED → CANARY → LIVE → RETIRED`` + ``FAILED`` from anywhere;
  strategy-level SHADOW was demolished by SHADOW-DEMOLITION-02)
* :mod:`regime_detector`   — deterministic market-regime classification
* :mod:`scheduler`         — bar-aligned cadence (when each strategy runs)
* :mod:`orchestrator`      — activates strategies based on regime + lifecycle
* :mod:`conflict_resolver` — resolves conflicting signals on the same symbol

Every module here is pure-Python, IO-free, and clock-free. They are
consumed by ``IntelligenceEngine`` and the upcoming signal pipeline
(Phase 3) without crossing engine boundaries (INV-08).
"""

from intelligence_engine.strategy_runtime.conflict_resolver import (
    ConflictResolution,
    ConflictResolver,
)
from intelligence_engine.strategy_runtime.orchestrator import (
    StrategyOrchestrator,
    StrategyRecord,
)
from intelligence_engine.strategy_runtime.regime_detector import (
    MarketRegime,
    RegimeDetector,
    RegimeReading,
)
from intelligence_engine.strategy_runtime.scheduler import (
    StrategyScheduler,
    StrategyTick,
)
from intelligence_engine.strategy_runtime.state_machine import (
    LEGAL_STRATEGY_TRANSITIONS,
    StrategyLifecycleError,
    StrategyState,
    StrategyStateMachine,
)

__all__ = [
    "ConflictResolution",
    "ConflictResolver",
    "LEGAL_STRATEGY_TRANSITIONS",
    "MarketRegime",
    "RegimeDetector",
    "RegimeReading",
    "StrategyLifecycleError",
    "StrategyOrchestrator",
    "StrategyRecord",
    "StrategyScheduler",
    "StrategyState",
    "StrategyStateMachine",
    "StrategyTick",
]
