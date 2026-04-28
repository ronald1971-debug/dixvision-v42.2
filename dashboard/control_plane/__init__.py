"""Dashboard control-plane modules (Phase 6, 5 immutable widgets).

Per Build Compiler Spec §6, these modules are the dashboard's Python
backend. They implement two kinds of behaviour:

1. **Read projections** — pure functions of engine state, ledger
   reader output, or registry. No side effects, no PRNG, no clock.
2. **Request seam** — the only writes the dashboard ever performs are
   :class:`~core.contracts.governance.OperatorRequest` instances handed
   to the GOV-CP-07 :class:`OperatorInterfaceBridge`. That bridge is
   the dashboard's sole write path into the system (INV-12, INV-37).

Authority lint rule **B7** (dashboard isolation) restricts imports to
``core.contracts``, ``core.coherence`` (read-only projections),
``governance_engine.control_plane`` (Protocol surfaces only), and
peer engines' ``check_self``-style read APIs. The dashboard may not
import private modules of any engine.
"""

from dashboard.control_plane.decision_trace import DecisionTracePanel
from dashboard.control_plane.engine_status_grid import (
    EngineHealthRow,
    EngineStatusGrid,
)
from dashboard.control_plane.memecoin_control_panel import (
    MemecoinControlPanel,
    MemecoinSubsystemStatus,
)
from dashboard.control_plane.mode_control_bar import ModeControlBar
from dashboard.control_plane.router import ControlPlaneRouter, RouteOutcome
from dashboard.control_plane.strategy_lifecycle_panel import (
    StrategyLifecyclePanel,
    StrategyRow,
)

__all__ = (
    "ControlPlaneRouter",
    "DecisionTracePanel",
    "EngineHealthRow",
    "EngineStatusGrid",
    "MemecoinControlPanel",
    "MemecoinSubsystemStatus",
    "ModeControlBar",
    "RouteOutcome",
    "StrategyLifecyclePanel",
    "StrategyRow",
)
