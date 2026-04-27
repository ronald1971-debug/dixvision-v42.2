"""Hot-path execution package — Phase 2 / EXEC-11.

Modules in this package are subject to lint rule **T1** (fast-path
purity, INV-17): no imports from ``governance_engine`` or any other
runtime engine; only ``core`` / ``core.contracts`` / ``state.ledger.reader``
are allowed in addition to the standard library.

The hot path runs every signal through a deterministic risk gate
without any IO. It is the *micro* counterpart to ``ExecutionEngine``;
it never blocks, never allocates large objects, and never logs to
disk. Auditing is performed slow-path, by Governance, off the bus.
"""

from execution_engine.hot_path.fast_execute import (
    FastExecutor,
    HotPathDecision,
    HotPathOutcome,
    RiskSnapshot,
)

__all__ = [
    "FastExecutor",
    "HotPathDecision",
    "HotPathOutcome",
    "RiskSnapshot",
]
