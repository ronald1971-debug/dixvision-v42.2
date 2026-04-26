"""OFFLINE-ENGINE-01 Learning (Phase E0 shell).

Scheduler-driven. Reads ledger via ``state.ledger.reader``. Emits
``UPDATE_PROPOSED`` (sub-type of ``SystemEvent``) only.

**Lint rule L1 forbids importing from ``evolution_engine``** even though
both engines share a single offline Python process. Sharing a process
boundary does NOT mean sharing a domain boundary.
"""

from learning_engine.engine import LearningEngine

__all__ = ["LearningEngine"]
