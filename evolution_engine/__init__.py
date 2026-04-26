"""OFFLINE-ENGINE-02 Evolution (Phase E0 shell).

Scheduler-driven. Hosts skill graph, intelligence loops, and the patch
pipeline (GOV-G18). Emits ``UPDATE_PROPOSED`` only.

**Lint rule L1 forbids importing from ``learning_engine``** even though
both engines share a single offline Python process.
"""

from evolution_engine.engine import EvolutionEngine

__all__ = ["EvolutionEngine"]
