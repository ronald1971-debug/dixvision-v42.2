"""Broker adapters for the Execution Engine (Phase E1).

Adapters convert :class:`SignalEvent` (after Governance approval) into
:class:`ExecutionEvent`. The Phase E1 adapter is the deterministic
:class:`PaperBroker`; live exchange adapters land in later phases.
"""

from execution_engine.adapters.base import BrokerAdapter
from execution_engine.adapters.paper import PaperBroker

__all__ = ["BrokerAdapter", "PaperBroker"]
