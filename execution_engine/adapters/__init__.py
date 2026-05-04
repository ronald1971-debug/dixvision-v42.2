"""Broker adapters for the Execution Engine (Phase E1).

Adapters convert :class:`SignalEvent` (after Governance approval) into
:class:`ExecutionEvent`. The Phase E1 adapter is the deterministic
:class:`PaperBroker`; live exchange adapters land in later phases.
"""

from execution_engine.adapters._live_base import (
    AdapterState,
    AdapterStatus,
    LiveAdapterBase,
)
from execution_engine.adapters.base import BrokerAdapter
from execution_engine.adapters.hummingbot import HummingbotAdapter
from execution_engine.adapters.paper import PaperBroker
from execution_engine.adapters.pumpfun import PumpFunAdapter
from execution_engine.adapters.registry import (
    AdapterRegistry,
    default_registry,
)

# UniswapX needs ``eth-account`` for EIP-712 signing. That dep lives in
# the optional ``[evm]`` / ``[dev]`` extras so the base launcher can
# boot without it. Re-export ``UniswapXAdapter`` only when its
# dependency chain imports cleanly; otherwise ``UniswapXAdapter`` is
# ``None`` and ``default_registry()`` skips registering it.
try:
    from execution_engine.adapters.uniswapx import UniswapXAdapter
except ImportError:
    UniswapXAdapter = None  # type: ignore[assignment,misc]

__all__ = [
    "AdapterRegistry",
    "AdapterState",
    "AdapterStatus",
    "BrokerAdapter",
    "HummingbotAdapter",
    "LiveAdapterBase",
    "PaperBroker",
    "PumpFunAdapter",
    "UniswapXAdapter",
    "default_registry",
]
