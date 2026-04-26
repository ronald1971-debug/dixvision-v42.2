"""RUNTIME-ENGINE-02 Execution (Phase E0 shell).

Owner of hot-path order routing, exchange/broker adapters, and
slippage/protection plugins. Strictly deterministic. Subject to lint rules
T1, B1, W1, L3.
"""

from execution_engine.engine import ExecutionEngine

__all__ = ["ExecutionEngine"]
