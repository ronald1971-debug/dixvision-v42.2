"""Simulation package — the SIM-XX module surface.

The simulation engine runs at a slower cadence than the trading hot path
(see executive_summary §296). Its outputs (RealitySummary,
SimulationSnapshot) are read cached by the meta-controller; the T1 ≤1ms
hot-path budget is never compromised.

This package is intentionally a leaf — it imports only from
:mod:`core.contracts` and the standard library. Engine cross-imports
would violate INV-08 (engine isolation).
"""
