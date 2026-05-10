"""Simulation engine — slow-cadence sim primitives consumed off-hot-path.

This package collects the canonical-tier simulation primitives adapted
from external open-source backtesters (hftbacktest, nautilus_trader,
…). Every module here is a pure-data leaf: no engine cross-imports, no
clock, no PRNG except via explicit ``seed`` arguments, no IO.

The simulation engine outputs are read by the meta-controller's
scoring layer and the strategy arena. They never run on the trading
hot path (T1 ≤1ms budget) — that's the simulation tier rule from the
master canonical PART 1 ("OFFLINE tier — can use ML, must emit only
``UPDATE_PROPOSED`` events; never called from ``hot_path/``").
"""
