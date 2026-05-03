"""DIX VISION v42.2 system primitives — Tier-0 safety surface.

Currently exports:

* :mod:`system.time_source` — canonical monotonic clock + sequence
  authority (T0-4). Re-introduced after the rust-deletion (PR #116)
  removed the polyglot dual-backend implementation. P0-1a of the
  PHASE6 action plan.

The module-level package marker is intentionally minimal — Tier-0
primitives must remain free of side effects at import time so they
load cleanly during bootstrap before any engine is constructed.
"""
