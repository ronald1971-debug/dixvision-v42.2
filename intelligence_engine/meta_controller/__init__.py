"""Meta-Controller — H1 conceptual split (Phase 6.T1b/T1e).

Internal layout (manifest_v3.1_delta.md §H1):

* ``perception/`` — regime classification & hysteresis (T1e).
* ``evaluation/`` — confidence engine + debate round (T1b).
* ``allocation/`` — position sizer (T1b).
* ``policy/`` — execution policy + INV-48 fallback + INV-52 shadow (T1b).

Phase 6.T1e ships ``perception/`` only. The remaining sub-packages land
in 6.T1b on a separate branch.

Authority lint:

* B1 — no cross-runtime-engine direct imports (intelligence_engine
  cannot import execution_engine / system_engine / governance_engine).
* L3 — no learning / evolution imports.
* The package depends only on ``core.contracts`` and ``core.coherence``.
"""

__all__: list[str] = []
