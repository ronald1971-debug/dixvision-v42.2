# DIX VISION v42.2

Engine + plugin runtime for a dual-domain (Indira / Dyon), governance-mediated,
event-sourced trading system.

## Architecture (binding)

* **6 engines, 2 tiers.** See `docs/total_recall_index.md §39`.
  * RUNTIME (per-tick canonical bus, strictly deterministic):
    * `intelligence_engine` (RUNTIME-ENGINE-01)
    * `execution_engine` (RUNTIME-ENGINE-02)
    * `system_engine` (RUNTIME-ENGINE-03)
    * `governance_engine` (RUNTIME-ENGINE-04)
  * OFFLINE (scheduler-driven, share one process, emit `UPDATE_PROPOSED`
    only):
    * `learning_engine` (OFFLINE-ENGINE-01)
    * `evolution_engine` (OFFLINE-ENGINE-02)
* **4 canonical events.** Defined once in `core/contracts/events.py` (and
  mirrored in `contracts/events.proto`):
  `SignalEvent` · `ExecutionEvent` · `SystemEvent` · `HazardEvent`.
* **Authority lint rules** (CI-enforced, `tools/authority_lint.py`):
  T1 · C2 · C3 · W1 · **L1** · **L2** · **L3** · **B1**.

## Phase E0 deliverables (this commit)

* `core/contracts/engine.py` — `Engine` / `RuntimeEngine` / `OfflineEngine` /
  `Plugin` Protocols.
* `core/contracts/events.py` + `contracts/events.proto` — 4 typed events.
* Six engine shells, one per runtime + offline engine.
* `registry/engines.yaml`, `registry/plugins.yaml` — declarative truth.
* `tools/authority_lint.py` — full rule set including L1/L2/L3/B1.
* `tests/` — engine instantiation + lint rule unit tests.
* `.github/workflows/ci.yml` — ruff + authority_lint + pytest.

## Running

```bash
python -m pip install -e ".[dev]"
ruff check .
python tools/authority_lint.py --strict .
pytest -q
```

## References

* `docs/total_recall_index.md` — every architectural ID.
* `manifest.md` — single source of truth.
* `build_plan.md` — phase-by-phase build order.
* `MAPPING.md` — layer → engine → plugin lossless mapping proof.
