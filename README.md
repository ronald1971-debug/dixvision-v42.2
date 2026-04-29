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

### Running on Windows (one-click launcher)

A double-click launcher + desktop shortcut are shipped under
`scripts/windows/`. First-time setup (PowerShell, run from the repo root):

```powershell
# 1. (one time) install the desktop shortcut
powershell -ExecutionPolicy Bypass -File scripts\windows\install_desktop_shortcut.ps1

# 2. double-click "DIX VISION" on your desktop — or run the .bat directly
scripts\windows\start_dixvision.bat
```

The launcher is **idempotent**: the first run creates a `.venv`,
`pip install -e ".[dev]"` and writes a marker file under `.venv/`;
subsequent runs skip setup and just (re)start the FastAPI control-plane
harness on `http://127.0.0.1:8080/`, opening the page in your default
browser. Use `scripts\windows\stop_dixvision.bat` for a clean shutdown
(force-kills whatever is listening on port 8080).

Requirements: Python 3.12 from python.org *or* `winget install
Python.Python.3.12`. If a `pip install` fails because of missing C
build tools, run `winget install Microsoft.VisualStudio.2022.BuildTools`
once and retry.

## References

* `docs/total_recall_index.md` — every architectural ID.
* `manifest.md` — single source of truth.
* `build_plan.md` — phase-by-phase build order.
* `MAPPING.md` — layer → engine → plugin lossless mapping proof.
