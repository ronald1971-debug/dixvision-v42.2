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

The repo ships **two equivalent install paths**, kept in lockstep:

* `requirements.txt` — runtime-only deps (PyYAML, fastapi, uvicorn,
  pydantic, websockets). Auto-installed by the Windows launcher and any
  CI/CD that just needs to *run* the harness.
* `requirements-dev.txt` — runtime + pytest, ruff, httpx. What you need
  to also run the test suite + linters.
* `pyproject.toml` `[project.optional-dependencies]` — Python-native
  alternative (`.[dev]` / `.[ui]`). Same versions as the requirement
  files; bump both in the same commit.

```bash
# either path works; pick one
python -m pip install -r requirements-dev.txt && python -m pip install -e .
# or
python -m pip install -e ".[dev]"

ruff check .
python tools/authority_lint.py --strict .
pytest -q
```

#### Faster: install with `uv` (recommended)

[`uv`](https://github.com/astral-sh/uv) is significantly faster than
pip for resolution + install and is the **recommended** tool when
working on the codebase. The Windows launcher still uses pip for
first-run simplicity (no extra prerequisite), so this is opt-in:

```bash
# one-time: install uv (Linux / macOS / WSL)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows: irm https://astral.sh/uv/install.ps1 | iex

# create a venv and install deps
uv venv .venv -p 3.12
uv pip install -r requirements-dev.txt
uv pip install -e .
```

The full dependency policy (what's in, what's deferred, and what's
explicitly NOT recommended due to invariant conflicts like INV-15
replay determinism) is documented in
[`docs/dependency_roadmap.md`](docs/dependency_roadmap.md).

### Running on Windows (one-click launcher)

A double-click launcher + auto-installed desktop shortcut are shipped
under `scripts/windows/`. First-time setup is literally one step:

```bat
scripts\windows\start_dixvision.bat
```

That's it. On the first run the launcher creates a `.venv`,
`pip install -r requirements-dev.txt` + `pip install -e .`, writes a
marker file under `.venv/`, **drops a "DIX VISION" shortcut on your
desktop**, and starts the FastAPI control-plane harness on
`http://127.0.0.1:8080/` (opening the page in your default browser).

Subsequent runs skip the install steps and just (re)start the harness.
The desktop-shortcut step is idempotent and self-healing: every launch
overwrites `Desktop\DIX VISION.lnk`, so re-running the .bat fixes a
deleted shortcut.

If you prefer to install the shortcut without launching the harness
(e.g. on a setup pass), the standalone PowerShell entry point is still
available:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\install_desktop_shortcut.ps1
```

Use `scripts\windows\stop_dixvision.bat` for a clean shutdown
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
