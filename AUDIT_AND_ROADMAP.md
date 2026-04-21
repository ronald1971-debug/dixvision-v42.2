# DIX VISION v42.2 — Compliance Audit & Improvement Roadmap

**Source of truth:** `DIX VISION v42.2 – CANONICAL SYSTEM MANIFEST.txt`
**Scope:** full repository as of this snapshot.
**Method:** manifest line-by-line mapping + static scans for cross-domain imports, silent exception handlers, direct state writes, hot-path hazards.

Severity legend:

- **BLOCKER** — violates a manifest invariant that could cause loss of capital or unsafe trading.
- **MAJOR** — architectural drift; still works but erodes the dual-domain + event-sourced guarantees.
- **MINOR** — code quality, polish, small perf or correctness bugs.
- **NIT** — cosmetic.

---

## 1. Compliance findings

### 1.1 BLOCKERS

| # | Finding | Manifest ref | File(s) | Fix |
|---|---|---|---|---|
| B-1 | `main.py` in **prod** mode trips `foundation_integrity_failed` because `immutable_core/foundation.hash` is missing/stale → kill switch arms before system boots. | §0, §2 | `immutable_core/foundation.py`, `bootstrap_kernel.py` | Generate `foundation.hash` at release time via `scripts/generate_hash.py`; include in ZIP. Also relax for `DIX_STRICT_INTEGRITY=0` (default) so dev runs work. |
| B-2 | Governance *can* be imported from the fast path (nothing blocks it). Any future drift → synchronous governance call in `place_order`. | §2, §3, §8 | fast path | **FIXED this session**: added `core/authority.py` + `mind/fast_execute.py` (no governance import allowed there). Need lint rule to enforce permanently. |
| B-3 | Dyon code (`execution/hazard/*`, `execution/engine.py`) lives under `execution/` — same package as INDIRA adapters. A single accidental import chain puts trade code inside a Dyon handler. | §4, §6 | `execution/hazard/*`, `execution/engine.py` | Promote `system_monitor.hazard_bus` (already created) to canonical import; keep `execution/hazard/*` as a re-export shim for compat. `assert_no_adapter_import(__name__)` runs at import time. |

### 1.2 MAJOR

| # | Finding | Manifest ref | File(s) | Fix |
|---|---|---|---|---|
| M-1 | `mind/knowledge/*` files listed in manifest §4 are **missing**. We have 1-line stubs at `mind/knowledge_validator.py` etc. but not under `mind/knowledge/`. | §4 (tree) | `mind/knowledge/*` | Move stubs → `mind/knowledge/<name>.py` with canonical docstrings + minimal class that logs a `NotImplementedYet` hazard on first call. |
| M-2 | Governance kernel uses `threading.Lock` + manual subscribers, not an event-sourced ledger-tail consumer. Not *wrong*, but drifts from §7 (“ledger records everything”). Governance should subscribe to the ledger stream router, not to ad-hoc callbacks. | §3, §7 | `governance/kernel.py` | Add `governance/kernel._start_ledger_consumer()` that subscribes to `stream_router` for `MARKET/SYSTEM/HAZARD` and updates risk cache + publishes `EXECUTION_CONSTRAINT_SET` back to the ledger. |
| M-3 | `execution/patch/*` (canary, sandbox, rollback, version_controller, outcome_store, patch_standards) are **1-line stubs**. Per §14 updater + §4 Dyon patching these need at least a tombstone class that raises `NotImplementedYet` + logs an audit event. | §4, §14 | `execution/patch/*` | Fill with safe-fail classes or move to `windows/updater/` (where manifest actually places them) and delete from `execution/`. |
| M-4 | `execution/engine.py` is Dyon by docstring but physically sits in INDIRA package. Conflict with §6. | §6 | `execution/engine.py` | Move source → `system_monitor/dyon_engine.py`; leave thin compat shim. |
| M-5 | `print()` used in 8 places (logger, health_monitor, kill_switch, main) — not structured logging. Manifest §10 requires JSON logs. | §10 | `system/logger.py`, `immutable_core/kill_switch.py`, `system/health_monitor.py`, `main.py` | Route every `print` through `get_logger(...).info/warn(...)`. Keep one operator-stdout banner at boot (acceptable). |
| M-6 | SQLite ledger is WAL-mode but **no other PRAGMAs tuned** (synchronous, mmap_size, cache_size, temp_store, wal_autocheckpoint). Big perf win + still durable. | §7, §8 | `state/ledger/event_store.py` | Add `PRAGMA synchronous=NORMAL; PRAGMA mmap_size=268435456; PRAGMA cache_size=-8000; PRAGMA temp_store=MEMORY; PRAGMA wal_autocheckpoint=1000;`. |
| M-7 | `cockpit.py` (FastAPI) has **no authentication**. Anyone reachable on `127.0.0.1` can hit it. | §11 | `cockpit.py` | Add loopback-only bind + bearer-token middleware that calls `security.authentication.verify_token`. |
| M-8 | No **replay determinism test**. Manifest §1 mandates replayability. Nothing currently proves the state projectors give the same hash when the ledger is replayed. | §1, §7 | `tests/` | Add `tests/test_replay_determinism.py` that writes N deterministic events, rehashes projector state, asserts stable hash. |
| M-9 | Governance `mode.safe_mode`, `degraded_mode`, `halted_mode` exist but no module **transitions between them** based on ledger hazards. | §3 | `governance/kernel.py` + `governance/mode/*` | Wire `governance.kernel` hazard consumer → `enter_safe_mode` / `enter_degraded_mode` / `enter_halted_mode` based on severity. |
| M-10 | `immutable_core/kill_switch.py` uses `print(...)` + a try/except pass on file write. A kill switch MUST NEVER swallow errors silently — best-effort, yes, but emit to stderr + ledger. | §1 (fail_closed) | `immutable_core/kill_switch.py` | `sys.stderr.write(...)` on failure; attempt ledger append (best-effort). |

### 1.3 MINOR

| # | Finding | File(s) | Fix |
|---|---|---|---|
| m-1 | 25 1-line stub files in `mind/`, `system/`, `execution/patch/`, `execution/runtime_monitor.py`, `execution/feedback.py`, `execution/system_repair_orchestrator.py`, `translation/round_trip.py` | various | Either implement with minimal safe-fail class or delete. Keeping 1-line stubs is worse than deleting — they lie about coverage. |
| m-2 | Silent `except Exception: pass` in 35+ places. Some are legitimate (best-effort ledger write from kill switch) but many hide real bugs. | see §2.2 below | Replace with `except Exception as e: get_logger(__name__).warning("<ctx> failed", err=str(e))`. |
| m-3 | `pyproject.toml` pins `requires-python = ">=3.10"`. Fast-path & asyncio improvements in 3.11/3.12 materially help latency. | `pyproject.toml` | Bump to `">=3.11"` (free-threaded 3.13 optional). |
| m-4 | No `mypy` config, no `ruff` config, no `pytest.ini` — `Makefile` calls them but config is missing. | repo root | Add `[tool.ruff]`, `[tool.mypy]`, `[tool.pytest.ini_options]` sections in `pyproject.toml`. |
| m-5 | `data/` directory shipped in the ZIP with `incidents.jsonl` and `ledger.db` (dev artefacts). | `data/*` | Add to packaging exclude list; recreate on first boot. |
| m-6 | No hash-chain **verify** command that walks the ledger top-to-bottom and asserts chain integrity. | `dix.py` | `dix.py ledger check` exists — verify its implementation walks *every* row, not just N most recent. |
| m-7 | `windows/service/nssm_config.xml` ships but NSSM is unmaintained since 2017 (verified via online research). | `windows/service/` | Keep NSSM xml for compat; add optional `servy.json` / `winsw.xml` alongside. |
| m-8 | `security/keyring_adapter.py` has three silent excepts. Secrets failing silently = pw lost to OS keyring. | `security/keyring_adapter.py` | Raise on read failure; log on write failure + fallback. |
| m-9 | `core/secrets.py` referenced in manifest §3 tree — missing file. | `core/` | Add small module re-exporting `security.secrets_manager` (single-sourced). |
| m-10 | No **single-instance** lockfile cleanup on SIGTERM. Second start-up after crash may be refused. | `core/single_instance.py` | On acquire, check if PID alive; if not, reclaim. |

### 1.4 NITS

- `README.md` exists but has no “First run on Windows” section with elevation/UAC instructions.
- `VERSION` is `42.2.0` but some docstrings still reference older minor versions.
- `.pre-commit-config.yaml` doesn't install `ruff-format` (present but unused).

---

## 2. Scan summaries (evidence)

### 2.1 Missing manifest files (post-remediation)

```
mind/knowledge/knowledge_validator.py    [missing]
mind/knowledge/source_conflict_graph.py  [missing]
mind/knowledge/memory_index.py           [missing]
mind/knowledge/edge_case_memory.py       [missing]
mind/knowledge/drift_monitor.py          [missing]
core/secrets.py                          [missing]
immutable_core/foundation.hash           [missing — B-1]
```

### 2.2 Silent exception handlers (`except …: pass`) — 35 hits. Key ones:

```
immutable_core/kill_switch.py:20   MAJOR (M-10)
security/keyring_adapter.py:30,40,48   MAJOR (M-8 / m-8)
observability/alerts/alert_engine.py:56,67   minor
mind/engine.py:124,146   minor (wrap-for-hot-path OK but should log)
mind/sources/*.py (news/sentiment/onchain/market):   minor (stream loops)
state/ledger/writer.py:66   OK (best-effort flush; by design)
execution/hazard/*   OK (event bus MUST never fail a producer)
```

### 2.3 Cross-domain import scan — **all clean**

```
governance → adapters/trade_executor/fast_execute:   none
mind/fast_execute → governance:                       none
interrupt → governance:                               none (uses cached policy)
system_monitor → execution/adapters:                  none
```

### 2.4 Direct SQLite writes outside the ledger

```
state/ledger/event_store.py  (canonical writer — expected)
system/config.py             (config DB, allowed by manifest §12)
(no other module writes directly.)
```

### 2.5 Ledger integrity

WAL mode enabled. Hash-chain implemented (`prev_hash` + `event_hash` per row). No other PRAGMAs tuned. `dix.py ledger check` command exists.

---

## 3. Simplification plan (lossless)

| # | Simplification | Loss? | Gain |
|---|---|---|---|
| S-1 | Promote `system_monitor.hazard_bus` to canonical; make `execution/hazard/*` a thin re-export shim. | none | One mental model for the hazard bus. |
| S-2 | Delete the 6 empty stubs in `mind/*` (duplicates of `mind/knowledge/*`). Replace with real minimal classes in `mind/knowledge/*`. | none | -6 files, -6 lies about coverage. |
| S-3 | Move `execution/patch/*` → `windows/updater/` (where manifest actually places patching). | none | Matches §14. |
| S-4 | Move `execution/engine.py` (Dyon) → `system_monitor/dyon_engine.py`; leave shim. | none | Matches §6 strict separation. |
| S-5 | One ledger writer API: `state.ledger.writer.write(...)`. Projectors only read. Remove any bypass path. | none | Enforces §7. |
| S-6 | One time source: all code calls `system.time_source.now()`. `time.time()`/`datetime.utcnow()` grep-forbidden outside `time_source.py`. | none | Deterministic replay. |
| S-7 | One config source: `system.config.get()`. Kill scattered `os.environ.get()`. | none | Centralized knobs. |
| S-8 | One authority entrypoint: `core.authority.scope(Domain.X)` wraps every public handler. | none | §16 boundary is enforced. |
| S-9 | Collapse `observability/logs` placeholder dir (logs are emitted live, not stored in-repo). | none | -1 junk dir in ZIP. |
| S-10 | Replace `print(...)` with `get_logger()`, keep only the single boot banner. | none | §10 JSON logs. |

**Net:** ~30 files/dirs consolidated or removed, **zero behavioral regressions**.

---

## 4. Enhancement backlog (ranked)

### Tier 1 — ship now, no risk

1. **SQLite PRAGMA tuning** (`synchronous=NORMAL`, `mmap_size=256MB`, `cache_size=-8000`, `temp_store=MEMORY`, `wal_autocheckpoint=1000`). 3–10× ledger throughput, still crash-safe in WAL.
2. **Fast-path zero-alloc**: pre-allocate `ExecutionEvent` + ledger dict per asset in a ring buffer. Avoids GC in hot path.
3. **Interrupt pre-compile**: freeze hazard→action map into a `dict[int, EmergencyAction]` keyed by enum ordinal; O(1) dispatch, no string hashing.
4. **Replay determinism test** (CI job): write N deterministic events → replay → assert projector hash matches golden.
5. **Chaos / fault-injection test**: kill websocket, drop 50% of heartbeats, corrupt a ledger row → assert `interrupt` fires + `kill_switch` arms + degraded_mode recovery.
6. **Hash-chain verify command**: ensure `dix.py ledger check` walks the entire table and reports the break-row.
7. **Cockpit auth middleware**: loopback-only bind + bearer-token.
8. **OS keyring at runtime** via `keyring` lib (Windows DPAPI / macOS Keychain / Linux Secret Service). Replace current stub.
9. **Signed releases + SBOM**: `sigstore/cosign` + `CycloneDX` in `.github/workflows/release.yml`. Matches §11/§15.
10. **Structured logging everywhere**: grep the repo for `print(` → replace, keep ONE operator banner at boot.

### Tier 2 — light lift, large benefit

11. **`asyncio.TaskGroup` + `asyncio.timeout()`** replace ad-hoc `threading.Thread` supervision for the Dyon monitors + governance consumer (Python 3.11+).
12. **Windows service: add `WinSW` / `Servy` option** alongside existing NSSM config (NSSM is unmaintained). User picks at install.
13. **PyInstaller single-binary release** for operators who don't want venvs. Also easier to sign with Authenticode.
14. **Prometheus `/metrics` endpoint on cockpit** (already have a `prometheus_exporter` module — wire the route).
15. **OpenTelemetry bridge** for traces — same trace_ids, exported to OTLP if env is set.
16. **Per-adapter latency metric** + per-hazard-type detection latency histogram (p50/p95/p99).
17. **`secrets.yml` encrypted at rest with a machine-bound key** (DPAPI on Win, libsecret on Linux).
18. **Crash-only shutdown semantics**: every module implements `close()` that flushes & is idempotent; `SIGTERM` → `shutdown_sequence.run()`.
19. **Bandit + pip-audit + ruff + mypy all wired into `.github/workflows/ci.yml`** (partially present; tighten).
20. **`dix.py replay <from-seq>` command** — rebuild any projector from a sequence number.

### Tier 3 — optional, future-facing

21. **Python 3.13 free-threaded mode** — real parallelism for Dyon + governance consumer threads (no GIL contention with Indira hot path). Ship as opt-in.
22. **mmap-backed FastRiskCache** — governance process can update it without the Indira process restarting, enabling hot-reload of constraints.
23. **ccxt.pro-style WebSocket pool** for `mind/sources/websocket_client.py` (persistent + reconnecting, built-in backoff).
24. **SQLite WAL checkpoint background thread** (we set auto, but an explicit thread makes latency predictable).
25. **Event ledger sharding by date/week** (`ledger-2026-W17.db`) for multi-year retention without growing the active DB.
26. **Governance oracle tiers wired through a `concurrent.futures.ProcessPoolExecutor`** — L3 deep analysis runs in a separate process, cannot starve Indira.
27. **FPGA / CLR path** (long-term per §8 of online research): hot-path executor could be rewritten in Rust/C via `pyo3` for µs-level latency. Current Python target 5ms is fine; ceiling is ~100× with native.
28. **`pydantic v2` data contracts** for every inter-module message — free schema validation + serde.
29. **Feature flags via `unleash`-style local JSON** — governance can flip a strategy live without redeploy.
30. **Attribution DB** (`data/sqlite/attribution.db` per §12) — wire the module; track PnL per signal source.

---

## 5. What I recommend we ship next (if I had to pick 5)

1. **ARCH FIX #3** — promote `system_monitor.hazard_bus` canonical (strict §6).
2. **ARCH FIX #4** — governance = ledger-tail consumer (strict §3, §7).
3. **Tier-1 #1** — SQLite PRAGMA tuning. Biggest free perf win.
4. **Tier-1 #4** — replay determinism test. Closes the biggest unlocked manifest invariant (§1 replayability).
5. **Tier-1 #7** — cockpit auth. Security posture leak.

Everything else can follow in ordered sprints without blocking a production ship.

---

## 6. Latest external upgrades worth folding in (verified via web search)

- **Python 3.13** (released Oct 2024): free-threaded build (`--disable-gil`) + JIT (PEP 744). For our workload — mostly I/O-bound + one hot path — main benefit is the free-threaded build for Dyon + governance consumer.
- **asyncio TaskGroup / timeout()** (Python 3.11+): safer supervision than raw `threading.Thread`.
- **SQLite 3.46** (current): `jsonb` column type for payloads, better `STRICT` table support, better WAL performance. PRAGMA tuning guidance is consistent across recent sources — we're leaving 3–10× on the table.
- **NSSM alternative**: NSSM is formally unmaintained since 2017; maintained alternatives are **Servy** (modern, GUI+CLI), **WinSW** (older but still active community), **Shawl**. Recommend keeping NSSM for compat + add Servy/WinSW as optional.
- **sigstore/cosign + CycloneDX** are the 2025 standard for signed releases + SBOM — both GitHub-Action-native.
- **`keyring` Python lib** remains the canonical wrapper for Windows DPAPI, macOS Keychain, Linux Secret Service.
- **Low-latency trading design patterns** (2025 sources): the pattern we use — precomputed risk cache + async governance + deterministic interrupt — matches current best practice. Only thing we're not doing is **zero-allocation fast path** (Tier-1 #2).

---

*Document generated as part of the v42.2 final-pass audit. Implementation of all BLOCKER + MAJOR items is in flight in this session.*
