# Dependency Roadmap

Single source of truth for which third-party packages land in DIX
VISION, when, and why. The shipping `requirements.txt` /
`requirements-dev.txt` is intentionally minimal — every dependency
is an audit + maintenance + Windows-install cost, and several
"obvious" libraries actively conflict with the system's invariants.

Status as of branch `c` (Dashboard-2026 wave-01 / PR #69):

| File | Direct deps |
|---|---|
| `requirements.txt` | 5 (PyYAML, fastapi, uvicorn, pydantic, websockets) |
| `requirements-dev.txt` | +3 (pytest, ruff, httpx) |

Total = 8 direct deps. Pip pulls a few transitive deps automatically;
none of those are pinned because they are floating contracts of the
direct deps and pinning them creates upgrade hell.

## Adopted now

| Tool | Where | Why |
|---|---|---|
| **`uv`** (Astral) | recommended in README as the preferred install path; pip still works | uv is significantly faster for resolution + install and supports lockfile-based reproducibility. Adopting now is a tooling change, not a code change. The `.bat` launcher continues to use pip for first-run Windows simplicity; advanced users can `uv pip sync requirements-dev.txt`. |

## Deferred (separate PR with explicit user approval)

These all need their own PR with tests, threat-model / replay-equivalence proof, and a feature flag. Adding any of them to wave-01 is scope creep.

| Dep | Replaces | Wave | Why deferred |
|---|---|---|---|
| **`mypy`** | n/a (additive) | wave-02 dev-tooling | Adding the package without `[tool.mypy]` config + CI step + initial type-fix sweep is a no-op. Lands as a focused dev-tooling PR alongside ruff strict-mode bumps. |
| **PostgreSQL + `SQLAlchemy[asyncio]` + `Alembic`** | in-memory + file-backed ledger | "ledger persistence" wave | The ledger is currently deterministic + replayable per INV-15. Migrating to Postgres needs migrations, replay-equivalence tests, hot ↔ cold tier semantics. Not a wave-01 thing. |
| **`redis` (or `aiokafka`)** | in-process event bus | "scaling" wave | Adds a Windows install dependency for a single-operator launcher. No real scaling driver yet. Defer until either multi-process or remote-agent mode lands. |
| **Docker + `docker-compose`** | bare-metal Python venv | alongside Postgres / Redis | Useful when there's actually a stateful service to start. Premature now. |
| **`python-dotenv`** | YAML config | n/a | The system uses YAML for everything that operators edit; secrets go to a vault (future). Adding `.env` would create a third config surface. |
| **`cryptography`** | n/a | wallet vault wave | Pulls a heavy MSVC build dependency on Windows. Not needed for paper trading. Comes with the wallet wave. |
| **`python-jose` / `PyJWT` / `passlib` / `argon2-cffi`** | n/a | auth wave | Single-operator system today. Auth wave is its own PR with explicit threat model first. |
| **`uvloop` + `httptools`** | default uvicorn loop | n/a (probably skip permanently) | Linux-only / WSL-only. Would break the Windows launcher (PR #66). The default loop is fast enough for a single-operator dashboard. |
| **PyO3 + `maturin`** | n/a | polyglot revival wave | Already scaffolded in PRs #15–#21 (paused). Reviving them is the polyglot wave; per the architectural opinion, this happens BEFORE LIVE mode. |
| **`langgraph` + `langchain-core`** | n/a | wave-03 (cognitive) | See [`dashboard_2026_wave03_cognitive_plan.md`](dashboard_2026_wave03_cognitive_plan.md). Authority lint rule **B24** + invariant **INV-67** are already in place to prevent leakage. |
| **`langsmith`** | n/a | wave-03+ (only if self-hosted) | SaaS — covered by wave-03 plan as off-by-default / self-hosted only. |

## Frontend stack — wave-02

The full TypeScript / React stack lives in
[`dashboard_2026_wave03_cognitive_plan.md`](dashboard_2026_wave03_cognitive_plan.md)
and a separate wave-02 plan doc that ships with the wave-02 PR.
Briefly:

| Dep | Wave | Notes |
|---|---|---|
| Vite + React 19 + TypeScript | wave-02 | Vite, **not Next.js**, per architectural opinion (FastAPI already ships JSON; SSR/SSG/RSC don't fit a single-operator real-time terminal). |
| Tailwind CSS + shadcn/ui | wave-02 | Or Radix primitives. |
| TanStack Query | wave-02 | Server-state management for coherence projections. |
| Zustand or Jotai | wave-02 | Lightweight client state. |
| TradingView Lightweight Charts v5 | wave-02 | Per-form widget charts. |
| Zod | wave-02 | Generated FROM Pydantic via `datamodel-code-generator` — never hand-written. Type drift between FastAPI contracts and the React client is the #1 way these systems rot. |
| WebSocket / SSE client | wave-02 | Live coherence projections. |
| `lucide-react` icons | wave-02 | |

## Actively NOT recommended (architectural conflict)

These are great libraries in their own context, but adding them
breaks specific invariants of this system:

| Dep | Conflict |
|---|---|
| **`numpy` / `pandas` / `polars`** | INV-15 (replay determinism). Numeric paths are pure Python by design — pure-Python ops are byte-identical across platforms; numpy can drift on FP across CPU vendors / BLAS versions / kernel-level FMA settings. The very reason this system has reproducible audit ledgers is that it stays out of vectorised libs. If a hot path needs numerics later, it goes to **Rust** (PyO3), not numpy. |
| **`TA-Lib`** | Same FP-determinism problem + a C dependency that breaks Windows wheel installs. If indicators are needed: pure-Python `ta` package, or implement deterministically in Rust as a microstructure plugin. |
| **`pandas-ta`** | Depends on pandas. See above. |
| **`ccxt`** | Adds connectors for ~100 venues we don't use. Per-venue adapters in `execution_engine/adapters/` are smaller, auditable, and registry-bound (SCVS). Would conflict with W1 memecoin isolation if not very carefully wrapped. |
| **`alpaca-py`, `ib_insync`** | Same per-venue pattern — when we wire Alpaca / IBKR, ship a thin SCVS-registered adapter, not a SDK that spans the whole engine. |
| **`langsmith`** in production runtime | SaaS — sending DecisionTrace / patch proposals / chat transcripts to a third-party observability service is a governance violation by default. |

## Why "selectively" matters

The list above isn't an opinion about "good libraries" — most are
great. The system has invariants that make several of them an
active liability:

* **INV-15** rules out non-deterministic numerics.
* **SCVS-01** (no phantom sources) rules out fat SDKs without
  per-venue registration.
* **W1** (memecoin isolation) rules out shared SDKs that span all
  venues.
* **Single-operator scope** + **Windows launcher** rules out
  adding daemons (Postgres / Redis) without a real driver.
* **INV-67** (this PR) quarantines LangChain / LangGraph to
  cognitive subsystems only.
* **INV-56** (Triad Lock) rules out "smart governance" libraries
  that walk graphs to make authority decisions.

Adding a dep that conflicts with one of these isn't a tradeoff to
weigh — it's a regression of an invariant, which means deleting
half the existing audit machinery to accommodate the new dep. Not
worth it.
