# DIX VISION v42.2 — Janus-Sentinel

**Version:** 42.2.0
**Status:** PRODUCTION_READY_DEPLOYABLE
**Architecture:** Contract-First + Dual-Domain Split Authority + Event-Sourced Ledger + Fast Path + Hazard Interrupt

## Architectural Truth (locked)

| Component   | Role                                                       |
|-------------|------------------------------------------------------------|
| INDIRA      | Market intelligence + decision + trade execution authority |
| DYON        | System monitoring + hazard detection (sensor only)         |
| GOVERNANCE  | Control plane — defines rules, never in hot path           |
| LEDGER      | Immutable append-only memory (all events, all time)        |
| INTERRUPT   | Deterministic emergency-action path                        |

**Triggers are NOT authority. Policies ARE authority.**

## Quick start (Linux / macOS)

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python startup_test.py        # smoke test — 15 checks
python tests/test_all.py      # full integration suite
python dix.py verify          # foundation + governance gate, exit
python main.py --dev          # run the system (Ctrl+C to stop)
```

## Quick start (Windows)

```powershell
.\run.ps1          # installs deps in .venv and launches main.py
.\build.ps1        # creates redistributable zip
.\windows\installer\setup.ps1   # installs NSSM service + tray
```

## CLI

```
python dix.py start [--dev]   # start system
python dix.py verify          # verify integrity only
python dix.py status          # print current state snapshot
python dix.py ledger check    # verify event chain integrity
```

## Directory map (canonical)

```
immutable_core/   # Lean4-verified axioms, foundation hash, kill switch
core/             # Contracts, registry, bootstrap, runtime context
mind/             # INDIRA — market reasoning + fast-path execution
execution/        # Shared execution engine + exchange adapters + hazard bus
system_monitor/   # DYON — sensors, hazard detection, emitters
governance/       # Control plane — kernel, risk, policy, modes, oracles
interrupt/        # Deterministic emergency executor
state/ledger/     # Append-only event store with hash-chained integrity
state/projectors/ # Read-side projections (market/portfolio/system/governance)
enforcement/      # Runtime guardian, decorators, resource enforcer
translation/      # Typed intent model + validator (no free-text execution)
observability/    # Prometheus, structured logging, tracing, alerts, cockpit
security/         # Secrets, keyring (DPAPI), encryption, auth, audit trail
system/           # Config, state, time, logger, audit, fast risk cache
windows/          # NSSM service, PowerShell installer, tray app, updater
tests/            # Integration tests
data/             # Runtime data (ledger.db, audit.jsonl, incidents.jsonl, ...)
```

## Safety axioms (Lean4 verified)

- `max_drawdown = 4%` (hard floor; operator override ≤ 15%)
- `max_loss_per_trade = 1%`
- `fail_closed = true`
- Credentials never leave the machine
- Every event is deterministic + replayable
- Runtime core mutation, martingale, unbounded leverage: forbidden

## Flow summary

```
Market flow:   Market Data  → Indira      → Execution → Ledger
System flow:   Telemetry    → Dyon        → Governance → System Action → Ledger
Hazard flow:   Dyon         → SYSTEM_HAZARD → Governance → Interrupt Executor → Ledger
```
