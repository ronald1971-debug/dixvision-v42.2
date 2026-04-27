# DIX v42.2 — Canonical Directory Tree (System Reference)

This file is the architectural source of truth for the DIX v42.2 layout.
It is **descriptive of the steady-state shape**, not a refactor instruction
for in-flight phases. Engine directories currently live at the repository
root (e.g. `intelligence_engine/`, `execution_engine/`, …) rather than
under an `engines/` umbrella; both layouts represent the same ENGINE
boundary contracts (`core/contracts/engine.py`, lint rules `B1`/`L1`/
`L2`/`L3`). The umbrella is a documentation convention.

References:

- `manifest.md` — invariants, ENGINE model, GOV-CP-01..07
- `build_plan.md` — phase-by-phase delivery plan (E0..E9)
- `docs/total_recall_index.md` — IND-L01..L31, DYN-L01..L24, HAZ-01..12,
  CORE-01..31, EXEC-01..14, NEUR-01..03, SAFE-01..27, DASH-01..32
- `MAPPING.md` — layer-id → plugin-slot mapping

```
dix_vision/
│
├── core/                                  # CORE-01..31 (foundation)
│   ├── bootstrap_kernel.py                # CORE-02
│   ├── registry.py                        # CORE-03
│   ├── registry_lock.py
│   ├── time_authority.py                  # CORE-08 (T0-04)
│   ├── fast_risk_cache.py                 # CORE-06 (T0-01)
│   ├── translation/                       # CORE-15 (SAFE-25)
│   │   ├── translator.py
│   │   ├── schemas.py
│   │   └── validator.py
│   ├── contracts/                         # Protocol layer (INV-08)
│   │   ├── engine.py
│   │   ├── plugin.py
│   │   ├── event.py
│   │   ├── risk.py
│   │   ├── governance.py
│   │   └── execution.py
│   ├── config/                            # DYN-CFG-01..04
│   │   ├── manager.py
│   │   ├── watcher.py
│   │   ├── versioning.py
│   │   └── fallback.py
│   └── safety/
│       ├── kill_switch.py                 # SAFE-01 / SAFE-09
│       ├── compute_budget.py
│       └── network_guard.py               # SAFE-24
│
├── contracts/                             # Protobuf (PR-14 LOCKED)
│   ├── events.proto                       # EVT-01..04
│   ├── execution.proto
│   ├── governance.proto
│   └── system.proto
│
├── engines/                               # ENGINE MODEL (binding)
│
│   ├── intelligence_engine/               # ENGINE-01 (Indira)
│   │   ├── engine.py
│   │   ├── process.py
│   │   ├── plugin_slots/
│   │   │   ├── microstructure/
│   │   │   ├── alpha/
│   │   │   ├── alt_data/
│   │   │   ├── memory/
│   │   │   ├── multi_timeframe/
│   │   │   ├── transfer/
│   │   │   ├── cognition/
│   │   │   └── agents/
│   │   ├── plugins/                       # IND-L01..L31
│   │   │   ├── market_microstructure.py
│   │   │   ├── rag_engine.py
│   │   │   ├── ral_engine.py
│   │   │   ├── finmem_memory.py
│   │   │   ├── alpha_agent_pool.py
│   │   │   ├── multi_timeframe.py
│   │   │   ├── drl_execution.py
│   │   │   └── ...
│   │   └── intent_producer.py             # CORE-27
│
│   ├── execution_engine/                  # ENGINE-02
│   │   ├── engine.py
│   │   ├── fast_execute.py                # T1-lint-pure
│   │   ├── adapter_router.py              # EXEC-01
│   │   ├── adapters/                      # EXEC-02
│   │   │   ├── binance.py
│   │   │   ├── coinbase.py
│   │   │   ├── kraken.py
│   │   │   └── memecoin/                  # isolated
│   │   ├── protections/
│   │   │   ├── circuit_breaker.py         # SAFE-23
│   │   │   └── slippage_guard.py
│   │   └── feedback/
│   │       ├── slippage.py
│   │       ├── latency.py
│   │       └── fill_rate.py
│
│   ├── learning_engine/                   # ENGINE-03 (offline)
│   │   ├── engine.py
│   │   ├── evaluator.py
│   │   ├── trainer.py
│   │   ├── distillation.py
│   │   ├── experience/
│   │   └── emit_update_event.py
│
│   ├── system_engine/                     # ENGINE-04 (Dyon)
│   │   ├── engine.py
│   │   ├── hazard_sensors/                # HAZ-01..12
│   │   │   ├── ws_timeout.py
│   │   │   ├── exchange_unreachable.py
│   │   │   ├── stale_data.py
│   │   │   ├── memory_overflow.py
│   │   │   └── clock_drift.py
│   │   ├── health_monitors/
│   │   │   ├── heartbeat.py
│   │   │   ├── liveness.py
│   │   │   └── watchdog.py
│   │   ├── anomaly_detection/             # NEUR-02
│   │   │   └── neuromorphic_detector.py
│   │   └── state/
│   │       ├── system_state.py
│   │       └── drift_monitor.py           # CORE-18
│
│   ├── evolution_engine/                  # ENGINE-05
│   │   ├── engine.py
│   │   ├── skill_graph/
│   │   ├── patch_pipeline/
│   │   │   ├── sandbox.py
│   │   │   ├── static_analysis.py
│   │   │   ├── backtest.py
│   │   │   ├── shadow.py
│   │   │   ├── canary.py
│   │   │   └── rollback.py
│   │   └── emit_patch_event.py
│
│   └── governance_engine/                 # ENGINE-06 (AUTHORITY)
│       ├── engine.py
│       ├── control_plane/                 # GOV-CP-01..07
│       │   ├── event_classifier.py        # CP-04
│       │   ├── policy_engine.py           # CP-01
│       │   ├── risk_evaluator.py          # CP-02
│       │   ├── compliance_validator.py    # CP-06
│       │   ├── state_transition.py        # CP-03
│       │   ├── operator_bridge.py         # CP-07
│       │   └── ledger_writer.py           # CP-05
│       ├── modules/                       # GOV-G01..G18
│       │   ├── constraint_loader.py
│       │   ├── emergency_policy.py
│       │   ├── trust_engine.py
│       │   ├── patch_gate.py
│       │   └── audit_replay.py
│       └── fast_risk_updater.py
│
├── state/                                 # LEDGER + DBs
│   ├── ledger/
│   │   ├── append.py                      # LEDGER-01
│   │   ├── hash_chain.py
│   │   ├── indexer.py
│   │   ├── snapshots.py                   # LEDGER-08
│   │   └── reconstructor.py               # CORE-07
│   ├── databases/                         # DB-01..26
│   └── knowledge_store/                   # CORE-24
│
├── registry/                              # REG-01..14 (SOURCE OF TRUTH)
│   ├── plugins.yaml
│   ├── layers.yaml
│   ├── risk.yaml
│   ├── feature_flags.yaml
│   ├── strategies.yaml
│   └── enforcement_policies.yaml
│
├── cockpit/                               # COCKPIT SYSTEM
│   ├── api/
│   ├── websocket/
│   ├── voices/
│   │   ├── indira.py
│   │   ├── dyon.py
│   │   ├── governance.py
│   │   └── devin.py
│   └── reflection/
│
├── dashboard/                             # DASHBOARD OS
│   ├── os_layer/
│   │   ├── mode_manager.py                # MANUAL / SEMI / AUTO
│   │   ├── session_controller.py
│   │   ├── operator_gate.py               # INV-12 enforcement
│   │   └── state_sync.py
│   │
│   ├── layouts/
│   │   ├── default_4pane.py               # DASH-32
│   │   ├── memecoin_tab.py                # DASH-27
│   │   └── advanced_workspace.py
│   │
│   ├── widgets/
│   │   ├── decision_trace.py              # DASH-04
│   │   ├── risk_view.py                   # DASH-05
│   │   ├── portfolio_view.py              # DASH-06
│   │   ├── system_health.py               # DASH-07
│   │   ├── governance_panel.py            # DASH-08
│   │   ├── latency_monitor.py             # DASH-10
│   │   └── plugin_manager.py
│   │
│   ├── trading_modes/
│   │   ├── manual_mode.py
│   │   ├── semi_auto_mode.py
│   │   └── full_auto_mode.py
│   │
│   └── memecoin/                          # FULL ISOLATION
│       ├── sniper.py                      # EXEC-12
│       ├── copy_trader.py                 # EXEC-13
│       ├── signal_trader.py               # EXEC-14
│       ├── safety_stack.py                # SAFE-13
│       └── burner_wallet.py               # INV-20
│
├── sensory/                               # NEURO + WEB AUTOLEARN
│   ├── neuromorphic/
│   │   ├── indira_signal.py               # NEUR-01
│   │   ├── dyon_anomaly.py                # NEUR-02
│   │   └── governance_risk.py             # NEUR-03
│   └── web_autolearn/
│       ├── crawler.py
│       ├── filter.py
│       ├── curator.py
│       └── approval_queue.py
│
├── execution/                             # SHARED INFRA (non-engine logic)
│   ├── async_bus.py                       # EXEC-05
│   ├── event_emitter.py                   # EXEC-04
│   ├── severity_classifier.py             # EXEC-06
│   └── chaos_engine.py                    # EXEC-07
│
├── scripts/
│   ├── profile_hot_path.py                # CI-10
│   ├── verify.py
│   └── dix_cli.py                         # plugin + mode control
│
├── tests/                                 # TEST-01..20
│   ├── test_replay.py
│   ├── test_hazard_flow.py
│   ├── test_latency.py
│   ├── test_governance.py
│   ├── test_neuromorphic.py
│   └── ...
│
├── deploy/
│   ├── setup.ps1
│   ├── dix-update.bat
│   ├── docker/
│   └── service/
│
├── immutable_core/
│   ├── foundation.py
│   ├── foundation.hash
│   ├── safety_axioms.lean
│   ├── hazard_axioms.lean
│   └── neuromorphic_axioms.lean
│
└── VERSION
```

## Notes on the current code layout vs. this tree

* Engine packages currently live at the repo root (`intelligence_engine/`,
  `execution_engine/`, `learning_engine/`, `system_engine/`,
  `evolution_engine/`, `governance_engine/`). They are imported under those
  paths everywhere — `core/contracts/engine.py`, `tools/authority_lint.py`
  (rules `B1`, `L1`, `L2`, `L3`), `tests/`, `ui/server.py`. This is an
  identical model to placing them under `engines/`; the umbrella is a
  documentation convention, not a code change.
* `dashboard/`, `sensory/`, `immutable_core/` are reserved namespaces in
  the spec. Their initial implementations land in later phases
  (E6/E7/E8) per `build_plan.md`. Until they ship, leaving them out of
  the code tree is the correct state — they would otherwise be empty
  packages that the lint and import tools would flag.
* `state/ledger/` is implemented as a flat package
  (`state/ledger/store.py`, `state/ledger/reader.py`,
  `state/ledger/__init__.py`) at Phase E3. The spec breakdown
  (`append.py` / `hash_chain.py` / `indexer.py` / `snapshots.py` /
  `reconstructor.py`) is the steady-state shape for E5+.
