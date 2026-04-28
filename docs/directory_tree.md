# DIX v42.2 — Canonical Directory Tree (System Reference, v3.2)

This file is the architectural source of truth for the DIX v42.2 layout. It
**describes the steady-state shape** of the repository — every directory
and module that is canonical under the v42.2 specification, regardless of
whether it is implemented yet.

This is **v3.2 of the canonical tree**, integrating:

1. `manifest.md §A` (engine-led layout) — the binding base
2. The 22 addon directives (Coherence Layer, Mode Engine, Drift Oracle,
   Causal Graph, Meta-Adaptation Bridge, Dashboard OS, hard 3-domain
   isolation, drift killers, plugin budgets, dual-speed system, …)
3. The 10 institutional-grade additions (A–J): Portfolio Brain,
   Strategy Orchestrator, Execution Lifecycle FSM, Market Data
   Normalizer, Simulation Engine, Real-Time Risk Engine, Performance &
   Alpha-Decay Tracking, Data Versioning, Strategy Registry split,
   Operator Audit
4. The 20 extras directives (operator decisions A1 / B1 / C2 / D1 / E1 / F1):
   - Tier 1 follow-ons after Phase 6: Belief State, Pressure Vector,
     Meta-Controller, Confidence Engine, Reward Shaping
   - `agents/` namespace alongside `plugins/` (per C2)
   - **Phase 10: Intelligence Depth Layer** (per E1, after Phase 9):
     Simulation vPro, Trader Intelligence System (full F1), Macro Regime
     Engine, Cross-Asset Coupling, Strategic Execution + Market Impact,
     trader-intelligence proto contract
5. The v3.1 fold-in (operator decisions G1 / G2 / G3 / G4):
   - **System Intent Engine** (read-only projection in `core/coherence/`,
     operator-written via GOV-CP-07) — Phase 6.T1d
   - **Opponent Model** (`intelligence_engine/opponent_model/`,
     extends Trader Intelligence) — Phase 10.10
   - **Reflexive Simulation Layer** (`simulation/reflexive_layer/`,
     market-reacts-to-you) — Phase 10.11
   - **Strategy Genetics** (`evolution_engine/genetic/`,
     mutation/crossover/inheritance) — Phase 10.12
   - **Regret / Counterfactual Memory** (`state/memory_tensor/regret/`,
     missed-opportunity tracking) — Phase 10.13
   - **Internal Debate Round** (`meta_controller/evaluation/debate_round.py`,
     deterministic agent stance scoring — NOT meta-RL) — Phase 10.14
   - **Time Hierarchy + Dynamic Identity** doctrine (manifest §X,
     no new modules — emergent property of existing FSMs)
6. The v3.2 stress-stabilization (operator decisions I1 / I2 / I3 /
   I4 / I5 / I6 / I7):
   - **Meta-Controller `O(1)` fallback lane** (`FALLBACK_POLICY` +
     `_fallback_lane()` in `meta_controller/policy/execution_policy.py`,
     INV-48) — Phase 6.T1b
   - **Regime hysteresis activation** (extends `regime_detector.py` +
     new `registry/regime_hysteresis.yaml`, INV-49) — Phase 6.T1e
   - **Cross-signal entropy in Pressure Vector `uncertainty`**
     (`pressure_vector.py` derivation, INV-50, +
     `registry/pressure.yaml`) — Phase 6.T1a
   - **Typed `agent_context` schema** (`SignalEvent.agent_context:
     Mapping[str, str]` + `registry/agent_context_keys.yaml` allowlist,
     B15) — Phase 10.8
   - **Richer `SimulationOutcome` payload** (`failure_modes`,
     `regime_performance_map`, `adversarial_breakdowns` —
     `simulation/strategy_arena/simulation_outcome.py`) — Phase 10.1
   - **Archetype lifecycle** (`{state, decay_rate, performance_score}`
     in `registry/trader_archetypes.yaml` +
     `intelligence_engine/strategy_runtime/archetype_lifecycle.py`,
     INV-51) — Phase 10.2–10.4
   - **PolicyEngine constant-time decision table** (`I7` reframed —
     internal precompile in `governance_engine/control_plane/
     policy_engine.py`, no parallel approval path) — Phase 7

References:

- `manifest.md` — invariants, ENGINE-01..06 model, GOV-CP-01..07,
  PLUGIN-ACT-01..07, authority lint rules
- `build_plan.md` — phase-by-phase delivery plan (E0..E9 + v2 steps 8..13)
- `docs/total_recall_index.md` — IND-L01..L31, DYN-L01..L24, HAZ-01..12,
  CORE-01..31, EXEC-01..14, NEUR-01..03, SAFE-01..27, DASH-01..32
- `MAPPING.md` — layer-id → plugin-slot mapping

Annotation legend:

- **[EXISTS]** — present on `main` today (Phases 0–5 shipped)
- **[NEW v1]** — added by the 22 addons (System Coherence Layer,
  Dashboard OS, hard 3-domain isolation, drift killers)
- **[NEW v2-A..J]** — added by the 10 institutional-grade additions
- **[NEW v3-T1]** — Tier 1 extras follow-on (after Phase 6, fits inside
  existing engines, no spec change)
- **[NEW v3-P10]** — Phase 10 Intelligence Depth Layer (extras Tier 2,
  formal phase append after Phase 9)
- **[NEW v3.1]** — v3.1 fold-in (Intent Engine, Opponent Model,
  Reflexive Sim, Strategy Genetics, Regret Memory, Internal Debate)
- **[NEW v3.2]** — v3.2 stress-stabilization (fallback lane,
  hysteresis, entropy uncertainty, agent_context schema, richer
  simulation outcome, archetype lifecycle, PolicyEngine constant-time table)
- otherwise — canonical per `manifest.md §A`, not yet implemented

```text
dixvision-v42.2/
├── README.md                                                  # INFRA-05  [EXISTS]
├── pyproject.toml                                             # INFRA-01  [EXISTS]
├── VERSION                                                    # INFRA-04
├── .github/workflows/
│   ├── ci.yml                                                 # CI-01     [EXISTS]
│   ├── release.yml                                            # CI-02
│   ├── rust.yml                                               # CI-03 (deferred)
│   └── sandbox.yml                                            # CI-04, TEST-17
│
├── contracts/                                                 # PR-14, INV-08
│   ├── events.proto                                           # EVT-01..04  [EXISTS]
│   ├── execution.proto
│   ├── governance.proto
│   ├── ledger.proto
│   ├── market.proto
│   ├── system.proto
│   └── trader_intelligence.proto                              # [NEW v3-P10] TraderProfile, StrategyAtom, ComposedStrategy, MetaControllerState
│
├── core/
│   ├── __init__.py                                            # [EXISTS]
│   ├── bootstrap_kernel.py                                    # CORE-02
│   ├── registry.py                                            # CORE-03
│   ├── secrets.py                                             # DEPLOY-11
│   ├── contracts/                                             # CORE-04
│   │   ├── __init__.py                                        # [EXISTS]
│   │   ├── engine.py                                          # ENGINE-01..06 protocols  [EXISTS]
│   │   ├── events.py                                          # EVT pydantic             [EXISTS]
│   │   ├── market.py                                          # MarketTick               [EXISTS]
│   │   ├── risk.py                                            # IRiskCache, IRiskConstraints
│   │   ├── ledger.py                                          # ILedger
│   │   ├── governance.py                                      # IGovernanceHazardSink, SystemMode enum
│   │   └── execution.py                                       # IExecutionAdapter
│   │
│   └── coherence/                                             # [NEW v1] System Coherence Layer (addon §1)
│       ├── __init__.py                                        # [NEW v1]
│       ├── engine.py                                          # SCL-01 — global interpretation
│       ├── causal_graph.py                                    # SCL-02 — trade→outcome→update edges
│       ├── mode_engine.py                                     # SCL-03 — read-only Protocol/types (only Governance writes mode)
│       ├── drift_oracle.py                                    # SCL-04 — DRIFT_VECTOR computation
│       ├── meta_adaptation.py                                 # SCL-05 — Learning↔Evolution unifier
│       ├── belief_state.py                                    # [NEW v3-T1] BELIEF_STATE_VECTOR (regime, vol, liq, conf, hypotheses) — frozen, read-only projection
│       ├── performance_pressure.py                            # [NEW v3-T1] PRESSURE_VECTOR (perf/risk/drift/latency/uncertainty) — derived from existing sensors
│       └── system_intent.py                                   # [NEW v3.1] INTENT_VECTOR (objective, focus, risk_mode, horizon) — read-only; operator proposes via GOV-CP-07, state_transition_manager (GOV-CP-03) writes IntentTransition event
│
├── immutable_core/                                            # SAFE-06, axioms
│   ├── foundation.hash                                        # SAFE-06
│   ├── kill_switch.py                                         # CORE-09
│   ├── safety_axioms.lean                                     # S1..S10
│   ├── hazard_axioms.lean                                     # H1..H10
│   ├── neuromorphic_axioms.lean                               # N1..N8
│   └── system_identity.py                                     # CORE-13
│
├── intelligence_engine/                                       # ENGINE-01 (Indira)  [EXISTS]
│   ├── __init__.py                                            # [EXISTS]
│   ├── engine.py                                              # [EXISTS]
│   ├── charter/indira.py                                      # CORE-30
│   ├── intent_producer.py                                     # CORE-27
│   ├── signal_pipeline.py                                     # IND-SP-01 [EXISTS, Phase 3]
│   ├── learning_interface.py                                  # IND-LI-01 [EXISTS, Phase 3]
│   ├── plugins/                                               # IND-L0x stateless feature plugins (existing taxonomy)
│   │   ├── __init__.py                                        # [EXISTS]
│   │   ├── microstructure/                                    # IND-L02, L22, L23, L24
│   │   │   ├── __init__.py                                    # [EXISTS]
│   │   │   └── microstructure_v1.py                           # IND-L02 v1  [EXISTS]
│   │   ├── alpha/                                             # IND-L18, L15, L19
│   │   ├── alt_data/                                          # IND-L08, L03
│   │   ├── memory/                                            # IND-L21, L06, L26
│   │   ├── multi_timeframe/                                   # IND-L20
│   │   ├── transfer/                                          # IND-L27, L28
│   │   ├── cognition/                                         # IND-L10, L11, L13
│   │   └── agent/                                             # IND-L14, L16, L17
│   ├── agents/                                                # [NEW v3-P10, per C2] specialised stateful agents (distinct from stateless plugins)
│   │   ├── __init__.py
│   │   ├── scalper.py                                         # AGT-01 — high-frequency intra-bar agent
│   │   ├── swing_trader.py                                    # AGT-02 — multi-bar swing agent
│   │   ├── macro.py                                           # AGT-03 — macro/regime-driven agent
│   │   ├── liquidity_provider.py                              # AGT-04 — passive liquidity agent
│   │   └── adversarial_observer.py                            # AGT-05 — read-only adversarial probe (no orders)
│   ├── portfolio/                                             # [NEW v2-A] Portfolio Brain — coordinated portfolio
│   │   ├── allocator.py                                       # capital allocation across strategies
│   │   ├── exposure_manager.py                                # cross-asset exposure control
│   │   ├── correlation_engine.py                              # correlation + clustering
│   │   ├── risk_parity.py                                     # portfolio balancing
│   │   └── capital_scheduler.py                               # capital rotation logic
│   ├── strategy_runtime/                                      # [NEW v2-B] Strategy Orchestrator [EXISTS, Phase 3]
│   │   ├── orchestrator.py                                    # IND-ORC-01 — regime+lifecycle gating  [EXISTS]
│   │   ├── scheduler.py                                       # IND-SCH-01 — bar-aligned cadence    [EXISTS]
│   │   ├── regime_detector.py                                 # IND-REG-01 — runtime regime tags    [EXISTS] (extended in 6.T1e for INV-49 hysteresis [NEW v3.2])
│   │   ├── archetype_lifecycle.py                             # ARCH-LC-01 — {state, decay_rate, performance_score} per archetype; offline-only auto-demotion [NEW v3.2 — INV-51]
│   │   ├── state_machine.py                                   # IND-SLM-01 — strategy lifecycle FSM [EXISTS]
│   │   └── conflict_resolver.py                               # IND-CFR-01 — resolves conflicting signals [EXISTS]
│   ├── meta_controller/                                       # [NEW v3-T1] Meta-Controller (sits BETWEEN orchestrator and conflict_resolver; per B1 keeps both). v3.1 sub-package layout per H1 (audit separation, NOT a new engine boundary)
│   │   ├── __init__.py
│   │   ├── perception/                                        # [NEW v3.1] Regime / context perception
│   │   │   ├── __init__.py
│   │   │   └── regime_router.py                               # MC-01 — routes by Belief State regime
│   │   ├── evaluation/                                        # [NEW v3.1] Selection + confidence + debate
│   │   │   ├── __init__.py
│   │   │   ├── strategy_selector.py                           # MC-02 — picks eligible strategies
│   │   │   ├── confidence_engine.py                           # MC-03 — composite confidence (Sharpe + Bayesian + Entropy + Stability + Alignment + safety mods)
│   │   │   └── debate_round.py                                # [NEW v3.1] MC-06 — deterministic stance/scoring round across agents (NOT meta-RL); feeds confidence_engine
│   │   ├── allocation/                                        # [NEW v3.1] Position sizing
│   │   │   ├── __init__.py
│   │   │   └── position_sizer.py                              # MC-04 — Kelly / vol-target / pressure-adjusted size
│   │   └── policy/                                            # [NEW v3.1] Final SKIP / SHADOW / EXECUTE gate
│   │       ├── __init__.py
│   │       └── execution_policy.py                            # MC-05 — final SKIP / SHADOW / EXECUTE decision; precomputed FALLBACK_POLICY + _fallback_lane() returned when latency budget exceeded or upstream stale [NEW v3.2 — INV-48]
│   ├── macro/                                                 # [NEW v3-P10] Macro Regime Engine
│   │   ├── regime_classifier.py                               # MAC-01 — HMM/Bayesian regime switching
│   │   ├── hidden_state_detector.py                           # MAC-02 — latent state inference
│   │   ├── latent_embedder.py                                 # MAC-03 — deterministic offline embeddings
│   │   └── macro_event_aligner.py                             # MAC-04 — aligns macro releases to bars
│   ├── cross_asset/                                           # [NEW v3-P10] Cross-Asset Coupling
│   │   ├── correlation_matrix.py                              # XAS-01 — rolling correlation
│   │   ├── lead_lag.py                                        # XAS-02 — lead/lag detection
│   │   ├── contagion_detector.py                              # XAS-03 — cross-asset shock propagation
│   │   └── basket_constructor.py                              # XAS-04 — synthetic basket builder
│   ├── opponent_model/                                        # [NEW v3.1] Real-time opponent / crowd modelling (extends Trader Intelligence)
│   │   ├── __init__.py
│   │   ├── behavior_predictor.py                              # OPP-01 — predicts likely trader actions from microstructure
│   │   ├── crowd_density.py                                   # OPP-02 — estimates positioning crowdedness
│   │   └── strategy_detector.py                               # OPP-03 — infers in-market strategy populations
│   └── meta/                                                  # [NEW v3-P10] Trader Intelligence consumer (reads archetypes, synthesises strategies)
│       ├── trader_archetypes.py                               # TI-CONS-01 — loads registry/trader_archetypes.yaml
│       ├── strategy_synthesizer.py                            # TI-CONS-02 — composes archetypes into ComposedStrategy
│       └── archetype_arena.py                                 # TI-CONS-03 — Darwinian capital competition between archetypes
│
├── execution_engine/                                          # ENGINE-02  [EXISTS]
│   ├── __init__.py                                            # [EXISTS]
│   ├── engine.py                                              # [EXISTS]
│   ├── hot_path/
│   │   ├── time_authority.py                                  # CORE-08, T0-04
│   │   ├── fast_risk_cache.py                                 # CORE-06, T0-01
│   │   └── fast_execute.py                                    # EXEC-11, T1-pure  [EXISTS, Phase 2]
│   ├── adapters/
│   │   ├── __init__.py                                        # [EXISTS]
│   │   ├── base.py                                            # EXEC-02   [EXISTS]
│   │   ├── paper.py                                           #          [EXISTS]
│   │   ├── router.py                                          # EXEC-01 hard-domain router  [EXISTS, Phase 2]
│   │   ├── binance.py · coinbase.py · kraken.py
│   │   ├── oanda.py · ig.py · ibkr.py · alpaca.py
│   │   └── memecoin/                                          # EXEC-12..14 (separate-process candidate)
│   ├── protections/
│   │   ├── circuit_breaker.py                                 # T0-08, SAFE-23
│   │   ├── runtime_monitor.py                                 # EXEC-08  [EXISTS, Phase 2]
│   │   ├── reconciliation.py                                  # EXEC-10
│   │   └── feedback.py                                        # EXEC-09  [EXISTS, Phase 5]
│   ├── lifecycle/                                             # [NEW v2-C] Order State Machine — real broker realism  [EXISTS, Phase 2]
│   │   ├── order_state_machine.py                             # EXEC-LC-01 FSM: NEW→PENDING→PARTIAL→FILLED→CLOSED→ERROR
│   │   ├── fill_handler.py                                    # EXEC-LC-02
│   │   ├── sl_tp_manager.py                                   # EXEC-LC-03 stop-loss / take-profit lifecycle
│   │   ├── retry_logic.py                                     # EXEC-LC-04
│   │   └── partial_fill_resolver.py                           # EXEC-LC-05
│   ├── market_data/                                           # [NEW v2-D] Canonical market state (replay==live)
│   │   ├── normalizer.py
│   │   ├── aggregator.py
│   │   ├── latency_tracker.py
│   │   └── book_builder.py
│   ├── strategic_execution/                                   # [NEW v3-P10] Strategic execution + market impact
│   │   ├── __init__.py
│   │   ├── adversarial_executor.py                            # SE-01 — game-theoretic order placement
│   │   ├── optimal_execution.py                               # SE-02 — Almgren-Chriss style optimal trajectory
│   │   └── market_impact/
│   │       ├── model.py                                       # SE-03 — square-root impact model
│   │       ├── depth_estimator.py                             # SE-04 — book-depth estimator
│   │       └── slippage_curve.py                              # SE-05 — historical slippage curve fitter
│   └── domains/                                               # [NEW v1] Hard 3-domain isolation
│       ├── __init__.py
│       ├── normal/                                            # standard Indira+execution
│       ├── copy_trading/                                      # external wallet mirror, isolated
│       └── memecoin/                                          # burner wallet, strict caps, isolated process
│
├── learning_engine/                                           # ENGINE-03 (offline)  [EXISTS]
│   ├── __init__.py                                            # [EXISTS]
│   ├── engine.py                                              # [EXISTS]
│   ├── lanes/
│   │   ├── self_learning_loop.py                              # IND-L04
│   │   ├── ral.py                                             # IND-L07
│   │   ├── policy_distillation.py                             # IND-L12
│   │   ├── continual_distillation.py                          # DYN-L22
│   │   ├── federated.py                                       # IND-L31, DYN-L24
│   │   ├── experience_base.py                                 # IND-L30, DYN-L23
│   │   └── patch_outcome_feedback.py                          # DYN-L02
│   ├── update_emitter.py                                      # → GOV-G18  [EXISTS, Phase 5]
│   ├── trader_abstraction/                                    # [NEW v3-P10] Trader Intelligence learning side
│   │   ├── __init__.py
│   │   ├── extractor.py                                       # TI-LRN-01 — extracts behaviour primitives
│   │   ├── normalizer.py                                      # TI-LRN-02 — schema-normalises into TraderProfile
│   │   ├── encoder.py                                         # TI-LRN-03 — encodes into StrategyAtom
│   │   └── embedder.py                                        # TI-LRN-04 — deterministic offline embedding (fixed seed + checkpoint, ledgered)
│   └── performance_analysis/                                  # [NEW v2-G] Alpha decay + execution quality
│       ├── alpha_decay.py
│       ├── execution_quality.py
│       ├── slippage_analysis.py
│       ├── latency_impact.py
│       ├── pnl_attribution.py
│       └── reward_shaping.py                                  # [NEW v3-T1] kills naive PnL=reward; risk-adjusted reward composition
│
├── system_engine/                                             # ENGINE-04 (Dyon)  [EXISTS]
│   ├── __init__.py                                            # [EXISTS]
│   ├── engine.py                                              # [EXISTS]
│   ├── charter/dyon.py                                        # CORE-30
│   ├── hazard_sensors/                                        # HAZ-01..12
│   │   ├── sensor_array.py
│   │   ├── ws_timeout.py · exchange_unreachable.py · stale_data.py
│   │   ├── memory_overflow.py · clock_drift.py
│   │   ├── neuromorphic_detector.py                           # NEUR-02 (rule-based stub v1)
│   │   ├── market_anomaly.py
│   │   └── system_anomaly.py
│   ├── health_monitors/
│   │   ├── heartbeat.py · liveness.py · watchdog.py
│   │   ├── api_changelogs.py · github_trending.py · stack_overflow.py
│   │   └── repo_discovery.py
│   └── state/
│       ├── system_state.py
│       ├── drift_monitor.py                                   # CORE-18 (feeds core/coherence/drift_oracle.py)
│       ├── homeostasis.py                                     # CORE-19
│       ├── anomaly_detector.py                                # CORE-20
│       ├── runtime_guardian.py                                # CORE-10
│       └── kill_switch_runtime.py                             # T0-09
│
├── evolution_engine/                                          # ENGINE-05 (offline)  [EXISTS]
│   ├── __init__.py                                            # [EXISTS]
│   ├── engine.py                                              # [EXISTS]
│   ├── intelligence_loops/                                    # DYN-L01, L03..L08, L19
│   ├── skill_graph/                                           # DYN-L14..L17
│   ├── genetic/                                               # [NEW v3.1] Strategy Genetics — patch-pipeline-gated
│   │   ├── __init__.py
│   │   ├── mutation_operators.py                              # GEN-01 — parameter / structural mutations
│   │   ├── crossover.py                                       # GEN-02 — strategy crossover
│   │   └── fitness_inheritance.py                             # GEN-03 — inherited fitness accounting
│   └── patch_pipeline/                                        # GOV-G18, EXEC-15, DYN-L18, L21
│       ├── pipeline.py
│       ├── sandbox.py · static_analysis.py · backtest.py
│       ├── shadow.py · canary.py · rollback.py
│       └── critique_loop.py
│
├── governance_engine/                                         # ENGINE-06  [EXISTS]
│   ├── __init__.py                                            # [EXISTS]
│   ├── engine.py                                              # [EXISTS]
│   ├── charter/governance.py                                  # CORE-30
│   ├── control_plane/                                         # GOV-CP-01..07
│   │   ├── policy_engine.py                                   # GOV-CP-01 — v3.2: precompiles a frozen O(1) decision table at __init__; emits POLICY_TABLE_INSTALLED ledger row; fail-closed on hash mismatch (SAFE-47) [NEW v3.2 — I7 reframed]
│   │   ├── risk_evaluator.py                                  # GOV-CP-02
│   │   ├── state_transition_manager.py                        # GOV-CP-03 (only writer of system mode)
│   │   ├── event_classifier.py                                # GOV-CP-04
│   │   ├── ledger_authority_writer.py                         # GOV-CP-05
│   │   ├── compliance_validator.py                            # GOV-CP-06
│   │   └── operator_interface_bridge.py                       # GOV-CP-07
│   ├── services/                                              # adjacent (non-pipeline)
│   │   ├── trust_engine.py                                    # GOV-G13
│   │   ├── liveness_watchdog.py                               # GOV-G09
│   │   ├── triple_window_dry_run.py                           # GOV-G15
│   │   ├── overconfidence_guardrail.py                        # GOV-G17
│   │   ├── audit_replay.py                                    # GOV-G17
│   │   └── patch_pipeline.py                                  # GOV-G18
│   ├── plugin_lifecycle/                                      # PLUGIN-ACT-01..07
│   │   ├── registry_loader.py
│   │   ├── activation_gate.py
│   │   ├── lifecycle_emitter.py
│   │   └── hot_reload_signal.py
│   └── risk_engine/                                           # [NEW v2-F] Real-time risk evaluator (cache ≠ intelligence)
│       ├── real_time_risk.py
│       ├── position_limits.py
│       ├── drawdown_guard.py
│       ├── exposure_limits.py
│       └── kill_conditions.py
│
├── sensory/                                                   # NEUR-01..04, WEBLEARN-01..10
│   ├── neuromorphic/
│   │   ├── indira_signal.py                                   # NEUR-01
│   │   ├── dyon_anomaly.py                                    # NEUR-02
│   │   └── governance_risk.py                                 # NEUR-03
│   └── web_autolearn/
│       ├── crawler.py                                         # WEBLEARN-01 (Playwright)
│       ├── ai_filter.py                                       # WEBLEARN-02
│       ├── curator.py                                         # WEBLEARN-03
│       ├── pending_buffer.py                                  # WEBLEARN-04, HITL-07
│       ├── seeds.yaml                                         # WEBLEARN-10
│       └── trader_intelligence/                               # [NEW v3-P10] Trader Intelligence ingestion side
│           ├── __init__.py
│           ├── crawler.py                                     # TI-ING-01 — crawls trader profiles (governed seed list)
│           ├── profile_extractor.py                           # TI-ING-02 — pulls structured trader behaviour
│           ├── behavior_analyzer.py                           # TI-ING-03 — derives behaviour primitives
│           ├── performance_validator.py                       # TI-ING-04 — validates against verified PnL
│           └── archetype_publisher.py                         # TI-ING-05 — publishes WEB_SIGNAL_EVENT for HITL gate
│
├── state/
│   ├── __init__.py                                            # [EXISTS]
│   ├── ledger/
│   │   ├── __init__.py                                        # [EXISTS]
│   │   ├── reader.py                                          # LEDGER-stub  [EXISTS]
│   │   ├── append.py · event_store.py                         # LEDGER-01
│   │   ├── hot_store.py · cold_store.py                       # LEDGER-03..04
│   │   ├── hash_chain.py · indexer.py                         # LEDGER-05
│   │   ├── integrity.py · event_types.py                      # LEDGER-06..07
│   │   ├── snapshots.py                                       # LEDGER-08
│   │   └── reconstructor.py                                   # CORE-07
│   ├── databases/                                             # DB-01..26
│   ├── knowledge_store.py                                     # CORE-24, T0-11
│   ├── memory_tensor/                                         # [NEW v1] Unified market+decision+system+outcome
│   │   ├── episodic.py                                        # episodic memory (per-trade outcomes)
│   │   ├── semantic.py                                        # semantic memory (market knowledge)
│   │   ├── procedural.py                                      # procedural memory (strategy procedures)
│   │   ├── meta_memory.py                                     # meta memory (what works vs doesn't)
│   │   ├── trader_patterns/                                   # [NEW v3-P10] persisted TraderProfile + StrategyAtom store
│   │   │   ├── profile_store.py
│   │   │   ├── atom_store.py
│   │   │   └── archetype_store.py
│   │   └── regret/                                            # [NEW v3.1] Regret / counterfactual memory
│   │       ├── __init__.py
│   │       ├── missed_opportunity.py                          # RGT-01 — paths not taken (renamed from REG-01 to avoid collision with registry/ REG-01..14)
│   │       ├── almost_trades.py                               # RGT-02 — near-miss tracking
│   │       └── regret_log.py                                  # RGT-03 — append-only regret events
│   └── data_versioning/                                       # [NEW v2-H] Snapshot + feature versioning
│       ├── market_snapshots.py
│       ├── feature_store.py
│       └── dataset_registry.py
│
├── registry/                                                  # REG-01..14, source of truth
│   ├── plugins.yaml                                           # PLUGIN-ACT-01  [EXISTS]
│   ├── engines.yaml                                           # REG-02         [EXISTS]
│   ├── layers.yaml · risk.yaml · feature_flags.yaml
│   ├── enforcement_policies.yaml · governance_ruleset.yaml · alerts.yaml
│   ├── budgets.yaml                                           # [NEW v1] plugin budgets per engine
│   ├── trader_archetypes.yaml                                 # [NEW v3-P10] 30 seed traders × 5 dimensions → 300 archetypes catalog (v3.2: each row also declares {state, decay_rate, performance_score} — INV-51)
│   ├── agents.yaml                                            # [NEW v3-P10] agent registry (scalper / swing / macro / liquidity / adversarial)
│   ├── agent_context_keys.yaml                                # [NEW v3.2] allowlist of typed `SignalEvent.agent_context` keys (B15) — horizon / conviction_type / memory_ref / regime_assumption / confidence_band
│   ├── regime_hysteresis.yaml                                 # [NEW v3.2] persistence_ticks + confidence_delta thresholds for INV-49 hysteresis
│   ├── pressure.yaml                                          # [NEW v3.2] α, β coefficients for entropy-aware uncertainty + entropy_high_water + entropy_high_water_modifier (INV-50, SAFE-43)
│   └── strategies/                                            # [NEW v2-I] Strategy registry split
│       ├── definitions.yaml
│       ├── lifecycle.yaml
│       └── performance.yaml
│
├── translation/                                               # CORE-15, SAFE-25
│   ├── intent_to_patch.py
│   ├── round_trip_validator.py
│   └── audit_writer.py                                        # DB-14
│
├── enforcement/
│   ├── decorators.py                                          # CORE-11
│   └── runtime_guardian.py                                    # CORE-10
│
├── integrity/
│   └── verify_boot.py                                         # CORE-12, FAIL-16
│
├── execution/                                                 # SHARED INFRA (non-engine)
│   ├── async_bus.py                                           # EXEC-05 (single bus)
│   ├── fast_lane.py                                           # [NEW v1] segmented bus
│   ├── hazard_lane.py                                         # [NEW v1]
│   ├── offline_lane.py                                        # [NEW v1]
│   ├── event_emitter.py                                       # EXEC-04, HAZ-04
│   ├── severity_classifier.py                                 # EXEC-06
│   └── chaos_engine.py                                        # EXEC-07
│
├── simulation/                                                # [NEW v2-E + v3-P10] First-class simulation engine — vPro
│   ├── __init__.py
│   ├── engine.py                                              # SIM-00 — top-level entry point (parallel realities + arena)
│   ├── backtester.py                                          # SIM-01 — historical backtest
│   ├── event_replayer.py                                      # SIM-02 — uses tools/replay_validator
│   ├── scenario_generator.py                                  # SIM-03 — deterministic scenario PRNG (caller-supplied seed)
│   ├── slippage_model.py                                      # SIM-04
│   ├── latency_model.py                                       # SIM-05
│   ├── market_state_adapter.py                                # [NEW v3-P10] SIM-06 — frozen state builder
│   ├── parallel_runner.py                                     # [NEW v3-P10] SIM-07 — runs N realities deterministically
│   ├── scoring_engine.py                                      # [NEW v3-P10] SIM-08 — scores per-trader/per-archetype outcomes
│   ├── state_snapshot.py                                      # [NEW v3-P10] SIM-09 — ledger-safe SimulationSnapshot
│   ├── strategy_arena/                                        # [NEW v3-P10] Darwinian competition (slow cadence, publishes ranking)
│   │   ├── arena.py                                           # SIM-10 — competition harness
│   │   ├── capital_allocator.py                               # SIM-11 — capital flows by score
│   │   ├── kill_underperformers.py                            # SIM-12 — retires losing strategies
│   │   ├── simulation_outcome.py                              # [NEW v3.2] richer payload: ranking + failure_modes + regime_performance_map + adversarial_breakdowns (SystemEvent.simulation_outcome subtype, off-bus, seed-locked)
│   │   └── promotion_engine.py                                # SIM-13 — graduates winners (PROPOSED→SHADOW)
│   ├── adversarial/                                           # [NEW v3-P10] Adversarial market simulation layer
│   │   ├── liquidity_attacker.py                              # SIM-14
│   │   ├── stop_hunter.py                                     # SIM-15
│   │   └── flash_crash_synth.py                               # SIM-16
│   └── reflexive_layer/                                       # [NEW v3.1] Reflexivity — market reacts to YOUR orders
│       ├── __init__.py
│       ├── impact_feedback.py                                 # REFL-01 — own-order price impact loop
│       ├── liquidity_decay.py                                 # REFL-02 — liquidity drying up under our flow
│       └── crowd_density_sim.py                               # REFL-03 — alpha decay due to popularity / crowding
│
├── tools/
│   ├── __init__.py                                            # [EXISTS]
│   ├── authority_lint.py                                      # CORE-31, CI-05  [EXISTS]
│   ├── contract_diff.py                                       # LEDGER-12
│   ├── replay_validator.py                                    # TEST-01 helper
│   ├── config_validator.py                                    # DYN-CFG-02 helper
│   └── enforcement_matrix.py                                  # [NEW v1] invariant→4-layer map
│
├── scripts/
│   ├── diagnostics.py                                         # TEST-12
│   ├── profile_hot_path.py                                    # CI-10
│   ├── run_chaos_day.py                                       # EXEC-07, TEST-08
│   ├── verify.py                                              # TEST-15
│   └── dix_cli.py                                             # plugin + mode CLI
│
├── tests/                                                     # TEST-01..20  [EXISTS partial]
│   ├── __init__.py                                            # [EXISTS]
│   ├── test_engine_contracts.py                               # [EXISTS]
│   ├── test_authority_lint.py                                 # TEST-18  [EXISTS]
│   ├── test_execution_engine.py                               # [EXISTS]
│   ├── test_intelligence_engine.py                            # [EXISTS]
│   ├── test_ui_server.py                                      # [EXISTS]
│   ├── test_replay.py                                         # TEST-01
│   ├── test_hazard_flow.py                                    # TEST-02
│   ├── test_latency.py · test_governance.py
│   ├── test_neuromorphic.py
│   └── drift_killers/                                         # [NEW v1]
│       ├── test_replay_gate.py
│       ├── test_behavior_diff.py
│       ├── test_registry_lock.py
│       ├── test_snapshot_boundary.py
│       └── test_no_hidden_channels.py
│
├── ui/                                                        # FastAPI test harness  [EXISTS]
│   ├── __init__.py                                            # [EXISTS]
│   ├── server.py                                              # [EXISTS]
│   └── static/{index.html, app.js, styles.css}                # [EXISTS]
│
├── cockpit/                                                   # COCKPIT-01..11
│   ├── app.py · auth.py · llm.py · pairing.py · qr.py         # COCKPIT-01..03
│   ├── charter/devin.py                                       # CORE-30
│   ├── audit/                                                 # [NEW v2-J] Operator decision logging — full HITL trace
│   │   ├── operator_actions.py
│   │   ├── override_log.py
│   │   └── decision_diff.py
│   ├── api/
│   │   ├── status.py · risk.py · charters.py · ai.py
│   │   ├── autonomy.py · operator.py
│   │   ├── custom_strategies.py · weekly_scout.py
│   │   └── mode.py                                            # [NEW v1] Dashboard OS — request-only
│   ├── widgets/
│   │   ├── plugin_manager.py                                  # PLUGIN-ACT-03
│   │   ├── kill_switch.py                                     # COCKPIT-01
│   │   ├── master_sliders.py                                  # COCKPIT-02..04
│   │   ├── decision_trace.py                                  # COCKPIT-05 (causal-chain enabled)
│   │   ├── risk_view.py                                       # COCKPIT-06
│   │   ├── portfolio_view.py                                  # COCKPIT-07
│   │   ├── system_health.py                                   # COCKPIT-08
│   │   ├── alert_center.py                                    # COCKPIT-09
│   │   └── governance_panel.py                                # COCKPIT-10
│   └── cli/
│       └── dix_plugin.py                                      # PLUGIN-ACT-04
│
├── dashboard/                                                 # DASH-01..32 (TypeScript) + Dashboard OS
│   ├── package.json
│   ├── pnpm-lock.yaml
│   ├── os_layer/                                              # [NEW v1] DOS-CORE
│   │   ├── kernel.ts                                          # DASH-00 — event subscription, projection, routing
│   │   ├── state_projection.ts                                # EVENT → UI state
│   │   ├── control_plane_router.ts                            # all user actions → governance
│   │   ├── mode_aware_controller.ts                           # enforces UI based on system mode
│   │   ├── temporal_layer.ts                                  # LIVE / REPLAY / SNAPSHOT / SIMULATION
│   │   ├── session_controller.ts
│   │   ├── operator_gate.ts                                   # INV-12 enforcement
│   │   └── state_sync.ts
│   ├── trading_modes/                                         # [NEW v1] UI bindings for mode behavior
│   │   ├── manual_mode.ts
│   │   ├── semi_auto_mode.ts
│   │   ├── auto_mode.ts
│   │   └── safe_locked_mode.ts
│   └── src/
│       ├── App.tsx
│       ├── GlobalHeader.tsx                                   # DASH-01
│       ├── ModeControlBar.tsx                                 # DASH-02 — Phase 6 IMMUTABLE WIDGET 1 (request-only, GOV-CP-03 writes)
│       ├── EngineStatusGrid.tsx                               # DASH-EG-01 — Phase 6 IMMUTABLE WIDGET 2 (6 engines × {alive,degraded,halted,offline})
│       ├── WorkspaceGrid.tsx                                  # DASH-03 (mode-aware)
│       ├── DecisionTrace.tsx                                  # DASH-04 — Phase 6 IMMUTABLE WIDGET 3 (causal chain rendering)
│       ├── StrategyLifecyclePanel.tsx                         # DASH-SLP-01 — Phase 6 IMMUTABLE WIDGET 4 (strategy FSM viewer)
│       ├── RiskView.tsx                                       # DASH-05 (unified RISK_STATE_VECTOR)
│       ├── PortfolioView.tsx                                  # DASH-06
│       ├── SystemHealth.tsx                                   # DASH-07
│       ├── GovernancePanel.tsx                                # DASH-08
│       ├── EvolutionMonitor.tsx                               # DASH-09
│       ├── LatencyMonitor.tsx                                 # DASH-10
│       ├── PerformanceMetrics.tsx · TradeJournal.tsx · AlertCenter.tsx
│       ├── WorkspaceManager.tsx · ReportingSuite.tsx
│       ├── DriftMonitor.tsx                                   # [NEW v1]
│       ├── CognitionPanel.tsx                                 # [NEW v1]
│       ├── TimeControl.tsx                                    # [NEW v1]
│       ├── BeliefStateView.tsx                                # [NEW v3-T1] renders core/coherence/belief_state.py snapshot
│       ├── PressureMeter.tsx                                  # [NEW v3-T1] renders core/coherence/performance_pressure.py vector
│       ├── MetaControllerView.tsx                             # [NEW v3-T1] confidence/sizing visibility
│       ├── ArchetypeArena.tsx                                 # [NEW v3-P10] strategy_arena live ranking
│       ├── memecoin/                                          # DASH-27 (3 sub-modes)
│       │   └── MemecoinControlPanel.tsx                       # DASH-MCP-01 — Phase 6 IMMUTABLE WIDGET 5 (isolated process control)
│       ├── per_form/                                          # DASH-28 (Forex / Stocks / Crypto / Memecoin)
│       ├── self_reflection.tsx                                # DASH-29
│       └── grafana_panel.tsx                                  # DASH-31
│
├── mobile_pwa/                                                # DASH-25, DEPLOY-14
├── cloud/                                                     # DEPLOY-13
├── windows/                                                   # DEPLOY-01..12
├── deploy/
│   ├── docker/ · service/
│   ├── setup.ps1 · dix-update.bat
│
└── docs/
    ├── PR2_SPEC.md                                            # [EXISTS]
    ├── directory_tree.md                                      # this file
    ├── total_recall_index.md
    ├── coverage_report.md
    └── enforcement_matrix.md                                  # [NEW v1]
```

## Build phasing (Build Compiler Spec §2 — locked sequence)

The phase-by-phase delivery is in `build_plan.md`. Updated to integrate
v3 (Tier 1 follow-ons + Phase 10):

| Phase / Step | Scope | Status |
|---|---|---|
| Phase 0 | Bootstrap core (contracts, ledger, registry, time, event bus) | DONE (PR #14, #15, #23) |
| Phase 1 | Governance core (GOV-CP-01..07, Mode FSM, OperatorBridge) | DONE (PR #28) |
| Phase 2 | Execution core (adapters, lifecycle FSM, hot path, runtime monitor) | DONE (PR #29) |
| Phase 3 | Indira (signal_pipeline, microstructure, strategy_runtime, learning_interface) | DONE (PR #30, #31) |
| Phase 4 | Dyon (HAZ-01..12, health monitors, system state, patch pipeline) | DONE (PR #32, #33) |
| Phase 5 | Learning + Evolution closed loop | DONE (PR #34) |
| **Phase 6** | **Dashboard OS Control Plane** — 5 IMMUTABLE WIDGETS per spec §6 | **DONE (PR #37)** |
| Phase 6.T1a | Tier 1 follow-on: Belief State + Pressure Vector (`core/coherence/`) — entropy-aware uncertainty (INV-50) [v3.2] | **NEXT** |
| Phase 6.T1b | Tier 1 follow-on: Meta-Controller + Confidence Engine (`intelligence_engine/meta_controller/`) — INV-48 fallback lane in `policy/execution_policy.py` [v3.2] | after 6.T1a |
| Phase 6.T1c | Tier 1 follow-on: Reward shaping (`learning_engine/performance_analysis/reward_shaping.py`) | after 6.T1b |
| Phase 6.T1d | v3.1 fold-in: System Intent Engine (`core/coherence/system_intent.py`, GOV-CP-07 setter) | after 6.T1c |
| Phase 6.T1e | v3.2 fold-in: regime hysteresis activation (`regime_detector.py` + `registry/regime_hysteresis.yaml`, INV-49) | after 6.T1d |
| Phase 7 | Asset systems (forex, stocks, crypto, memecoin isolated process) + PolicyEngine constant-time decision table (I7 reframed) [v3.2] | locked spec |
| Phase 8 | Neuromorphic + AutoLearn (sensors, web autolearn, anomaly adapters) | locked spec |
| Phase 9 | Optimization layer (Rust ports if measured) | locked spec |
| **Phase 10** | **Intelligence Depth Layer** — Simulation vPro + Trader Intelligence (full F1) + Macro Regime + Cross-Asset + Strategic Execution + `agents/` | **NEW (per E1)** |
| Phase 10.1 | Simulation vPro — adds richer `SimulationOutcome` (failure_modes + regime_performance_map + adversarial_breakdowns) [v3.2] | within Phase 10 |
| Phase 10.2–10.4 | Trader Intelligence ingest/offline/consumer + archetype lifecycle (`archetype_lifecycle.py`, INV-51) [v3.2] | within Phase 10 |
| Phase 10.8 | `agents/` namespace activation + typed `SignalEvent.agent_context` schema + B15 lint (`registry/agent_context_keys.yaml`) [v3.2] | within Phase 10 |
| Phase 10.10 | v3.1 fold-in: Opponent Model (`intelligence_engine/opponent_model/`) | within Phase 10 |
| Phase 10.11 | v3.1 fold-in: Reflexive Simulation Layer (`simulation/reflexive_layer/`) | within Phase 10 |
| Phase 10.12 | v3.1 fold-in: Strategy Genetics (`evolution_engine/genetic/`) | within Phase 10 |
| Phase 10.13 | v3.1 fold-in: Regret / Counterfactual Memory (`state/memory_tensor/regret/`) | within Phase 10 |
| Phase 10.14 | v3.1 fold-in: Internal Debate Round (`meta_controller/evaluation/debate_round.py`) | within Phase 10 |

Legacy v2 13-step build remains a sub-decomposition reference in
`build_plan.md` for non-engine items (drift killers, registry split,
operator audit).

Every phase lands as its own PR. Each PR ends with a green CI gate.
Build Compiler Spec §1.1 freeze rules apply to every phase: no engine
renames, no domain collapses, no module removals, additive only.

## Architectural invariants reinforced by this tree

1. **Engines are sealed boxes.** No engine imports another engine; only
   `core/contracts/` is shared. Lint rules `T1`, `B1`, `L1`, `L2`, `L3`
   enforce.
2. **Coherence is a layer, not an engine.** `core/coherence/` *binds*
   engines via event interception; it never modifies engine code. New
   lint rule `B2` (Step 4) reserves cross-engine import privilege to
   `core/coherence/`.
3. **Governance is the only authority.** Every state mutation
   (mode, plugin lifecycle, risk amend, patch deploy, learning update)
   traverses GOV-CP-01..07 and lands as a ledger row.
4. **Hard 3-domain isolation.** NORMAL / COPY-TRADING / MEMECOIN are
   separated under `execution_engine/domains/`; memecoin runs in its own
   process with a burner wallet (INV-20, SAFE-13).
5. **Replay determinism.** All offline engines (Learning, Evolution)
   read the ledger via `state/ledger/reader.py` only; never reach into
   runtime engine state. Data versioning (v2-H) guarantees that
   replay sees the same market data as live ran on.
6. **Coordinated portfolio.** v2-A + v2-B turn "many independent
   strategy outputs" into "one coordinated portfolio decision".
7. **Real broker realism.** v2-C + v2-D + v2-F provide the order
   lifecycle, normalised market state, and real-time risk evaluation
   needed for non-paper execution.
8. **Belief State + Pressure Vector are derived projections.** v3-T1
   `core/coherence/belief_state.py` and `performance_pressure.py` read
   existing engine state via L3 protocols; they never write engine
   state. Governance remains the only authority.
9. **Meta-Controller composes with Strategy Orchestrator (per B1).**
   Pipeline: `signal_pipeline → orchestrator (lifecycle gate) →
   meta_controller (regime route + selector + confidence + sizer +
   policy) → conflict_resolver (vote)`. Both modules retained, distinct
   responsibilities.
10. **Trader Intelligence is governed sensory data.** v3-P10 ingestion
    (`sensory/web_autolearn/trader_intelligence/`) emits
    `WEB_SIGNAL_EVENT` through HITL gate; learning side
    (`learning_engine/trader_abstraction/`) builds embeddings offline
    with fixed seed + ledgered checkpoint; consumer side
    (`intelligence_engine/meta/`) reads `registry/trader_archetypes.yaml`.
    Engines never reach into raw web data.
11. **Simulation runs on slower cadence than hot path.** v3-P10
    `simulation/strategy_arena/` publishes a `StrategyRanking` snapshot
    that the meta-controller reads cached. T1 ≤1ms hot-path budget
    preserved.
12. **Determinism preserved across all v3 additions.** Scenario
    generation uses caller-supplied PRNG seeds; embeddings produced
    offline with fixed seed + checkpoint hash ledgered; agents are
    pure-function-of-state with no clocks; no pure RL (INV-15).
13. **Intent is operator-written, system-read (v3.1).**
    `core/coherence/system_intent.py` is a frozen read-only projection.
    The operator writes `IntentTransition` events through GOV-CP-07
    (HITL gate); meta-controller reads intent via L3 Protocol. The
    system never auto-mutates its own mission. Governance remains the
    only authority.
14. **Internal debate is deterministic, not meta-RL (v3.1).**
    `meta_controller/evaluation/debate_round.py` runs a deterministic stance +
    confidence scoring round across stateful `agents/`. No learned
    coordinator, no policy-gradient meta-controller. Output feeds
    `confidence_engine`. INV-15 replay determinism preserved.
15. **Time hierarchy is layered, not new (v3.1).** Existing FSMs
    already span ms (hot_path) → sec/min (strategy_runtime) →
    hour/day (portfolio + arena cadence) → day/week
    (evolution_engine) → week/month (System Intent + GOV-G18 patch
    cadence). v3.1 documents this, no new modules.
16. **Dynamic identity is emergent (v3.1).** "From trend follower
    → mean reversion" is the active subset of LIVE strategies under
    the current regime + intent — produced by Strategy Lifecycle FSM
    + Strategy Arena + meta-controller `regime_router` reading
    `system_intent`. No new identity engine.
