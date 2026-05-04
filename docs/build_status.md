# DIX VISION v42.2 — Build Status (auto-generated)
_Generated against `docs/directory_tree.md`. Total tree entries: 492; on disk: 131; missing: 361._
_Coverage: 26%_

Legend:
- ✅ on disk
- ❌ missing
- 🟡 stub-only (file exists but is a phantom forward-decl)

---

## Per-package gap summary (descending)

| Package | Implemented | Missing | Gap % |
|---|---|---|---|
| `intelligence_engine/` | 29 | 47 | 61% |
| `dashboard/` | 0 | 45 | 100% |
| `simulation/` | 0 | 27 | 100% |
| `state/` | 5 | 26 | 83% |
| `execution_engine/` | 20 | 24 | 54% |
| `cockpit/` | 0 | 24 | 100% |
| `learning_engine/` | 8 | 22 | 73% |
| `governance_engine/` | 12 | 18 | 60% |
| `sensory/` | 0 | 18 | 100% |
| `registry/` | 4 | 16 | 80% |
| `core/` | 11 | 13 | 54% |
| `system_engine/` | 12 | 10 | 45% |
| `evolution_engine/` | 5 | 10 | 66% |
| `tests/` | 7 | 10 | 58% |
| `execution/` | 0 | 8 | 100% |
| `contracts/` | 2 | 6 | 75% |
| `immutable_core/` | 1 | 6 | 85% |
| `scripts/` | 1 | 5 | 83% |
| `tools/` | 3 | 4 | 57% |
| `translation/` | 0 | 4 | 100% |
| `.github/` | 2 | 3 | 60% |
| `deploy/` | 0 | 3 | 100% |
| `integrity/` | 0 | 2 | 100% |
| `enforcement/` | 1 | 2 | 66% |
| `mobile_pwa/` | 0 | 1 | 100% |
| `coverage_report.md/` | 0 | 1 | 100% |
| `VERSION/` | 0 | 1 | 100% |
| `directory_tree.md/` | 0 | 1 | 100% |
| `total_recall_index.md/` | 0 | 1 | 100% |
| `windows/` | 0 | 1 | 100% |
| `cloud/` | 0 | 1 | 100% |
| `enforcement_matrix.md/` | 0 | 1 | 100% |
| `pyproject.toml/` | 1 | 0 | 0% |
| `README.md/` | 1 | 0 | 0% |
| `PR2_SPEC.md/` | 1 | 0 | 0% |
| `docs/` | 1 | 0 | 0% |
| `ui/` | 4 | 0 | 0% |

---

## Missing entries (full list, grouped by package)

### `intelligence_engine/` (47 missing)

- ❌ `intelligence_engine/agents` _NEW v3-P10, per C2_
- ❌ `intelligence_engine/agents/__init__.py`
- ❌ `intelligence_engine/agents/_base.py` _NEW v3.3_
- ❌ `intelligence_engine/agents/adversarial_observer.py`
- ❌ `intelligence_engine/agents/liquidity_provider.py`
- ❌ `intelligence_engine/agents/macro.py`
- ❌ `intelligence_engine/agents/scalper.py`
- ❌ `intelligence_engine/agents/swing_trader.py`
- ❌ `intelligence_engine/archetype_arena.py`
- ❌ `intelligence_engine/charter/indira.py`
- ❌ `intelligence_engine/cross_asset` _NEW v3-P10_
- ❌ `intelligence_engine/cross_asset/basket_constructor.py`
- ❌ `intelligence_engine/cross_asset/contagion_detector.py`
- ❌ `intelligence_engine/cross_asset/correlation_matrix.py`
- ❌ `intelligence_engine/cross_asset/lead_lag.py`
- ❌ `intelligence_engine/intent_producer.py`
- ❌ `intelligence_engine/macro` _NEW v3-P10_
- ❌ `intelligence_engine/macro/hidden_state_detector.py`
- ❌ `intelligence_engine/macro/latent_embedder.py`
- ❌ `intelligence_engine/macro/macro_event_aligner.py`
- ❌ `intelligence_engine/macro/regime_classifier.py`
- ❌ `intelligence_engine/meta` _NEW v3-P10_
- ❌ `intelligence_engine/meta_controller/evaluation/debate_round.py` _NEW v3.1_
- ❌ `intelligence_engine/meta_controller/evaluation/strategy_selector.py`
- ❌ `intelligence_engine/meta_controller/execution_policy.py` _NEW v3.2 — INV-48_
- ❌ `intelligence_engine/meta_controller/shadow_policy.py` _NEW v3.3_
- ❌ `intelligence_engine/opponent_model` _NEW v3.1_
- ❌ `intelligence_engine/opponent_model/__init__.py`
- ❌ `intelligence_engine/opponent_model/behavior_predictor.py`
- ❌ `intelligence_engine/opponent_model/crowd_density.py`
- ❌ `intelligence_engine/opponent_model/strategy_detector.py`
- ❌ `intelligence_engine/plugins/agent`
- ❌ `intelligence_engine/plugins/alpha`
- ❌ `intelligence_engine/plugins/alt_data`
- ❌ `intelligence_engine/plugins/cognition`
- ❌ `intelligence_engine/plugins/memory`
- ❌ `intelligence_engine/plugins/multi_timeframe`
- ❌ `intelligence_engine/plugins/transfer`
- ❌ `intelligence_engine/portfolio` _NEW v2-A_
- ❌ `intelligence_engine/portfolio/allocator.py`
- ❌ `intelligence_engine/portfolio/capital_scheduler.py`
- ❌ `intelligence_engine/portfolio/correlation_engine.py`
- ❌ `intelligence_engine/portfolio/exposure_manager.py`
- ❌ `intelligence_engine/portfolio/risk_parity.py`
- ❌ `intelligence_engine/strategy_runtime/archetype_lifecycle.py` _NEW v3.2 — INV-51_
- ❌ `intelligence_engine/strategy_synthesizer.py`
- ❌ `intelligence_engine/trader_archetypes.py`

### `dashboard/` (45 missing)

- ❌ `dashboard`
- ❌ `dashboard/App.tsx`
- ❌ `dashboard/ArchetypeArena.tsx` _NEW v3-P10_
- ❌ `dashboard/BeliefStateView.tsx` _NEW v3-T1_
- ❌ `dashboard/CognitionPanel.tsx` _NEW v1_
- ❌ `dashboard/DecisionTrace.tsx`
- ❌ `dashboard/DriftMonitor.tsx` _NEW v1_
- ❌ `dashboard/EngineStatusGrid.tsx`
- ❌ `dashboard/EvolutionMonitor.tsx`
- ❌ `dashboard/GlobalHeader.tsx`
- ❌ `dashboard/GovernancePanel.tsx`
- ❌ `dashboard/LatencyMonitor.tsx`
- ❌ `dashboard/MetaControllerView.tsx` _NEW v3-T1_
- ❌ `dashboard/ModeControlBar.tsx`
- ❌ `dashboard/PerformanceMetrics.tsx`
- ❌ `dashboard/PortfolioView.tsx`
- ❌ `dashboard/PressureMeter.tsx` _NEW v3-T1_
- ❌ `dashboard/RiskView.tsx`
- ❌ `dashboard/StrategyLifecyclePanel.tsx`
- ❌ `dashboard/SystemHealth.tsx`
- ❌ `dashboard/TimeControl.tsx` _NEW v1_
- ❌ `dashboard/WorkspaceGrid.tsx`
- ❌ `dashboard/WorkspaceManager.tsx`
- ❌ `dashboard/grafana_panel.tsx`
- ❌ `dashboard/memecoin`
- ❌ `dashboard/memecoin/MemecoinControlPanel.tsx`
- ❌ `dashboard/os_layer` _NEW v1_
- ❌ `dashboard/os_layer/control_plane_router.ts`
- ❌ `dashboard/os_layer/kernel.ts`
- ❌ `dashboard/os_layer/mode_aware_controller.ts`
- ❌ `dashboard/os_layer/operator_gate.ts`
- ❌ `dashboard/os_layer/session_controller.ts`
- ❌ `dashboard/os_layer/state_projection.ts`
- ❌ `dashboard/os_layer/state_sync.ts`
- ❌ `dashboard/os_layer/temporal_layer.ts`
- ❌ `dashboard/package.json`
- ❌ `dashboard/per_form`
- ❌ `dashboard/pnpm-lock.yaml`
- ❌ `dashboard/self_reflection.tsx`
- ❌ `dashboard/src`
- ❌ `dashboard/trading_modes` _NEW v1_
- ❌ `dashboard/trading_modes/auto_mode.ts`
- ❌ `dashboard/trading_modes/manual_mode.ts`
- ❌ `dashboard/trading_modes/safe_locked_mode.ts`
- ❌ `dashboard/trading_modes/semi_auto_mode.ts`

### `simulation/` (27 missing)

- ❌ `simulation` _NEW v2-E + v3-P10_
- ❌ `simulation/__init__.py`
- ❌ `simulation/__init__.py`
- ❌ `simulation/adversarial` _NEW v3-P10_
- ❌ `simulation/adversarial/flash_crash_synth.py`
- ❌ `simulation/adversarial/liquidity_attacker.py`
- ❌ `simulation/adversarial/stop_hunter.py`
- ❌ `simulation/backtester.py`
- ❌ `simulation/crowd_density_sim.py`
- ❌ `simulation/engine.py`
- ❌ `simulation/event_replayer.py`
- ❌ `simulation/impact_feedback.py`
- ❌ `simulation/latency_model.py`
- ❌ `simulation/liquidity_decay.py`
- ❌ `simulation/market_state_adapter.py` _NEW v3-P10_
- ❌ `simulation/parallel_runner.py` _NEW v3-P10_
- ❌ `simulation/reflexive_layer` _NEW v3.1_
- ❌ `simulation/scenario_generator.py`
- ❌ `simulation/scoring_engine.py` _NEW v3-P10_
- ❌ `simulation/slippage_model.py`
- ❌ `simulation/state_snapshot.py` _NEW v3-P10_
- ❌ `simulation/strategy_arena` _NEW v3-P10_
- ❌ `simulation/strategy_arena/arena.py`
- ❌ `simulation/strategy_arena/capital_allocator.py`
- ❌ `simulation/strategy_arena/kill_underperformers.py`
- ❌ `simulation/strategy_arena/promotion_engine.py`
- ❌ `simulation/strategy_arena/simulation_outcome.py` _NEW v3.2_

### `state/` (26 missing)

- ❌ `state/data_versioning` _NEW v2-H_
- ❌ `state/databases`
- ❌ `state/dataset_registry.py`
- ❌ `state/feature_store.py`
- ❌ `state/knowledge_store.py`
- ❌ `state/ledger/append.py`
- ❌ `state/ledger/hash_chain.py`
- ❌ `state/ledger/hot_store.py`
- ❌ `state/ledger/integrity.py`
- ❌ `state/ledger/reconstructor.py`
- ❌ `state/ledger/snapshots.py`
- ❌ `state/market_snapshots.py`
- ❌ `state/memory_tensor` _NEW v1_
- ❌ `state/memory_tensor/__init__.py`
- ❌ `state/memory_tensor/almost_trades.py`
- ❌ `state/memory_tensor/episodic.py`
- ❌ `state/memory_tensor/meta_memory.py`
- ❌ `state/memory_tensor/missed_opportunity.py`
- ❌ `state/memory_tensor/procedural.py`
- ❌ `state/memory_tensor/regret` _NEW v3.1_
- ❌ `state/memory_tensor/regret_log.py`
- ❌ `state/memory_tensor/semantic.py`
- ❌ `state/memory_tensor/trader_patterns` _NEW v3-P10_
- ❌ `state/memory_tensor/trader_patterns/archetype_store.py`
- ❌ `state/memory_tensor/trader_patterns/atom_store.py`
- ❌ `state/memory_tensor/trader_patterns/profile_store.py`

### `execution_engine/` (24 missing)

- ❌ `execution_engine/adapters/binance.py`
- ❌ `execution_engine/adapters/memecoin`
- ❌ `execution_engine/adapters/oanda.py`
- ❌ `execution_engine/copy_trading`
- ❌ `execution_engine/domains` _NEW v1_
- ❌ `execution_engine/hot_path/fast_risk_cache.py`
- ❌ `execution_engine/hot_path/time_authority.py`
- ❌ `execution_engine/market_data` _NEW v2-D_
- ❌ `execution_engine/market_data/aggregator.py`
- ❌ `execution_engine/market_data/book_builder.py`
- ❌ `execution_engine/market_data/latency_tracker.py`
- ❌ `execution_engine/market_data/normalizer.py`
- ❌ `execution_engine/memecoin`
- ❌ `execution_engine/normal`
- ❌ `execution_engine/protections/circuit_breaker.py`
- ❌ `execution_engine/protections/reconciliation.py`
- ❌ `execution_engine/strategic_execution` _NEW v3-P10_
- ❌ `execution_engine/strategic_execution/__init__.py`
- ❌ `execution_engine/strategic_execution/adversarial_executor.py`
- ❌ `execution_engine/strategic_execution/depth_estimator.py`
- ❌ `execution_engine/strategic_execution/market_impact`
- ❌ `execution_engine/strategic_execution/model.py`
- ❌ `execution_engine/strategic_execution/optimal_execution.py`
- ❌ `execution_engine/strategic_execution/slippage_curve.py`

### `cockpit/` (24 missing)

- ❌ `cockpit`
- ❌ `cockpit/api`
- ❌ `cockpit/api/autonomy.py`
- ❌ `cockpit/api/custom_strategies.py`
- ❌ `cockpit/api/mode.py` _NEW v1_
- ❌ `cockpit/api/status.py`
- ❌ `cockpit/app.py`
- ❌ `cockpit/audit` _NEW v2-J_
- ❌ `cockpit/audit/decision_diff.py`
- ❌ `cockpit/audit/operator_actions.py`
- ❌ `cockpit/audit/override_log.py`
- ❌ `cockpit/charter/devin.py`
- ❌ `cockpit/cli`
- ❌ `cockpit/dix_plugin.py`
- ❌ `cockpit/widgets`
- ❌ `cockpit/widgets/alert_center.py`
- ❌ `cockpit/widgets/decision_trace.py`
- ❌ `cockpit/widgets/governance_panel.py`
- ❌ `cockpit/widgets/kill_switch.py`
- ❌ `cockpit/widgets/master_sliders.py`
- ❌ `cockpit/widgets/plugin_manager.py`
- ❌ `cockpit/widgets/portfolio_view.py`
- ❌ `cockpit/widgets/risk_view.py`
- ❌ `cockpit/widgets/system_health.py`

### `learning_engine/` (22 missing)

- ❌ `learning_engine/coherence_calibrator.py` _NEW v3.3_
- ❌ `learning_engine/lanes/continual_distillation.py`
- ❌ `learning_engine/lanes/experience_base.py`
- ❌ `learning_engine/lanes/federated.py`
- ❌ `learning_engine/lanes/policy_distillation.py`
- ❌ `learning_engine/lanes/ral.py`
- ❌ `learning_engine/lanes/self_learning_loop.py`
- ❌ `learning_engine/performance_analysis` _NEW v2-G_
- ❌ `learning_engine/performance_analysis/alpha_decay.py`
- ❌ `learning_engine/performance_analysis/archetype_evaluator.py` _NEW v3.2_
- ❌ `learning_engine/performance_analysis/execution_quality.py`
- ❌ `learning_engine/performance_analysis/latency_impact.py`
- ❌ `learning_engine/performance_analysis/pnl_attribution.py`
- ❌ `learning_engine/performance_analysis/reward_shaping.py` _NEW v3-T1_
- ❌ `learning_engine/performance_analysis/slippage_analysis.py`
- ❌ `learning_engine/sim_realism_tracker.py` _NEW v3.3_
- ❌ `learning_engine/trader_abstraction` _NEW v3-P10_
- ❌ `learning_engine/trader_abstraction/__init__.py`
- ❌ `learning_engine/trader_abstraction/embedder.py`
- ❌ `learning_engine/trader_abstraction/encoder.py`
- ❌ `learning_engine/trader_abstraction/extractor.py`
- ❌ `learning_engine/trader_abstraction/normalizer.py`

### `governance_engine/` (18 missing)

- ❌ `governance_engine/charter/governance.py`
- ❌ `governance_engine/drawdown_guard.py`
- ❌ `governance_engine/exposure_limits.py`
- ❌ `governance_engine/kill_conditions.py`
- ❌ `governance_engine/plugin_lifecycle`
- ❌ `governance_engine/plugin_lifecycle/activation_gate.py`
- ❌ `governance_engine/plugin_lifecycle/hot_reload_signal.py`
- ❌ `governance_engine/plugin_lifecycle/lifecycle_emitter.py`
- ❌ `governance_engine/plugin_lifecycle/registry_loader.py`
- ❌ `governance_engine/position_limits.py`
- ❌ `governance_engine/real_time_risk.py`
- ❌ `governance_engine/risk_engine` _NEW v2-F_
- ❌ `governance_engine/services/audit_replay.py`
- ❌ `governance_engine/services/liveness_watchdog.py`
- ❌ `governance_engine/services/overconfidence_guardrail.py`
- ❌ `governance_engine/services/patch_pipeline.py`
- ❌ `governance_engine/services/triple_window_dry_run.py`
- ❌ `governance_engine/services/trust_engine.py`

### `sensory/` (18 missing)

- ❌ `sensory`
- ❌ `sensory/__init__.py`
- ❌ `sensory/ai_filter.py`
- ❌ `sensory/archetype_publisher.py`
- ❌ `sensory/behavior_analyzer.py`
- ❌ `sensory/crawler.py`
- ❌ `sensory/crawler.py`
- ❌ `sensory/curator.py`
- ❌ `sensory/neuromorphic`
- ❌ `sensory/neuromorphic/dyon_anomaly.py`
- ❌ `sensory/neuromorphic/governance_risk.py`
- ❌ `sensory/neuromorphic/indira_signal.py`
- ❌ `sensory/pending_buffer.py`
- ❌ `sensory/performance_validator.py`
- ❌ `sensory/profile_extractor.py`
- ❌ `sensory/seeds.yaml`
- ❌ `sensory/trader_intelligence` _NEW v3-P10_
- ❌ `sensory/web_autolearn`

### `registry/` (16 missing)

- ❌ `registry/agent_context_keys.yaml` _NEW v3.2_
- ❌ `registry/agent_rationale_tags.yaml` _NEW v3.3_
- ❌ `registry/agent_state_keys.yaml` _NEW v3.3_
- ❌ `registry/agents.yaml` _NEW v3-P10_
- ❌ `registry/budgets.yaml` _NEW v1_
- ❌ `registry/calibration.yaml` _NEW v3.3_
- ❌ `registry/definitions.yaml`
- ❌ `registry/enforcement_policies.yaml`
- ❌ `registry/layers.yaml`
- ❌ `registry/lifecycle.yaml`
- ❌ `registry/meta_controller.yaml` _NEW v3.3_
- ❌ `registry/performance.yaml`
- ❌ `registry/regime_hysteresis.yaml` _NEW v3.2_
- ❌ `registry/reward_components.yaml` _NEW v3.3_
- ❌ `registry/strategies` _NEW v2-I_
- ❌ `registry/trader_archetypes.yaml` _NEW v3-P10_

### `core/` (13 missing)

- ❌ `core/belief_state.py` _NEW v3-T1_
- ❌ `core/bootstrap_kernel.py`
- ❌ `core/causal_graph.py`
- ❌ `core/contracts/execution.py`
- ❌ `core/contracts/ledger.py`
- ❌ `core/drift_oracle.py`
- ❌ `core/engine.py`
- ❌ `core/meta_adaptation.py`
- ❌ `core/mode_engine.py`
- ❌ `core/performance_pressure.py` _NEW v3-T1_
- ❌ `core/registry.py`
- ❌ `core/secrets.py`
- ❌ `core/system_intent.py` _NEW v3.1_

### `system_engine/` (10 missing)

- ❌ `system_engine/anomaly_detector.py`
- ❌ `system_engine/charter/dyon.py`
- ❌ `system_engine/drift_monitor.py`
- ❌ `system_engine/hazard_sensors/neuromorphic_detector.py`
- ❌ `system_engine/health_monitors/api_changelogs.py`
- ❌ `system_engine/health_monitors/repo_discovery.py`
- ❌ `system_engine/homeostasis.py`
- ❌ `system_engine/kill_switch_runtime.py`
- ❌ `system_engine/runtime_guardian.py`
- ❌ `system_engine/system_state.py`

### `evolution_engine/` (10 missing)

- ❌ `evolution_engine/critique_loop.py`
- ❌ `evolution_engine/genetic` _NEW v3.1_
- ❌ `evolution_engine/genetic/__init__.py`
- ❌ `evolution_engine/genetic/crossover.py`
- ❌ `evolution_engine/genetic/fitness_inheritance.py`
- ❌ `evolution_engine/genetic/mutation_operators.py`
- ❌ `evolution_engine/pipeline.py`
- ❌ `evolution_engine/sandbox.py`
- ❌ `evolution_engine/shadow.py`
- ❌ `evolution_engine/skill_graph`

### `tests/` (10 missing)

- ❌ `tests/drift_killers` _NEW v1_
- ❌ `tests/test_behavior_diff.py`
- ❌ `tests/test_hazard_flow.py`
- ❌ `tests/test_latency.py`
- ❌ `tests/test_neuromorphic.py`
- ❌ `tests/test_no_hidden_channels.py`
- ❌ `tests/test_registry_lock.py`
- ❌ `tests/test_replay.py`
- ❌ `tests/test_replay_gate.py`
- ❌ `tests/test_snapshot_boundary.py`

### `execution/` (8 missing)

- ❌ `execution`
- ❌ `execution/async_bus.py`
- ❌ `execution/chaos_engine.py`
- ❌ `execution/event_emitter.py`
- ❌ `execution/fast_lane.py` _NEW v1_
- ❌ `execution/hazard_lane.py` _NEW v1_
- ❌ `execution/offline_lane.py` _NEW v1_
- ❌ `execution/severity_classifier.py`

### `contracts/` (6 missing)

- ❌ `contracts/execution.proto`
- ❌ `contracts/governance.proto`
- ❌ `contracts/ledger.proto`
- ❌ `contracts/market.proto`
- ❌ `contracts/system.proto`
- ❌ `contracts/trader_intelligence.proto` _NEW v3-P10_

### `immutable_core/` (6 missing)

- ❌ `immutable_core/foundation.hash`
- ❌ `immutable_core/hazard_axioms.lean`
- ❌ `immutable_core/kill_switch.py`
- ❌ `immutable_core/neuromorphic_axioms.lean`
- ❌ `immutable_core/safety_axioms.lean`
- ❌ `immutable_core/system_identity.py`

### `scripts/` (5 missing)

- ❌ `scripts/diagnostics.py`
- ❌ `scripts/dix_cli.py`
- ❌ `scripts/profile_hot_path.py`
- ❌ `scripts/run_chaos_day.py`
- ❌ `scripts/verify.py`

### `tools/` (4 missing)

- ❌ `tools/config_validator.py`
- ❌ `tools/contract_diff.py`
- ❌ `tools/enforcement_matrix.py` _NEW v1_
- ❌ `tools/replay_validator.py`

### `translation/` (4 missing)

- ❌ `translation`
- ❌ `translation/audit_writer.py`
- ❌ `translation/intent_to_patch.py`
- ❌ `translation/round_trip_validator.py`

### `.github/` (3 missing)

- ❌ `.github/workflows/release.yml`
- ❌ `.github/workflows/rust.yml`
- ❌ `.github/workflows/sandbox.yml`

### `deploy/` (3 missing)

- ❌ `deploy`
- ❌ `deploy/docker`
- ❌ `deploy/setup.ps1`

### `integrity/` (2 missing)

- ❌ `integrity`
- ❌ `integrity/verify_boot.py`

### `enforcement/` (2 missing)

- ❌ `enforcement/decorators.py`
- ❌ `enforcement/runtime_guardian.py`

### `mobile_pwa/` (1 missing)

- ❌ `mobile_pwa`

### `coverage_report.md/` (1 missing)

- ❌ `coverage_report.md`

### `VERSION/` (1 missing)

- ❌ `VERSION`

### `directory_tree.md/` (1 missing)

- ❌ `directory_tree.md`

### `total_recall_index.md/` (1 missing)

- ❌ `total_recall_index.md`

### `windows/` (1 missing)

- ❌ `windows`

### `cloud/` (1 missing)

- ❌ `cloud`

### `enforcement_matrix.md/` (1 missing)

- ❌ `enforcement_matrix.md` _NEW v1_

