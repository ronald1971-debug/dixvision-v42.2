# execution_engine/ — order lifecycle + adapters (28 files)

## Purpose

The **only** authority that turns a governance-approved
`ExecutionIntent` into an exchange-side order. INV-56 Triad Lock
executor.

Sub-packages:

* `engine.py` (401 lines) — `ExecutionEngine.execute(intent)` is the
  single chokepoint. AuthorityGuard verifies the HMAC-signed
  `governance_decision_id` before any adapter is called.
* `lifecycle/` — `order_state_machine.py`, `retry_logic.py`,
  `fill_handler.py`, `partial_fill_resolver.py`, `sl_tp_manager.py`.
* `protections/` — `feedback.py` (FeedbackCollector → TradeOutcome),
  `runtime_monitor.py`.
* `hot_path/fast_execute.py` — strict purity, no clock reads, no
  PRNG. Fast path for canary + live.
* `adapters/` — `paper.py`, `pumpfun.py`, `hummingbot.py`, `uniswapx.py`,
  `_uniswapx_signer.py`, `_uniswapx_quote.py`, `_hummingbot_gateway.py`,
  `base.py`, `_live_base.py`, `registry.py`, `router.py`.
* `strategic/almgren_chriss.py` — strategic execution scheduler
  (INV-62, SAFE-61/62, PERF-03).
* `execution_gate.py` — final gate before adapter dispatch.

## Wiring

* `ui/server.py` builds the engine and routes
  `governance.approved_intents` → `engine.execute(intent)`.
* `ExecutionEvent` flows back to `intelligence_engine` via
  `IntelligenceFeedbackSink` (PR #140) and to `governance_engine`
  for ledger persistence.
* `AdapterRegistry` (PR #152) maps venue ID → adapter; the
  `UniswapXAdapter` is now lazy-imported (PR #178) so the harness can
  boot without `eth-account` installed.

## Static-analysis result

* 28 files, 15 with findings — **all 15 are ruff-format drift only**
  (FORMAT rule).
* No orphan modules. No semantic findings.

## Deep-read observations

* `engine.py` — `execute(intent: ExecutionIntent)` is the **only**
  public mutating method (HARDEN-02 PR #79). Legacy
  `process(SignalEvent)` was deleted in PR #88 and B27 lint forbids
  its return. AuthorityGuard verifies HMAC signature (PR #170/#171)
  or refuses.
* `adapters/paper.py` — deterministic fills, slippage_bps, no clock
  reads. **Paper-S2 will extend this with latency, fee, virtual
  balance ledger, partial fills, fill ring** (queued).
* `adapters/_uniswapx_signer.py` — EIP-712 signer using `eth_account`
  (lazy-loaded). UniswapX is the only EVM-trade path on disk;
  `UniswapV3Adapter` + `RaydiumAdapter` + `PancakeSwapAdapter` are
  queued under the memecoin execution layer build queue.
* `protections/feedback.py` — FeedbackCollector projects
  ExecutionEvent → TradeOutcome → both `intelligence` and
  `learning_engine`. PR #143 ensured hazard-throttled REJECTs also
  feed the learning loop (else WeightAdjuster would never learn from
  hazard-driven failures).

## Risks / gaps

* `_uniswapx_signer.py` reads private key from env var; **must be
  refactored before LIVE memecoin execution** to a `KeyStore` (OS
  keyring + Ledger HW wallet). Tracked under the memecoin execution
  layer queue.
* Paper adapter has no fee/latency model; PaperBroker upgrade
  (Paper-S2) is the next planned PR.

## Verdict

**HEALTHY chokepoint, partial coverage on adapters.** Triad Lock
executor is bullet-proof; LIVE memecoin path needs the keystore +
MEV-policy work before promotion.
