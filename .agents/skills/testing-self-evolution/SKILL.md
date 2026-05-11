# Testing the P0 self-evolution path (HARDEN-04 + learning override)

This skill covers end-to-end testing of the closed learning loop's freeze
gate (HARDEN-04 / `LearningEvolutionFreezePolicy`), the operator override
toggle, and the surrounding mode FSM + authority-ledger audit surface.
Use this whenever a PR touches:

- `core/contracts/learning_evolution_freeze.py`
- `learning_engine/loops/closed_loop.py` (sample/update builders, policy supplier)
- `evolution_engine/loops/structural_loop.py`
- `governance_engine/control_plane/state_transition_manager.py`
- `governance_engine/control_plane/operator_interface_bridge.py`
- `core/contracts/operator_consent.py`
- Any `/api/operator/learning-override`, `/api/operator/action/mode`, or `/api/admin/learning/*` route in `ui/server.py`

## Contract under test

The loop is **frozen** iff `not (mode is SystemMode.LIVE AND operator_override is True)`.

- Boot seed defaults `operator_override=True` (PR #376). Override the boot seed with `DIXVISION_LEARNING_OVERRIDE=false`.
- `ClosedLearningLoop` is constructed with concrete `sample_builder` / `update_builder` (PR #377). Without these, unfrozen ticks still emit zero samples.
- `OperatorStrategyCounts.shadow` was deleted in PR #378; `GET /api/operator/summary.strategies` must only project `{proposed, canary, live, retired, failed}`.

## Booting the harness for testing

```bash
rm -f /tmp/test_p0_ledger.db    # fresh ledger per run
cd <repo root>
DIX_LEARNING_DEBUG_TICK=1 \          # exposes POST /api/admin/learning/tick
DIXVISION_LEDGER_PATH=/tmp/test_p0_ledger.db \
DIX_HARNESS_APPROVER_ENABLED=1 \     # auto-approves operator requests in the harness
uvicorn ui.server:app --host 127.0.0.1 --port 8080 --log-level warning &
```

Without `DIX_LEARNING_DEBUG_TICK=1` the `/api/admin/learning/tick` route returns 404 — the harness path is env-gated for safety.

## Quick-reference routes

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/operator/learning-override` | Read `{enabled, mode, is_freeze_active}` projection |
| POST | `/api/operator/learning-override` | Flip the override flag (audited under `STATE.lock`) |
| POST | `/api/operator/action/mode` | Drive the mode FSM through `OperatorInterfaceBridge` |
| POST | `/api/admin/learning/tick` | Tick the closed + structural loops (debug-gated) |
| GET | `/api/operator/summary` | Engine + strategies projection (PR #378 SHADOW residue check) |

## The SAFE→PAPER consent envelope (gotcha)

Per Hardening-S1 item 8, `CONSENT_REQUIRED_EDGES = {(SAFE, PAPER), (LIVE, AUTO)}`.
The state transition manager calls `OperatorConsentValidator.validate(..., live_policy_hash=self._policy.table_hash)`. You must construct an `OperatorConsent` with `policy_hash` equal to the **policy table hash**, which is logged as the very first ledger row (`POLICY_TABLE_INSTALLED`) and is **not** any of the four `POLICY_HASHES_BOUND` registry digests.

Extract it from the ledger:

```bash
POLICY_HASH=$(python3 -c "
import sqlite3, json
c = sqlite3.connect('/tmp/test_p0_ledger.db')
row = c.execute(\"SELECT payload FROM authority_ledger WHERE kind='POLICY_TABLE_INSTALLED'\").fetchone()
print(json.loads(row[0])['table_hash'])
")
```

Then drive SAFE→PAPER:

```bash
NOW_NS=$(python3 -c 'import time; print(time.time_ns())')
curl -s -X POST http://127.0.0.1:8080/api/operator/action/mode \
  -H 'Content-Type: application/json' \
  -d "{\"target_mode\":\"PAPER\",\"requestor\":\"test\",\"operator_authorized\":true,
        \"consent_operator_id\":\"test-op\",\"consent_policy_hash\":\"$POLICY_HASH\",
        \"consent_nonce\":\"n1\",\"consent_ts_ns\":$NOW_NS}"
```

PAPER→CANARY and CANARY→LIVE do **not** require consent; just send `operator_authorized: true`.

Failure modes that have bitten testing before:
- `CONSENT_MISSING` — payload omitted one of the four consent fields.
- `CONSENT_POLICY_HASH_MISMATCH` — used a `POLICY_HASHES_BOUND` digest instead of `POLICY_TABLE_INSTALLED.table_hash`.
- `CONSENT_STALE` — `consent_ts_ns` outside the freshness window. Always compute it fresh per request.
- Nonce reuse → `CONSENT_REPLAY`. Pick a new nonce per envelope.

## Adversarial assertions worth running

For any PR touching the freeze path, run these eight checks. They are pure curl and complete in a few seconds.

1. Boot GET — `enabled=true, mode=SAFE, is_freeze_active=true`.
2. Tick at SAFE — `closed_learning.frozen=true`, `operator_override=true`.
3. POST `enabled=false` — confirm GET reflects + tick reflects (under `STATE.lock`).
4. Mode chain SAFE→PAPER (with consent) → CANARY → LIVE — final GET `is_freeze_active=false`.
5. Tick at LIVE + override=true — both `closed_learning.frozen` and `structural_evolution.frozen` are `false`.
6. Flip override OFF at LIVE — next tick is frozen again (guards against BUG_0001/BUG_0002: stale supplier reads).
7. SQLite ledger contains ≥ N `OPERATOR_LEARNING_OVERRIDE_CHANGED` rows, each carrying `previous/next/mode/requestor/reason`.
8. `/api/operator/summary.strategies` has no `shadow` key.

## Direct ledger introspection

```bash
python3 -c "
import sqlite3
c = sqlite3.connect('/tmp/test_p0_ledger.db')
for row in c.execute('SELECT seq, kind, payload FROM authority_ledger ORDER BY seq'):
    print(row)
"
```

The ledger schema is `(seq, ts_ns, kind, payload TEXT, prev_hash, hash_chain)` — `payload` is a JSON string, not a JSON column.

## What this skill does NOT cover

- Builder-emission depth (`submitted_samples > 0`, `emitted_events > 0`). No HTTP route exists to inject a `TradeOutcome` into `STATE.feedback_collector`. Use `tests/test_pr_z2_wire_builders.py` instead — it covers the builder contract directly.
- LIVE→AUTO transition (also requires consent; same envelope shape as SAFE→PAPER).
- LOCKED-mode behaviour — out of scope; covered by `/api/operator/action/unlock` tests elsewhere.

## Devin Secrets Needed

None. The whole test runs against a local harness with a transient SQLite ledger and the `DIX_HARNESS_APPROVER_ENABLED=1` auto-approver. No external credentials required.
