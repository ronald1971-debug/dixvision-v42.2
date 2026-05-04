# sensory.web_autolearn — operator activation guide

This document is the canonical activation path for the
`sensory/web_autolearn/` pipeline. It is intentionally brief: the
pipeline is engine-agnostic, so the operator surface is just two
files plus one harness flag.

## What this pipeline is (and is not)

`sensory.web_autolearn` is the **curator-side** ingestion pipeline.
It is *not* a SCVS source. SCVS providers (Reuters, X, Reddit, FRED,
BLS, etc.) live in `registry/data_source_registry.yaml` and are
credentialed via `scripts/check_credentials`.

The autolearn pipeline runs on top of those providers — it polls
public-facing URLs the operator picks, runs them through the
relevance filter, applies admissibility rules, and pushes admitted
items into the HITL `PendingBuffer`. It does **not** emit
`SignalEvent` directly: every admitted item still requires operator
approval before it can fan into Indira.

```
seeds.yaml ─► Crawler (Playwright) ─► AIFilter ─► Curator ─► PendingBuffer ─► [HITL] ─► NewsFanout
                                                                                          │
                                                                                          └─► SignalEvent
```

Because it is HITL-gated, it is safe to run with no seeds at all —
the pipeline degrades to a no-op rather than emitting noisy or
unauthenticated rows.

## Activation

### 1. Edit `sensory/web_autolearn/seeds.yaml`

Uncomment one or more of the starter seeds. The starter rows ship
disabled by default so a fresh clone never emits anything.

The starter rows cover four mainstream news sources (CoinDesk,
Decrypt, CoinTelegraph, The Block) plus a placeholder X aggregator.
The four news rows are RSS feeds and require **no credentials**;
they fetch over plain HTTPS. The X row is a placeholder for an
operator-managed webhook aggregator (Zapier / Make / n8n) — replace
the example URL with your own endpoint before uncommenting.

Each row's schema is documented in the file header. Briefly:

| field       | required | type     | notes                                                  |
| ----------- | -------- | -------- | ------------------------------------------------------ |
| `url`       | yes      | str      | canonical URL the crawler fetches                       |
| `topic`     | yes      | str      | carried into `CuratedItem.seed_topic`                  |
| `keywords`  | no       | list[str]| relevance keywords for the AIFilter (case-insensitive) |
| `min_score` | no       | float    | curator threshold in `[0.0, 1.0]` (default `0.0`)      |
| `allow`     | no       | list[str]| if non-empty, at least one must appear in title+body   |
| `deny`      | no       | list[str]| any match drops the item                               |
| `tags`      | no       | list[str]| carry-through tags applied to admitted items           |

### 2. Verify the YAML parses

The smoke test re-runs the parser against the on-disk file:

```bash
python -m pytest tests/sensory/web_autolearn/test_seeds_yaml.py -v
```

A typo in `seeds.yaml` will trip the `CuratorRules.from_mapping`
contract here (e.g. `deny: sponsored` instead of `deny: [sponsored]`)
before any harness boot.

### 3. Boot the harness with the autolearn pipeline enabled

The pipeline is wired via the same `_State` bootstrap as the rest of
the harness. There is no separate flag — once `seeds.yaml` has at
least one uncommented seed and the harness boots, the crawler picks
up the file at startup.

For Windows operators the `.bat` launchers boot the harness
unconditionally, so the only manual step is editing `seeds.yaml`.

For Linux / direct uvicorn launches:

```bash
DIXVISION_LEDGER_PATH=$HOME/.dixvision/governance.db \
  uvicorn ui.server:app --host 127.0.0.1 --port 8080
```

### 4. Approve admitted items

Admitted items land in the HITL `PendingBuffer`. Open the operator
dashboard (`http://127.0.0.1:8080/dash2/`) and review pending rows
under the Sensory pane. Approving an item fans it into `NewsFanout`,
which projects to a `SignalEvent` and runs the `NewsShockSensor`
hazard probe.

Rejecting an item drops it without a `SignalEvent` write. All
approve / reject events land in the authority ledger as
`OPERATOR_SETTINGS_CHANGED` rows.

## Disabling the pipeline

Comment every row in `seeds.yaml` (or delete the file's body so it
parses to `None`). The pipeline degrades to a no-op on the next
harness boot. Existing pending items are kept until the operator
explicitly clears them.

## Troubleshooting

| symptom                                     | likely cause                                                           |
| ------------------------------------------- | ---------------------------------------------------------------------- |
| `seed rule topic must be non-empty`         | a `topic:` line is missing or empty                                    |
| `seed 'foo' allow must be a list, not a string` | wrote `allow: sponsored` instead of `allow: [sponsored]`                 |
| no admitted items, but feed is live         | `min_score` too high, or `keywords` don't match the feed's vocabulary  |
| crawler returns `fetched_ok=False`          | URL is unreachable from the host network; check firewall / proxy       |
| HTTP 429 from a provider                    | back off the seed; the crawler does not throttle below the seed level  |

## Related files

- `sensory/web_autolearn/contracts.py` — value types (`RawDocument`,
  `FilteredItem`, `CuratedItem`).
- `sensory/web_autolearn/crawler.py` — Playwright-based fetcher.
- `sensory/web_autolearn/ai_filter.py` — `KeywordAIFilter` (the
  default; operators may replace with a model-backed filter later).
- `sensory/web_autolearn/curator.py` — `CuratorRules` parser and
  admissibility logic.
- `sensory/web_autolearn/pending_buffer.py` — HITL queue.
- `tests/sensory/web_autolearn/test_seeds_yaml.py` — parser smoke
  test (run before every commit that edits `seeds.yaml`).
