# DIX VISION v42.2 — Indira Web-Autolearn Spec

**Status:** reference specification (no runtime code yet)
**Owners:** Indira (perception) · Operator (approval gate)
**Implementation phase:** Phase 1 (trader_knowledge store) → Phase 2 (crawler)
**Governing axioms:** N1 (no decision authority) · N2 (event outputs only) ·
N4 (ledger every output) · N5 (dead-man) · E1–E5 (bounded self-improvement)

This document specifies the **Indira web-autolearn subsystem**: an
autonomous, Playwright-driven, AI-filtered knowledge crawler that ingests
trading education, market reference material, platform documentation,
and research — then surfaces distilled, operator-approved snippets into
the `trader_knowledge` store where strategy plugins can query them.

Nothing in this subsystem is allowed to:

- execute a trade
- approve or modify a trade
- change governance rules
- modify the fast path
- alter the neuromorphic topology
- write directly to any store the strategy engine reads from without
  the operator-approval gate in §5

Its job is **advisory retrieval**, nothing more. It is the same
authority class as the neuromorphic sensors: observe, distill, advise.

---

## 0. Non-negotiables

1. **Playwright, sandboxed.** All crawling runs in a locked-down,
   disposable Playwright container with no access to any DIX private
   key, no access to any broker/exchange API key, no write access to
   any store except the **pending** knowledge buffer.
2. **AI filter is advisory.** The LLM-based content filter suggests
   which snippets are worth keeping; an operator must approve before a
   snippet enters the active `trader_knowledge` pool.
3. **Ledger every crawl.** Each URL fetch, each distilled snippet, each
   approve/reject — all ledgered.
4. **`authority_lint` C3.** The crawler process may not import
   `execution.*`, `governance.*`, `mind.fast_execute`, or any wallet /
   key resolver. Static-analysis enforced.
5. **Rate limits + politeness.** Respect robots.txt; hard per-domain
   request-per-second cap; back-off on any non-200.
6. **Dead-man.** Crawler + filter + curator each expose `check_self()`.
   If any goes silent beyond 3× heartbeat, the subsystem halts and
   emits a `WEB_AUTOLEARN_HAZARD` event.
7. **No login automation without explicit operator opt-in per site.**
   No scraping of sites that require payment or violate ToS.
8. **Source attribution always preserved.** Every snippet carries its
   URL, fetch timestamp, fetch commit hash of the filter prompt, and
   operator-approval signature.

---

## 1. Architecture

```
+---------------------+     +--------------+     +-----------+
| Seed URL list       | --> |   Crawler    | --> |  Raw HTML |
| (docs/seeds.yaml)   |     | (Playwright) |     |  store    |
+---------------------+     +--------------+     +-----+-----+
                                                       |
                                                       v
                                                +--------------+
                                                |  AI Filter   |
                                                |  (LLM)       |
                                                +------+-------+
                                                       |
                                                       v
                                                +--------------+
                                                |  Curator     |
                                                |  (dedupe +   |
                                                |   normalize) |
                                                +------+-------+
                                                       |
                                                       v
                                                +--------------+
                                                |  Pending     |
                                                |  buffer      |
                                                +------+-------+
                                                       |
                                              operator approval
                                                       |
                                                       v
                                                +--------------+
                                                | trader_      |
                                                | knowledge    |
                                                | (active)     |
                                                +------+-------+
                                                       |
                                                       v
                                   strategy plugins query via RAG
```

Five components, one direction of flow, single approval gate.

---

## 2. Components

### 2.1 Crawler (`mind/autolearn/crawler.py` — Phase 2)

- Playwright (Chromium), headless, sandboxed profile.
- No persistent cookies unless operator-opted-in per site.
- Fetch budget: configurable per domain; hard global cap.
- Extracts main content (`readability` / `trafilatura`) + preserves
  source URL, HTTP headers, fetch timestamp.
- Emits `WEB_AUTOLEARN_FETCH` events to the ledger.
- Implements N5 `check_self()`.

### 2.2 AI Filter (`mind/autolearn/filter.py` — Phase 2)

- Local LLM (Ollama) preferred; off-box LLM allowed only if the
  operator provisions an API key and explicitly approves outbound data
  egress.
- Prompt is a pinned artifact in `prompts/autolearn_filter.md` with a
  commit hash; the hash is stored with every snippet.
- Outputs one of: `KEEP` · `DROP` · `FLAG_FOR_REVIEW`.
- Emits `WEB_AUTOLEARN_FILTER` events to the ledger.
- Implements N5 `check_self()`.

### 2.3 Curator (`mind/autolearn/curator.py` — Phase 2)

- Deduplicates by content hash.
- Normalizes snippet schema (title, body, source URL, tags, confidence).
- Tags by taxonomy (see §4).
- Writes to **pending buffer** only.
- Implements N5 `check_self()`.

### 2.4 Pending Buffer (`data/autolearn/pending/*.jsonl`)

- Append-only JSONL files.
- Survives restarts.
- Visible from cockpit "Knowledge Review" tab.

### 2.5 Operator Approval Gate (cockpit)

- Cockpit tab: `/autolearn/review`.
- Shows one pending snippet at a time with source URL, filter output,
  taxonomy tags, confidence score.
- Operator actions: **Approve** (promote to `trader_knowledge`) /
  **Reject** (drop, remember hash to avoid re-ingestion) / **Edit &
  Approve**.
- Approval is signed (operator TOTP) and ledgered.

### 2.6 Trader Knowledge Store (`trader_knowledge/`)

- Write-once, append-only snippet store.
- Indexed for retrieval (Phase 1 builds this).
- Queried by strategy plugins via agentic RAG layer (§6).

---

## 3. Seed URL List (`docs/seeds.yaml`)

Seed list is a version-controlled YAML file. Editing requires a PR.

```yaml
# docs/seeds.yaml (structure; actual file lands in Phase 2)
general_education:
  - https://www.investopedia.com/
  - https://www.babypips.com/
  - https://www.cmegroup.com/education.html
  - https://academy.binance.com/en
  - https://www.quantconnect.com/docs/v2
  - https://menthorq.com/academy/
  - https://menthorq.com/guide/
  - https://predictindicators.ai/blog
  - https://arxiv.org/list/q-fin/recent
  - https://www.sec.gov/edgar
  - https://fred.stlouisfed.org/

memecoin_education:
  - https://memegateway.com/
  - https://99bitcoins.com/
  - https://bingx.com/en/learn
  - https://mudrex.com/learn
  - https://docs.gmgn.ai/
  - https://www.pumpdotfun.com/     # docs subsections only
  - https://station.jup.ag/docs
  - https://www.bullx.io/learn
  - https://trojan.app/docs
  - https://docs.axiom.trade/

forex_and_crypto_markets:
  - https://menthorq.com/academy/forex/
  - https://www.tradealgo.com/
  - https://www.brokeranalysis.com/reviews/
  - https://www.forexfactory.com/calendar

protocols_and_exchanges:
  - https://hyperliquid.xyz/docs
  - https://docs.dydx.exchange/
  - https://station.jup.ag/docs
  - https://docs.jito.wtf/
  - https://developers.binance.com/
  - https://docs.cdp.coinbase.com/
  - https://docs.kraken.com/
  - https://www.interactivebrokers.com/campus/
  - https://alpaca.markets/docs/

safety_and_onchain_forensics:
  - https://rugcheck.xyz/docs
  - https://honeypot.is/
  - https://gopluslabs.io/
  - https://de.fi/
  - https://skynet.certik.com/
  - https://solscan.io/docs
  - https://birdeye.so/docs
  - https://docs.dexscreener.com/
  - https://docs.dextools.io/

research_and_theory:
  - https://arxiv.org/list/q-fin.TR/recent
  - https://arxiv.org/list/q-fin.PM/recent
  - https://ssrn.com/
  - https://www.risk.net/
```

New seed URLs are added via PR; the PR itself is the audit trail.

Any URL that requires login or payment is marked with an explicit
`requires_operator_session: true` flag and is **off** by default.

---

## 4. Taxonomy (tagging schema)

Every distilled snippet is tagged along these axes:

- **Domain:** `forex | crypto_major | crypto_memecoin | equities | options | futures | macro | microstructure`
- **Topic:** `strategy | risk | safety | execution | market_structure | indicator | tool | glossary`
- **Depth:** `beginner | intermediate | advanced | research`
- **Actionable:** `yes | no` — does this snippet describe something a
  strategy could cite or use?
- **Source-trust:** `high | medium | low` — based on operator-maintained
  source whitelist.

Taxonomy is version-controlled in `docs/autolearn_taxonomy.md` and
evolves via PR.

---

## 5. Operator Approval Gate (explicit requirements)

Before any snippet reaches `trader_knowledge`:

1. The crawler fetch event is ledgered.
2. The AI filter output is ledgered.
3. The curator's normalized snippet is written to the pending buffer.
4. The operator reviews via cockpit.
5. The operator's Approve / Reject / Edit+Approve action is TOTP-signed
   and ledgered.
6. Only after operator approval does the snippet enter
   `trader_knowledge`.

Rejected snippet content hashes are remembered so the crawler does not
re-present them.

---

## 6. Agentic RAG Layer (consumption side)

Strategy plugins and the cockpit UI can query `trader_knowledge` for
context. This is an advisory retrieval layer, not a decision layer.

- Query API: `trader_knowledge.query(context: dict, k: int) -> list[Snippet]`.
- Inputs: free-text question + optional structured filters (domain,
  topic, depth, tags).
- Output: ranked snippets with source URL, tags, confidence.
- **Plugins may cite snippets in their decision rationale (logged) but
  may never treat the snippet as authoritative input.** Governance
  still decides via the same deterministic rules.

Example strategy usage:

```python
ctx = trader_knowledge.query(
    {"topic": "0dte_gamma", "domain": "options", "depth": "advanced"},
    k=3,
)
# ctx is displayed in the decision rationale; it does not bypass
# governance thresholds, position caps, or the neuromorphic risk advisor.
```

---

## 7. Authority Boundaries (`authority_lint` C3)

New `authority_lint` rule **C3**:

```
FORBIDDEN import edges from mind/autolearn/*:
  - execution.*
  - governance.*
  - mind.fast_execute
  - any wallet / private-key resolver
  - any adapter that can place an order
```

Violations fail CI. This is the static guarantee that the crawler and
RAG layer cannot decide or execute.

---

## 8. Failure Modes & Responses

| Failure | Response |
|--------|----------|
| Crawler goes silent (N5 stale) | Subsystem halts; `WEB_AUTOLEARN_HAZARD` emitted; operator notified |
| AI filter goes silent | Same as above |
| Curator goes silent | Same as above |
| Pending buffer disk-full | Halt ingestion; `WEB_AUTOLEARN_HAZARD` |
| Seed URL domain returns 403 / captcha | Back off; notify operator; do not retry aggressively |
| Source flagged as compromised (e.g. known malware domain) | Quarantine all snippets from that domain; emit `WEB_AUTOLEARN_QUARANTINE` |
| LLM filter output malformed | Drop snippet; log; count against filter health score |
| Operator rejects X consecutive snippets from a source | Auto-demote source trust; surface to operator for seed-list edit |

---

## 9. Phased Build-Out

| Phase | What lands |
|-------|-----------|
| 1 | `trader_knowledge` store + retrieval API (read side only) |
| 2 | Crawler + AI filter + curator + pending buffer |
| 2 | `mind/autolearn/*` package + `authority_lint` C3 rule |
| 2 | Cockpit `/autolearn/review` tab (operator approval gate) |
| 2 | `docs/seeds.yaml` with initial seed list (§3) |
| 4 | Integration with the governance kernel's rationale logger (RAG citations surface in decision logs) |
| 7 | Observability: Prometheus metrics — crawl rate, filter accept-rate,
    operator-approve rate, per-source trust score drift |

---

## 10. Bounded Self-Improvement — What This Subsystem May / May Not Change

Per axioms **E1–E5** (to be codified in `immutable_core/evolution_axioms.lean`):

| Action | Allowed? |
|--------|---------|
| Ingest a new URL | Yes (operator-approved seed) |
| Promote a snippet to active `trader_knowledge` | Yes (operator-approved) |
| Demote a source's trust score | Yes (bounded, ledgered) |
| Surface snippets into strategy decision rationales | Yes (advisory) |
| Update feature weights based on retrieved snippets | Yes — sandbox-gated, bounded, backtest-required, operator-approved (Phase 4+) |
| Synthesize a new strategy template citing snippets | Yes — sandbox-gated, backtest-proven, operator-approved promote chain |
| Modify any governance rule | **No** — governance is immutable at runtime |
| Modify the fast path | **No** — frozen by axiom |
| Modify the neuromorphic topology | **No** — axiom N8 |
| Modify kill-switch / dead-man / wallet policy | **No** — locked |
| Place or approve a trade | **No** — ever |
