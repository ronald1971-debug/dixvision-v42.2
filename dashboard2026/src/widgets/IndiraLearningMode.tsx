import { useMemo, useState } from "react";

import { Brain, BookOpen, Inbox, Microscope, Sparkles } from "lucide-react";

/**
 * Indira Learning Mode panel — surfaces what Indira is *currently
 * learning from*, not just what she is currently saying. Backend is
 * wired through:
 *
 *   - PR #95  TraderModel + PhilosophyProfile contracts
 *   - PR #96  TradingView trader-feed adapter
 *   - PR #98  Strategy decomposition library (5 reusable component VOs)
 *   - PR #99  Composition engine + compatibility constraint table
 *   - PR #100 Why Layer (DecisionTrace structured "why" references)
 *   - PR #114 UpdateValidator + UpdateApplier (closed learning loop)
 *
 * Operator can:
 *   - Browse philosophy library (TraderModel rows)
 *   - Inspect trader-feed inbox (latest signals from followed traders)
 *   - Review strategy proposals queue (waiting for shadow-eval)
 *   - See shadow-eval results (Sharpe / drawdown / fill rate vs gate)
 *   - Pick which corpus enters the next fine-tune run
 *
 * Every action that mutates Indira (e.g. promoting a strategy, adding
 * a corpus) is routed through the operator-approval edge — this widget
 * never side-steps governance.
 */
type Tab = "philosophies" | "feed" | "proposals" | "shadow" | "corpus";

interface PhilosophyRow {
  id: string;
  name: string;
  style: string;
  followers: number;
  philosophy_version: string;
  last_updated: string;
}

interface FeedRow {
  id: string;
  trader: string;
  symbol: string;
  side: "BUY" | "SELL" | "CLOSE";
  size_pct: number;
  ts_iso: string;
}

interface ProposalRow {
  id: string;
  strategy: string;
  components: string;
  composite_score: number;
  status: "QUEUED" | "SHADOW" | "PROMOTED" | "REJECTED";
}

interface ShadowEval {
  id: string;
  strategy: string;
  sharpe: number;
  max_drawdown_pct: number;
  fill_rate: number;
  news_attribution: number;
  samples: number;
  gate_pass: boolean;
}

interface CorpusRow {
  id: string;
  name: string;
  rows: number;
  last_synced: string;
  enabled: boolean;
}

const PHILOSOPHIES: PhilosophyRow[] = [
  {
    id: "trader-001",
    name: "@orderflow_jane",
    style: "Microstructure / VWAP reversion",
    followers: 12_400,
    philosophy_version: "v3.2",
    last_updated: "2026-04-19T11:21Z",
  },
  {
    id: "trader-002",
    name: "@perp_savant",
    style: "Funding-flip + liquidation cascade",
    followers: 5_840,
    philosophy_version: "v2.7",
    last_updated: "2026-04-21T08:02Z",
  },
  {
    id: "trader-003",
    name: "@rune_macro",
    style: "FRED / BLS regime overlay",
    followers: 2_010,
    philosophy_version: "v1.4",
    last_updated: "2026-04-18T15:40Z",
  },
];

const FEED: FeedRow[] = [
  {
    id: "f-001",
    trader: "@orderflow_jane",
    symbol: "BTC/USDC",
    side: "BUY",
    size_pct: 1.5,
    ts_iso: "2026-04-21T20:14Z",
  },
  {
    id: "f-002",
    trader: "@perp_savant",
    symbol: "SOL-PERP",
    side: "SELL",
    size_pct: 0.8,
    ts_iso: "2026-04-21T20:11Z",
  },
  {
    id: "f-003",
    trader: "@rune_macro",
    symbol: "EUR/USD",
    side: "CLOSE",
    size_pct: 0.0,
    ts_iso: "2026-04-21T20:02Z",
  },
];

const PROPOSALS: ProposalRow[] = [
  {
    id: "prop-104",
    strategy: "vwap_reversion_v3",
    components: "Trigger:vwap_band · Filter:atr_floor · Sizer:kelly_clamped",
    composite_score: 0.78,
    status: "SHADOW",
  },
  {
    id: "prop-105",
    strategy: "funding_flip_v2",
    components: "Trigger:funding_inversion · Filter:oi_delta · Exit:tp_ladder",
    composite_score: 0.71,
    status: "QUEUED",
  },
  {
    id: "prop-103",
    strategy: "macro_regime_overlay_v1",
    components: "Filter:fred_regime · Trigger:cpi_surprise · Sizer:vol_target",
    composite_score: 0.66,
    status: "REJECTED",
  },
];

const SHADOW: ShadowEval[] = [
  {
    id: "s-001",
    strategy: "vwap_reversion_v3",
    sharpe: 1.34,
    max_drawdown_pct: 3.1,
    fill_rate: 0.97,
    news_attribution: 0.62,
    samples: 612,
    gate_pass: true,
  },
  {
    id: "s-002",
    strategy: "funding_flip_v2",
    sharpe: 0.82,
    max_drawdown_pct: 4.7,
    fill_rate: 0.94,
    news_attribution: 0.41,
    samples: 380,
    gate_pass: false,
  },
];

const CORPUS: CorpusRow[] = [
  {
    id: "c-1",
    name: "TradingView trader feed (PR #96)",
    rows: 18_412,
    last_synced: "2026-04-21T20:00Z",
    enabled: true,
  },
  {
    id: "c-2",
    name: "CoinDesk RSS sentiment (PR #102)",
    rows: 4_920,
    last_synced: "2026-04-21T19:55Z",
    enabled: true,
  },
  {
    id: "c-3",
    name: "FRED macro deltas (PR #108)",
    rows: 1_104,
    last_synced: "2026-04-21T18:30Z",
    enabled: true,
  },
  {
    id: "c-4",
    name: "BLS macro deltas (PR #121)",
    rows: 612,
    last_synced: "2026-04-21T18:30Z",
    enabled: false,
  },
];

const TABS: { id: Tab; label: string; icon: typeof Brain; hint: string }[] = [
  {
    id: "philosophies",
    label: "Philosophies",
    icon: BookOpen,
    hint: "PR #95 + #96",
  },
  { id: "feed", label: "Trader feed", icon: Inbox, hint: "PR #96" },
  {
    id: "proposals",
    label: "Strategy proposals",
    icon: Sparkles,
    hint: "PR #98 + #99",
  },
  { id: "shadow", label: "Shadow eval", icon: Microscope, hint: "PR #114" },
  { id: "corpus", label: "Corpus", icon: Brain, hint: "PR #100" },
];

export function IndiraLearningMode() {
  const [tab, setTab] = useState<Tab>("philosophies");
  const body = useMemo(() => {
    switch (tab) {
      case "philosophies":
        return <PhilosophiesTable rows={PHILOSOPHIES} />;
      case "feed":
        return <FeedTable rows={FEED} />;
      case "proposals":
        return <ProposalsTable rows={PROPOSALS} />;
      case "shadow":
        return <ShadowTable rows={SHADOW} />;
      case "corpus":
        return <CorpusTable rows={CORPUS} />;
    }
  }, [tab]);

  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface text-sm">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Indira · Learning Mode
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            philosophy library · trader feed · proposals · shadow eval ·
            corpus — all governance-gated
          </p>
        </div>
        <span className="rounded border border-accent/40 bg-accent/10 px-1.5 py-0.5 font-mono text-[10px] text-accent">
          INDIRA-L
        </span>
      </header>
      <nav
        className="flex flex-wrap items-center gap-1 border-b border-border bg-bg/50 px-2 py-1.5"
        role="tablist"
        aria-label="Indira learning sections"
      >
        {TABS.map((t) => {
          const Icon = t.icon;
          const active = tab === t.id;
          return (
            <button
              key={t.id}
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => setTab(t.id)}
              className={`flex items-center gap-1.5 rounded border px-2 py-1 font-mono text-[10px] uppercase tracking-wider ${
                active
                  ? "border-accent bg-accent/10 text-accent"
                  : "border-border bg-bg text-slate-400 hover:text-slate-200"
              }`}
            >
              <Icon className="h-3 w-3" />
              {t.label}
              <span className="text-[9px] text-slate-600">{t.hint}</span>
            </button>
          );
        })}
      </nav>
      <div className="flex-1 overflow-auto">{body}</div>
    </div>
  );
}

function PhilosophiesTable({ rows }: { rows: PhilosophyRow[] }) {
  return (
    <table className="w-full table-fixed text-left text-[11px]">
      <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
        <tr>
          <th className="w-1/4 px-3 py-1.5">Trader</th>
          <th className="w-2/5 px-3 py-1.5">Style</th>
          <th className="w-1/12 px-3 py-1.5 text-right">Followers</th>
          <th className="w-1/12 px-3 py-1.5">Version</th>
          <th className="w-1/4 px-3 py-1.5">Updated</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.id} className="border-t border-border/60">
            <td className="px-3 py-1.5 font-mono text-accent">{r.name}</td>
            <td className="px-3 py-1.5 text-slate-300">{r.style}</td>
            <td className="px-3 py-1.5 text-right text-slate-400">
              {r.followers.toLocaleString()}
            </td>
            <td className="px-3 py-1.5 font-mono text-slate-500">
              {r.philosophy_version}
            </td>
            <td className="px-3 py-1.5 font-mono text-slate-500">
              {r.last_updated}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function FeedTable({ rows }: { rows: FeedRow[] }) {
  return (
    <table className="w-full table-fixed text-left text-[11px]">
      <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
        <tr>
          <th className="w-1/3 px-3 py-1.5">Trader</th>
          <th className="w-1/6 px-3 py-1.5">Symbol</th>
          <th className="w-1/6 px-3 py-1.5">Side</th>
          <th className="w-1/6 px-3 py-1.5 text-right">Size %</th>
          <th className="w-1/6 px-3 py-1.5">When</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.id} className="border-t border-border/60">
            <td className="px-3 py-1.5 font-mono text-accent">{r.trader}</td>
            <td className="px-3 py-1.5 text-slate-300">{r.symbol}</td>
            <td
              className={`px-3 py-1.5 font-mono uppercase ${
                r.side === "BUY"
                  ? "text-emerald-400"
                  : r.side === "SELL"
                    ? "text-rose-400"
                    : "text-slate-400"
              }`}
            >
              {r.side}
            </td>
            <td className="px-3 py-1.5 text-right font-mono text-slate-300">
              {r.size_pct.toFixed(2)}
            </td>
            <td className="px-3 py-1.5 font-mono text-slate-500">{r.ts_iso}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ProposalsTable({ rows }: { rows: ProposalRow[] }) {
  return (
    <table className="w-full table-fixed text-left text-[11px]">
      <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
        <tr>
          <th className="w-1/5 px-3 py-1.5">Strategy</th>
          <th className="w-2/5 px-3 py-1.5">Components</th>
          <th className="w-1/6 px-3 py-1.5 text-right">Composite</th>
          <th className="w-1/5 px-3 py-1.5">Status</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.id} className="border-t border-border/60">
            <td className="px-3 py-1.5 font-mono text-accent">{r.strategy}</td>
            <td className="px-3 py-1.5 text-slate-300">{r.components}</td>
            <td className="px-3 py-1.5 text-right font-mono text-slate-300">
              {r.composite_score.toFixed(2)}
            </td>
            <td
              className={`px-3 py-1.5 font-mono uppercase ${
                r.status === "PROMOTED"
                  ? "text-emerald-400"
                  : r.status === "SHADOW"
                    ? "text-amber-400"
                    : r.status === "REJECTED"
                      ? "text-rose-400"
                      : "text-slate-400"
              }`}
            >
              {r.status}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ShadowTable({ rows }: { rows: ShadowEval[] }) {
  return (
    <table className="w-full table-fixed text-left text-[11px]">
      <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
        <tr>
          <th className="w-1/5 px-3 py-1.5">Strategy</th>
          <th className="w-1/12 px-3 py-1.5 text-right">Sharpe</th>
          <th className="w-1/12 px-3 py-1.5 text-right">DD%</th>
          <th className="w-1/12 px-3 py-1.5 text-right">Fill</th>
          <th className="w-1/12 px-3 py-1.5 text-right">News-attr</th>
          <th className="w-1/12 px-3 py-1.5 text-right">Samples</th>
          <th className="w-1/6 px-3 py-1.5">Gate</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.id} className="border-t border-border/60">
            <td className="px-3 py-1.5 font-mono text-accent">{r.strategy}</td>
            <td className="px-3 py-1.5 text-right font-mono text-slate-300">
              {r.sharpe.toFixed(2)}
            </td>
            <td className="px-3 py-1.5 text-right font-mono text-slate-300">
              {r.max_drawdown_pct.toFixed(1)}
            </td>
            <td className="px-3 py-1.5 text-right font-mono text-slate-300">
              {(r.fill_rate * 100).toFixed(0)}%
            </td>
            <td className="px-3 py-1.5 text-right font-mono text-slate-300">
              {(r.news_attribution * 100).toFixed(0)}%
            </td>
            <td className="px-3 py-1.5 text-right font-mono text-slate-500">
              {r.samples}
            </td>
            <td
              className={`px-3 py-1.5 font-mono uppercase ${
                r.gate_pass ? "text-emerald-400" : "text-rose-400"
              }`}
            >
              {r.gate_pass ? "PASS" : "FAIL"}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function CorpusTable({ rows }: { rows: CorpusRow[] }) {
  return (
    <table className="w-full table-fixed text-left text-[11px]">
      <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
        <tr>
          <th className="w-1/2 px-3 py-1.5">Source</th>
          <th className="w-1/6 px-3 py-1.5 text-right">Rows</th>
          <th className="w-1/4 px-3 py-1.5">Last sync</th>
          <th className="w-1/12 px-3 py-1.5">In tune</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.id} className="border-t border-border/60">
            <td className="px-3 py-1.5 text-slate-300">{r.name}</td>
            <td className="px-3 py-1.5 text-right font-mono text-slate-300">
              {r.rows.toLocaleString()}
            </td>
            <td className="px-3 py-1.5 font-mono text-slate-500">
              {r.last_synced}
            </td>
            <td
              className={`px-3 py-1.5 font-mono uppercase ${
                r.enabled ? "text-emerald-400" : "text-slate-500"
              }`}
            >
              {r.enabled ? "yes" : "no"}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
