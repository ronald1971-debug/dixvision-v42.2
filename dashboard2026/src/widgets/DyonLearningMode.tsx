import { useMemo, useState } from "react";

import { AlertTriangle, FlaskConical, GitMerge, Wrench } from "lucide-react";

/**
 * Dyon Learning Mode panel — surfaces what Dyon is *currently
 * learning to fix*, not just what Dyon is currently saying. Backend
 * is wired through:
 *
 *   - PR #32  hazard sensors HAZ-01..12 + health monitors
 *   - PR #33  sandbox / memory_overflow / anomaly_detector hardening
 *   - PR #65  patch pipeline orchestrator + ledger surface (INV-66)
 *   - PR #114 UpdateValidator + UpdateApplier (closed learning loop)
 *
 * Operator can:
 *   - Browse hazard journal (HAZ-01..12) — what fired, when, severity
 *   - Inspect patch proposals queue — what Dyon wants to change
 *   - Review sandbox runs — coverage / regression / lint results
 *   - See promotion ledger — what landed on main vs reverted
 *
 * Every promotion goes through the operator-approval edge — this
 * widget shows the queue but never side-steps the gate.
 */
type Tab = "hazards" | "patches" | "sandbox" | "promotions";

interface HazardRow {
  id: string;
  code: string;
  severity: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";
  message: string;
  ts_iso: string;
  acknowledged: boolean;
}

interface PatchProposal {
  id: string;
  hazard_code: string;
  module: string;
  diff_summary: string;
  status: "QUEUED" | "SANDBOX" | "AWAITING_APPROVAL" | "MERGED" | "REVERTED";
}

interface SandboxRun {
  id: string;
  patch_id: string;
  coverage_delta: number;
  regression: boolean;
  lint_clean: boolean;
  duration_ms: number;
}

interface PromotionRow {
  id: string;
  patch_id: string;
  result: "MERGED" | "REVERTED";
  reason: string;
  ts_iso: string;
}

const HAZARDS: HazardRow[] = [
  {
    id: "h-001",
    code: "HAZ-LATENCY-P99",
    severity: "MEDIUM",
    message: "Order ack p99 = 142ms (target ≤ 100ms) on Binance perp leg",
    ts_iso: "2026-04-21T20:14Z",
    acknowledged: false,
  },
  {
    id: "h-002",
    code: "HAZ-NEWS-SHOCK",
    severity: "HIGH",
    message: "CoinDesk burst (>5 items / 60s) — auto-throttle engaged",
    ts_iso: "2026-04-21T19:58Z",
    acknowledged: true,
  },
  {
    id: "h-003",
    code: "HAZ-MEMORY",
    severity: "LOW",
    message: "ledger hot-ring 73% full — compaction triggered",
    ts_iso: "2026-04-21T19:42Z",
    acknowledged: true,
  },
  {
    id: "h-004",
    code: "HAZ-DRIFT-MODEL",
    severity: "HIGH",
    message: "drift composite = 0.61 (warn 0.50, fail 0.75)",
    ts_iso: "2026-04-21T19:30Z",
    acknowledged: false,
  },
];

const PATCHES: PatchProposal[] = [
  {
    id: "patch-118",
    hazard_code: "HAZ-LATENCY-P99",
    module: "execution_engine/hot_path",
    diff_summary: "switch order_ack callback to ring_buffer (-23ms p99)",
    status: "AWAITING_APPROVAL",
  },
  {
    id: "patch-119",
    hazard_code: "HAZ-DRIFT-MODEL",
    module: "intelligence_engine/regime_router",
    diff_summary: "tighten hysteresis band (0.6 → 0.45) for vol regimes",
    status: "SANDBOX",
  },
  {
    id: "patch-120",
    hazard_code: "HAZ-MEMORY",
    module: "core/ledger/hot_ring",
    diff_summary: "raise compaction trigger from 70% → 85%",
    status: "QUEUED",
  },
];

const SANDBOX: SandboxRun[] = [
  {
    id: "sb-001",
    patch_id: "patch-118",
    coverage_delta: 0.012,
    regression: false,
    lint_clean: true,
    duration_ms: 4_810,
  },
  {
    id: "sb-002",
    patch_id: "patch-119",
    coverage_delta: -0.004,
    regression: false,
    lint_clean: true,
    duration_ms: 6_120,
  },
];

const PROMOTIONS: PromotionRow[] = [
  {
    id: "pr-117",
    patch_id: "patch-117",
    result: "MERGED",
    reason: "all gates passed · operator approved",
    ts_iso: "2026-04-21T18:11Z",
  },
  {
    id: "pr-116",
    patch_id: "patch-116",
    result: "REVERTED",
    reason: "post-merge HAZ-LATENCY-P99 spike",
    ts_iso: "2026-04-21T17:02Z",
  },
];

const TABS: { id: Tab; label: string; icon: typeof Wrench; hint: string }[] = [
  { id: "hazards", label: "Hazard journal", icon: AlertTriangle, hint: "PR #32" },
  { id: "patches", label: "Patch proposals", icon: Wrench, hint: "PR #65" },
  { id: "sandbox", label: "Sandbox runs", icon: FlaskConical, hint: "PR #33" },
  { id: "promotions", label: "Promotions", icon: GitMerge, hint: "PR #114" },
];

export function DyonLearningMode() {
  const [tab, setTab] = useState<Tab>("hazards");
  const body = useMemo(() => {
    switch (tab) {
      case "hazards":
        return <HazardsTable rows={HAZARDS} />;
      case "patches":
        return <PatchesTable rows={PATCHES} />;
      case "sandbox":
        return <SandboxTable rows={SANDBOX} />;
      case "promotions":
        return <PromotionsTable rows={PROMOTIONS} />;
    }
  }, [tab]);

  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface text-sm">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Dyon · Learning Mode
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            hazard journal · patch proposals · sandbox runs · promotion ledger
            — every merge gated
          </p>
        </div>
        <span className="rounded border border-accent/40 bg-accent/10 px-1.5 py-0.5 font-mono text-[10px] text-accent">
          DYON-L
        </span>
      </header>
      <nav
        className="flex flex-wrap items-center gap-1 border-b border-border bg-bg/50 px-2 py-1.5"
        role="tablist"
        aria-label="Dyon learning sections"
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

function severityClass(severity: HazardRow["severity"]): string {
  switch (severity) {
    case "CRITICAL":
      return "text-rose-500";
    case "HIGH":
      return "text-rose-400";
    case "MEDIUM":
      return "text-amber-400";
    case "LOW":
      return "text-slate-400";
  }
}

function HazardsTable({ rows }: { rows: HazardRow[] }) {
  return (
    <table className="w-full table-fixed text-left text-[11px]">
      <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
        <tr>
          <th className="w-1/5 px-3 py-1.5">Code</th>
          <th className="w-1/12 px-3 py-1.5">Sev</th>
          <th className="w-2/5 px-3 py-1.5">Message</th>
          <th className="w-1/6 px-3 py-1.5">When</th>
          <th className="w-1/12 px-3 py-1.5">Ack</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.id} className="border-t border-border/60">
            <td className="px-3 py-1.5 font-mono text-accent">{r.code}</td>
            <td
              className={`px-3 py-1.5 font-mono uppercase ${severityClass(r.severity)}`}
            >
              {r.severity}
            </td>
            <td className="px-3 py-1.5 text-slate-300">{r.message}</td>
            <td className="px-3 py-1.5 font-mono text-slate-500">{r.ts_iso}</td>
            <td
              className={`px-3 py-1.5 font-mono uppercase ${
                r.acknowledged ? "text-emerald-400" : "text-rose-400"
              }`}
            >
              {r.acknowledged ? "yes" : "no"}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function PatchesTable({ rows }: { rows: PatchProposal[] }) {
  return (
    <table className="w-full table-fixed text-left text-[11px]">
      <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
        <tr>
          <th className="w-1/12 px-3 py-1.5">Patch</th>
          <th className="w-1/6 px-3 py-1.5">Hazard</th>
          <th className="w-1/4 px-3 py-1.5">Module</th>
          <th className="w-1/3 px-3 py-1.5">Diff</th>
          <th className="w-1/6 px-3 py-1.5">Status</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.id} className="border-t border-border/60">
            <td className="px-3 py-1.5 font-mono text-accent">{r.id}</td>
            <td className="px-3 py-1.5 font-mono text-slate-400">
              {r.hazard_code}
            </td>
            <td className="px-3 py-1.5 font-mono text-slate-300">
              {r.module}
            </td>
            <td className="px-3 py-1.5 text-slate-300">{r.diff_summary}</td>
            <td
              className={`px-3 py-1.5 font-mono uppercase ${
                r.status === "MERGED"
                  ? "text-emerald-400"
                  : r.status === "AWAITING_APPROVAL"
                    ? "text-amber-400"
                    : r.status === "REVERTED"
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

function SandboxTable({ rows }: { rows: SandboxRun[] }) {
  return (
    <table className="w-full table-fixed text-left text-[11px]">
      <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
        <tr>
          <th className="w-1/6 px-3 py-1.5">Run</th>
          <th className="w-1/6 px-3 py-1.5">Patch</th>
          <th className="w-1/6 px-3 py-1.5 text-right">Δ Coverage</th>
          <th className="w-1/6 px-3 py-1.5">Regression</th>
          <th className="w-1/6 px-3 py-1.5">Lint</th>
          <th className="w-1/6 px-3 py-1.5 text-right">Duration</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.id} className="border-t border-border/60">
            <td className="px-3 py-1.5 font-mono text-accent">{r.id}</td>
            <td className="px-3 py-1.5 font-mono text-slate-400">
              {r.patch_id}
            </td>
            <td
              className={`px-3 py-1.5 text-right font-mono ${
                r.coverage_delta >= 0 ? "text-emerald-400" : "text-rose-400"
              }`}
            >
              {(r.coverage_delta * 100).toFixed(2)}%
            </td>
            <td
              className={`px-3 py-1.5 font-mono uppercase ${
                r.regression ? "text-rose-400" : "text-emerald-400"
              }`}
            >
              {r.regression ? "yes" : "no"}
            </td>
            <td
              className={`px-3 py-1.5 font-mono uppercase ${
                r.lint_clean ? "text-emerald-400" : "text-rose-400"
              }`}
            >
              {r.lint_clean ? "clean" : "dirty"}
            </td>
            <td className="px-3 py-1.5 text-right font-mono text-slate-400">
              {r.duration_ms.toLocaleString()}ms
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function PromotionsTable({ rows }: { rows: PromotionRow[] }) {
  return (
    <table className="w-full table-fixed text-left text-[11px]">
      <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
        <tr>
          <th className="w-1/6 px-3 py-1.5">Promotion</th>
          <th className="w-1/6 px-3 py-1.5">Patch</th>
          <th className="w-1/6 px-3 py-1.5">Result</th>
          <th className="w-1/3 px-3 py-1.5">Reason</th>
          <th className="w-1/6 px-3 py-1.5">When</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.id} className="border-t border-border/60">
            <td className="px-3 py-1.5 font-mono text-accent">{r.id}</td>
            <td className="px-3 py-1.5 font-mono text-slate-400">
              {r.patch_id}
            </td>
            <td
              className={`px-3 py-1.5 font-mono uppercase ${
                r.result === "MERGED" ? "text-emerald-400" : "text-rose-400"
              }`}
            >
              {r.result}
            </td>
            <td className="px-3 py-1.5 text-slate-300">{r.reason}</td>
            <td className="px-3 py-1.5 font-mono text-slate-500">{r.ts_iso}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
