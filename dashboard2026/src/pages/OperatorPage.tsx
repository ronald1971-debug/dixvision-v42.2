import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { fetchOperatorSummary, postOperatorKill } from "@/api/operator";
import { AdapterStatusGrid } from "@/components/AdapterStatusGrid";
import { EngineBucketBadge } from "@/components/EngineBucketBadge";
import { HotkeyConfigurator } from "@/components/HotkeyConfigurator";
import { PopoutButton } from "@/components/PopoutButton";
import type {
  OperatorActionResponse,
  OperatorStrategyCounts,
} from "@/types/generated/api";

const STRATEGY_LABELS: Array<[keyof OperatorStrategyCounts, string]> = [
  ["proposed", "PROPOSED"],
  ["shadow", "SHADOW"],
  ["canary", "CANARY"],
  ["live", "LIVE"],
  ["retired", "RETIRED"],
  ["failed", "FAILED"],
];

export function OperatorPage() {
  const queryClient = useQueryClient();
  const { data, isPending, isError, error, refetch, isFetching } = useQuery({
    queryKey: ["operator", "summary"],
    queryFn: ({ signal }) => fetchOperatorSummary(signal),
    refetchInterval: 5_000,
  });

  const [killReason, setKillReason] = useState("operator kill");
  const [actionLog, setActionLog] = useState<
    Array<{ ts: string; approved: boolean; summary: string }>
  >([]);

  const killMutation = useMutation({
    mutationFn: () => postOperatorKill({ reason: killReason }),
    onSuccess: (resp: OperatorActionResponse) => {
      setActionLog((rows) => [
        {
          ts: new Date().toLocaleTimeString(),
          approved: resp.approved,
          summary: resp.summary,
        },
        ...rows.slice(0, 9),
      ]);
      queryClient.invalidateQueries({ queryKey: ["operator", "summary"] });
    },
    onError: (err: Error) => {
      setActionLog((rows) => [
        {
          ts: new Date().toLocaleTimeString(),
          approved: false,
          summary: `request failed: ${err.message}`,
        },
        ...rows.slice(0, 9),
      ]);
    },
  });

  return (
    <section className="max-w-6xl mx-auto space-y-5">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">
            Operator control plane
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Mode FSM, engine health, strategy counts, and the kill
            switch. Read projection of the Phase 6 widgets, refreshed
            every 5 s.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <PopoutButton route="operator" />
          <button
            type="button"
            onClick={() => refetch()}
            className="rounded border border-border bg-surface px-3 py-1.5 text-xs hover:border-accent disabled:opacity-50"
            disabled={isFetching}
          >
            {isFetching ? "refreshing…" : "refresh"}
          </button>
        </div>
      </div>

      {isPending && <p className="text-sm text-slate-400">Loading…</p>}

      {isError && (
        <div className="rounded border border-danger/40 bg-danger/10 p-3 text-sm text-danger">
          Failed to load operator summary: {(error as Error).message}
        </div>
      )}

      {data && (
        <>
          <ModeCard data={data.mode} />
          <EnginesCard rows={data.engines} />
          <AdapterStatusGrid />
          <StrategiesCard counts={data.strategies} />
          <MemecoinCard data={data.memecoin} />
          <DecisionCountCard count={data.decision_chain_count} />
          <KillCard
            reason={killReason}
            onReasonChange={setKillReason}
            onKill={() => killMutation.mutate()}
            isSubmitting={killMutation.isPending}
            log={actionLog}
            isLocked={data.mode.is_locked}
          />
          <HotkeyConfigurator />
        </>
      )}
    </section>
  );
}

function ModeCard({
  data,
}: {
  data: { current_mode: string; legal_targets: string[]; is_locked: boolean };
}) {
  return (
    <div className="rounded border border-border bg-surface p-4">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
        Mode FSM <span className="ml-2 text-slate-600">DASH-02</span>
      </h2>
      <div className="grid grid-cols-3 gap-4">
        <Tile label="current mode" value={data.current_mode} />
        <Tile
          label="legal targets"
          value={data.legal_targets.join(", ") || "—"}
        />
        <Tile
          label="locked"
          value={data.is_locked ? "yes" : "no"}
          tone={data.is_locked ? "danger" : "ok"}
        />
      </div>
    </div>
  );
}

function EnginesCard({
  rows,
}: {
  rows: Array<{
    engine_name: string;
    bucket: string;
    detail: string;
    plugin_count: number;
  }>;
}) {
  return (
    <div className="rounded border border-border bg-surface p-4">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
        Engine status <span className="ml-2 text-slate-600">DASH-EG-01</span>
      </h2>
      {rows.length === 0 ? (
        <p className="text-xs text-slate-500">no engines registered</p>
      ) : (
        <table className="w-full text-left text-sm">
          <thead className="text-xs uppercase text-slate-500">
            <tr>
              <th className="px-3 py-2">engine</th>
              <th className="px-3 py-2">bucket</th>
              <th className="px-3 py-2">detail</th>
              <th className="px-3 py-2 text-right">plugins</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.engine_name} className="border-t border-border">
                <td className="px-3 py-2 font-mono text-xs">
                  {row.engine_name}
                </td>
                <td className="px-3 py-2">
                  <EngineBucketBadge bucket={row.bucket} />
                </td>
                <td className="px-3 py-2 text-xs text-slate-400">
                  {row.detail || "—"}
                </td>
                <td className="px-3 py-2 text-right font-mono text-xs">
                  {row.plugin_count}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function StrategiesCard({ counts }: { counts: OperatorStrategyCounts }) {
  return (
    <div className="rounded border border-border bg-surface p-4">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
        Strategy lifecycle{" "}
        <span className="ml-2 text-slate-600">DASH-SLP-01</span>
      </h2>
      <div className="grid grid-cols-6 gap-2 font-mono text-xs">
        {STRATEGY_LABELS.map(([key, label]) => (
          <Tile key={key} label={label.toLowerCase()} value={counts[key]} />
        ))}
      </div>
    </div>
  );
}

function MemecoinCard({
  data,
}: {
  data: { enabled: boolean; killed: boolean; summary: string };
}) {
  return (
    <div className="rounded border border-border bg-surface p-4">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
        Memecoin subsystem{" "}
        <span className="ml-2 text-slate-600">DASH-MCP-01</span>
      </h2>
      <div className="grid grid-cols-3 gap-4">
        <Tile
          label="enabled"
          value={data.enabled ? "yes" : "no"}
          tone={data.enabled ? "ok" : undefined}
        />
        <Tile
          label="killed"
          value={data.killed ? "yes" : "no"}
          tone={data.killed ? "danger" : undefined}
        />
        <Tile label="summary" value={data.summary || "—"} />
      </div>
    </div>
  );
}

function DecisionCountCard({ count }: { count: number }) {
  return (
    <div className="rounded border border-border bg-surface p-4">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
        Decision trace <span className="ml-2 text-slate-600">DASH-04</span>
      </h2>
      <Tile label="symbols traced" value={count} />
      <p className="mt-2 text-xs text-slate-500">
        Per-event detail rolls up on the legacy <code>/operator</code>{" "}
        page; richer trace view is queued for a follow-up wave-02 PR.
      </p>
    </div>
  );
}

function KillCard({
  reason,
  onReasonChange,
  onKill,
  isSubmitting,
  log,
  isLocked,
}: {
  reason: string;
  onReasonChange: (v: string) => void;
  onKill: () => void;
  isSubmitting: boolean;
  log: Array<{ ts: string; approved: boolean; summary: string }>;
  isLocked: boolean;
}) {
  return (
    <div className="rounded border border-danger/40 bg-danger/5 p-4">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-danger">
        Kill switch
      </h2>
      <p className="mb-3 text-xs text-slate-400">
        Submits a <code>REQUEST_KILL</code> through{" "}
        <code>ControlPlaneRouter</code> →{" "}
        <code>OperatorInterfaceBridge</code> (GOV-CP-07). The decision
        Governance returns is logged below verbatim.
        {isLocked && (
          <span className="ml-1 text-warn">
            System is already LOCKED — a fresh kill will likely be a
            no-op.
          </span>
        )}
      </p>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!isSubmitting) onKill();
        }}
        className="mb-3 flex flex-wrap items-end gap-2"
      >
        <label className="flex flex-col text-xs text-slate-400">
          reason
          <input
            type="text"
            value={reason}
            onChange={(e) => onReasonChange(e.target.value)}
            className="mt-1 w-72 rounded border border-border bg-surface px-2 py-1 font-mono text-xs text-slate-200"
            maxLength={512}
          />
        </label>
        <button
          type="submit"
          disabled={isSubmitting || reason.trim().length === 0}
          className="rounded border border-danger bg-danger/20 px-4 py-1.5 text-xs font-semibold text-danger hover:bg-danger/30 disabled:opacity-50"
        >
          {isSubmitting ? "submitting…" : "KILL"}
        </button>
      </form>

      {log.length === 0 ? (
        <p className="text-xs text-slate-500">no operator actions yet</p>
      ) : (
        <ul className="space-y-1 font-mono text-xs">
          {log.map((row, idx) => (
            <li
              key={idx}
              className={
                row.approved ? "text-ok" : "text-danger"
              }
            >
              [{row.ts}] {row.approved ? "OK" : "DENY"} — {row.summary}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Tile({
  label,
  value,
  tone,
}: {
  label: string;
  value: string | number;
  tone?: "ok" | "warn" | "danger";
}) {
  const toneClass =
    tone === "ok"
      ? "text-ok"
      : tone === "warn"
        ? "text-warn"
        : tone === "danger"
          ? "text-danger"
          : "text-slate-200";
  return (
    <div className="rounded border border-border bg-bg px-3 py-2">
      <div className="text-[10px] uppercase tracking-wide text-slate-500">
        {label}
      </div>
      <div className={`mt-1 font-mono text-sm ${toneClass}`}>{value}</div>
    </div>
  );
}
