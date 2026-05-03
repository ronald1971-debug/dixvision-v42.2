import { useQuery } from "@tanstack/react-query";

import { fetchSources, type SourceRow } from "@/api/governance";

/**
 * Tier-1 governance widget — SCVS Source-liveness grid.
 *
 * SCVS Phase 2 (PR #57) FSM: every external source declares a
 * liveness threshold; the runtime tracks heartbeats / data and
 * escalates a hazard if a critical source goes stale. This panel
 * surfaces the per-source row: provider, auth, status, gap,
 * critical flag.
 *
 * P0-4: SourceManager runtime instance is not yet constructed in
 * `ui.server.STATE`, so `backend_wired=false` initially. Even in
 * that mode the registry rows still render so the operator can
 * audit which sources are declared and their thresholds.
 */
export function SCVSLivenessGrid() {
  const { data, isPending, isError, error, isFetching, refetch } = useQuery({
    queryKey: ["governance", "sources"],
    queryFn: ({ signal }) => fetchSources(signal),
    refetchInterval: 5_000,
  });

  return (
    <section className="flex flex-col h-full rounded border border-border bg-surface">
      <header className="flex items-center justify-between border-b border-border px-3 py-2">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">
            SCVS source liveness
          </h2>
          <p className="text-[11px] text-slate-500">
            Per-source FSM (PR #57 — INV-58)
          </p>
        </div>
        <button
          type="button"
          onClick={() => refetch()}
          disabled={isFetching}
          className="rounded border border-border bg-bg px-2 py-1 text-[11px] hover:border-accent disabled:opacity-50"
        >
          {isFetching ? "…" : "refresh"}
        </button>
      </header>

      <div className="flex-1 overflow-auto p-2 text-xs">
        {isPending && <p className="px-2 py-1 text-slate-500">loading sources…</p>}
        {isError && (
          <p className="px-2 py-1 text-rose-400">
            failed: {(error as Error).message}
          </p>
        )}
        {data && !data.registry_loaded && (
          <p className="px-2 py-1 text-rose-400">
            SourceRegistry failed to load — bidirectional-closure lint may have rejected the manifest
          </p>
        )}
        {data && data.registry_loaded && (
          <table className="w-full border-collapse text-[11px]">
            <thead>
              <tr className="text-[10px] uppercase tracking-wider text-slate-500">
                <th className="px-2 py-1 text-left">source</th>
                <th className="px-2 py-1 text-left">cat.</th>
                <th className="px-2 py-1 text-left">provider</th>
                <th className="px-2 py-1 text-left">auth</th>
                <th className="px-2 py-1 text-left">status</th>
                <th className="px-2 py-1 text-right">gap</th>
                <th className="px-2 py-1 text-right">th. (ms)</th>
                <th className="px-2 py-1 text-center">crit.</th>
              </tr>
            </thead>
            <tbody>
              {data.rows.map((r) => (
                <SourceRowEl key={r.source_id} row={r} backendWired={data.backend_wired} />
              ))}
            </tbody>
          </table>
        )}
      </div>
      {data && !data.backend_wired && (
        <div className="border-t border-border px-3 py-1.5 text-[10px] text-amber-400/80">
          backend not wired — see ZIP analysis P0-4. SourceManager runtime
          instance is not constructed at boot, so liveness status reads as
          UNKNOWN. Registry rows render from the manifest only.
        </div>
      )}
    </section>
  );
}

function SourceRowEl({
  row,
  backendWired,
}: {
  row: SourceRow;
  backendWired: boolean;
}) {
  const status = backendWired ? row.status.toUpperCase() : "UNKNOWN";
  const gapText = formatGap(row.gap_ns);
  return (
    <tr className="border-t border-border/60 hover:bg-bg/40">
      <td className="px-2 py-1 font-mono text-slate-200" title={row.source_id}>
        {row.source_id}
      </td>
      <td className="px-2 py-1 text-slate-400">{row.category}</td>
      <td className="px-2 py-1 text-slate-300">{row.provider}</td>
      <td className="px-2 py-1 text-slate-400">{row.auth}</td>
      <td className="px-2 py-1">
        <StatusChip status={status} />
      </td>
      <td className="px-2 py-1 text-right font-mono text-slate-400">
        {backendWired ? gapText : "—"}
      </td>
      <td className="px-2 py-1 text-right font-mono text-slate-400">
        {row.liveness_threshold_ms.toLocaleString()}
      </td>
      <td className="px-2 py-1 text-center">
        {row.critical && (
          <span
            className="rounded bg-rose-500/20 px-1.5 py-0.5 font-mono text-[9px] text-rose-300"
            title="critical source — staleness escalates a hazard"
          >
            CRIT
          </span>
        )}
      </td>
    </tr>
  );
}

function formatGap(ns: number): string {
  if (ns <= 0) return "—";
  const ms = ns / 1_000_000;
  if (ms < 1_000) return `${ms.toFixed(0)}ms`;
  if (ms < 60_000) return `${(ms / 1_000).toFixed(1)}s`;
  return `${(ms / 60_000).toFixed(1)}m`;
}

function StatusChip({ status }: { status: string }) {
  const tone =
    status === "FRESH" || status === "OK" || status === "ALIVE"
      ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
      : status === "STALE"
        ? "border-amber-500/40 bg-amber-500/10 text-amber-300"
        : status === "DEAD" || status === "FAILED"
          ? "border-rose-500/40 bg-rose-500/10 text-rose-300"
          : "border-slate-600/40 bg-slate-800/40 text-slate-400";
  return (
    <span
      className={`rounded border px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider ${tone}`}
    >
      {status}
    </span>
  );
}
