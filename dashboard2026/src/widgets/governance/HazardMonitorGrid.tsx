import { useQuery } from "@tanstack/react-query";

import {
  fetchHazards,
  type HazardEvent,
  type HazardTaxonomyRow,
} from "@/api/governance";

/**
 * Tier-1 governance widget — Hazard Monitor (HAZ-01..13 grid).
 *
 * Surfaces the frozen hazard taxonomy plus recent events from the
 * hazard sensor array. Once the array is constructed at boot
 * (P0-2), the recent-events feed populates from the canonical
 * event bus.
 */
export function HazardMonitorGrid() {
  const { data, isPending, isError, error, isFetching, refetch } = useQuery({
    queryKey: ["governance", "hazards"],
    queryFn: ({ signal }) => fetchHazards(signal),
    refetchInterval: 5_000,
  });

  return (
    <section className="flex flex-col h-full rounded border border-border bg-surface">
      <header className="flex items-center justify-between border-b border-border px-3 py-2">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">
            Hazard monitor
          </h2>
          <p className="text-[11px] text-slate-500">
            HAZ-01..13 taxonomy + recent events
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

      <div className="flex flex-1 overflow-hidden text-xs">
        <div className="w-1/2 overflow-auto border-r border-border p-2">
          <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">
            taxonomy
          </div>
          {isPending && <p className="text-slate-500">loading…</p>}
          {isError && (
            <p className="text-rose-400">
              failed: {(error as Error).message}
            </p>
          )}
          {data && (
            <ul className="flex flex-col gap-1">
              {data.taxonomy.map((t) => (
                <TaxonomyRow
                  key={t.code}
                  row={t}
                  recent={data.recent.filter((e) => e.code === t.code).length}
                />
              ))}
            </ul>
          )}
        </div>
        <div className="w-1/2 overflow-auto p-2">
          <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">
            recent events
          </div>
          {data && data.recent.length === 0 && (
            <p className="text-slate-500">
              {data.backend_wired
                ? "no recent hazard events"
                : "backend not wired — sensor array not constructed at boot (P0-2)"}
            </p>
          )}
          {data && data.recent.length > 0 && (
            <ul className="flex flex-col gap-1.5">
              {data.recent.map((e, i) => (
                <EventRow key={i} event={e} />
              ))}
            </ul>
          )}
        </div>
      </div>
    </section>
  );
}

function TaxonomyRow({
  row,
  recent,
}: {
  row: HazardTaxonomyRow;
  recent: number;
}) {
  return (
    <li className="rounded border border-border bg-bg/40 px-2 py-1">
      <div className="flex items-baseline justify-between">
        <span className="font-mono text-[10px] text-accent">{row.code}</span>
        <span className="text-[10px] text-slate-500">{recent} recent</span>
      </div>
      <div className="text-[11px] text-slate-200">{row.label}</div>
      <p className="mt-0.5 text-[10px] leading-snug text-slate-500">
        {row.description}
      </p>
    </li>
  );
}

function EventRow({ event }: { event: HazardEvent }) {
  return (
    <li className="rounded border border-border bg-bg/40 px-2 py-1">
      <div className="flex items-baseline justify-between gap-2">
        <span className="font-mono text-[10px] text-accent">{event.code}</span>
        <SeverityChip severity={event.severity} />
        <span className="ml-auto text-[10px] text-slate-500">
          {event.source}
        </span>
      </div>
      <div className="text-[11px] text-slate-200">{event.summary}</div>
      <div className="text-[10px] text-slate-500">ts_ns {event.ts_ns}</div>
    </li>
  );
}

function SeverityChip({ severity }: { severity: string }) {
  const s = severity.toUpperCase();
  const tone =
    s === "CRITICAL"
      ? "border-rose-600/40 bg-rose-600/15 text-rose-300"
      : s === "HIGH"
        ? "border-rose-500/40 bg-rose-500/10 text-rose-300"
        : s === "MEDIUM" || s === "MED"
          ? "border-amber-500/40 bg-amber-500/10 text-amber-300"
          : s === "LOW" || s === "INFO"
            ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
            : "border-slate-600/40 bg-slate-800/40 text-slate-400";
  return (
    <span
      className={`rounded border px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider ${tone}`}
    >
      {s}
    </span>
  );
}
