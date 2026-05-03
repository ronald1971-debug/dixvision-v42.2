import { useQuery } from "@tanstack/react-query";

import { fetchDrift, type DriftComponent } from "@/api/governance";

/**
 * Tier-1 governance widget — Continuous Drift Oracle (PR #125).
 *
 * Reviewer #4 finding 3 + Reviewer #5 AUTO safeguards: a continuous
 * drift composite over four axes (model / exec / latency / causal)
 * gates AUTO mode. When the composite breaches its threshold the
 * Mode FSM auto-downgrades AUTO → LIVE → SHADOW.
 *
 * P0-7 wiring is incomplete on the live repo (no DriftMonitor
 * instance is constructed in `ui.server.STATE` per metric, and
 * nothing aggregates the readings into a composite). The panel
 * therefore renders the four expected components with their
 * threshold and waits for the runtime to attach. When wired the
 * composite + per-component values render in real time.
 */
export function DriftOraclePanel() {
  const { data, isPending, isError, error, isFetching, refetch } = useQuery({
    queryKey: ["governance", "drift"],
    queryFn: ({ signal }) => fetchDrift(signal),
    refetchInterval: 5_000,
  });

  return (
    <section className="flex flex-col h-full rounded border border-border bg-surface">
      <header className="flex items-center justify-between border-b border-border px-3 py-2">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">
            Drift oracle
          </h2>
          <p className="text-[11px] text-slate-500">
            Continuous AUTO-mode gate (PR #125 — Reviewer #4 finding 3)
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

      <div className="flex-1 overflow-auto p-3 text-xs space-y-3">
        {isPending && <p className="text-slate-500">loading drift…</p>}
        {isError && (
          <p className="text-rose-400">
            failed: {(error as Error).message}
          </p>
        )}
        {data && (
          <>
            <CompositePill
              backendWired={data.backend_wired}
              composite={data.composite}
              threshold={data.downgrade_threshold}
            />

            <div>
              <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">
                components
              </div>
              <ul className="flex flex-col gap-1.5">
                {data.expected_components.map((c) => (
                  <ComponentRow
                    key={c.id}
                    spec={c}
                    live={data.components.find((x) => x.id === c.id)}
                  />
                ))}
              </ul>
            </div>

            {!data.backend_wired && (
              <p className="text-[10px] text-slate-500 leading-snug">
                backend not wired — see ZIP analysis P0-7. The
                <code className="mx-1 text-slate-400">DriftMonitor</code>
                primitive exists at
                <code className="mx-1 text-slate-400">
                  system_engine/state/drift_monitor.py
                </code>
                but no per-metric instances or composite aggregator are
                constructed at boot. Once wired the composite + four
                component values populate from live readings.
              </p>
            )}
          </>
        )}
      </div>
    </section>
  );
}

function CompositePill({
  backendWired,
  composite,
  threshold,
}: {
  backendWired: boolean;
  composite: number | null;
  threshold: number;
}) {
  if (!backendWired || composite === null) {
    return (
      <div className="rounded border border-slate-600/40 bg-slate-800/40 px-2 py-2 text-[11px]">
        <div className="flex items-baseline justify-between">
          <span className="text-slate-400">composite</span>
          <span className="font-mono text-slate-500">—</span>
        </div>
        <div className="mt-1 text-[10px] text-slate-500">
          downgrade threshold: {threshold.toFixed(2)}
        </div>
      </div>
    );
  }
  const breached = composite >= threshold;
  const tone = breached ? "rose" : "emerald";
  return (
    <div
      className={`rounded border border-${tone}-500/40 bg-${tone}-500/10 px-2 py-2 text-[11px]`}
    >
      <div className="flex items-baseline justify-between">
        <span className={`text-${tone}-300`}>composite</span>
        <span className={`font-mono text-${tone}-300`}>
          {composite.toFixed(3)}
        </span>
      </div>
      <div className="mt-1 text-[10px] text-slate-400">
        downgrade threshold: {threshold.toFixed(2)} —{" "}
        {breached ? "BREACH" : "OK"}
      </div>
    </div>
  );
}

function ComponentRow({
  spec,
  live,
}: {
  spec: DriftComponent;
  live: DriftComponent | undefined;
}) {
  const value = live?.value ?? null;
  const breached = value !== null && value >= spec.threshold;
  return (
    <li className="rounded border border-border bg-bg/40 px-2 py-1.5">
      <div className="flex items-baseline justify-between">
        <span className="text-slate-200">{spec.label}</span>
        <span
          className={`font-mono text-[10px] ${
            value === null
              ? "text-slate-500"
              : breached
                ? "text-rose-400"
                : "text-emerald-400"
          }`}
        >
          {value === null ? "—" : value.toFixed(3)} / {spec.threshold.toFixed(2)}
        </span>
      </div>
      <p className="mt-0.5 text-[10px] leading-snug text-slate-500">
        {spec.description}
      </p>
    </li>
  );
}
