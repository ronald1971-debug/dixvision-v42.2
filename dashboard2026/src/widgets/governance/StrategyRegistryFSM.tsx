import { useQuery } from "@tanstack/react-query";

import {
  fetchStrategies,
  type StrategiesByState,
  type StrategyRow,
} from "@/api/governance";

const FSM_STATES = [
  "PROPOSED",
  "PAPER",
  "SHADOW",
  "CANARY",
  "LIVE",
  "DECAY",
  "RETIRED",
  "FAILED",
] as const;

type FsmState = (typeof FSM_STATES)[number];

/**
 * Tier-1 governance widget — Strategy registry FSM panel.
 *
 * PR #113 ledger-replayed FSM lifecycle: PROPOSED → PAPER → SHADOW →
 * CANARY → LIVE → DECAY → RETIRED, with FAILED as a side branch. The
 * panel reads `/api/dashboard/strategies` and groups strategies by
 * state across the lifecycle ribbon so the operator can see at a
 * glance how many strategies are at each promotion stage.
 */
export function StrategyRegistryFSM() {
  const { data, isPending, isError, error, isFetching, refetch } = useQuery({
    queryKey: ["governance", "strategies"],
    queryFn: ({ signal }) => fetchStrategies(signal),
    refetchInterval: 6_000,
  });

  return (
    <section className="flex flex-col h-full rounded border border-border bg-surface">
      <header className="flex items-center justify-between border-b border-border px-3 py-2">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">
            Strategy registry FSM
          </h2>
          <p className="text-[11px] text-slate-500">
            PROPOSED → PAPER → SHADOW → CANARY → LIVE → DECAY (PR #113)
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

      <div className="flex-1 overflow-auto p-3 text-xs">
        {isPending && <p className="text-slate-500">loading strategies…</p>}
        {isError && (
          <p className="text-rose-400">
            failed: {(error as Error).message}
          </p>
        )}
        {data && (
          <div className="space-y-3">
            <FsmRibbon data={data} />
            <ul className="flex flex-col gap-2">
              {FSM_STATES.map((state) => (
                <StateGroup
                  key={state}
                  state={state}
                  rows={collectState(data, state)}
                />
              ))}
            </ul>
          </div>
        )}
      </div>
    </section>
  );
}

function collectState(data: StrategiesByState, state: FsmState): StrategyRow[] {
  const rows: StrategyRow[] = [];
  for (const key of Object.keys(data)) {
    if (key.toUpperCase() === state) {
      rows.push(...(data[key] ?? []));
    }
  }
  return rows;
}

function FsmRibbon({ data }: { data: StrategiesByState }) {
  return (
    <div className="flex items-stretch gap-1 overflow-x-auto">
      {FSM_STATES.map((state) => {
        const count = collectState(data, state).length;
        return (
          <div
            key={state}
            className={`flex flex-1 min-w-[80px] flex-col items-center gap-0.5 rounded border px-2 py-1.5 text-center ${stateChrome(state)}`}
          >
            <span className="font-mono text-[10px] uppercase tracking-wider">
              {state}
            </span>
            <span className="font-mono text-base font-semibold">{count}</span>
          </div>
        );
      })}
    </div>
  );
}

function StateGroup({ state, rows }: { state: FsmState; rows: StrategyRow[] }) {
  if (rows.length === 0) return null;
  return (
    <li className="rounded border border-border bg-bg/40 px-2 py-1.5">
      <div
        className={`flex items-center gap-2 text-[10px] uppercase tracking-wider`}
      >
        <span className={chipText(state)}>{state}</span>
        <span className="text-slate-500">({rows.length})</span>
      </div>
      <ul className="mt-1 flex flex-col gap-1">
        {rows.slice(0, 8).map((r) => (
          <li
            key={String(r.strategy_id)}
            className="flex items-baseline justify-between gap-2 text-[11px]"
          >
            <span className="font-mono text-slate-300 truncate">
              {String(r.strategy_id)}
            </span>
            <span className="text-[10px] text-slate-500">
              {Object.keys(r).length} field
            </span>
          </li>
        ))}
        {rows.length > 8 && (
          <li className="text-[10px] text-slate-500">
            +{rows.length - 8} more…
          </li>
        )}
      </ul>
    </li>
  );
}

function stateChrome(state: FsmState): string {
  switch (state) {
    case "PROPOSED":
      return "border-slate-600/40 bg-slate-800/40 text-slate-300";
    case "PAPER":
      return "border-sky-500/40 bg-sky-500/10 text-sky-300";
    case "SHADOW":
      return "border-indigo-500/40 bg-indigo-500/10 text-indigo-300";
    case "CANARY":
      return "border-amber-500/40 bg-amber-500/10 text-amber-300";
    case "LIVE":
      return "border-emerald-500/40 bg-emerald-500/10 text-emerald-300";
    case "DECAY":
      return "border-zinc-500/40 bg-zinc-500/10 text-zinc-300";
    case "RETIRED":
      return "border-slate-500/40 bg-slate-500/10 text-slate-400";
    case "FAILED":
      return "border-rose-500/40 bg-rose-500/10 text-rose-300";
  }
}

function chipText(state: FsmState): string {
  switch (state) {
    case "LIVE":
      return "text-emerald-300";
    case "CANARY":
      return "text-amber-300";
    case "SHADOW":
      return "text-indigo-300";
    case "PAPER":
      return "text-sky-300";
    case "FAILED":
      return "text-rose-300";
    default:
      return "text-slate-400";
  }
}
