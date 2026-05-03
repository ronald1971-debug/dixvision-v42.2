import { useQuery } from "@tanstack/react-query";

import { apiUrl } from "@/api/base";

type AdapterRow = {
  name: string;
  venue: string;
  state:
    | "DISCONNECTED"
    | "CONNECTING"
    | "READY"
    | "DEGRADED"
    | "HALTED";
  detail: string;
  last_heartbeat_ns: number;
};

type AdapterListResponse = {
  count: number;
  adapters: AdapterRow[];
};

async function fetchAdapters(signal?: AbortSignal): Promise<AdapterListResponse> {
  const res = await fetch(apiUrl("/api/execution/adapters"), { signal });
  if (!res.ok) {
    throw new Error(`adapters fetch failed: ${res.status}`);
  }
  return (await res.json()) as AdapterListResponse;
}

const STATE_TONE: Record<AdapterRow["state"], string> = {
  READY: "border-emerald-500/40 text-emerald-400",
  CONNECTING: "border-amber-500/40 text-amber-400",
  DISCONNECTED: "border-slate-600/40 text-slate-400",
  DEGRADED: "border-amber-500/40 text-amber-400",
  HALTED: "border-rose-500/40 text-rose-400",
};

function relativeHeartbeat(ns: number): string {
  if (ns <= 0) return "—";
  const ms = Date.now() - Math.floor(ns / 1_000_000);
  if (ms < 0) return "future";
  if (ms < 1_000) return `${ms}ms ago`;
  if (ms < 60_000) return `${Math.floor(ms / 1000)}s ago`;
  if (ms < 3_600_000) return `${Math.floor(ms / 60_000)}m ago`;
  return `${Math.floor(ms / 3_600_000)}h ago`;
}

export function AdapterStatusGrid() {
  const { data, isPending, isError, error, isFetching } = useQuery({
    queryKey: ["execution", "adapters"],
    queryFn: ({ signal }) => fetchAdapters(signal),
    refetchInterval: 5_000,
  });

  return (
    <div className="rounded border border-border bg-surface p-4">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
        Execution adapters{" "}
        <span className="ml-2 text-slate-600">DASH-EXEC-01</span>
        {isFetching && (
          <span className="ml-2 text-xs text-slate-500">refreshing…</span>
        )}
      </h2>
      {isPending && <p className="text-xs text-slate-500">loading…</p>}
      {isError && (
        <p className="text-xs text-rose-400">
          failed: {(error as Error).message}
        </p>
      )}
      {data && data.adapters.length === 0 && (
        <p className="text-xs text-slate-500">no adapters registered</p>
      )}
      {data && data.adapters.length > 0 && (
        <table className="w-full text-left text-sm">
          <thead className="text-xs uppercase text-slate-500">
            <tr>
              <th className="px-3 py-2">name</th>
              <th className="px-3 py-2">venue</th>
              <th className="px-3 py-2">state</th>
              <th className="px-3 py-2">detail</th>
              <th className="px-3 py-2">heartbeat</th>
            </tr>
          </thead>
          <tbody>
            {data.adapters.map((row) => (
              <tr key={row.name} className="border-t border-border">
                <td className="px-3 py-2 font-mono text-xs">{row.name}</td>
                <td className="px-3 py-2 font-mono text-xs text-slate-400">
                  {row.venue}
                </td>
                <td className="px-3 py-2">
                  <span
                    className={`rounded border px-2 py-0.5 text-xs ${STATE_TONE[row.state]}`}
                  >
                    {row.state}
                  </span>
                </td>
                <td className="px-3 py-2 text-xs text-slate-400">
                  {row.detail || "—"}
                </td>
                <td className="px-3 py-2 text-xs text-slate-500">
                  {relativeHeartbeat(row.last_heartbeat_ns)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
