/**
 * Perps widget — per-venue oracle vs execution price spread.
 *
 * Surfaces oracle-price-vs-exec-price divergence per venue with a
 * configurable alarm threshold (default 25 bps). Polls
 * ``/api/dashboard/perps/oracle?symbol=<sym>`` every 2 s; falls
 * back to a deterministic skeleton when the route 404s.
 */
import { useQuery } from "@tanstack/react-query";

import { apiUrl } from "@/api/base";
import { WidgetStatusChip } from "@/components/WidgetStatusChip";

interface OracleRow {
  venue: string;
  oracle_price: number;
  exec_price: number;
  divergence_bps: number;
}

interface OracleSnapshot {
  symbol: string;
  rows: OracleRow[];
  alarm_bps: number;
  ts_iso: string;
}

const FALLBACK: OracleSnapshot = {
  symbol: "BTC-PERP",
  rows: [
    {
      venue: "Hyperliquid",
      oracle_price: 71_412.4,
      exec_price: 71_408.2,
      divergence_bps: -0.6,
    },
    {
      venue: "dYdX",
      oracle_price: 71_415.1,
      exec_price: 71_437.6,
      divergence_bps: 3.2,
    },
    {
      venue: "Drift",
      oracle_price: 71_410.0,
      exec_price: 71_558.2,
      divergence_bps: 20.8,
    },
  ],
  alarm_bps: 25,
  ts_iso: new Date().toISOString(),
};

async function fetchOracle(
  symbol: string,
  signal?: AbortSignal,
): Promise<{ snap: OracleSnapshot; live: boolean }> {
  try {
    const res = await fetch(
      apiUrl(`/api/dashboard/perps/oracle?symbol=${encodeURIComponent(symbol)}`),
      { signal },
    );
    if (!res.ok) throw new Error(`status ${res.status}`);
    return { snap: (await res.json()) as OracleSnapshot, live: true };
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw err;
    }
    return { snap: { ...FALLBACK, symbol }, live: false };
  }
}

export function OracleSpread({ symbol = "BTC-PERP" }: { symbol?: string }) {
  const { data } = useQuery({
    queryKey: ["dashboard", "perps", "oracle", symbol],
    queryFn: ({ signal }) => fetchOracle(symbol, signal),
    refetchInterval: 2_000,
    initialData: { snap: { ...FALLBACK, symbol }, live: false },
  });
  const { snap, live } = data;
  const breach = snap.rows.some(
    (r) => Math.abs(r.divergence_bps) >= snap.alarm_bps,
  );
  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Per-venue Oracle · {snap.symbol}
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            oracle price vs exec price · divergence alarm ≥{" "}
            {snap.alarm_bps} bps
          </p>
        </div>
        <WidgetStatusChip mode={live ? "live" : "mock"} />
      </header>
      {breach && (
        <div
          role="alert"
          className="border-b border-rose-500/60 bg-rose-500/10 px-3 py-1 text-[11px] text-rose-300"
        >
          ⚠ divergence ≥ {snap.alarm_bps} bps on at least one venue
        </div>
      )}
      <div className="flex-1 overflow-auto p-3">
        <table className="w-full font-mono text-[11px]">
          <thead className="text-[10px] uppercase tracking-wider text-slate-500">
            <tr>
              <th className="px-1 py-1 text-left">venue</th>
              <th className="px-1 py-1 text-right">oracle</th>
              <th className="px-1 py-1 text-right">exec</th>
              <th className="px-1 py-1 text-right">div</th>
            </tr>
          </thead>
          <tbody>
            {snap.rows.map((r) => {
              const breached = Math.abs(r.divergence_bps) >= snap.alarm_bps;
              return (
                <tr
                  key={r.venue}
                  className={
                    breached
                      ? "text-rose-300"
                      : r.divergence_bps > 0
                        ? "text-amber-200"
                        : "text-slate-300"
                  }
                >
                  <td className="px-1 py-1">{r.venue}</td>
                  <td className="px-1 py-1 text-right tabular-nums">
                    {r.oracle_price.toFixed(1)}
                  </td>
                  <td className="px-1 py-1 text-right tabular-nums">
                    {r.exec_price.toFixed(1)}
                  </td>
                  <td className="px-1 py-1 text-right tabular-nums">
                    {r.divergence_bps >= 0 ? "+" : ""}
                    {r.divergence_bps.toFixed(1)} bps
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
