/**
 * Perps widget — per-venue funding-rate table.
 *
 * Current funding rate, time-to-next-funding, and cumulative-funding
 * PnL since position open. Polls
 * ``/api/dashboard/perps/funding?symbol=<sym>`` every 5 s; falls
 * back to a deterministic skeleton when the route 404s.
 */
import { useQuery } from "@tanstack/react-query";

import { apiUrl } from "@/api/base";
import { WidgetStatusChip } from "@/components/WidgetStatusChip";

interface FundingRow {
  venue: string;
  current_rate_bps: number;
  next_funding_ts_iso: string;
  cum_funding_pnl_usd: number;
}

interface FundingSnapshot {
  symbol: string;
  rows: FundingRow[];
  ts_iso: string;
}

const NOW = Date.now();
const FALLBACK: FundingSnapshot = {
  symbol: "BTC-PERP",
  rows: [
    {
      venue: "Hyperliquid",
      current_rate_bps: 1.2,
      next_funding_ts_iso: new Date(NOW + 47 * 60_000).toISOString(),
      cum_funding_pnl_usd: -38.4,
    },
    {
      venue: "dYdX",
      current_rate_bps: 0.8,
      next_funding_ts_iso: new Date(NOW + 23 * 60_000).toISOString(),
      cum_funding_pnl_usd: -12.7,
    },
    {
      venue: "Drift",
      current_rate_bps: -0.4,
      next_funding_ts_iso: new Date(NOW + 11 * 60_000).toISOString(),
      cum_funding_pnl_usd: 6.1,
    },
  ],
  ts_iso: new Date(NOW).toISOString(),
};

async function fetchFunding(
  symbol: string,
  signal?: AbortSignal,
): Promise<{ snap: FundingSnapshot; live: boolean }> {
  try {
    const res = await fetch(
      apiUrl(`/api/dashboard/perps/funding?symbol=${encodeURIComponent(symbol)}`),
      { signal },
    );
    if (!res.ok) throw new Error(`status ${res.status}`);
    return { snap: (await res.json()) as FundingSnapshot, live: true };
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw err;
    }
    return { snap: { ...FALLBACK, symbol }, live: false };
  }
}

export function FundingTable({ symbol = "BTC-PERP" }: { symbol?: string }) {
  const { data } = useQuery({
    queryKey: ["dashboard", "perps", "funding", symbol],
    queryFn: ({ signal }) => fetchFunding(symbol, signal),
    refetchInterval: 5_000,
    initialData: { snap: { ...FALLBACK, symbol }, live: false },
  });
  const { snap, live } = data;
  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Funding Table · {snap.symbol}
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            per-venue · next-funding countdown · cumulative-funding PnL
          </p>
        </div>
        <WidgetStatusChip mode={live ? "live" : "mock"} />
      </header>
      <div className="flex-1 overflow-auto p-3">
        <table className="w-full font-mono text-[11px]">
          <thead className="text-[10px] uppercase tracking-wider text-slate-500">
            <tr>
              <th className="px-1 py-1 text-left">venue</th>
              <th className="px-1 py-1 text-right">rate</th>
              <th className="px-1 py-1 text-right">next</th>
              <th className="px-1 py-1 text-right">cum PnL</th>
            </tr>
          </thead>
          <tbody>
            {snap.rows.map((r) => (
              <tr key={r.venue} className="text-slate-300">
                <td className="px-1 py-1">{r.venue}</td>
                <td
                  className={`px-1 py-1 text-right ${
                    r.current_rate_bps > 0
                      ? "text-emerald-300"
                      : r.current_rate_bps < 0
                        ? "text-rose-300"
                        : "text-slate-400"
                  }`}
                >
                  {r.current_rate_bps.toFixed(2)} bps
                </td>
                <td className="px-1 py-1 text-right text-slate-500">
                  {fmtCountdown(r.next_funding_ts_iso)}
                </td>
                <td
                  className={`px-1 py-1 text-right ${
                    r.cum_funding_pnl_usd >= 0
                      ? "text-emerald-300"
                      : "text-rose-300"
                  }`}
                >
                  ${r.cum_funding_pnl_usd.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function fmtCountdown(iso: string): string {
  const ms = new Date(iso).getTime() - Date.now();
  if (ms <= 0) return "now";
  const m = Math.floor(ms / 60_000);
  const s = Math.floor((ms % 60_000) / 1_000);
  return `${m}m${s.toString().padStart(2, "0")}s`;
}
