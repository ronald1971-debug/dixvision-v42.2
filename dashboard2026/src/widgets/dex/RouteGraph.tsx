/**
 * DEX widget — solver-auction route table.
 *
 * Operators want to see, per swap, what Jupiter Juno, 1inch
 * Fusion+, and CowSwap solver auctions are quoting for the same
 * intent so they can sanity-check the route the OrderForm will
 * pick. The widget polls ``/api/dashboard/dex/route?symbol=<sym>``
 * every 5 s; when the route returns a 404 it falls back to a
 * deterministic skeleton so the surface remains legible end-to-end
 * (mirroring the pattern in ``CoherencePanel``).
 *
 * The status chip flips between ``live`` and ``mock`` so the
 * operator always knows whether the numbers are real venue quotes
 * or the deterministic skeleton.
 */
import { useQuery } from "@tanstack/react-query";

import { apiUrl } from "@/api/base";
import { WidgetStatusChip } from "@/components/WidgetStatusChip";

interface RouteQuote {
  venue: string;
  in_token: string;
  out_token: string;
  in_amount: number;
  out_amount: number;
  price_impact_bps: number;
  est_fill_ms: number;
}

interface RouteSnapshot {
  symbol: string;
  quotes: RouteQuote[];
  best_venue: string;
  ts_iso: string;
}

const FALLBACK: RouteSnapshot = {
  symbol: "SOL/USDC",
  quotes: [
    {
      venue: "Jupiter Juno",
      in_token: "SOL",
      out_token: "USDC",
      in_amount: 100,
      out_amount: 14_812.4,
      price_impact_bps: 3.2,
      est_fill_ms: 410,
    },
    {
      venue: "1inch Fusion+",
      in_token: "SOL",
      out_token: "USDC",
      in_amount: 100,
      out_amount: 14_809.1,
      price_impact_bps: 5.4,
      est_fill_ms: 620,
    },
    {
      venue: "CowSwap solver",
      in_token: "SOL",
      out_token: "USDC",
      in_amount: 100,
      out_amount: 14_807.6,
      price_impact_bps: 6.1,
      est_fill_ms: 1_180,
    },
  ],
  best_venue: "Jupiter Juno",
  ts_iso: new Date().toISOString(),
};

async function fetchRoute(
  symbol: string,
  signal?: AbortSignal,
): Promise<{ snap: RouteSnapshot; live: boolean }> {
  try {
    const res = await fetch(
      apiUrl(`/api/dashboard/dex/route?symbol=${encodeURIComponent(symbol)}`),
      { signal },
    );
    if (!res.ok) throw new Error(`status ${res.status}`);
    return { snap: (await res.json()) as RouteSnapshot, live: true };
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw err;
    }
    return { snap: { ...FALLBACK, symbol }, live: false };
  }
}

export function RouteGraph({ symbol = "SOL/USDC" }: { symbol?: string }) {
  const { data } = useQuery({
    queryKey: ["dashboard", "dex", "route", symbol],
    queryFn: ({ signal }) => fetchRoute(symbol, signal),
    refetchInterval: 5_000,
    initialData: { snap: { ...FALLBACK, symbol }, live: false },
  });
  const { snap, live } = data;
  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Route Graph · {snap.symbol}
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            Jupiter Juno · 1inch Fusion+ · CowSwap solver auction
          </p>
        </div>
        <WidgetStatusChip mode={live ? "live" : "mock"} />
      </header>
      <div className="flex-1 overflow-auto p-3 text-[12px]">
        <table className="w-full font-mono text-[11px]">
          <thead className="text-[10px] uppercase tracking-wider text-slate-500">
            <tr>
              <th className="px-1 py-1 text-left">venue</th>
              <th className="px-1 py-1 text-right">out</th>
              <th className="px-1 py-1 text-right">impact</th>
              <th className="px-1 py-1 text-right">fill</th>
            </tr>
          </thead>
          <tbody>
            {snap.quotes.map((q) => (
              <tr
                key={q.venue}
                className={
                  q.venue === snap.best_venue
                    ? "border-l-2 border-emerald-500/60 text-emerald-300"
                    : "text-slate-300"
                }
              >
                <td className="px-1 py-1">{q.venue}</td>
                <td className="px-1 py-1 text-right">
                  {q.out_amount.toFixed(2)}
                </td>
                <td className="px-1 py-1 text-right text-slate-400">
                  {q.price_impact_bps.toFixed(1)} bps
                </td>
                <td className="px-1 py-1 text-right text-slate-500">
                  {q.est_fill_ms} ms
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="mt-2 text-[10px] text-slate-500">
          best: <span className="text-emerald-300">{snap.best_venue}</span>
        </div>
      </div>
    </div>
  );
}


