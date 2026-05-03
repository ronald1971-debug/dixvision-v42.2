import { useState } from "react";

/**
 * H-track widget — Hot movers / gainers / losers / volume / volatility.
 *
 * Backend hook: ``GET /api/market/movers?bucket=gainers|losers|volume|vol``
 * reads from a daily-rollup of the canonical price+volume fanout.
 */
type Bucket = "gainers" | "losers" | "volume" | "vol";

interface Mover {
  symbol: string;
  pct24: number;
  vol24m: number;
  vol_rank: number;
  vola_rank: number;
}

const ALL: Mover[] = [
  { symbol: "WIF-USDT", pct24: 12.4, vol24m: 320, vol_rank: 6, vola_rank: 1 },
  { symbol: "PEPE-USDT", pct24: 8.7, vol24m: 410, vol_rank: 5, vola_rank: 2 },
  { symbol: "BONK-USDT", pct24: 7.2, vol24m: 180, vol_rank: 9, vola_rank: 3 },
  { symbol: "SOL-USDT", pct24: 3.84, vol24m: 4_550, vol_rank: 3, vola_rank: 7 },
  { symbol: "BTC-USDT", pct24: 1.42, vol24m: 38_400, vol_rank: 1, vola_rank: 12 },
  { symbol: "DOGE-USDT", pct24: 0.7, vol24m: 690, vol_rank: 4, vola_rank: 9 },
  { symbol: "ETH-USDT", pct24: -0.21, vol24m: 18_200, vol_rank: 2, vola_rank: 11 },
  { symbol: "MATIC-USDT", pct24: -2.3, vol24m: 280, vol_rank: 7, vola_rank: 8 },
  { symbol: "ADA-USDT", pct24: -3.4, vol24m: 220, vol_rank: 8, vola_rank: 6 },
  { symbol: "SUI-USDT", pct24: -4.1, vol24m: 150, vol_rank: 10, vola_rank: 5 },
  { symbol: "ARB-USDT", pct24: -5.6, vol24m: 90, vol_rank: 11, vola_rank: 4 },
];

function rows(bucket: Bucket): Mover[] {
  switch (bucket) {
    case "gainers":
      return [...ALL].sort((a, b) => b.pct24 - a.pct24).slice(0, 6);
    case "losers":
      return [...ALL].sort((a, b) => a.pct24 - b.pct24).slice(0, 6);
    case "volume":
      return [...ALL].sort((a, b) => a.vol_rank - b.vol_rank).slice(0, 6);
    case "vol":
      return [...ALL].sort((a, b) => a.vola_rank - b.vola_rank).slice(0, 6);
  }
}

export function HotMovers() {
  const [bucket, setBucket] = useState<Bucket>("gainers");
  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Hot movers
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            top 6 · 24h window
          </p>
        </div>
        <div className="flex gap-1 font-mono text-[10px] uppercase tracking-wider">
          {(["gainers", "losers", "volume", "vol"] as Bucket[]).map((b) => (
            <button
              key={b}
              type="button"
              onClick={() => setBucket(b)}
              className={`rounded border px-2 py-0.5 ${
                bucket === b
                  ? "border-accent/40 bg-accent/10 text-accent"
                  : "border-border bg-bg/40 text-slate-400 hover:border-accent hover:text-accent"
              }`}
            >
              {b}
            </button>
          ))}
        </div>
      </header>
      <div className="flex-1 overflow-auto">
        <table className="w-full font-mono text-[11px] text-slate-300">
          <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
            <tr className="border-b border-border">
              <th className="px-3 py-1.5 text-left">#</th>
              <th className="px-3 py-1.5 text-left">symbol</th>
              <th className="px-3 py-1.5 text-right">24h %</th>
              <th className="px-3 py-1.5 text-right">vol $M</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/40">
            {rows(bucket).map((r, i) => (
              <tr key={r.symbol}>
                <td className="px-3 py-1 text-slate-500">{i + 1}</td>
                <td className="px-3 py-1 text-slate-200">{r.symbol}</td>
                <td
                  className={`px-3 py-1 text-right ${
                    r.pct24 >= 0 ? "text-emerald-400" : "text-rose-400"
                  }`}
                >
                  {r.pct24 >= 0 ? "+" : ""}
                  {r.pct24.toFixed(2)}%
                </td>
                <td className="px-3 py-1 text-right">
                  {r.vol24m.toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
