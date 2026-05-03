import { useEffect, useState } from "react";

/**
 * H-track widget — Watchlist.
 *
 * User-curated symbol list with last/Δ/24h-vol/spark column.
 * Backend hook: ``GET /api/market/watchlist?symbols=...`` reads the
 * fan-out of the canonical Binance/Kraken/Bybit price pumps
 * (PR #67 + PR #102 fanout). Today seeded deterministically with
 * tiny drift so the user can see the spark refresh.
 */
interface Row {
  symbol: string;
  last: number;
  pct24: number;
  vol24_usd: number;
  spark: number[];
}

const SEED: Row[] = [
  { symbol: "BTC-USDT", last: 67_812, pct24: 1.42, vol24_usd: 38_400_000_000, spark: [66_900, 67_010, 66_980, 67_120, 67_240, 67_400, 67_350, 67_812] },
  { symbol: "ETH-USDT", last: 3_534, pct24: -0.21, vol24_usd: 18_200_000_000, spark: [3_550, 3_540, 3_540, 3_530, 3_525, 3_535, 3_520, 3_534] },
  { symbol: "SOL-USDT", last: 145.10, pct24: 3.84, vol24_usd: 4_550_000_000, spark: [139.80, 140.20, 140.90, 142.50, 143.20, 144.10, 144.60, 145.10] },
  { symbol: "AVAX-USDT", last: 31.65, pct24: 0.92, vol24_usd: 540_000_000, spark: [31.20, 31.30, 31.10, 31.20, 31.40, 31.50, 31.60, 31.65] },
  { symbol: "WIF-USDT", last: 2.41, pct24: 12.4, vol24_usd: 320_000_000, spark: [2.10, 2.18, 2.22, 2.28, 2.30, 2.34, 2.38, 2.41] },
  { symbol: "MATIC-USDT", last: 0.512, pct24: -2.3, vol24_usd: 280_000_000, spark: [0.521, 0.520, 0.518, 0.516, 0.514, 0.515, 0.513, 0.512] },
  { symbol: "DOGE-USDT", last: 0.142, pct24: 0.7, vol24_usd: 690_000_000, spark: [0.141, 0.140, 0.140, 0.141, 0.142, 0.142, 0.141, 0.142] },
  { symbol: "LINK-USDT", last: 14.20, pct24: 1.1, vol24_usd: 410_000_000, spark: [14.05, 14.10, 14.08, 14.12, 14.15, 14.18, 14.22, 14.20] },
];

export function Watchlist() {
  const [rows, setRows] = useState(SEED);

  useEffect(() => {
    const t = setInterval(() => {
      setRows((prev) =>
        prev.map((r) => {
          // tiny deterministic drift based on row order so the spark animates without PRNG
          const drift = (r.spark.length % 3) === 0 ? 0.0008 : -0.0006;
          const next = r.last * (1 + drift);
          const spark = [...r.spark.slice(1), next];
          return { ...r, last: next, spark };
        }),
      );
    }, 1500);
    return () => clearInterval(t);
  }, []);

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Watchlist
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          curated · last/24h%/24h-vol · live tick
        </p>
      </header>
      <div className="flex-1 overflow-auto">
        <table className="w-full font-mono text-[11px] text-slate-300">
          <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
            <tr className="border-b border-border">
              <th className="px-3 py-1.5 text-left">symbol</th>
              <th className="px-3 py-1.5 text-right">last</th>
              <th className="px-3 py-1.5 text-right">24h %</th>
              <th className="px-3 py-1.5 text-right">24h vol</th>
              <th className="px-3 py-1.5">spark</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/40">
            {rows.map((r) => (
              <tr key={r.symbol}>
                <td className="px-3 py-1 text-slate-200">{r.symbol}</td>
                <td className="px-3 py-1 text-right">
                  {r.last < 1
                    ? r.last.toFixed(4)
                    : r.last < 100
                      ? r.last.toFixed(2)
                      : r.last.toLocaleString(undefined, {
                          maximumFractionDigits: 0,
                        })}
                </td>
                <td
                  className={`px-3 py-1 text-right ${
                    r.pct24 >= 0 ? "text-emerald-400" : "text-rose-400"
                  }`}
                >
                  {r.pct24 >= 0 ? "+" : ""}
                  {r.pct24.toFixed(2)}%
                </td>
                <td className="px-3 py-1 text-right text-slate-400">
                  {(r.vol24_usd / 1_000_000).toFixed(0)}M
                </td>
                <td className="px-3 py-1">
                  <Spark values={r.spark} positive={r.pct24 >= 0} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Spark({ values, positive }: { values: number[]; positive: boolean }) {
  const w = 80;
  const h = 16;
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = Math.max(hi - lo, 1e-9);
  const xs = (i: number) => (i / (values.length - 1)) * w;
  const ys = (v: number) => h - ((v - lo) / span) * (h - 2) - 1;
  const path = values
    .map((v, i) => `${i === 0 ? "M" : "L"}${xs(i).toFixed(1)},${ys(v).toFixed(1)}`)
    .join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="h-4 w-20">
      <path
        d={path}
        fill="none"
        stroke={positive ? "#10b981" : "#f43f5e"}
        strokeWidth={1}
      />
    </svg>
  );
}
