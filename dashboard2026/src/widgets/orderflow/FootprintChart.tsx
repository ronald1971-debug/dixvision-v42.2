import { useMemo } from "react";

import { useEventStream } from "@/state/realtime";

interface Trade {
  side: "BUY" | "SELL" | string;
  price: number;
  size: number;
  venue?: string;
}

/**
 * Tier-2 order-flow widget — Footprint chart.
 *
 * Buckets recent trades into price levels (rounded to a tick step)
 * and shows per-level buy vs sell aggressor volume side-by-side.
 * The side with the larger volume is highlighted; the cumulative
 * delta per row is shown on the right.
 */
const TICK_STEP = 0.05;
const ROWS = 24;

export function FootprintChart() {
  const trades = useEventStream<Trade>("ticks", [], 500);

  const rows = useMemo(() => buildFootprint(trades), [trades]);

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Footprint
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          per-price aggressor split · last {trades.length} ticks · step{" "}
          {TICK_STEP}
        </p>
      </header>
      <div className="flex-1 overflow-auto p-1 text-[11px]">
        {rows.length === 0 ? (
          <p className="px-2 py-1 text-slate-500">waiting for ticks…</p>
        ) : (
          <table className="w-full border-collapse font-mono">
            <thead>
              <tr className="text-[9px] uppercase tracking-wider text-slate-500">
                <th className="px-2 py-1 text-right">buy</th>
                <th className="px-2 py-1 text-center">price</th>
                <th className="px-2 py-1 text-left">sell</th>
                <th className="px-2 py-1 text-right">Δ</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <FootprintRow key={r.price.toFixed(4)} row={r} />
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}

interface Row {
  price: number;
  buy: number;
  sell: number;
  max: number;
}

function buildFootprint(trades: Trade[]): Row[] {
  const buckets = new Map<number, { buy: number; sell: number }>();
  for (const t of trades) {
    if (!Number.isFinite(t.price)) continue;
    const bucket = Math.round(t.price / TICK_STEP) * TICK_STEP;
    const cur = buckets.get(bucket) ?? { buy: 0, sell: 0 };
    if (String(t.side).toUpperCase() === "BUY") cur.buy += t.size;
    else cur.sell += t.size;
    buckets.set(bucket, cur);
  }
  if (buckets.size === 0) return [];
  const sorted = Array.from(buckets.entries())
    .map(([price, v]) => ({ price, buy: v.buy, sell: v.sell }))
    .sort((a, b) => b.price - a.price)
    .slice(0, ROWS);
  const max = Math.max(1, ...sorted.flatMap((r) => [r.buy, r.sell]));
  return sorted.map((r) => ({ ...r, max }));
}

function FootprintRow({ row }: { row: Row }) {
  const delta = row.buy - row.sell;
  const bias = row.buy > row.sell ? "buy" : row.sell > row.buy ? "sell" : "flat";
  const buyW = (row.buy / row.max) * 100;
  const sellW = (row.sell / row.max) * 100;
  return (
    <tr className="border-t border-border/40">
      <td className="relative px-2 py-0.5 text-right text-emerald-300">
        <span
          aria-hidden
          className="absolute right-0 top-0 h-full bg-emerald-500/15"
          style={{ width: `${buyW}%` }}
        />
        <span className="relative">{row.buy}</span>
      </td>
      <td
        className={`px-2 py-0.5 text-center font-semibold ${
          bias === "buy"
            ? "text-emerald-300"
            : bias === "sell"
              ? "text-rose-300"
              : "text-slate-300"
        }`}
      >
        {row.price.toFixed(2)}
      </td>
      <td className="relative px-2 py-0.5 text-left text-rose-300">
        <span
          aria-hidden
          className="absolute left-0 top-0 h-full bg-rose-500/15"
          style={{ width: `${sellW}%` }}
        />
        <span className="relative">{row.sell}</span>
      </td>
      <td
        className={`px-2 py-0.5 text-right ${
          delta > 0
            ? "text-emerald-300"
            : delta < 0
              ? "text-rose-300"
              : "text-slate-400"
        }`}
      >
        {delta > 0 ? "+" : ""}
        {delta}
      </td>
    </tr>
  );
}
