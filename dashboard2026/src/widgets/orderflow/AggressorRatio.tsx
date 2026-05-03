import { useMemo } from "react";

import { useEventStream } from "@/state/realtime";

interface Trade {
  side: "BUY" | "SELL" | string;
  size: number;
}

/**
 * Tier-2 order-flow widget — Aggressor ratio.
 *
 * Splits the rolling window of trades into buy-aggressor vs
 * sell-aggressor share by total size. Renders a horizontal bar
 * with both sides plus a numeric breakdown.
 */
const WINDOW = 300;

export function AggressorRatio() {
  const trades = useEventStream<Trade>("ticks", [], WINDOW);

  const stats = useMemo(() => {
    let buy = 0;
    let sell = 0;
    for (const t of trades) {
      if (String(t.side).toUpperCase() === "BUY") buy += t.size;
      else sell += t.size;
    }
    const total = buy + sell;
    return { buy, sell, total };
  }, [trades]);

  const buyPct = stats.total === 0 ? 50 : (stats.buy / stats.total) * 100;
  const dominant =
    stats.buy > stats.sell ? "BUY" : stats.sell > stats.buy ? "SELL" : "BAL";

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Aggressor ratio
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          last {trades.length}/{WINDOW} ticks · dominant{" "}
          <span
            className={
              dominant === "BUY"
                ? "text-emerald-300"
                : dominant === "SELL"
                  ? "text-rose-300"
                  : "text-slate-300"
            }
          >
            {dominant}
          </span>
        </p>
      </header>
      <div className="flex flex-1 flex-col justify-center gap-3 px-3">
        <div className="h-3 w-full overflow-hidden rounded border border-border bg-bg/40">
          <div className="flex h-full">
            <div
              className="h-full bg-emerald-500/70"
              style={{ width: `${buyPct}%` }}
            />
            <div
              className="h-full bg-rose-500/70"
              style={{ width: `${100 - buyPct}%` }}
            />
          </div>
        </div>
        <div className="grid grid-cols-3 gap-2 text-center font-mono text-[11px]">
          <Stat label="buy" value={stats.buy} tone="emerald" />
          <Stat label="total" value={stats.total} tone="slate" />
          <Stat label="sell" value={stats.sell} tone="rose" />
        </div>
        <div className="text-center font-mono text-[11px] text-slate-300">
          {buyPct.toFixed(1)}% buy / {(100 - buyPct).toFixed(1)}% sell
        </div>
      </div>
    </section>
  );
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone: "emerald" | "rose" | "slate";
}) {
  const cls =
    tone === "emerald"
      ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
      : tone === "rose"
        ? "border-rose-500/40 bg-rose-500/10 text-rose-300"
        : "border-slate-600/40 bg-slate-800/40 text-slate-300";
  return (
    <div className={`rounded border px-2 py-1 ${cls}`}>
      <div className="text-[9px] uppercase tracking-wider text-slate-400">
        {label}
      </div>
      <div className="text-sm">{value}</div>
    </div>
  );
}
