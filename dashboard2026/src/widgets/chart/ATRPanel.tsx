import { useMemo } from "react";

/**
 * Average-True-Range sub-pane (period 14). Volatility band shown as
 * filled area + last-value badge; the same hook used by the SL/TP
 * builder's ATR-multiplier stop pre-fill.
 */
function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

function seedOf(s: string): number {
  let h = 0;
  for (const ch of s) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return h;
}

export function ATRPanel({
  symbol,
  period = 14,
  count = 120,
}: {
  symbol: string;
  period?: number;
  count?: number;
}) {
  const series = useMemo(() => {
    const r = rng(seedOf(`atr:${symbol}:${period}:${count}`));
    let base = 0.6 + r() * 1.8;
    const out: number[] = [];
    for (let i = 0; i < count; i += 1) {
      base += (r() - 0.5) * 0.15;
      base = Math.max(0.05, base);
      out.push(base);
    }
    return out;
  }, [symbol, period, count]);

  const last = series[series.length - 1] ?? 0;
  const max = Math.max(...series);
  const min = Math.min(...series);
  const W = 100;
  const H = 100;
  const span = Math.max(0.0001, max - min);

  const linePath = series
    .map((v, i) => {
      const x = (i / (series.length - 1)) * W;
      const y = H - ((v - min) / span) * H;
      return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");

  const areaPath = `${linePath} L${W},${H} L0,${H} Z`;

  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            ATR ({period})
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            volatility · {symbol}
          </p>
        </div>
        <span className="rounded border border-amber-500/40 bg-amber-500/10 px-1.5 py-0.5 font-mono text-[11px] text-amber-300">
          {last.toFixed(2)}
        </span>
      </header>
      <div className="flex-1 px-2 py-2">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="none"
          className="h-full w-full"
        >
          <path d={areaPath} fill="#fbbf2420" />
          <path d={linePath} stroke="#fbbf24" strokeWidth="1" fill="none" vectorEffect="non-scaling-stroke" />
        </svg>
      </div>
    </div>
  );
}
