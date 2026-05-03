import { useMemo } from "react";

/**
 * Relative-Strength-Index sub-pane (period 14). Presentational mock —
 * deterministic seed off `symbol`, replaced by the SSE-bridge stream
 * once the live pump lands.
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

export function RSIPanel({
  symbol,
  period = 14,
  count = 120,
}: {
  symbol: string;
  period?: number;
  count?: number;
}) {
  const series = useMemo(() => {
    const r = rng(seedOf(`rsi:${symbol}:${period}:${count}`));
    let v = 50 + (r() - 0.5) * 30;
    const out: number[] = [];
    for (let i = 0; i < count; i += 1) {
      v += (r() - 0.5) * 12;
      v = Math.max(2, Math.min(98, v));
      out.push(v);
    }
    return out;
  }, [symbol, period, count]);

  const last = series[series.length - 1] ?? 50;
  const tone =
    last >= 70
      ? "text-rose-300 border-rose-500/40 bg-rose-500/10"
      : last <= 30
        ? "text-emerald-300 border-emerald-500/40 bg-emerald-500/10"
        : "text-slate-300 border-slate-600/40 bg-slate-700/30";
  const max = Math.max(...series);
  const min = Math.min(...series);

  // SVG path
  const W = 100;
  const H = 100;
  const path = series
    .map((y, i) => {
      const x = (i / (series.length - 1)) * W;
      const ny = H - ((y - 0) / 100) * H;
      return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${ny.toFixed(2)}`;
    })
    .join(" ");

  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            RSI ({period})
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            relative strength · {symbol}
          </p>
        </div>
        <span
          className={`rounded border px-1.5 py-0.5 font-mono text-[11px] ${tone}`}
        >
          {last.toFixed(1)}
        </span>
      </header>
      <div className="relative flex-1 px-2 py-2">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="none"
          className="h-full w-full"
        >
          <line x1="0" y1={H * 0.3} x2={W} y2={H * 0.3} stroke="#f5325030" strokeDasharray="2,2" />
          <line x1="0" y1={H * 0.7} x2={W} y2={H * 0.7} stroke="#10b98130" strokeDasharray="2,2" />
          <line x1="0" y1={H * 0.5} x2={W} y2={H * 0.5} stroke="#64748b40" />
          <path d={path} stroke="#a78bfa" strokeWidth="1" fill="none" vectorEffect="non-scaling-stroke" />
        </svg>
        <div className="absolute right-2 top-2 flex gap-1 font-mono text-[10px] text-slate-500">
          <span>hi {max.toFixed(0)}</span>
          <span>lo {min.toFixed(0)}</span>
        </div>
      </div>
    </div>
  );
}
