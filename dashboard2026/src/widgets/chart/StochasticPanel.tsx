import { useMemo } from "react";

/**
 * Stochastic oscillator sub-pane (14, 3, 3). %K (fast) + %D (slow).
 * Mock series; replaced by the SSE bridge once live candles ship.
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

function smooth(values: number[], period: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < values.length; i += 1) {
    const a = Math.max(0, i - period + 1);
    let sum = 0;
    for (let j = a; j <= i; j += 1) sum += values[j];
    out.push(sum / (i - a + 1));
  }
  return out;
}

export function StochasticPanel({
  symbol,
  count = 120,
}: {
  symbol: string;
  count?: number;
}) {
  const { k, d } = useMemo(() => {
    const r = rng(seedOf(`stoch:${symbol}:${count}`));
    let v = 50;
    const series: number[] = [];
    for (let i = 0; i < count; i += 1) {
      v += (r() - 0.5) * 18;
      v = Math.max(2, Math.min(98, v));
      series.push(v);
    }
    const dv = smooth(series, 3);
    return { k: series, d: dv };
  }, [symbol, count]);

  const lastK = k[k.length - 1] ?? 50;
  const tone =
    lastK >= 80
      ? "text-rose-300 border-rose-500/40 bg-rose-500/10"
      : lastK <= 20
        ? "text-emerald-300 border-emerald-500/40 bg-emerald-500/10"
        : "text-slate-300 border-slate-600/40 bg-slate-700/30";

  const W = 100;
  const H = 100;
  const path = (s: number[]) =>
    s
      .map((y, i) => {
        const x = (i / (s.length - 1)) * W;
        const ny = H - (y / 100) * H;
        return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${ny.toFixed(2)}`;
      })
      .join(" ");

  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Stochastic (14, 3, 3)
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            %K + %D · {symbol}
          </p>
        </div>
        <span
          className={`rounded border px-1.5 py-0.5 font-mono text-[11px] ${tone}`}
        >
          %K {lastK.toFixed(1)}
        </span>
      </header>
      <div className="flex-1 px-2 py-2">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="none"
          className="h-full w-full"
        >
          <line x1="0" y1={H * 0.2} x2={W} y2={H * 0.2} stroke="#f5325030" strokeDasharray="2,2" />
          <line x1="0" y1={H * 0.8} x2={W} y2={H * 0.8} stroke="#10b98130" strokeDasharray="2,2" />
          <path d={path(k)} stroke="#60a5fa" strokeWidth="1" fill="none" vectorEffect="non-scaling-stroke" />
          <path d={path(d)} stroke="#fbbf24" strokeWidth="1" fill="none" vectorEffect="non-scaling-stroke" />
        </svg>
      </div>
    </div>
  );
}
