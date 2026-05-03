import { useMemo } from "react";

/**
 * Average-Directional-Index sub-pane (period 14). Three-line trend
 * strength: ADX (white), +DI (green), -DI (red). ADX > 25 conventionally
 * marks a trending market.
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

export function ADXPanel({
  symbol,
  period = 14,
  count = 120,
}: {
  symbol: string;
  period?: number;
  count?: number;
}) {
  const { adx, pdi, ndi } = useMemo(() => {
    const r = rng(seedOf(`adx:${symbol}:${period}:${count}`));
    let a = 18 + r() * 14;
    let p = 22;
    let n = 18;
    const adxArr: number[] = [];
    const pArr: number[] = [];
    const nArr: number[] = [];
    for (let i = 0; i < count; i += 1) {
      a += (r() - 0.5) * 4;
      a = Math.max(5, Math.min(85, a));
      p += (r() - 0.5) * 5;
      n += (r() - 0.5) * 5;
      p = Math.max(2, Math.min(80, p));
      n = Math.max(2, Math.min(80, n));
      adxArr.push(a);
      pArr.push(p);
      nArr.push(n);
    }
    return { adx: adxArr, pdi: pArr, ndi: nArr };
  }, [symbol, period, count]);

  const lastA = adx[adx.length - 1] ?? 20;
  const trending = lastA > 25;
  const tone = trending
    ? "text-emerald-300 border-emerald-500/40 bg-emerald-500/10"
    : "text-slate-300 border-slate-600/40 bg-slate-700/30";

  const W = 100;
  const H = 100;
  const path = (s: number[], scale = 100) =>
    s
      .map((v, i) => {
        const x = (i / (s.length - 1)) * W;
        const y = H - (v / scale) * H;
        return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");

  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            ADX ({period})
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            trend strength · {symbol}
          </p>
        </div>
        <span
          className={`rounded border px-1.5 py-0.5 font-mono text-[11px] ${tone}`}
        >
          {trending ? "trending" : "ranging"} · {lastA.toFixed(0)}
        </span>
      </header>
      <div className="flex-1 px-2 py-2">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="none"
          className="h-full w-full"
        >
          <line x1="0" y1={H * 0.75} x2={W} y2={H * 0.75} stroke="#64748b40" strokeDasharray="2,2" />
          <path d={path(pdi)} stroke="#10b981" strokeWidth="1" fill="none" vectorEffect="non-scaling-stroke" />
          <path d={path(ndi)} stroke="#f43f5e" strokeWidth="1" fill="none" vectorEffect="non-scaling-stroke" />
          <path d={path(adx)} stroke="#e2e8f0" strokeWidth="1.2" fill="none" vectorEffect="non-scaling-stroke" />
        </svg>
        <div className="flex gap-3 px-1 pt-1 font-mono text-[10px]">
          <span className="text-slate-300">ADX</span>
          <span className="text-emerald-300">+DI</span>
          <span className="text-rose-300">-DI</span>
        </div>
      </div>
    </div>
  );
}
