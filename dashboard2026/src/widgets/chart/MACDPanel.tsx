import { useMemo } from "react";

/**
 * MACD sub-pane (12, 26, 9). Presentational mock; histogram colored
 * green when MACD is rising vs. signal, red when falling.
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

function ema(values: number[], period: number): number[] {
  const k = 2 / (period + 1);
  const out: number[] = [];
  let prev = values[0] ?? 0;
  for (let i = 0; i < values.length; i += 1) {
    prev = i === 0 ? values[i] : values[i] * k + prev * (1 - k);
    out.push(prev);
  }
  return out;
}

export function MACDPanel({
  symbol,
  count = 120,
}: {
  symbol: string;
  count?: number;
}) {
  const { macd, signal, hist } = useMemo(() => {
    const r = rng(seedOf(`macd:${symbol}:${count}`));
    let p = 100;
    const close: number[] = [];
    for (let i = 0; i < count; i += 1) {
      p += (r() - 0.5) * 2;
      close.push(p);
    }
    const fast = ema(close, 12);
    const slow = ema(close, 26);
    const m = fast.map((v, i) => v - slow[i]);
    const s = ema(m, 9);
    const h = m.map((v, i) => v - s[i]);
    return { macd: m, signal: s, hist: h };
  }, [symbol, count]);

  const lastM = macd[macd.length - 1] ?? 0;
  const lastS = signal[signal.length - 1] ?? 0;
  const cross = lastM > lastS ? "bullish" : "bearish";
  const crossTone =
    lastM > lastS
      ? "text-emerald-300 border-emerald-500/40 bg-emerald-500/10"
      : "text-rose-300 border-rose-500/40 bg-rose-500/10";

  const W = 100;
  const H = 100;
  const all = [...macd, ...signal, ...hist];
  const min = Math.min(...all);
  const max = Math.max(...all);
  const span = Math.max(0.0001, max - min);
  const y = (v: number) => H - ((v - min) / span) * H;

  const macdPath = macd
    .map((v, i) => `${i === 0 ? "M" : "L"}${((i / (macd.length - 1)) * W).toFixed(2)},${y(v).toFixed(2)}`)
    .join(" ");
  const sigPath = signal
    .map((v, i) => `${i === 0 ? "M" : "L"}${((i / (signal.length - 1)) * W).toFixed(2)},${y(v).toFixed(2)}`)
    .join(" ");

  const zero = y(0);

  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            MACD (12, 26, 9)
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            momentum · {symbol}
          </p>
        </div>
        <span
          className={`rounded border px-1.5 py-0.5 font-mono text-[11px] ${crossTone}`}
        >
          {cross}
        </span>
      </header>
      <div className="flex-1 px-2 py-2">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="none"
          className="h-full w-full"
        >
          <line x1="0" y1={zero} x2={W} y2={zero} stroke="#64748b40" />
          {hist.map((h, i) => {
            const bw = W / hist.length;
            const yh = y(h);
            const top = h >= 0 ? yh : zero;
            const height = Math.abs(yh - zero);
            const fill = h >= 0 ? "#10b98180" : "#f5325080";
            return (
              <rect
                key={i}
                x={i * bw}
                y={top}
                width={bw * 0.8}
                height={Math.max(0.4, height)}
                fill={fill}
              />
            );
          })}
          <path d={macdPath} stroke="#60a5fa" strokeWidth="1" fill="none" vectorEffect="non-scaling-stroke" />
          <path d={sigPath} stroke="#fbbf24" strokeWidth="1" fill="none" vectorEffect="non-scaling-stroke" />
        </svg>
      </div>
    </div>
  );
}
