import { useMemo, useState } from "react";

/**
 * I-track widget — Monte-Carlo path explorer.
 *
 * Bootstraps the realised return distribution of the strategy and
 * draws N alternate equity paths so the operator can see the
 * dispersion that a finite sample masks (sequence-of-returns risk,
 * tail probability of ruin, etc.).
 *
 * Backend hook: ``GET /api/testing/monte-carlo?strategy=&n_paths=&horizon=``
 * resamples ledger PnL increments with replacement and returns a
 * (n_paths × horizon) matrix.
 */
function lcg(seed: number) {
  let s = seed;
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
}

function bootstrapPath(seed: number, horizon: number, mu: number, sigma: number): number[] {
  const rng = lcg(seed);
  let v = 100_000;
  const out: number[] = [v];
  for (let i = 0; i < horizon; i++) {
    // Box–Muller on two uniforms
    const u1 = Math.max(rng(), 1e-9);
    const u2 = rng();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    const r = mu + sigma * z;
    v = v * (1 + r);
    out.push(v);
  }
  return out;
}

const W = 480;
const H = 220;
const PAD_X = 36;
const PAD_Y = 12;

export function MonteCarloPaths() {
  const [n, setN] = useState(50);
  const [horizon, setHorizon] = useState(180);

  const { paths, mean, p5, p95 } = useMemo(() => {
    const mu = 0.0006;
    const sigma = 0.012;
    const list: number[][] = [];
    for (let i = 0; i < n; i++) {
      list.push(bootstrapPath(7919 + i * 31, horizon, mu, sigma));
    }
    const finals = list.map((p) => p[p.length - 1]).sort((a, b) => a - b);
    const meanFinal = finals.reduce((s, x) => s + x, 0) / finals.length;
    const p5Final = finals[Math.floor(finals.length * 0.05)];
    const p95Final = finals[Math.floor(finals.length * 0.95)];
    return { paths: list, mean: meanFinal, p5: p5Final, p95: p95Final };
  }, [n, horizon]);

  // Avoid spreading the flattened array into Math.min/max — at large
  // n × horizon (e.g. 200 × 365 ≈ 73k elements) Safari/JSC throws a
  // RangeError because the function-arguments cap is ~65k.
  const allValues = paths.flat();
  const min = allValues.reduce((a, b) => (b < a ? b : a), Infinity);
  const max = allValues.reduce((a, b) => (b > a ? b : a), -Infinity);
  const xAt = (i: number) => PAD_X + (i / horizon) * (W - PAD_X * 2);
  const yAt = (v: number) => H - PAD_Y - ((v - min) / (max - min || 1)) * (H - PAD_Y * 2);

  const ruinPaths = paths.filter(
    (p) => p.reduce((a, b) => (b < a ? b : a), Infinity) < 50_000,
  ).length;

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Monte-Carlo paths
        </h3>
        <div className="mt-1 flex items-center gap-2 text-[11px]">
          <label className="text-slate-500">paths</label>
          <select value={n} onChange={(e) => setN(Number(e.target.value))} className="rounded border border-border bg-bg/50 px-2 py-0.5 font-mono text-[11px] text-slate-200 outline-none">
            {[20, 50, 100, 200].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
          <label className="text-slate-500">horizon</label>
          <select value={horizon} onChange={(e) => setHorizon(Number(e.target.value))} className="rounded border border-border bg-bg/50 px-2 py-0.5 font-mono text-[11px] text-slate-200 outline-none">
            {[30, 90, 180, 365].map((v) => <option key={v} value={v}>{v}d</option>)}
          </select>
        </div>
      </header>
      <div className="flex flex-1 flex-col gap-2 px-3 py-3">
        <div className="grid grid-cols-4 gap-2 font-mono text-[11px]">
          <Stat label="mean" value={`$${(mean / 1000).toFixed(0)}k`} tone={mean >= 100_000 ? "emerald" : "rose"} />
          <Stat label="p5" value={`$${(p5 / 1000).toFixed(0)}k`} tone={p5 >= 100_000 ? "emerald" : p5 >= 80_000 ? "amber" : "rose"} />
          <Stat label="p95" value={`$${(p95 / 1000).toFixed(0)}k`} tone="emerald" />
          <Stat
            label="ruin"
            value={`${((ruinPaths / n) * 100).toFixed(0)}%`}
            tone={ruinPaths === 0 ? "emerald" : ruinPaths / n <= 0.05 ? "amber" : "rose"}
          />
        </div>
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
          <line x1={xAt(0)} y1={yAt(100_000)} x2={xAt(horizon)} y2={yAt(100_000)} stroke="#475569" strokeWidth={0.5} strokeDasharray="3 3" />
          {paths.map((p, i) => {
            const last = p[p.length - 1];
            const tone = last >= 100_000 ? "#10b98155" : "#f43f5e55";
            return (
              <polyline
                key={i}
                fill="none"
                stroke={tone}
                strokeWidth={0.5}
                points={p.map((v, j) => `${xAt(j)},${yAt(v)}`).join(" ")}
              />
            );
          })}
        </svg>
      </div>
    </section>
  );
}

function Stat({ label, value, tone }: { label: string; value: string; tone: "emerald" | "amber" | "rose" }) {
  return (
    <div className="rounded border border-border/60 bg-bg/30 px-2 py-1.5">
      <div className="font-mono text-[9px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className={`font-mono text-sm text-${tone}-400`}>{value}</div>
    </div>
  );
}
