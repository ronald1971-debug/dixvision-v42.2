import { useMemo, useState } from "react";

/**
 * I-track widget — Equity-curve studio.
 *
 * Full strategy × symbol × date-range deterministic backtest viewer.
 * Generates a seeded equity path and renders the curve, drawdown
 * envelope, and core metrics (Sharpe, max DD, CAGR).
 *
 * Backend hook: ``GET /api/testing/equity-curve?strategy=&symbol=&start=&end=``
 * runs the canonical historical replay and returns the equity series
 * with per-tick PnL.
 */
type Range = "30d" | "90d" | "180d" | "365d";

const STRATS = ["strat_meanrev_v3", "strat_breakout_v2", "strat_funding_v1"];
const SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "AVAX-USDT"];

function seededCurve(seed: number, n: number): number[] {
  let v = 100_000;
  const out: number[] = [v];
  let r = seed;
  for (let i = 0; i < n; i++) {
    r = (r * 9301 + 49297) % 233280;
    const noise = (r / 233280 - 0.5) * 0.018;
    const trend = 0.0006;
    v = v * (1 + trend + noise);
    out.push(v);
  }
  return out;
}

function metrics(curve: number[]) {
  const ret = curve.slice(1).map((v, i) => v / curve[i] - 1);
  const mean = ret.reduce((s, x) => s + x, 0) / ret.length;
  const variance = ret.reduce((s, x) => s + (x - mean) ** 2, 0) / ret.length;
  const std = Math.sqrt(variance);
  const sharpe = (mean / std) * Math.sqrt(365);
  let peak = curve[0];
  let maxDD = 0;
  for (const v of curve) {
    if (v > peak) peak = v;
    const dd = (v - peak) / peak;
    if (dd < maxDD) maxDD = dd;
  }
  const cagr = (curve[curve.length - 1] / curve[0]) ** (365 / curve.length) - 1;
  return { sharpe, maxDD, cagr, finalPnl: curve[curve.length - 1] - curve[0] };
}

const W = 480;
const H = 200;
const PAD_X = 36;
const PAD_Y = 12;

export function EquityCurveStudio() {
  const [strat, setStrat] = useState(STRATS[0]);
  const [symbol, setSymbol] = useState(SYMBOLS[0]);
  const [range, setRange] = useState<Range>("90d");

  const days = range === "30d" ? 30 : range === "90d" ? 90 : range === "180d" ? 180 : 365;
  const seed =
    strat.length * 13 + symbol.length * 7 + days + STRATS.indexOf(strat) * 1009 + SYMBOLS.indexOf(symbol) * 53;

  const { curve, m } = useMemo(() => {
    const c = seededCurve(seed, days);
    return { curve: c, m: metrics(c) };
  }, [seed, days]);

  const min = Math.min(...curve);
  const max = Math.max(...curve);
  const x = (i: number) => PAD_X + (i / (curve.length - 1)) * (W - PAD_X * 2);
  const y = (v: number) => H - PAD_Y - ((v - min) / (max - min || 1)) * (H - PAD_Y * 2);

  let peak = curve[0];
  const ddCurve = curve.map((v) => {
    if (v > peak) peak = v;
    return (v - peak) / peak;
  });

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Equity-curve studio
        </h3>
        <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px]">
          <select value={strat} onChange={(e) => setStrat(e.target.value)} className="rounded border border-border bg-bg/50 px-2 py-0.5 font-mono text-[11px] text-slate-200 outline-none">
            {STRATS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
          <select value={symbol} onChange={(e) => setSymbol(e.target.value)} className="rounded border border-border bg-bg/50 px-2 py-0.5 font-mono text-[11px] text-slate-200 outline-none">
            {SYMBOLS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
          <div className="ml-auto flex gap-1">
            {(["30d", "90d", "180d", "365d"] as Range[]).map((r) => (
              <button
                key={r}
                onClick={() => setRange(r)}
                className={`rounded border px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider transition ${
                  range === r
                    ? "border-accent/60 bg-accent/20 text-accent"
                    : "border-border bg-bg/30 text-slate-400 hover:bg-bg/60"
                }`}
              >
                {r}
              </button>
            ))}
          </div>
        </div>
      </header>
      <div className="flex flex-1 flex-col gap-2 px-3 py-3">
        <div className="grid grid-cols-4 gap-2 font-mono text-[11px]">
          <Stat label="PnL" value={`$${(m.finalPnl / 1000).toFixed(1)}k`} tone={m.finalPnl >= 0 ? "emerald" : "rose"} />
          <Stat label="Sharpe" value={m.sharpe.toFixed(2)} tone={m.sharpe >= 1.5 ? "emerald" : m.sharpe >= 0.5 ? "amber" : "rose"} />
          <Stat label="Max DD" value={`${(m.maxDD * 100).toFixed(1)}%`} tone={m.maxDD >= -0.1 ? "emerald" : m.maxDD >= -0.2 ? "amber" : "rose"} />
          <Stat label="CAGR" value={`${(m.cagr * 100).toFixed(1)}%`} tone={m.cagr >= 0.2 ? "emerald" : m.cagr >= 0 ? "amber" : "rose"} />
        </div>
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
          <defs>
            <linearGradient id="eq-grad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#10b981" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#10b981" stopOpacity="0" />
            </linearGradient>
          </defs>
          <polygon
            fill="url(#eq-grad)"
            points={[`${x(0)},${H - PAD_Y}`, ...curve.map((v, i) => `${x(i)},${y(v)}`), `${x(curve.length - 1)},${H - PAD_Y}`].join(" ")}
          />
          <polyline fill="none" stroke="#34d399" strokeWidth={1.4} points={curve.map((v, i) => `${x(i)},${y(v)}`).join(" ")} />
          {ddCurve.map((dd, i) => {
            if (dd >= 0) return null;
            const ddH = Math.min(-dd * 80, 40);
            return <line key={i} x1={x(i)} y1={H - PAD_Y} x2={x(i)} y2={H - PAD_Y - ddH} stroke="#f43f5e" strokeOpacity={0.25} strokeWidth={1.2} />;
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
