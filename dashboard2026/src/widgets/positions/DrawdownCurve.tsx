import { useMemo } from "react";

/**
 * G-track widget — Drawdown curve.
 *
 * Underwater equity plot — distance from running peak.
 * Backend hook: ``GET /api/positions/drawdown?window=session|day|week|all``.
 * Today seeded from same intraday equity path as IntradayPnLCurve.
 */
const EQUITY = [
  100_000, 100_240, 100_480, 100_320, 100_110, 99_820, 99_690, 99_910,
  100_320, 100_540, 100_880, 101_120, 101_460, 101_842, 102_104,
];
const STAMPS = [
  "09:30",
  "10:00",
  "10:30",
  "11:00",
  "11:30",
  "12:00",
  "12:30",
  "13:00",
  "13:30",
  "14:00",
  "14:30",
  "15:00",
  "15:30",
  "16:00",
  "16:30",
];

export function DrawdownCurve() {
  const { path, area, ddCurr, ddMax, ddMaxAt, w, h } = useMemo(() => {
    const w = 600;
    const h = 200;
    const padX = 6;
    const padY = 8;
    let peak = EQUITY[0];
    const dd = EQUITY.map((v) => {
      peak = Math.max(peak, v);
      return ((v - peak) / peak) * 100;
    });
    const maxDD = Math.min(...dd);
    const maxDDIdx = dd.indexOf(maxDD);
    const xs = (i: number) =>
      padX + (i / (EQUITY.length - 1)) * (w - padX * 2);
    const ys = (d: number) => padY + (-d / Math.abs(Math.min(maxDD, -0.01))) * (h - padY * 2);
    const path = dd
      .map((d, i) => `${i === 0 ? "M" : "L"}${xs(i).toFixed(1)},${ys(d).toFixed(1)}`)
      .join(" ");
    const area =
      `M${xs(0).toFixed(1)},${padY} ` +
      dd.map((d, i) => `L${xs(i).toFixed(1)},${ys(d).toFixed(1)}`).join(" ") +
      ` L${xs(EQUITY.length - 1).toFixed(1)},${padY} Z`;
    return {
      path,
      area,
      ddCurr: dd[dd.length - 1],
      ddMax: maxDD,
      ddMaxAt: STAMPS[maxDDIdx],
      w,
      h,
    };
  }, []);

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Drawdown
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            underwater curve · % from running peak
          </p>
        </div>
        <div className="font-mono text-[11px] text-slate-300">
          <span className="text-[10px] uppercase tracking-wider text-slate-500">
            now
          </span>{" "}
          <span className={ddCurr < 0 ? "text-rose-400" : "text-slate-400"}>
            {ddCurr.toFixed(2)}%
          </span>
        </div>
      </header>
      <div className="flex-1 p-3">
        <svg
          viewBox={`0 0 ${w} ${h}`}
          preserveAspectRatio="none"
          className="h-full w-full"
        >
          <defs>
            <linearGradient id="dd-grad" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0" stopColor="#f43f5e" stopOpacity="0" />
              <stop offset="1" stopColor="#f43f5e" stopOpacity="0.3" />
            </linearGradient>
          </defs>
          <path d={area} fill="url(#dd-grad)" />
          <path d={path} fill="none" stroke="#f43f5e" strokeWidth={1.5} />
        </svg>
      </div>
      <footer className="grid grid-cols-2 gap-2 border-t border-border bg-bg/40 px-3 py-2 font-mono text-[10px] text-slate-400">
        <div>
          <span className="text-slate-500 uppercase tracking-wider">max DD</span>
          <span className="ml-1 text-rose-400">{ddMax.toFixed(2)}%</span>
        </div>
        <div className="text-right">
          <span className="text-slate-500 uppercase tracking-wider">at</span>
          <span className="ml-1">{ddMaxAt}</span>
        </div>
      </footer>
    </section>
  );
}
