import { useMemo } from "react";

/**
 * G-track widget — Intraday PnL curve.
 *
 * Equity curve since session open with marked-to-market PnL.
 * Backend hook: ``GET /api/positions/intraday_pnl?since=session_open``
 * reads from ``portfolio_engine.session_pnl``.
 */
const POINTS = [
  { t: "09:30", v: 0 },
  { t: "10:00", v: 240 },
  { t: "10:30", v: 480 },
  { t: "11:00", v: 320 },
  { t: "11:30", v: 110 },
  { t: "12:00", v: -180 },
  { t: "12:30", v: -310 },
  { t: "13:00", v: -90 },
  { t: "13:30", v: 320 },
  { t: "14:00", v: 540 },
  { t: "14:30", v: 880 },
  { t: "15:00", v: 1_120 },
  { t: "15:30", v: 1_460 },
  { t: "16:00", v: 1_842 },
  { t: "16:30", v: 2_104 },
];

export function IntradayPnLCurve() {
  const { path, area, hi, lo, last, w, h } = useMemo(() => {
    const w = 600;
    const h = 200;
    const padX = 6;
    const padY = 8;
    const xs = (i: number) =>
      padX + (i / (POINTS.length - 1)) * (w - padX * 2);
    const vs = POINTS.map((p) => p.v);
    const hi = Math.max(...vs);
    const lo = Math.min(...vs);
    const span = Math.max(hi - lo, 1);
    const ys = (v: number) =>
      h - padY - ((v - lo) / span) * (h - padY * 2);
    const path = POINTS.map((p, i) => `${i === 0 ? "M" : "L"}${xs(i).toFixed(1)},${ys(p.v).toFixed(1)}`).join(" ");
    const area =
      `M${xs(0).toFixed(1)},${(h - padY).toFixed(1)} ` +
      POINTS.map((p, i) => `L${xs(i).toFixed(1)},${ys(p.v).toFixed(1)}`).join(" ") +
      ` L${xs(POINTS.length - 1).toFixed(1)},${(h - padY).toFixed(1)} Z`;
    return { path, area, hi, lo, last: vs[vs.length - 1], w, h };
  }, []);

  const positive = last >= 0;
  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Intraday PnL
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            mark-to-market since session open
          </p>
        </div>
        <div className="font-mono text-[11px] text-slate-300">
          <span className="text-[10px] uppercase tracking-wider text-slate-500">
            now
          </span>{" "}
          <span className={positive ? "text-emerald-400" : "text-rose-400"}>
            {positive ? "+" : ""}
            {last.toLocaleString()} USD
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
            <linearGradient id="pnl-grad" x1="0" x2="0" y1="0" y2="1">
              <stop
                offset="0"
                stopColor={positive ? "#10b981" : "#f43f5e"}
                stopOpacity="0.3"
              />
              <stop
                offset="1"
                stopColor={positive ? "#10b981" : "#f43f5e"}
                stopOpacity="0"
              />
            </linearGradient>
          </defs>
          <path d={area} fill="url(#pnl-grad)" />
          <path
            d={path}
            fill="none"
            stroke={positive ? "#10b981" : "#f43f5e"}
            strokeWidth={1.5}
          />
        </svg>
      </div>
      <footer className="grid grid-cols-3 gap-2 border-t border-border bg-bg/40 px-3 py-2 font-mono text-[10px] text-slate-400">
        <div>
          <span className="text-slate-500 uppercase tracking-wider">peak</span>
          <span className="ml-1 text-emerald-400">
            +{hi.toLocaleString()}
          </span>
        </div>
        <div>
          <span className="text-slate-500 uppercase tracking-wider">trough</span>
          <span className="ml-1 text-rose-400">{lo.toLocaleString()}</span>
        </div>
        <div className="text-right">
          <span className="text-slate-500 uppercase tracking-wider">range</span>
          <span className="ml-1">{(hi - lo).toLocaleString()}</span>
        </div>
      </footer>
    </section>
  );
}
