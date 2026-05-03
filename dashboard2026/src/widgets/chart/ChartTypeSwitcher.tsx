import { useState } from "react";

/**
 * Chart-type switcher — lets the operator flip the main ChartPanel
 * between Candle / Heikin-Ashi / Line / Area / Renko / Range / Tick /
 * Footprint / CVD. The selection is persisted in localStorage; the
 * main ChartPanel reads it on mount (wired in a follow-up — for now
 * this widget is the operator-visible source of truth).
 */

const TYPES = [
  { id: "candle", label: "Candle", desc: "OHLC bars" },
  { id: "heikin", label: "Heikin Ashi", desc: "smoothed direction" },
  { id: "line", label: "Line", desc: "close only" },
  { id: "area", label: "Area", desc: "filled close" },
  { id: "renko", label: "Renko", desc: "fixed-brick price" },
  { id: "range", label: "Range", desc: "fixed range bars" },
  { id: "tick", label: "Tick", desc: "every print" },
  { id: "footprint", label: "Footprint", desc: "bid/ask volume" },
  { id: "cvd", label: "CVD", desc: "cumulative delta" },
] as const;

const KEY = "dash2:chart:type";

export function ChartTypeSwitcher() {
  const [active, setActive] = useState<string>(() => {
    if (typeof window === "undefined") return "candle";
    return window.localStorage.getItem(KEY) ?? "candle";
  });
  const pick = (id: string) => {
    setActive(id);
    if (typeof window !== "undefined") window.localStorage.setItem(KEY, id);
  };
  const current = TYPES.find((t) => t.id === active) ?? TYPES[0];

  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Chart Type
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">{current.desc}</p>
      </header>
      <div className="flex-1 overflow-auto px-2 py-2">
        <div className="grid grid-cols-3 gap-1">
          {TYPES.map((t) => {
            const on = t.id === active;
            return (
              <button
                key={t.id}
                type="button"
                onClick={() => pick(t.id)}
                title={t.desc}
                className={`rounded border px-2 py-1.5 text-left text-[11px] transition ${
                  on
                    ? "border-sky-500/60 bg-sky-500/15 text-sky-300"
                    : "border-border bg-slate-900/40 text-slate-300 hover:border-slate-500"
                }`}
              >
                <div className="font-medium">{t.label}</div>
                <div className="text-[9px] uppercase tracking-wider text-slate-500">
                  {t.desc}
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
