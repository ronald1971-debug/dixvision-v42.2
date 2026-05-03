import { useState } from "react";

/**
 * Drawing-tools rail — vertical icon strip rendered to the left of the
 * main ChartPanel. Captures the *intent* of the operator's selected
 * tool; the actual draw-on-chart wiring lands once Lightweight-Charts
 * exposes a stable plugin API for primitives. For now, selection state
 * is local + persisted to localStorage so other panels (e.g. the
 * SL/TP builder) can read it.
 */

const TOOLS = [
  { id: "cursor", label: "Cursor", glyph: "↖" },
  { id: "trendline", label: "Trendline", glyph: "╱" },
  { id: "channel", label: "Parallel Channel", glyph: "▱" },
  { id: "fib", label: "Fib Retracement", glyph: "ƒ" },
  { id: "fib-ext", label: "Fib Extension", glyph: "ƒ↑" },
  { id: "gann", label: "Gann Fan", glyph: "✱" },
  { id: "pitchfork", label: "Andrews Pitchfork", glyph: "⋔" },
  { id: "rectangle", label: "Rectangle", glyph: "▭" },
  { id: "ellipse", label: "Ellipse", glyph: "◯" },
  { id: "horizontal", label: "Horizontal Line", glyph: "─" },
  { id: "vertical", label: "Vertical Line", glyph: "│" },
  { id: "ray", label: "Ray", glyph: "→" },
  { id: "text", label: "Text Note", glyph: "T" },
  { id: "measure", label: "Measure", glyph: "📏" },
  { id: "magnet", label: "Magnet", glyph: "🧲" },
  { id: "lock", label: "Lock Drawings", glyph: "🔒" },
  { id: "trash", label: "Clear All", glyph: "✕" },
] as const;

const KEY = "dash2:chart:drawing-tool";

export function DrawingToolsRail() {
  const [active, setActive] = useState<string>(() => {
    if (typeof window === "undefined") return "cursor";
    return window.localStorage.getItem(KEY) ?? "cursor";
  });

  const pick = (id: string) => {
    setActive(id);
    if (typeof window !== "undefined") window.localStorage.setItem(KEY, id);
  };

  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Drawing Tools
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          {TOOLS.find((t) => t.id === active)?.label ?? "Cursor"}
        </p>
      </header>
      <div className="flex-1 overflow-auto px-2 py-2">
        <div className="grid grid-cols-3 gap-1">
          {TOOLS.map((t) => {
            const on = t.id === active;
            return (
              <button
                key={t.id}
                type="button"
                onClick={() => pick(t.id)}
                title={t.label}
                className={`flex h-9 items-center justify-center rounded border text-base transition ${
                  on
                    ? "border-sky-500/60 bg-sky-500/15 text-sky-300"
                    : "border-border bg-slate-900/40 text-slate-400 hover:border-slate-500 hover:text-slate-200"
                }`}
              >
                {t.glyph}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
