import { useEffect, useState } from "react";

/**
 * F-track widget — Order hotkeys panel.
 *
 * Lets the operator inspect (and remap) keyboard chords for fast
 * order entry / cancellation. Mirrors the prop-trader cockpit
 * (NinjaTrader / TradingView pro) hotkey configurator.
 *
 * Persistence is local only (``localStorage["dix-order-hotkeys"]``);
 * the backend adapter lands in a follow-up that hooks the bindings
 * to ``ExecutionIntent`` factories. Approval-edge gating (INV-72)
 * still applies — chords stage intents, they never auto-execute.
 */
interface Hotkey {
  id: string;
  label: string;
  action: string;
  chord: string;
}

const STORAGE_KEY = "dix-order-hotkeys-v1";

const DEFAULTS: Hotkey[] = [
  { id: "buy-mkt", label: "Market BUY at ask", action: "BUY MKT", chord: "B" },
  { id: "sell-mkt", label: "Market SELL at bid", action: "SELL MKT", chord: "S" },
  {
    id: "buy-mid",
    label: "BUY at mid",
    action: "BUY LMT mid",
    chord: "Shift+B",
  },
  {
    id: "sell-mid",
    label: "SELL at mid",
    action: "SELL LMT mid",
    chord: "Shift+S",
  },
  {
    id: "flatten",
    label: "Flatten symbol",
    action: "FLATTEN",
    chord: "Ctrl+F",
  },
  {
    id: "cancel-all",
    label: "Cancel all working",
    action: "CANCEL ALL",
    chord: "Esc",
  },
  {
    id: "tighten-stop",
    label: "Move SL to entry",
    action: "SL := entry",
    chord: "Ctrl+E",
  },
  {
    id: "scale-out-25",
    label: "Scale out 25%",
    action: "SCALE 25%",
    chord: "Ctrl+Shift+1",
  },
  {
    id: "scale-out-50",
    label: "Scale out 50%",
    action: "SCALE 50%",
    chord: "Ctrl+Shift+2",
  },
];

function load(): Hotkey[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULTS;
    const parsed = JSON.parse(raw) as Hotkey[];
    if (!Array.isArray(parsed)) return DEFAULTS;
    return parsed;
  } catch {
    return DEFAULTS;
  }
}

function save(rows: Hotkey[]) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(rows));
  } catch {
    /* localStorage may be unavailable */
  }
}

export function OrderHotkeysPanel() {
  const [rows, setRows] = useState<Hotkey[]>(() => load());
  const [recording, setRecording] = useState<string | null>(null);

  useEffect(() => {
    save(rows);
  }, [rows]);

  useEffect(() => {
    if (!recording) return;
    const onKey = (e: KeyboardEvent) => {
      if (
        e.key === "Shift" ||
        e.key === "Control" ||
        e.key === "Alt" ||
        e.key === "Meta"
      ) {
        return;
      }
      e.preventDefault();
      const parts: string[] = [];
      if (e.ctrlKey || e.metaKey) parts.push("Ctrl");
      if (e.altKey) parts.push("Alt");
      if (e.shiftKey) parts.push("Shift");
      const k = e.key.length === 1 ? e.key.toUpperCase() : e.key;
      parts.push(k);
      const chord = parts.join("+");
      setRows((prev) =>
        prev.map((r) => (r.id === recording ? { ...r, chord } : r)),
      );
      setRecording(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [recording]);

  const reset = () => setRows(DEFAULTS);

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Order hotkeys
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          chords stage intents · click "rebind" then press a chord · approval
          edge still gates execution
        </p>
      </header>
      <div className="flex-1 overflow-auto">
        <table className="w-full font-mono text-[11px] text-slate-300">
          <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
            <tr className="border-b border-border">
              <th className="px-3 py-1.5 text-left">label</th>
              <th className="px-3 py-1.5 text-left">action</th>
              <th className="px-3 py-1.5 text-left">chord</th>
              <th className="px-3 py-1.5"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/40">
            {rows.map((r) => (
              <tr key={r.id}>
                <td className="px-3 py-1 text-slate-200">{r.label}</td>
                <td className="px-3 py-1 text-slate-400">{r.action}</td>
                <td className="px-3 py-1">
                  <span
                    className={`inline-block rounded border px-1.5 py-0.5 text-[10px] uppercase tracking-wider ${
                      recording === r.id
                        ? "animate-pulse border-amber-500/40 bg-amber-500/10 text-amber-300"
                        : "border-border bg-bg/40 text-slate-300"
                    }`}
                  >
                    {recording === r.id ? "press chord…" : r.chord}
                  </span>
                </td>
                <td className="px-3 py-1 text-right">
                  <button
                    type="button"
                    onClick={() =>
                      setRecording((cur) => (cur === r.id ? null : r.id))
                    }
                    className="rounded border border-border bg-bg/40 px-2 py-0.5 text-[10px] uppercase tracking-wider text-slate-400 hover:border-accent hover:text-accent"
                  >
                    rebind
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <footer className="flex items-center gap-2 border-t border-border bg-bg/40 px-3 py-2 font-mono text-[10px] text-slate-500">
        <span>persisted in localStorage · {rows.length} bindings</span>
        <button
          type="button"
          onClick={reset}
          className="ml-auto rounded border border-border bg-bg/40 px-2 py-0.5 uppercase tracking-wider text-slate-400 hover:border-accent hover:text-accent"
        >
          reset defaults
        </button>
      </footer>
    </section>
  );
}
