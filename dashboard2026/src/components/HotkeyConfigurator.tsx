import { useState } from "react";

import {
  HOTKEY_DEFAULTS,
  resetHotkeys,
  setHotkey,
  useHotkeys,
  type HotkeyAction,
} from "@/state/hotkeys";
import { pushToast } from "@/state/toast";

/**
 * J-track operator-rebindable hotkey panel.
 *
 * Press *Capture* on a row, then any key combination to bind it.
 * ``Esc`` cancels capture without changing the binding. Defaults are
 * restorable per-row or globally via *Reset all*.
 */
export function HotkeyConfigurator() {
  const hotkeys = useHotkeys();
  const [capturing, setCapturing] = useState<HotkeyAction | null>(null);

  function startCapture(action: HotkeyAction) {
    setCapturing(action);
  }

  function captureKey(e: React.KeyboardEvent<HTMLButtonElement>) {
    if (!capturing) return;
    e.preventDefault();
    e.stopPropagation();
    if (e.key === "Escape") {
      setCapturing(null);
      return;
    }
    if (["Control", "Shift", "Alt", "Meta"].includes(e.key)) return;
    const parts: string[] = [];
    if (e.ctrlKey || e.metaKey) parts.push("ctrl");
    if (e.shiftKey) parts.push("shift");
    if (e.altKey) parts.push("alt");
    parts.push(e.key.toLowerCase());
    const combo = parts.join("+");
    setHotkey(capturing, combo);
    setCapturing(null);
    pushToast(`Bound ${combo} → ${capturing}`, { tone: "success" });
  }

  return (
    <section className="rounded border border-border bg-surface">
      <header className="flex items-center justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Hotkey configurator
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            Press <span className="font-mono">capture</span> then any key combination · esc cancels
          </p>
        </div>
        <button
          type="button"
          onClick={() => {
            resetHotkeys();
            pushToast("Hotkeys reset to defaults", { tone: "info" });
          }}
          className="rounded border border-border px-2 py-1 text-[11px] uppercase tracking-wider text-slate-300 hover:border-accent/50"
        >
          Reset all
        </button>
      </header>
      <div className="divide-y divide-border">
        {HOTKEY_DEFAULTS.map((b) => {
          const combo = hotkeys[b.action];
          const isDefault = combo === b.combo;
          const isCapturing = capturing === b.action;
          return (
            <div
              key={b.action}
              className="flex items-center justify-between gap-3 px-3 py-2"
            >
              <div className="min-w-0">
                <div className="text-xs text-slate-200">{b.label}</div>
                <div className="font-mono text-[10px] text-slate-500">
                  {b.action}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onKeyDown={captureKey}
                  onClick={() => startCapture(b.action)}
                  className={`rounded border px-2 py-1 font-mono text-[11px] ${
                    isCapturing
                      ? "border-accent text-accent"
                      : "border-border text-slate-200 hover:border-accent/50"
                  }`}
                >
                  {isCapturing ? "press combo…" : combo}
                </button>
                {!isDefault && (
                  <button
                    type="button"
                    onClick={() => setHotkey(b.action, b.combo)}
                    className="text-[10px] uppercase tracking-wider text-slate-500 hover:text-slate-300"
                  >
                    reset
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
