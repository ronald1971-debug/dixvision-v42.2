import { useEffect, useMemo, useRef, useState } from "react";

import {
  ASSET_ROUTE_LIST,
  SYSTEM_ROUTE_LIST,
  type Route,
} from "@/router";

/**
 * Tier-7 cockpit command palette (Ctrl-K / Cmd-K).
 *
 * Single keyboard-driven launcher for jumping between routes plus a
 * small set of operator actions (toggle sidebar, kill switch, open
 * approval queue). Subset filtering is plain `startsWith` /
 * `includes` against label tokens — no fuzzy library needed.
 *
 * Actions execute synchronously and close the palette. The palette
 * is mounted at the App level and toggled via a global hotkey
 * listener registered on `document` so it works regardless of which
 * widget has focus.
 */
export interface CommandAction {
  id: string;
  label: string;
  group: "Navigate" | "System";
  run: () => void;
  hint?: string;
}

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
  onNavigate: (route: Route) => void;
  extraActions?: CommandAction[];
}

const ROUTE_LABELS: Record<Route, string> = {
  spot: "Spot",
  perps: "Perps",
  dex: "DEX",
  memecoin: "Memecoin",
  forex: "Forex",
  stocks: "Stocks",
  nft: "NFT",
  operator: "Operator",
  credentials: "Credentials",
  chat: "Chat",
  indira: "Indira learn",
  dyon: "Dyon learn",
  testing: "Testing & Eval",
  governance: "Governance",
  risk: "Risk & Greeks",
};

function buildNavActions(onNavigate: (route: Route) => void): CommandAction[] {
  const all: Route[] = [...ASSET_ROUTE_LIST, ...SYSTEM_ROUTE_LIST];
  return all.map((r) => ({
    id: `nav:${r}`,
    label: `Go to ${ROUTE_LABELS[r]}`,
    group: "Navigate",
    hint: `#/${r}`,
    run: () => onNavigate(r),
  }));
}

export function CommandPalette({
  open,
  onClose,
  onNavigate,
  extraActions = [],
}: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const actions = useMemo(
    () => [...buildNavActions(onNavigate), ...extraActions],
    [onNavigate, extraActions],
  );

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (q === "") return actions;
    return actions.filter((a) => a.label.toLowerCase().includes(q));
  }, [actions, query]);

  useEffect(() => {
    if (!open) {
      setQuery("");
      setCursor(0);
      return;
    }
    const t = setTimeout(() => inputRef.current?.focus(), 0);
    return () => clearTimeout(t);
  }, [open]);

  useEffect(() => {
    setCursor(0);
  }, [query]);

  if (!open) return null;

  const grouped = (() => {
    const groups: Record<string, CommandAction[]> = {};
    filtered.forEach((a) => {
      groups[a.group] ??= [];
      groups[a.group].push(a);
    });
    return groups;
  })();
  // Single source of truth for cursor: items in the same order they
  // are rendered (grouped order), so keyboard Enter, visual highlight,
  // and onMouseEnter all index into the same array. Without this,
  // non-contiguous group order in `filtered` would let hover set the
  // cursor to one item while Enter fires a different one.
  const flatGrouped: CommandAction[] = Object.values(grouped).flat();

  function handleKey(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Escape") {
      e.preventDefault();
      onClose();
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setCursor((c) => Math.min(flatGrouped.length - 1, c + 1));
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      setCursor((c) => Math.max(0, c - 1));
      return;
    }
    if (e.key === "Enter") {
      e.preventDefault();
      const target = flatGrouped[cursor];
      if (target) {
        target.run();
        onClose();
      }
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/60 px-4 pt-24"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
    >
      <div
        className="w-full max-w-xl overflow-hidden rounded-lg border border-border bg-surface shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Type to navigate or run a system command…"
          className="w-full border-b border-border bg-transparent px-4 py-3 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none"
        />
        <div className="max-h-[50vh] overflow-auto py-1">
          {filtered.length === 0 && (
            <div className="px-4 py-6 text-center text-xs text-slate-500">
              No matches
            </div>
          )}
          {Object.entries(grouped).map(([group, items]) => (
            <div key={group} className="py-1">
              <div className="px-4 pb-1 pt-2 text-[10px] font-semibold uppercase tracking-widest text-slate-500">
                {group}
              </div>
              {items.map((a) => {
                const idx = flatGrouped.indexOf(a);
                const isActive = idx === cursor;
                return (
                  <button
                    key={a.id}
                    type="button"
                    onMouseEnter={() => setCursor(idx)}
                    onClick={() => {
                      a.run();
                      onClose();
                    }}
                    className={`flex w-full items-center justify-between px-4 py-2 text-left text-sm ${
                      isActive
                        ? "bg-accent/15 text-slate-50"
                        : "text-slate-200 hover:bg-accent/10"
                    }`}
                  >
                    <span>{a.label}</span>
                    {a.hint && (
                      <span className="font-mono text-[10px] text-slate-500">
                        {a.hint}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          ))}
        </div>
        <div className="flex items-center justify-between border-t border-border bg-bg px-4 py-2 text-[10px] text-slate-500">
          <span>↵ run · ↑↓ select · esc close</span>
          <span className="font-mono">⌘K / Ctrl+K</span>
        </div>
      </div>
    </div>
  );
}
