import { useEffect, useSyncExternalStore } from "react";

/**
 * J-track hotkey store — operator-rebindable global shortcuts.
 *
 * Each binding is a stringified shortcut like ``"ctrl+k"`` or
 * ``"shift+/"``. The store persists overrides to localStorage; the
 * default map is the source of truth and is restored on
 * ``resetHotkeys()``. Bindings are checked case-insensitively
 * against ``e.key`` (single character) so they survive layout
 * differences.
 */
export type HotkeyAction =
  | "toggle-palette"
  | "toggle-sidebar"
  | "go-operator"
  | "go-governance"
  | "go-testing"
  | "go-ai"
  | "kill-switch";

export interface HotkeyBinding {
  action: HotkeyAction;
  combo: string;
  label: string;
}

export const HOTKEY_DEFAULTS: readonly HotkeyBinding[] = [
  { action: "toggle-palette", combo: "ctrl+k", label: "Open command palette" },
  { action: "toggle-sidebar", combo: "ctrl+b", label: "Toggle sidebar" },
  { action: "go-operator", combo: "ctrl+1", label: "Go to operator" },
  { action: "go-governance", combo: "ctrl+2", label: "Go to governance" },
  { action: "go-testing", combo: "ctrl+3", label: "Go to testing" },
  { action: "go-ai", combo: "ctrl+4", label: "Go to AI" },
  { action: "kill-switch", combo: "ctrl+shift+k", label: "Kill switch" },
];

const KEY = "dix.dash2.hotkeys.v1";

type HotkeyMap = Record<HotkeyAction, string>;

function defaultMap(): HotkeyMap {
  const m = {} as HotkeyMap;
  for (const b of HOTKEY_DEFAULTS) m[b.action] = b.combo;
  return m;
}

function load(): HotkeyMap {
  if (typeof window === "undefined") return defaultMap();
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return defaultMap();
    const parsed: unknown = JSON.parse(raw);
    if (parsed === null || typeof parsed !== "object") return defaultMap();
    const o = parsed as Record<string, unknown>;
    const base = defaultMap();
    for (const a of Object.keys(base) as HotkeyAction[]) {
      const v = o[a];
      if (typeof v === "string" && v.trim() !== "") base[a] = v.toLowerCase();
    }
    return base;
  } catch {
    return defaultMap();
  }
}

let current: HotkeyMap = load();
const listeners = new Set<() => void>();

function emit() {
  for (const fn of listeners) fn();
}

function persist() {
  if (typeof window !== "undefined") {
    try {
      window.localStorage.setItem(KEY, JSON.stringify(current));
    } catch {
      /* ignore storage failures */
    }
  }
}

export function getHotkeys(): HotkeyMap {
  return current;
}

export function setHotkey(action: HotkeyAction, combo: string) {
  current = { ...current, [action]: combo.trim().toLowerCase() };
  persist();
  emit();
}

export function resetHotkeys() {
  current = defaultMap();
  persist();
  emit();
}

function subscribe(fn: () => void): () => void {
  listeners.add(fn);
  return () => {
    listeners.delete(fn);
  };
}

export function useHotkeys(): HotkeyMap {
  return useSyncExternalStore(subscribe, getHotkeys, getHotkeys);
}

export function comboMatches(combo: string, e: KeyboardEvent): boolean {
  const parts = combo.toLowerCase().split("+").map((p) => p.trim());
  const key = parts[parts.length - 1];
  if (!key) return false;
  const want = {
    ctrl: parts.includes("ctrl"),
    shift: parts.includes("shift"),
    alt: parts.includes("alt"),
    meta: parts.includes("meta") || parts.includes("cmd"),
  };
  // Treat ``ctrl`` as either ctrlKey or metaKey so the same combo works
  // on macOS without forcing a separate ⌘ binding.
  const ctrlOrMeta = e.ctrlKey || e.metaKey;
  if (want.ctrl !== ctrlOrMeta) return false;
  if (want.shift !== e.shiftKey) return false;
  if (want.alt !== e.altKey) return false;
  return e.key.toLowerCase() === key;
}

/**
 * Wires a single global handler for every registered hotkey. The
 * handler map is the only place ``e.preventDefault`` is called so
 * pop-out windows opting out of global hotkeys can simply skip the
 * hook.
 */
export function useGlobalHotkeys(handlers: Partial<Record<HotkeyAction, () => void>>) {
  const hotkeys = useHotkeys();
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      for (const action of Object.keys(handlers) as HotkeyAction[]) {
        const combo = hotkeys[action];
        if (combo && comboMatches(combo, e)) {
          const fn = handlers[action];
          if (fn) {
            e.preventDefault();
            fn();
          }
          return;
        }
      }
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [hotkeys, handlers]);
}
