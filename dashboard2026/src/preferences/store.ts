import { useEffect, useSyncExternalStore } from "react";

/**
 * Tier-7 user preferences — theme, density, layout profile.
 *
 * Persisted to `localStorage` under a single key so a refresh keeps
 * the operator's cockpit personal; defaults apply on first load.
 *
 * Layout profile is the PR-#2 spec §1.1 preset selector
 * (Conservative / Standard / Aggressive / Custom). The profile name
 * does NOT change widget code — it tells panels which dial preset
 * (size cap, drawdown floor, hazard tolerance, signal threshold) to
 * read from the constraint engine. Custom = operator-tuned override
 * persisted in the audit ledger.
 */
export type Theme = "dark" | "midnight" | "ash";
export type Density = "compact" | "normal" | "comfortable";
export type LayoutProfile =
  | "conservative"
  | "standard"
  | "aggressive"
  | "custom";

export interface Preferences {
  theme: Theme;
  density: Density;
  layoutProfile: LayoutProfile;
}

const KEY = "dix.dash2.preferences.v1";

export const DEFAULT_PREFERENCES: Preferences = {
  theme: "dark",
  density: "normal",
  layoutProfile: "standard",
};

const THEMES: readonly Theme[] = ["dark", "midnight", "ash"];
const DENSITIES: readonly Density[] = ["compact", "normal", "comfortable"];
const LAYOUTS: readonly LayoutProfile[] = [
  "conservative",
  "standard",
  "aggressive",
  "custom",
];

function isTheme(v: unknown): v is Theme {
  return typeof v === "string" && (THEMES as readonly string[]).includes(v);
}
function isDensity(v: unknown): v is Density {
  return typeof v === "string" && (DENSITIES as readonly string[]).includes(v);
}
function isLayout(v: unknown): v is LayoutProfile {
  return typeof v === "string" && (LAYOUTS as readonly string[]).includes(v);
}

function load(): Preferences {
  if (typeof window === "undefined") return DEFAULT_PREFERENCES;
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return DEFAULT_PREFERENCES;
    const parsed: unknown = JSON.parse(raw);
    if (parsed === null || typeof parsed !== "object") {
      return DEFAULT_PREFERENCES;
    }
    const o = parsed as Record<string, unknown>;
    return {
      theme: isTheme(o.theme) ? o.theme : DEFAULT_PREFERENCES.theme,
      density: isDensity(o.density) ? o.density : DEFAULT_PREFERENCES.density,
      layoutProfile: isLayout(o.layoutProfile)
        ? o.layoutProfile
        : DEFAULT_PREFERENCES.layoutProfile,
    };
  } catch {
    return DEFAULT_PREFERENCES;
  }
}

let current: Preferences = load();
const listeners = new Set<() => void>();

function emit() {
  for (const fn of listeners) fn();
}

export function getPreferences(): Preferences {
  return current;
}

export function setPreferences(patch: Partial<Preferences>) {
  current = { ...current, ...patch };
  if (typeof window !== "undefined") {
    try {
      window.localStorage.setItem(KEY, JSON.stringify(current));
    } catch {
      /* storage full / blocked — keep in-memory only */
    }
  }
  emit();
}

function subscribe(fn: () => void): () => void {
  listeners.add(fn);
  return () => {
    listeners.delete(fn);
  };
}

export function usePreferences(): Preferences {
  return useSyncExternalStore(subscribe, getPreferences, getPreferences);
}

/**
 * Effect-style hook that mirrors current preferences into HTML
 * `data-theme` / `data-density` attributes so Tailwind / CSS can
 * key off `[data-theme="ash"]` selectors without a global context.
 */
export function useApplyPreferences() {
  const prefs = usePreferences();
  useEffect(() => {
    const root = document.documentElement;
    root.setAttribute("data-theme", prefs.theme);
    root.setAttribute("data-density", prefs.density);
    root.setAttribute("data-layout-profile", prefs.layoutProfile);
  }, [prefs.theme, prefs.density, prefs.layoutProfile]);
}

export const THEME_OPTIONS = THEMES;
export const DENSITY_OPTIONS = DENSITIES;
export const LAYOUT_OPTIONS = LAYOUTS;
