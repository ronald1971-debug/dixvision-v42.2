import { Gauge, LayoutGrid, Palette, Rows3, type LucideIcon } from "lucide-react";

import {
  DENSITY_OPTIONS,
  LAYOUT_OPTIONS,
  THEME_OPTIONS,
  setPreferences,
  usePreferences,
  type Density,
  type LayoutProfile,
  type Theme,
} from "@/preferences/store";

/**
 * Tier-7 preferences bar — three pill rotators in the top header.
 *
 * Theme rotates through dark / midnight / ash (mirrored as
 * `data-theme=…` on `<html>` so future Tailwind dark variants and
 * widget-local CSS can key off it). Density rotates through
 * compact / normal / comfortable (mirrored as `data-density=…`,
 * widgets read it via class queries on the root). Layout profile
 * rotates Conservative / Standard / Aggressive / Custom — preset
 * dial values for size cap, drawdown floor, hazard tolerance, and
 * signal threshold. The profile name does not mutate widget code.
 */
function rotate<T>(arr: readonly T[], current: T): T {
  const idx = arr.indexOf(current);
  return arr[(idx + 1) % arr.length];
}

interface PillProps {
  label: string;
  value: string;
  onClick: () => void;
  icon: LucideIcon;
  hint?: string;
}

function Pill({ label, value, onClick, icon: Icon, hint }: PillProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={hint ?? `Cycle ${label}`}
      className="flex items-center gap-1.5 rounded border border-border bg-bg px-2 py-1 text-[10px] uppercase tracking-widest text-slate-300 hover:bg-surface hover:text-slate-100"
    >
      <Icon className="h-3.5 w-3.5" />
      <span className="font-mono text-slate-500">{label}</span>
      <span className="font-mono">{value}</span>
    </button>
  );
}

export function PreferencesBar() {
  const prefs = usePreferences();
  return (
    <div className="flex items-center gap-1.5">
      <Pill
        label="theme"
        value={prefs.theme}
        icon={Palette}
        onClick={() =>
          setPreferences({
            theme: rotate<Theme>(THEME_OPTIONS, prefs.theme),
          })
        }
      />
      <Pill
        label="dens"
        value={prefs.density}
        icon={Rows3}
        onClick={() =>
          setPreferences({
            density: rotate<Density>(DENSITY_OPTIONS, prefs.density),
          })
        }
      />
      <Pill
        label="layout"
        value={prefs.layoutProfile}
        icon={LayoutGrid}
        onClick={() =>
          setPreferences({
            layoutProfile: rotate<LayoutProfile>(
              LAYOUT_OPTIONS,
              prefs.layoutProfile,
            ),
          })
        }
        hint="Conservative ↔ Standard ↔ Aggressive ↔ Custom"
      />
      <span
        className="hidden items-center gap-1 rounded border border-border bg-bg px-2 py-1 font-mono text-[10px] uppercase tracking-widest text-slate-500 lg:flex"
        title="Open command palette"
      >
        <Gauge className="h-3.5 w-3.5" />
        <span>⌘K</span>
      </span>
    </div>
  );
}
