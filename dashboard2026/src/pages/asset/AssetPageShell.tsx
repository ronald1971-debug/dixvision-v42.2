import { type ReactNode } from "react";

export interface AssetPageShellProps {
  title: string;
  asset: string;
  description: string;
  children: ReactNode;
}

/**
 * Common chrome around a per-asset dashboard page: title, asset
 * label, short description of what the surface is for, and the
 * `react-grid-layout` body.
 *
 * The actual widget grid is owned by the page (so each asset class
 * can ship a different default layout), but the heading + identity
 * banner is shared so the cockpit feels consistent across surfaces.
 */
export function AssetPageShell({
  title,
  asset,
  description,
  children,
}: AssetPageShellProps) {
  return (
    <section className="flex h-full flex-col">
      <header className="mb-3 flex items-baseline justify-between">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">
            {title}{" "}
            <span className="ml-2 rounded border border-border bg-bg px-2 py-0.5 font-mono text-[11px] uppercase tracking-widest text-slate-400">
              {asset}
            </span>
          </h1>
          <p className="mt-1 text-xs text-slate-400">{description}</p>
        </div>
      </header>
      <div className="flex-1 overflow-auto pb-6">{children}</div>
    </section>
  );
}
