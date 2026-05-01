import { type ReactNode } from "react";

export interface PlaceholderWidgetProps {
  title: string;
  subtitle?: string;
  badge?: string;
  status?: "stub" | "wired" | "live";
  children?: ReactNode;
}

const STATUS_TONE: Record<NonNullable<PlaceholderWidgetProps["status"]>, string> = {
  stub: "bg-slate-700/40 border-slate-500/40 text-slate-300",
  wired: "bg-accent/10 border-accent/40 text-accent",
  live: "bg-emerald-500/10 border-emerald-500/40 text-emerald-400",
};

export function PlaceholderWidget({
  title,
  subtitle,
  badge,
  status = "stub",
  children,
}: PlaceholderWidgetProps) {
  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            {title}
          </h3>
          {subtitle && (
            <p className="mt-0.5 text-[11px] text-slate-500">{subtitle}</p>
          )}
        </div>
        <div className="flex items-center gap-1">
          {badge && (
            <span className="rounded border border-border bg-bg px-1.5 py-0.5 text-[10px] font-mono text-slate-500">
              {badge}
            </span>
          )}
          <span
            className={`rounded border px-1.5 py-0.5 text-[10px] font-mono uppercase ${STATUS_TONE[status]}`}
          >
            {status}
          </span>
        </div>
      </header>
      <div className="flex-1 overflow-auto p-3 text-sm text-slate-400">
        {children ?? (
          <div className="grid h-full place-items-center text-center text-xs text-slate-600">
            <div className="space-y-1">
              <div className="font-mono">{title}</div>
              <div>widget pending wiring</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
