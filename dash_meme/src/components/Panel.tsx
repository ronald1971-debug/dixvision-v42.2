import type { ReactNode } from "react";

export function Panel({
  title,
  right,
  children,
  className = "",
  bodyClassName = "",
}: {
  title: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section
      className={`flex flex-col overflow-hidden rounded border border-border bg-surface ${className}`}
    >
      <header className="flex h-8 shrink-0 items-center justify-between border-b border-hairline px-3 text-xs uppercase tracking-wide text-text-secondary">
        <span className="font-semibold text-text-primary">{title}</span>
        {right}
      </header>
      <div className={`flex-1 overflow-auto dex-scroll ${bodyClassName}`}>
        {children}
      </div>
    </section>
  );
}
