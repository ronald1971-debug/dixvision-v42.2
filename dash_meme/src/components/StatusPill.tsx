import type { ReactNode } from "react";

type Tone = "ok" | "warn" | "danger" | "info" | "neutral";

const TONE_CLASS: Record<Tone, string> = {
  ok: "bg-[var(--ok-soft)] text-ok",
  warn: "bg-[var(--warn-soft)] text-warn",
  danger: "bg-[var(--danger-soft)] text-danger",
  info: "bg-[var(--accent-soft)] text-accent",
  neutral: "bg-surface-raised text-text-secondary",
};

export function StatusPill({
  tone = "neutral",
  children,
  title,
}: {
  tone?: Tone;
  children: ReactNode;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs font-medium uppercase tracking-wide ${TONE_CLASS[tone]}`}
    >
      {children}
    </span>
  );
}
