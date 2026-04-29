import type { PresenceStateApi } from "@/types/generated/api";

const TEXT: Record<PresenceStateApi, string> = {
  present: "present",
  partial: "partial",
  missing: "missing",
};

const STYLE: Record<PresenceStateApi, string> = {
  present: "bg-ok/15 text-ok border-ok/40",
  partial: "bg-warn/15 text-warn border-warn/40",
  missing: "bg-danger/15 text-danger border-danger/40",
};

export function StateBadge({ state }: { state: PresenceStateApi }) {
  return (
    <span
      className={`inline-block rounded border px-2 py-0.5 font-mono text-xs ${STYLE[state]}`}
    >
      {TEXT[state]}
    </span>
  );
}
