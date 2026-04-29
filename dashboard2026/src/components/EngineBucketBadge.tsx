const STYLE: Record<string, string> = {
  alive: "bg-ok/15 text-ok border-ok/40",
  degraded: "bg-warn/15 text-warn border-warn/40",
  halted: "bg-danger/15 text-danger border-danger/40",
  offline: "bg-slate-700/30 text-slate-400 border-slate-700",
};

export function EngineBucketBadge({ bucket }: { bucket: string }) {
  const cls = STYLE[bucket] ?? STYLE.offline;
  return (
    <span
      className={`inline-block rounded border px-2 py-0.5 font-mono text-xs ${cls}`}
    >
      {bucket}
    </span>
  );
}
