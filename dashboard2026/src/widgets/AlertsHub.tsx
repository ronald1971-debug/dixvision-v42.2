interface Alert {
  id: string;
  ts_iso: string;
  severity: "info" | "warn" | "danger";
  text: string;
}

const MOCK: Alert[] = [
  {
    id: "a1",
    ts_iso: new Date().toISOString(),
    severity: "warn",
    text: "Helius p99 latency 510 ms > 350 ms — strategy `momentum_v3` throttled.",
  },
  {
    id: "a2",
    ts_iso: new Date(Date.now() - 60_000).toISOString(),
    severity: "info",
    text: "CoinDesk article `BTC ETF inflows hit YTD high` — sentiment +0.71.",
  },
  {
    id: "a3",
    ts_iso: new Date(Date.now() - 180_000).toISOString(),
    severity: "danger",
    text: "Rug-score for $TROLLBOX dropped to 38 — rug-trip SL armed.",
  },
];

export function AlertsHub() {
  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Alerts Hub
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            ledger-derived · per-widget sound rules pending
          </p>
        </div>
        <span className="rounded border border-amber-500/40 bg-amber-500/10 px-1.5 py-0.5 font-mono text-[10px] text-amber-300">
          {MOCK.length} active
        </span>
      </header>
      <ul className="flex-1 divide-y divide-border overflow-auto">
        {MOCK.map((a) => (
          <li
            key={a.id}
            className="flex items-baseline gap-2 px-3 py-1.5 text-[12px]"
          >
            <span
              className={`mt-1 h-2 w-2 shrink-0 rounded-full ${
                a.severity === "danger"
                  ? "bg-red-400"
                  : a.severity === "warn"
                    ? "bg-amber-300"
                    : "bg-accent"
              }`}
            />
            <span className="font-mono text-[10px] uppercase tracking-wider text-slate-500">
              {new Date(a.ts_iso).toLocaleTimeString()}
            </span>
            <span className="flex-1 text-slate-200">{a.text}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
