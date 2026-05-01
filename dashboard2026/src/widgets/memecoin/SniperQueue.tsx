/**
 * Sniper Queue (PR-#2 spec §3.4.3).
 *
 * Pre-signed bundle queue for Pump.fun launches + Raydium pool
 * creation + Uniswap V2/V3 PairCreated events. Mandatory filters
 * (LP size / liquidity locked / mint+freeze revoked / honeypot-safe
 * / dev-wallet share / social presence) are evaluated before a slot
 * goes hot.
 */
interface QueueRow {
  pair: string;
  status: "queued" | "armed" | "fired" | "rejected";
  filter_pass: number;
  filter_total: number;
  bundle_state: "pending" | "submitted" | "landed" | "dropped";
  age_s: number;
}

const MOCK: QueueRow[] = [
  {
    pair: "WIFCAT / SOL",
    status: "armed",
    filter_pass: 7,
    filter_total: 7,
    bundle_state: "submitted",
    age_s: 3,
  },
  {
    pair: "GIGABASED / SOL",
    status: "queued",
    filter_pass: 5,
    filter_total: 7,
    bundle_state: "pending",
    age_s: 9,
  },
  {
    pair: "$DOGAI / WETH",
    status: "fired",
    filter_pass: 7,
    filter_total: 7,
    bundle_state: "landed",
    age_s: 41,
  },
  {
    pair: "TROLLBOX / SOL",
    status: "rejected",
    filter_pass: 3,
    filter_total: 7,
    bundle_state: "dropped",
    age_s: 62,
  },
];

export function SniperQueue() {
  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Sniper Queue
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            Jito · Flashbots bundle · in-bundle SL/TP
          </p>
        </div>
        <span className="rounded border border-emerald-500/40 bg-emerald-500/10 px-1.5 py-0.5 font-mono text-[10px] text-emerald-300">
          armed
        </span>
      </header>
      <div className="flex-1 overflow-auto">
        <table className="w-full text-[11px] font-mono">
          <thead className="text-[10px] uppercase tracking-wider text-slate-500">
            <tr>
              <th className="px-2 py-1 text-left">pair</th>
              <th className="px-2 py-1 text-left">status</th>
              <th className="px-2 py-1 text-left">filters</th>
              <th className="px-2 py-1 text-left">bundle</th>
              <th className="px-2 py-1 text-right">age</th>
            </tr>
          </thead>
          <tbody>
            {MOCK.map((r, i) => (
              <tr key={i} className="border-t border-border">
                <td className="px-2 py-0.5 text-slate-200">{r.pair}</td>
                <td className="px-2 py-0.5">
                  <StatusChip status={r.status} />
                </td>
                <td className="px-2 py-0.5">
                  <span
                    className={
                      r.filter_pass === r.filter_total
                        ? "text-emerald-300"
                        : "text-amber-300"
                    }
                  >
                    {r.filter_pass}/{r.filter_total}
                  </span>
                </td>
                <td className="px-2 py-0.5 text-slate-300">
                  {r.bundle_state}
                </td>
                <td className="px-2 py-0.5 text-right text-slate-400">
                  {r.age_s}s
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function StatusChip({ status }: { status: QueueRow["status"] }) {
  const tone =
    status === "armed"
      ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
      : status === "queued"
        ? "border-accent/40 bg-accent/10 text-accent"
        : status === "fired"
          ? "border-amber-500/40 bg-amber-500/10 text-amber-300"
          : "border-red-500/40 bg-red-500/10 text-red-300";
  return (
    <span
      className={`rounded border px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider ${tone}`}
    >
      {status}
    </span>
  );
}
