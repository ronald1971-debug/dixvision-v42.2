/**
 * G-track widget — Funding-payment history.
 *
 * Cumulative funding paid/received on perp positions over time.
 * Backend hook: ``GET /api/positions/funding?since=...`` reads from
 * ``execution_engine.funding_ledger``.
 */
interface FundingTick {
  ts: string;
  symbol: string;
  rate_bps: number;
  notional_usd: number;
  pnl_usd: number;
  cumulative_usd: number;
}

const SEED: FundingTick[] = [
  { ts: "00:00", symbol: "BTC-USDT-PERP", rate_bps: 0.5, notional_usd: 80_000, pnl_usd: -4.0, cumulative_usd: -4.0 },
  { ts: "08:00", symbol: "BTC-USDT-PERP", rate_bps: 1.2, notional_usd: 80_000, pnl_usd: -9.6, cumulative_usd: -13.6 },
  { ts: "08:00", symbol: "ETH-USDT-PERP", rate_bps: 0.8, notional_usd: 60_000, pnl_usd: -4.8, cumulative_usd: -18.4 },
  { ts: "16:00", symbol: "BTC-USDT-PERP", rate_bps: -0.3, notional_usd: 80_000, pnl_usd: 2.4, cumulative_usd: -16.0 },
  { ts: "16:00", symbol: "ETH-USDT-PERP", rate_bps: 0.4, notional_usd: 60_000, pnl_usd: -2.4, cumulative_usd: -18.4 },
  { ts: "16:00", symbol: "SOL-USDT-PERP", rate_bps: 1.8, notional_usd: 40_000, pnl_usd: -7.2, cumulative_usd: -25.6 },
];

export function FundingHistory() {
  const last = SEED[SEED.length - 1].cumulative_usd;
  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Funding history
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            perp funding payments · 8h cycle
          </p>
        </div>
        <div className="font-mono text-[11px] text-slate-300">
          <span className="text-[10px] uppercase tracking-wider text-slate-500">
            cumulative
          </span>{" "}
          <span className={last < 0 ? "text-rose-400" : "text-emerald-400"}>
            {last.toFixed(2)} USD
          </span>
        </div>
      </header>
      <div className="flex-1 overflow-auto">
        <table className="w-full font-mono text-[11px] text-slate-300">
          <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
            <tr className="border-b border-border">
              <th className="px-3 py-1.5 text-left">time</th>
              <th className="px-3 py-1.5 text-left">symbol</th>
              <th className="px-3 py-1.5 text-right">rate</th>
              <th className="px-3 py-1.5 text-right">notional</th>
              <th className="px-3 py-1.5 text-right">pnl</th>
              <th className="px-3 py-1.5 text-right">cum.</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/40">
            {SEED.map((f, i) => (
              <tr key={i}>
                <td className="px-3 py-1 text-slate-500">{f.ts}</td>
                <td className="px-3 py-1 text-slate-200">{f.symbol}</td>
                <td
                  className={`px-3 py-1 text-right ${
                    f.rate_bps < 0 ? "text-emerald-400" : "text-rose-400"
                  }`}
                >
                  {f.rate_bps > 0 ? "+" : ""}
                  {f.rate_bps.toFixed(2)} bps
                </td>
                <td className="px-3 py-1 text-right">
                  {f.notional_usd.toLocaleString()}
                </td>
                <td
                  className={`px-3 py-1 text-right ${
                    f.pnl_usd < 0 ? "text-rose-400" : "text-emerald-400"
                  }`}
                >
                  {f.pnl_usd > 0 ? "+" : ""}
                  {f.pnl_usd.toFixed(2)}
                </td>
                <td
                  className={`px-3 py-1 text-right ${
                    f.cumulative_usd < 0
                      ? "text-rose-400"
                      : "text-emerald-400"
                  }`}
                >
                  {f.cumulative_usd.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
