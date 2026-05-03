/**
 * H-track widget — Long/Short ratio across top venues.
 *
 * Backend hook: ``GET /api/market/long-short?symbol=BTC-USDT``
 * pulls per-venue long/short account split (Binance, Bybit, OKX,
 * Bitget). Scrunches to a global percentage per side.
 */
interface Row {
  venue: string;
  long_pct: number;
  short_pct: number;
  accounts: number;
}

const SEED: Row[] = [
  { venue: "Binance", long_pct: 58.4, short_pct: 41.6, accounts: 142_310 },
  { venue: "Bybit", long_pct: 53.2, short_pct: 46.8, accounts: 87_840 },
  { venue: "OKX", long_pct: 61.0, short_pct: 39.0, accounts: 65_120 },
  { venue: "Bitget", long_pct: 49.7, short_pct: 50.3, accounts: 28_760 },
];

export function LongShortRatio() {
  const totalAccounts = SEED.reduce((s, r) => s + r.accounts, 0);
  const aggLong =
    SEED.reduce((s, r) => s + r.long_pct * r.accounts, 0) / totalAccounts;
  const aggShort = 100 - aggLong;

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Long / Short ratio
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          BTC-USDT · accounts-weighted across 4 venues
        </p>
      </header>
      <div className="flex flex-1 flex-col gap-3 px-3 py-3">
        <div>
          <div className="flex items-baseline justify-between font-mono text-[10px] uppercase tracking-wider">
            <span className="text-emerald-400">long {aggLong.toFixed(1)}%</span>
            <span className="text-rose-400">{aggShort.toFixed(1)}% short</span>
          </div>
          <div className="mt-1 flex h-3 overflow-hidden rounded">
            <div
              className="bg-emerald-500/70"
              style={{ width: `${aggLong}%` }}
            />
            <div
              className="bg-rose-500/70"
              style={{ width: `${aggShort}%` }}
            />
          </div>
        </div>
        <div className="flex-1 overflow-auto">
          <table className="w-full font-mono text-[11px] text-slate-300">
            <thead className="text-[10px] uppercase tracking-wider text-slate-500">
              <tr className="border-b border-border">
                <th className="px-1 py-1.5 text-left">venue</th>
                <th className="px-1 py-1.5 text-right">long</th>
                <th className="px-1 py-1.5 text-right">short</th>
                <th className="px-1 py-1.5 text-right">accounts</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/40">
              {SEED.map((r) => (
                <tr key={r.venue}>
                  <td className="px-1 py-1 text-slate-200">{r.venue}</td>
                  <td className="px-1 py-1 text-right text-emerald-400">
                    {r.long_pct.toFixed(1)}%
                  </td>
                  <td className="px-1 py-1 text-right text-rose-400">
                    {r.short_pct.toFixed(1)}%
                  </td>
                  <td className="px-1 py-1 text-right text-slate-400">
                    {(r.accounts / 1000).toFixed(1)}k
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
