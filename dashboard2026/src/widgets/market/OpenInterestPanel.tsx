/**
 * H-track widget — Open Interest panel.
 *
 * Per-venue OI ($M), 24h Δ, % of float. Backend hook:
 * ``GET /api/market/open-interest?symbol=...``.
 */
interface Row {
  venue: string;
  oi_usd_m: number;
  d24_pct: number;
  float_pct: number;
}

const SEED: Row[] = [
  { venue: "Binance", oi_usd_m: 8_410, d24_pct: 2.1, float_pct: 0.78 },
  { venue: "Bybit", oi_usd_m: 6_240, d24_pct: 4.7, float_pct: 0.58 },
  { venue: "OKX", oi_usd_m: 3_980, d24_pct: -1.4, float_pct: 0.37 },
  { venue: "dYdX", oi_usd_m: 1_120, d24_pct: 0.6, float_pct: 0.10 },
  { venue: "Hyperliquid", oi_usd_m: 2_840, d24_pct: 6.3, float_pct: 0.26 },
  { venue: "Deribit", oi_usd_m: 1_750, d24_pct: -0.8, float_pct: 0.16 },
];

export function OpenInterestPanel() {
  const total = SEED.reduce((s, r) => s + r.oi_usd_m, 0);
  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2 flex items-baseline justify-between">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Open interest
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            BTC-USDT perp · per-venue
          </p>
        </div>
        <div className="font-mono text-[11px] text-slate-300">
          Σ ${(total / 1000).toFixed(1)}B
        </div>
      </header>
      <div className="flex-1 overflow-auto">
        <table className="w-full font-mono text-[11px] text-slate-300">
          <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
            <tr className="border-b border-border">
              <th className="px-3 py-1.5 text-left">venue</th>
              <th className="px-3 py-1.5 text-right">OI $M</th>
              <th className="px-3 py-1.5 text-right">24h Δ</th>
              <th className="px-3 py-1.5">share</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/40">
            {SEED.map((r) => (
              <tr key={r.venue}>
                <td className="px-3 py-1 text-slate-200">{r.venue}</td>
                <td className="px-3 py-1 text-right">
                  {r.oi_usd_m.toLocaleString()}
                </td>
                <td
                  className={`px-3 py-1 text-right ${
                    r.d24_pct >= 0 ? "text-emerald-400" : "text-rose-400"
                  }`}
                >
                  {r.d24_pct >= 0 ? "+" : ""}
                  {r.d24_pct.toFixed(1)}%
                </td>
                <td className="px-3 py-1">
                  <div className="h-1.5 overflow-hidden rounded bg-bg/60">
                    <div
                      className="h-full bg-accent/60"
                      style={{ width: `${(r.oi_usd_m / total) * 100}%` }}
                    />
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
