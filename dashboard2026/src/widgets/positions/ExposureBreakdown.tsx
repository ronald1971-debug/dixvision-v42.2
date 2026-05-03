import { useState } from "react";

/**
 * G-track widget — Exposure breakdown.
 *
 * Pivots gross/net exposure by sector / venue / asset-class.
 * Backend hook: ``GET /api/positions/exposure?by=sector|venue|asset``
 * reads from ``portfolio_engine.exposure_pivot``.
 */
type Pivot = "sector" | "venue" | "asset";

interface Row {
  key: string;
  long_usd: number;
  short_usd: number;
}

const SEED: Record<Pivot, Row[]> = {
  sector: [
    { key: "L1", long_usd: 142_000, short_usd: 8_000 },
    { key: "DeFi", long_usd: 64_500, short_usd: 12_300 },
    { key: "AI", long_usd: 38_200, short_usd: 0 },
    { key: "Memecoin", long_usd: 21_500, short_usd: 6_000 },
    { key: "RWA", long_usd: 18_400, short_usd: 0 },
    { key: "FX", long_usd: 12_000, short_usd: 0 },
  ],
  venue: [
    { key: "binance", long_usd: 168_400, short_usd: 14_300 },
    { key: "kraken", long_usd: 54_200, short_usd: 0 },
    { key: "okx", long_usd: 38_900, short_usd: 6_000 },
    { key: "bybit", long_usd: 22_100, short_usd: 0 },
    { key: "uniswap-x", long_usd: 13_000, short_usd: 0 },
  ],
  asset: [
    { key: "BTC", long_usd: 84_300, short_usd: 0 },
    { key: "ETH", long_usd: 62_500, short_usd: 0 },
    { key: "SOL", long_usd: 41_200, short_usd: 6_000 },
    { key: "AVAX", long_usd: 22_400, short_usd: 0 },
    { key: "WIF", long_usd: 21_500, short_usd: 0 },
    { key: "MATIC", long_usd: 0, short_usd: 8_300 },
    { key: "USDT", long_usd: 64_700, short_usd: 0 },
  ],
};

export function ExposureBreakdown() {
  const [pivot, setPivot] = useState<Pivot>("sector");
  const rows = SEED[pivot];
  const grossLong = rows.reduce((acc, r) => acc + r.long_usd, 0);
  const grossShort = rows.reduce((acc, r) => acc + r.short_usd, 0);
  const max = Math.max(...rows.map((r) => Math.max(r.long_usd, r.short_usd)), 1);

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Exposure breakdown
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            gross long {grossLong.toLocaleString()} · gross short{" "}
            {grossShort.toLocaleString()} · net{" "}
            {(grossLong - grossShort).toLocaleString()}
          </p>
        </div>
        <div className="flex gap-1 font-mono text-[10px] uppercase tracking-wider">
          {(["sector", "venue", "asset"] as Pivot[]).map((p) => (
            <button
              key={p}
              type="button"
              onClick={() => setPivot(p)}
              className={`rounded border px-2 py-0.5 ${
                pivot === p
                  ? "border-accent/40 bg-accent/10 text-accent"
                  : "border-border bg-bg/40 text-slate-400 hover:border-accent hover:text-accent"
              }`}
            >
              {p}
            </button>
          ))}
        </div>
      </header>
      <div className="flex-1 overflow-auto">
        <table className="w-full font-mono text-[11px] text-slate-300">
          <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
            <tr className="border-b border-border">
              <th className="px-3 py-1.5 text-left">{pivot}</th>
              <th className="px-3 py-1.5">long</th>
              <th className="px-3 py-1.5"></th>
              <th className="px-3 py-1.5">short</th>
              <th className="px-3 py-1.5 text-right">net</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/40">
            {rows.map((r) => {
              const longW = (r.long_usd / max) * 100;
              const shortW = (r.short_usd / max) * 100;
              return (
                <tr key={r.key}>
                  <td className="px-3 py-1 text-slate-200">{r.key}</td>
                  <td className="px-3 py-1 text-right text-emerald-400">
                    {r.long_usd > 0 ? r.long_usd.toLocaleString() : "—"}
                  </td>
                  <td className="px-3 py-1">
                    <div className="flex h-2 items-center">
                      <div className="flex w-1/2 justify-end pr-px">
                        <div
                          className="h-2 rounded-l bg-rose-400/50"
                          style={{ width: `${shortW}%` }}
                        />
                      </div>
                      <div className="flex w-1/2 justify-start pl-px">
                        <div
                          className="h-2 rounded-r bg-emerald-400/60"
                          style={{ width: `${longW}%` }}
                        />
                      </div>
                    </div>
                  </td>
                  <td className="px-3 py-1 text-right text-rose-400">
                    {r.short_usd > 0 ? r.short_usd.toLocaleString() : "—"}
                  </td>
                  <td
                    className={`px-3 py-1 text-right ${
                      r.long_usd - r.short_usd > 0
                        ? "text-emerald-400"
                        : r.long_usd - r.short_usd < 0
                          ? "text-rose-400"
                          : "text-slate-400"
                    }`}
                  >
                    {(r.long_usd - r.short_usd).toLocaleString()}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
