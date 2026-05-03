import { useEffect, useState } from "react";

/**
 * Tier-5 on-chain widget — Stablecoin supply.
 *
 * Total circulating supply of the four dominant stablecoins.
 * Direction matters more than level: net minting (positive
 * Δ24h) is canonically pro-risk because USDT/USDC growth has
 * historically front-run risk-on rallies. Net redemption is
 * canonically risk-off.
 *
 * Real wiring through the macro adapter (filed); mock drifts
 * via the same sine pattern as ExchangeFlows so it stays alive.
 */
interface Stable {
  ticker: string;
  name: string;
  supply_usd: number;
  delta_24h_usd: number;
  delta_7d_pct: number;
}

const SEED: Stable[] = [
  {
    ticker: "USDT",
    name: "Tether",
    supply_usd: 188_400_000_000,
    delta_24h_usd: 412_000_000,
    delta_7d_pct: 0.0094,
  },
  {
    ticker: "USDC",
    name: "Circle",
    supply_usd: 79_200_000_000,
    delta_24h_usd: 88_000_000,
    delta_7d_pct: 0.0041,
  },
  {
    ticker: "DAI",
    name: "MakerDAO",
    supply_usd: 4_800_000_000,
    delta_24h_usd: -12_000_000,
    delta_7d_pct: -0.0018,
  },
  {
    ticker: "FDUSD",
    name: "First Digital",
    supply_usd: 2_400_000_000,
    delta_24h_usd: 31_000_000,
    delta_7d_pct: 0.0212,
  },
];

export function StablecoinSupply() {
  const [rows, setRows] = useState<Stable[]>(SEED);

  useEffect(() => {
    const id = setInterval(() => {
      setRows((prev) =>
        prev.map((s) => {
          const drift = Math.sin(Date.now() / 7_000 + s.ticker.length) * 0.003;
          return {
            ...s,
            delta_24h_usd: s.delta_24h_usd + drift * 50_000_000,
            delta_7d_pct: s.delta_7d_pct + drift * 0.001,
          };
        }),
      );
    }, 5_000);
    return () => clearInterval(id);
  }, []);

  const totalSupply = rows.reduce((s, r) => s + r.supply_usd, 0);
  const total24h = rows.reduce((s, r) => s + r.delta_24h_usd, 0);
  const direction = total24h >= 0 ? "minting" : "redeeming";

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <div className="flex items-baseline justify-between">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Stablecoin supply
          </h3>
          <span
            className={`rounded border px-1.5 py-0.5 text-[10px] uppercase tracking-wider ${
              total24h >= 0
                ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
                : "border-rose-500/40 bg-rose-500/10 text-rose-300"
            }`}
          >
            {direction}
          </span>
        </div>
        <p className="mt-0.5 text-[11px] text-slate-500">
          aggregate ${(totalSupply / 1_000_000_000).toFixed(0)}B · 24h{" "}
          {total24h >= 0 ? "+" : ""}
          {(total24h / 1_000_000).toFixed(0)}M
        </p>
      </header>
      <table className="w-full flex-1 overflow-auto text-[11px]">
        <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
          <tr className="text-left">
            <th className="px-3 py-1">ticker</th>
            <th className="px-2 py-1 text-right">supply</th>
            <th className="px-2 py-1 text-right">24h</th>
            <th className="px-3 py-1 text-right">7d</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border/40 font-mono">
          {rows.map((s) => (
            <tr key={s.ticker}>
              <td className="px-3 py-1">
                <span className="font-semibold text-slate-200">
                  {s.ticker}
                </span>
                <span className="ml-2 text-[10px] text-slate-500">
                  {s.name}
                </span>
              </td>
              <td className="px-2 py-1 text-right text-slate-300">
                ${(s.supply_usd / 1_000_000_000).toFixed(2)}B
              </td>
              <td
                className={`px-2 py-1 text-right ${s.delta_24h_usd >= 0 ? "text-emerald-300" : "text-rose-300"}`}
              >
                {s.delta_24h_usd >= 0 ? "+" : ""}
                {(s.delta_24h_usd / 1_000_000).toFixed(0)}M
              </td>
              <td
                className={`px-3 py-1 text-right ${s.delta_7d_pct >= 0 ? "text-emerald-300" : "text-rose-300"}`}
              >
                {(s.delta_7d_pct * 100).toFixed(2)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
