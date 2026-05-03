import { useEffect, useState } from "react";

/**
 * Tier-5 on-chain widget — Exchange net-flows.
 *
 * Net inflow (deposits – withdrawals) per major CEX, in USD.
 * Negative = withdrawals winning = supply leaving exchanges
 * (canonically bullish for spot scarcity). Positive = inflows
 * winning = supply moving toward exchanges (canonically bearish
 * because deposits precede sells).
 *
 * Real wiring goes through the on-chain adapter (Glassnode /
 * CryptoQuant). Mock here drifts deterministically via a sine
 * keyed on (Date.now() / 6_000) so repeated renders feel alive.
 */
interface ExchangeRow {
  name: string;
  asset: string;
  net_24h_usd: number;
  in_24h_usd: number;
  out_24h_usd: number;
  reserves_usd: number;
  reserves_dod: number; // day-over-day % change in reserves
}

const SEED: ExchangeRow[] = [
  {
    name: "Binance",
    asset: "BTC",
    net_24h_usd: -82_400_000,
    in_24h_usd: 410_000_000,
    out_24h_usd: 492_400_000,
    reserves_usd: 36_220_000_000,
    reserves_dod: -0.0024,
  },
  {
    name: "Coinbase",
    asset: "BTC",
    net_24h_usd: -41_300_000,
    in_24h_usd: 188_700_000,
    out_24h_usd: 230_000_000,
    reserves_usd: 18_400_000_000,
    reserves_dod: -0.0022,
  },
  {
    name: "Kraken",
    asset: "ETH",
    net_24h_usd: -12_900_000,
    in_24h_usd: 81_000_000,
    out_24h_usd: 93_900_000,
    reserves_usd: 4_800_000_000,
    reserves_dod: -0.0027,
  },
  {
    name: "Bybit",
    asset: "BTC",
    net_24h_usd: 18_700_000,
    in_24h_usd: 142_000_000,
    out_24h_usd: 123_300_000,
    reserves_usd: 6_900_000_000,
    reserves_dod: 0.0027,
  },
  {
    name: "OKX",
    asset: "ETH",
    net_24h_usd: -5_600_000,
    in_24h_usd: 64_400_000,
    out_24h_usd: 70_000_000,
    reserves_usd: 5_300_000_000,
    reserves_dod: -0.0011,
  },
];

export function ExchangeFlows() {
  const [rows, setRows] = useState<ExchangeRow[]>(SEED);

  useEffect(() => {
    const id = setInterval(() => {
      setRows((prev) =>
        prev.map((r) => {
          const drift =
            Math.sin(Date.now() / 6_000 + r.name.length) * 0.005 - 0.0025;
          return {
            ...r,
            net_24h_usd: r.net_24h_usd + drift * 25_000_000,
            reserves_dod: r.reserves_dod + drift * 0.0002,
          };
        }),
      );
    }, 4_500);
    return () => clearInterval(id);
  }, []);

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Exchange net-flows
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          24h net inflow vs reserves · negative = supply leaving exchange
        </p>
      </header>
      <table className="w-full flex-1 overflow-auto text-[11px]">
        <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
          <tr className="text-left">
            <th className="px-3 py-1">venue</th>
            <th className="px-2 py-1">asset</th>
            <th className="px-2 py-1 text-right">net 24h</th>
            <th className="px-2 py-1 text-right">reserves</th>
            <th className="px-3 py-1 text-right">Δ d/d</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border/40 font-mono">
          {rows.map((r) => (
            <tr key={r.name + r.asset}>
              <td className="px-3 py-1 text-slate-200">{r.name}</td>
              <td className="px-2 py-1 text-slate-400">{r.asset}</td>
              <td
                className={`px-2 py-1 text-right ${r.net_24h_usd >= 0 ? "text-rose-300" : "text-emerald-300"}`}
              >
                {r.net_24h_usd >= 0 ? "+" : ""}
                {(r.net_24h_usd / 1_000_000).toFixed(1)}M
              </td>
              <td className="px-2 py-1 text-right text-slate-300">
                ${(r.reserves_usd / 1_000_000_000).toFixed(2)}B
              </td>
              <td
                className={`px-3 py-1 text-right ${r.reserves_dod >= 0 ? "text-rose-300" : "text-emerald-300"}`}
              >
                {(r.reserves_dod * 100).toFixed(2)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
