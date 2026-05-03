import { useEffect, useState } from "react";

/**
 * Tier-4 memecoin widget — Bundle detector.
 *
 * Detects coordinated buys at launch (the "bundle"): N wallets
 * funded by the same source within seconds of mint, all buying
 * within the same block — the canonical insider-pre-buy pattern
 * that ends with the bundle dumping into retail flow once volume
 * arrives.
 *
 * Each row surfaces:
 *   - n_wallets        : count in the bundle
 *   - cluster_share    : combined supply share at end of block 1
 *   - funding_source   : the wallet that funded the cluster
 *   - first_dump_block : block where any cluster wallet first sold
 *
 * `cluster_share >= 0.20` triggers `HAZ-BUNDLE`. SLTP engine
 * applies the dev-dump style stop loss until cluster share decays
 * below 0.10 (release). Operator can override but it's logged.
 */
interface Bundle {
  ticker: string;
  n_wallets: number;
  cluster_share: number;
  funding_source: string;
  first_dump_block: number | null;
  decaying: boolean;
}

const SEED: Bundle[] = [
  {
    ticker: "WIFCAT",
    n_wallets: 4,
    cluster_share: 0.07,
    funding_source: "5d4e…0a91",
    first_dump_block: null,
    decaying: false,
  },
  {
    ticker: "BONKER",
    n_wallets: 23,
    cluster_share: 0.41,
    funding_source: "9b1c…ae42",
    first_dump_block: 18,
    decaying: false,
  },
  {
    ticker: "TURBOX",
    n_wallets: 11,
    cluster_share: 0.18,
    funding_source: "2f88…6610",
    first_dump_block: null,
    decaying: false,
  },
  {
    ticker: "MOONOPUS",
    n_wallets: 31,
    cluster_share: 0.62,
    funding_source: "ee70…1124",
    first_dump_block: 4,
    decaying: false,
  },
  {
    ticker: "GIGAFROG",
    n_wallets: 6,
    cluster_share: 0.11,
    funding_source: "7c33…aa01",
    first_dump_block: null,
    decaying: true,
  },
];

export function BundleDetector() {
  const [rows, setRows] = useState<Bundle[]>(SEED);

  useEffect(() => {
    const id = setInterval(() => {
      setRows((prev) =>
        prev.map((b) => {
          const drift = b.decaying ? -0.01 : Math.sin(Date.now() / 5_000) * 0.005;
          return {
            ...b,
            cluster_share: Math.max(0, b.cluster_share + drift),
          };
        }),
      );
    }, 4_000);
    return () => clearInterval(id);
  }, []);

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Bundle detector
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          coordinated launch buys · cluster share ≥ 20% trips HAZ-BUNDLE
        </p>
      </header>
      <div className="flex-1 overflow-auto">
      <table className="w-full text-[11px]">
        <thead className="sticky top-0 bg-surface text-[10px] uppercase tracking-wider text-slate-500">
          <tr className="text-left">
            <th className="px-3 py-1">ticker</th>
            <th className="px-2 py-1 text-right">wallets</th>
            <th className="px-2 py-1 text-right">share</th>
            <th className="px-2 py-1">funder</th>
            <th className="px-3 py-1 text-right">first dump</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border/40 font-mono">
          {rows.map((b) => {
            const tripped = b.cluster_share >= 0.20;
            return (
              <tr key={b.ticker}>
                <td className="px-3 py-1 text-slate-200">
                  {b.ticker}
                  {tripped && (
                    <span className="ml-2 rounded bg-rose-500/20 px-1 py-0.5 text-[9px] uppercase text-rose-300">
                      HAZ-BUNDLE
                    </span>
                  )}
                </td>
                <td className="px-2 py-1 text-right text-slate-300">
                  {b.n_wallets}
                </td>
                <td className="px-2 py-1 text-right">
                  <span
                    className={
                      tripped ? "text-rose-300" : "text-slate-300"
                    }
                  >
                    {(b.cluster_share * 100).toFixed(1)}%
                  </span>
                </td>
                <td className="px-2 py-1 text-slate-500">
                  {b.funding_source}
                </td>
                <td className="px-3 py-1 text-right text-slate-400">
                  {b.first_dump_block !== null ? `blk +${b.first_dump_block}` : "—"}
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
