interface HolderRow {
  rank: number;
  address: string;
  share_pct: number;
  is_dev: boolean;
  is_lp: boolean;
}

const MOCK: HolderRow[] = [
  { rank: 1, address: "9zXM…Lp4q", share_pct: 11.2, is_dev: false, is_lp: true },
  { rank: 2, address: "7BvE…Wn21", share_pct: 4.8, is_dev: true, is_lp: false },
  { rank: 3, address: "FpRQ…Ab98", share_pct: 3.6, is_dev: false, is_lp: false },
  { rank: 4, address: "2VkJ…H7ee", share_pct: 2.9, is_dev: false, is_lp: false },
  { rank: 5, address: "JhwT…q3b1", share_pct: 2.1, is_dev: false, is_lp: false },
  { rank: 6, address: "KpQ4…Mn03", share_pct: 1.8, is_dev: false, is_lp: false },
  { rank: 7, address: "DcN9…Vz12", share_pct: 1.5, is_dev: false, is_lp: false },
  { rank: 8, address: "TmKp…rA5h", share_pct: 1.4, is_dev: false, is_lp: false },
  { rank: 9, address: "8dRb…W8Tn", share_pct: 1.3, is_dev: false, is_lp: false },
  { rank: 10, address: "Fq2H…3JxC", share_pct: 1.2, is_dev: false, is_lp: false },
];

export function HolderConcentration() {
  const top10 = MOCK.slice(0, 10).reduce((a, h) => a + h.share_pct, 0);
  const devShare = MOCK.filter((h) => h.is_dev).reduce(
    (a, h) => a + h.share_pct,
    0,
  );
  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Holder Concentration
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            top-10 {top10.toFixed(1)}% · dev wallet {devShare.toFixed(2)}%
          </p>
        </div>
        <span className="rounded border border-accent/40 bg-accent/10 px-1.5 py-0.5 font-mono text-[10px] text-accent">
          live
        </span>
      </header>
      <div className="flex-1 overflow-auto">
        <table className="w-full text-[11px] font-mono">
          <thead className="text-[10px] uppercase tracking-wider text-slate-500">
            <tr>
              <th className="px-2 py-1 text-left">#</th>
              <th className="px-2 py-1 text-left">address</th>
              <th className="px-2 py-1 text-right">share</th>
              <th className="px-2 py-1 text-right">flags</th>
            </tr>
          </thead>
          <tbody>
            {MOCK.map((h) => (
              <tr key={h.rank} className="border-t border-border">
                <td className="px-2 py-0.5 text-slate-500">{h.rank}</td>
                <td className="px-2 py-0.5 text-slate-200">{h.address}</td>
                <td className="px-2 py-0.5 text-right text-slate-300">
                  {h.share_pct.toFixed(2)}%
                </td>
                <td className="px-2 py-0.5 text-right text-[10px]">
                  {h.is_dev && (
                    <span className="rounded border border-amber-500/40 bg-amber-500/10 px-1 py-0.5 text-amber-300">
                      DEV
                    </span>
                  )}
                  {h.is_lp && (
                    <span className="ml-1 rounded border border-accent/40 bg-accent/10 px-1 py-0.5 text-accent">
                      LP
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
