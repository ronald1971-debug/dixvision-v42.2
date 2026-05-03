import { useMemo, useState } from "react";

interface LadderRow {
  bidLevel: string;
  pricePct: number; // % of floor
  bidsAhead: number;
  totalBids: number;
  estTimeToFill: string;
}

const FLOOR = 7.40;

const ROWS: LadderRow[] = [
  { bidLevel: "floor",        pricePct: 1.00, bidsAhead: 0,   totalBids: 12,  estTimeToFill: "instant" },
  { bidLevel: "floor -1%",    pricePct: 0.99, bidsAhead: 12,  totalBids: 38,  estTimeToFill: "~2m" },
  { bidLevel: "floor -2%",    pricePct: 0.98, bidsAhead: 50,  totalBids: 91,  estTimeToFill: "~10m" },
  { bidLevel: "floor -5%",    pricePct: 0.95, bidsAhead: 141, totalBids: 257, estTimeToFill: "~1h" },
  { bidLevel: "floor -10%",   pricePct: 0.90, bidsAhead: 398, totalBids: 612, estTimeToFill: "~6h" },
  { bidLevel: "floor -15%",   pricePct: 0.85, bidsAhead: 1010,totalBids: 1380,estTimeToFill: "~24h" },
];

export function BidLadder({ collection = "Pudgy" }: { collection?: string }) {
  const [selected, setSelected] = useState<string>("floor -2%");
  const [size, setSize] = useState<number>(1);

  const focus = useMemo(
    () => ROWS.find((r) => r.bidLevel === selected) ?? ROWS[0],
    [selected]
  );
  const bidPrice = FLOOR * focus.pricePct;
  const lockUp = bidPrice * size;

  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Collection-Bid Ladder · {collection}
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            Blur-style pool bid · floor Ξ {FLOOR.toFixed(2)}
          </p>
        </div>
        <span className="rounded border border-sky-500/40 bg-sky-500/10 px-1.5 py-0.5 font-mono text-[11px] text-sky-300">
          {selected}
        </span>
      </header>
      <div className="flex-1 overflow-auto">
        <table className="w-full text-[11px] font-mono">
          <thead className="text-[10px] uppercase tracking-wider text-slate-500">
            <tr>
              <th className="px-2 py-1 text-left">level</th>
              <th className="px-2 py-1 text-right">price (Ξ)</th>
              <th className="px-2 py-1 text-right">ahead</th>
              <th className="px-2 py-1 text-right">total</th>
              <th className="px-2 py-1 text-right">est fill</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map((r) => {
              const active = r.bidLevel === selected;
              return (
                <tr
                  key={r.bidLevel}
                  onClick={() => setSelected(r.bidLevel)}
                  className={`cursor-pointer border-t border-border ${
                    active ? "bg-sky-500/10" : "hover:bg-slate-800/40"
                  }`}
                >
                  <td className={`px-2 py-1 ${active ? "text-sky-300" : "text-slate-400"}`}>
                    {r.bidLevel}
                  </td>
                  <td className="px-2 py-1 text-right text-slate-100">
                    {(FLOOR * r.pricePct).toFixed(3)}
                  </td>
                  <td className="px-2 py-1 text-right text-slate-300">{r.bidsAhead}</td>
                  <td className="px-2 py-1 text-right text-slate-500">{r.totalBids}</td>
                  <td className="px-2 py-1 text-right text-slate-400">{r.estTimeToFill}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="border-t border-border px-3 py-2 font-mono text-[11px]">
        <label className="block">
          <div className="flex items-center justify-between">
            <span className="text-[10px] uppercase tracking-wider text-slate-500">
              size (NFTs)
            </span>
            <span className="text-slate-300">{size}</span>
          </div>
          <input
            type="range"
            min={1}
            max={20}
            value={size}
            onChange={(e) => setSize(Number(e.target.value))}
            className="mt-0.5 w-full accent-sky-500"
          />
        </label>
        <div className="mt-1 flex items-baseline justify-between">
          <span className="text-[10px] uppercase tracking-wider text-slate-500">
            lock-up
          </span>
          <span className="text-emerald-300">Ξ {lockUp.toFixed(3)}</span>
        </div>
        <button
          type="button"
          className="mt-2 w-full rounded border border-sky-500/40 bg-sky-500/10 px-2 py-1 text-[11px] text-sky-200"
        >
          place bid {selected}
        </button>
      </div>
    </div>
  );
}
