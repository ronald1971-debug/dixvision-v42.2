import { useMemo, useState } from "react";

interface Item {
  tokenId: number;
  rarityPct: number;
  price: number; // ETH
  marketplace: "Blur" | "OpenSea" | "Magic Eden" | "Tensor";
}

const POOL: Item[] = [
  { tokenId: 5631, rarityPct: 0.28, price: 32.5, marketplace: "Blur" },
  { tokenId: 8121, rarityPct: 0.45, price: 22.8, marketplace: "OpenSea" },
  { tokenId: 1283, rarityPct: 0.67, price: 26.4, marketplace: "Blur" },
  { tokenId: 9711, rarityPct: 1.42, price: 18.2, marketplace: "Tensor" },
  { tokenId: 3221, rarityPct: 3.12, price: 12.4, marketplace: "Blur" },
  { tokenId: 4012, rarityPct: 9.8,  price: 7.62, marketplace: "OpenSea" },
  { tokenId: 6611, rarityPct: 14.8, price: 7.40, marketplace: "Magic Eden" },
  { tokenId: 1009, rarityPct: 18.9, price: 7.55, marketplace: "Blur" },
  { tokenId: 2451, rarityPct: 22.1, price: 7.42, marketplace: "OpenSea" },
  { tokenId: 7188, rarityPct: 27.4, price: 7.38, marketplace: "Blur" },
];

export function SweepCart({ collection = "Pudgy" }: { collection?: string }) {
  const [maxRarity, setMaxRarity] = useState<number>(5);
  const [count, setCount] = useState<number>(5);

  const candidates = useMemo(() => {
    return [...POOL]
      .filter((i) => i.rarityPct <= maxRarity)
      .sort((a, b) => a.price - b.price)
      .slice(0, count);
  }, [maxRarity, count]);

  const subtotal = candidates.reduce((a, i) => a + i.price, 0);
  const fees = subtotal * 0.005; // 0.5% aggregator
  const total = subtotal + fees;

  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Sweep Cart · {collection}
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            multi-marketplace · trait-rarity filtered · cheapest-first
          </p>
        </div>
        <span className="rounded border border-amber-500/40 bg-amber-500/10 px-1.5 py-0.5 font-mono text-[11px] text-amber-300">
          Ξ {total.toFixed(3)}
        </span>
      </header>
      <div className="space-y-2 border-b border-border px-3 py-2 text-[11px]">
        <label className="block">
          <div className="flex items-center justify-between">
            <span className="text-[10px] uppercase tracking-wider text-slate-500">
              max rarity %
            </span>
            <span className="font-mono text-slate-300">{maxRarity.toFixed(1)}</span>
          </div>
          <input
            type="range"
            min={0.1}
            max={30}
            step={0.1}
            value={maxRarity}
            onChange={(e) => setMaxRarity(Number(e.target.value))}
            className="mt-0.5 w-full accent-emerald-500"
          />
        </label>
        <label className="block">
          <div className="flex items-center justify-between">
            <span className="text-[10px] uppercase tracking-wider text-slate-500">count</span>
            <span className="font-mono text-slate-300">{count}</span>
          </div>
          <input
            type="range"
            min={1}
            max={POOL.length}
            value={count}
            onChange={(e) => setCount(Number(e.target.value))}
            className="mt-0.5 w-full accent-emerald-500"
          />
        </label>
      </div>
      <div className="flex-1 overflow-auto">
        <table className="w-full text-[11px] font-mono">
          <thead className="text-[10px] uppercase tracking-wider text-slate-500">
            <tr>
              <th className="px-2 py-1 text-left">#</th>
              <th className="px-2 py-1 text-left">market</th>
              <th className="px-2 py-1 text-right">rarity</th>
              <th className="px-2 py-1 text-right">price</th>
            </tr>
          </thead>
          <tbody>
            {candidates.map((i) => (
              <tr key={i.tokenId} className="border-t border-border">
                <td className="px-2 py-1 text-slate-200">#{i.tokenId}</td>
                <td className="px-2 py-1 text-slate-400">{i.marketplace}</td>
                <td className="px-2 py-1 text-right text-slate-300">
                  {i.rarityPct.toFixed(2)}%
                </td>
                <td className="px-2 py-1 text-right text-slate-100">
                  {i.price.toFixed(2)}
                </td>
              </tr>
            ))}
            {candidates.length === 0 && (
              <tr>
                <td colSpan={4} className="px-2 py-3 text-center text-slate-500">
                  no items match filters
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="border-t border-border px-3 py-2 font-mono text-[11px]">
        <Row label="subtotal" v={`Ξ ${subtotal.toFixed(3)}`} />
        <Row label="aggregator fee (0.5%)" v={`Ξ ${fees.toFixed(3)}`} />
        <Row label="total" v={`Ξ ${total.toFixed(3)}`} tone="emerald" />
        <button
          type="button"
          disabled={candidates.length === 0}
          className="mt-2 w-full rounded border border-emerald-500/40 bg-emerald-500/10 px-2 py-1 text-[11px] text-emerald-200 disabled:opacity-40"
        >
          sweep {candidates.length} items
        </button>
      </div>
    </div>
  );
}

function Row({ label, v, tone }: { label: string; v: string; tone?: string }) {
  return (
    <div className="flex items-baseline justify-between">
      <span className="text-[10px] uppercase tracking-wider text-slate-500">{label}</span>
      <span className={tone === "emerald" ? "text-emerald-300" : "text-slate-200"}>{v}</span>
    </div>
  );
}
