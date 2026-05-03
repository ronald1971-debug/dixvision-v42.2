interface TraitRow {
  trait: string;
  value: string;
  count: number;
  rarityPct: number; // 0..100
  floor: number; // ETH
  delta24h: number; // %
}

const MOCK: TraitRow[] = [
  { trait: "Background", value: "Diamond",  count: 28,   rarityPct: 0.28, floor: 32.4, delta24h: +12.3 },
  { trait: "Background", value: "Gold",     count: 142,  rarityPct: 1.42, floor: 18.1, delta24h: +3.7 },
  { trait: "Background", value: "Pink",     count: 1480, rarityPct: 14.8, floor: 7.85, delta24h: -1.1 },
  { trait: "Hat",        value: "Crown",    count: 67,   rarityPct: 0.67, floor: 26.0, delta24h: +5.4 },
  { trait: "Hat",        value: "Pirate",   count: 312,  rarityPct: 3.12, floor: 12.2, delta24h: -2.8 },
  { trait: "Eyes",       value: "Laser",    count: 45,   rarityPct: 0.45, floor: 22.5, delta24h: +9.1 },
  { trait: "Eyes",       value: "Sleepy",   count: 980,  rarityPct: 9.80, floor: 7.62, delta24h: -0.4 },
  { trait: "Mouth",      value: "Cigar",    count: 88,   rarityPct: 0.88, floor: 17.4, delta24h: +1.6 },
  { trait: "Mouth",      value: "Smile",    count: 2210, rarityPct: 22.1, floor: 7.40, delta24h: -0.2 },
  { trait: "Outfit",     value: "Astronaut",count: 53,   rarityPct: 0.53, floor: 24.7, delta24h: +6.0 },
  { trait: "Outfit",     value: "Hoodie",   count: 1890, rarityPct: 18.9, floor: 7.55, delta24h: -0.6 },
];

function rarityTone(p: number): string {
  if (p < 0.5) return "border-violet-500/40 bg-violet-500/10 text-violet-300";
  if (p < 1.5) return "border-amber-500/40 bg-amber-500/10 text-amber-300";
  if (p < 5) return "border-sky-500/40 bg-sky-500/10 text-sky-300";
  return "border-slate-600/40 bg-slate-700/30 text-slate-400";
}

export function TraitFloorGrid({ collection = "Pudgy" }: { collection?: string }) {
  const rows = [...MOCK].sort((a, b) => a.rarityPct - b.rarityPct);
  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Trait-Floor Grid · {collection}
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            rarity-aware floors per trait · sorted by rarity ascending
          </p>
        </div>
      </header>
      <div className="flex-1 overflow-auto">
        <table className="w-full text-[11px] font-mono">
          <thead className="text-[10px] uppercase tracking-wider text-slate-500">
            <tr>
              <th className="px-2 py-1 text-left">trait</th>
              <th className="px-2 py-1 text-left">value</th>
              <th className="px-2 py-1 text-right">supply</th>
              <th className="px-2 py-1 text-left">rarity</th>
              <th className="px-2 py-1 text-right">floor (Ξ)</th>
              <th className="px-2 py-1 text-right">24h</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="border-t border-border">
                <td className="px-2 py-1 text-slate-400">{r.trait}</td>
                <td className="px-2 py-1 text-slate-200">{r.value}</td>
                <td className="px-2 py-1 text-right text-slate-300">{r.count}</td>
                <td className="px-2 py-1">
                  <span
                    className={`rounded border px-1 py-px text-[9px] ${rarityTone(r.rarityPct)}`}
                  >
                    {r.rarityPct.toFixed(2)}%
                  </span>
                </td>
                <td className="px-2 py-1 text-right text-slate-100">{r.floor.toFixed(2)}</td>
                <td
                  className={`px-2 py-1 text-right ${
                    r.delta24h >= 0 ? "text-emerald-300" : "text-rose-300"
                  }`}
                >
                  {r.delta24h >= 0 ? "+" : ""}
                  {r.delta24h.toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
