import { useEffect, useRef, useState } from "react";

/**
 * Tier-5 on-chain widget — Whale watcher.
 *
 * Live tape of USD-denominated large transfers (≥ $1M) across
 * BTC / ETH / SOL / TRX. Each row tags the kind:
 *   - cex_in   : transfer INTO a known exchange wallet
 *   - cex_out  : transfer OUT of a known exchange wallet
 *   - bridge   : crosschain bridge contract
 *   - p2p      : neither side is a known cluster
 *
 * Real wiring lives in the on-chain adapter (Glassnode / Arkham
 * keys; live verifier already shipped in PR #97). Today we render
 * a deterministic mock stream so the surface is alive.
 */
type Kind = "cex_in" | "cex_out" | "bridge" | "p2p";

interface Transfer {
  id: string;
  ts: number;
  chain: "BTC" | "ETH" | "SOL" | "TRX";
  amount_usd: number;
  asset: string;
  kind: Kind;
  from: string;
  to: string;
}

const CHAINS: Transfer["chain"][] = ["BTC", "ETH", "SOL", "TRX"];
const ASSETS: Record<Transfer["chain"], string[]> = {
  BTC: ["BTC"],
  ETH: ["ETH", "USDT", "USDC", "WBTC"],
  SOL: ["SOL", "USDC", "JUP"],
  TRX: ["TRX", "USDT"],
};
const KINDS: Kind[] = ["cex_in", "cex_out", "bridge", "p2p"];
const NAMES = ["Binance", "Coinbase", "Kraken", "Bybit", "OKX", "Wintermute"];

function fakeAddr(seed: number): string {
  return (
    ((seed * 9301 + 49297) % 233280)
      .toString(16)
      .padStart(4, "0")
      .slice(0, 4) +
    "…" +
    (seed % 65536).toString(16).padStart(4, "0")
  );
}

function newTransfer(seq: number, now: number): Transfer {
  const chain = CHAINS[seq % CHAINS.length];
  const assets = ASSETS[chain];
  const asset = assets[seq % assets.length];
  const kind = KINDS[seq % KINDS.length];
  const amount = 1_000_000 + (seq % 25) * 480_000;
  const knownName = NAMES[seq % NAMES.length];
  return {
    id: `${chain}-${seq}`,
    ts: now,
    chain,
    asset,
    amount_usd: amount,
    kind,
    from: kind === "cex_out" ? knownName : fakeAddr(seq * 13),
    to: kind === "cex_in" ? knownName : fakeAddr(seq * 17 + 11),
  };
}

const KIND_COLOR: Record<Kind, string> = {
  cex_in: "border-rose-500/40 bg-rose-500/10 text-rose-300",
  cex_out: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
  bridge: "border-violet-500/40 bg-violet-500/10 text-violet-300",
  p2p: "border-slate-500/40 bg-slate-500/10 text-slate-300",
};

export function WhaleWatcher() {
  const [feed, setFeed] = useState<Transfer[]>([]);
  const seq = useRef(0);

  useEffect(() => {
    const tick = () => {
      seq.current += 1;
      const now = Date.now();
      setFeed((prev) => [newTransfer(seq.current, now), ...prev].slice(0, 18));
    };
    for (let i = 0; i < 6; i += 1) tick();
    const id = setInterval(tick, 3_500);
    return () => clearInterval(id);
  }, []);

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Whale watcher
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          ≥ $1M transfers · cex flows · bridges · p2p
        </p>
      </header>
      <ul className="flex-1 divide-y divide-border/40 overflow-auto">
        {feed.map((t) => (
          <li
            key={t.id}
            className="grid grid-cols-[auto_1fr_auto] items-baseline gap-2 px-3 py-1.5 font-mono text-[11px] text-slate-300"
          >
            <span
              className={`rounded border px-1.5 py-0.5 text-[10px] uppercase ${KIND_COLOR[t.kind]}`}
            >
              {t.kind.replace("_", " ")}
            </span>
            <div className="min-w-0">
              <div className="flex items-baseline gap-2">
                <span className="font-semibold text-slate-200">
                  ${(t.amount_usd / 1_000_000).toFixed(2)}M
                </span>
                <span className="text-[10px] text-slate-500">
                  {t.asset} · {t.chain}
                </span>
              </div>
              <div className="truncate text-[10px] text-slate-500">
                {t.from} → {t.to}
              </div>
            </div>
            <span className="text-[10px] text-slate-500">
              {Math.max(0, Math.floor((Date.now() - t.ts) / 1000))}s
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}
