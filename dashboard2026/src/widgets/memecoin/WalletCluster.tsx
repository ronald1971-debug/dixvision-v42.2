import { useState } from "react";

/**
 * Tier-4 memecoin widget — Wallet cluster lens.
 *
 * Operator pastes a wallet address; the widget surfaces the
 * inferred cluster: every wallet that has been funded from this
 * one (or that funded this one) within a 24h window, with a
 * combined holdings snapshot per token.
 *
 * The signal answers two questions:
 *   1. Is this "single whale" actually one human?
 *   2. Are the bundles I'm seeing in BundleDetector all
 *      converging back to the same root funder?
 *
 * Real graph traversal lives in the on-chain adapter (Tier-5,
 * filed). Here we render a deterministic mock graph keyed on
 * the address hash so the panel stays useful without RPC.
 */
interface ClusterMember {
  addr: string;
  role: "funder" | "child" | "sibling";
  funded_via: string;
  age_h: number;
  active_holdings: { ticker: string; amount: number }[];
}

const TICKERS = ["BONK", "WIF", "POPCAT", "GIGAFROG", "TURBOX", "MOONOPUS"];

function hash(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i += 1) {
    h = (h * 31 + s.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

function deriveAddr(seed: string, i: number): string {
  const h = hash(seed + i).toString(16).padStart(8, "0");
  return `${h.slice(0, 4)}…${h.slice(4, 8)}`;
}

function deriveCluster(addr: string): ClusterMember[] {
  const h = hash(addr);
  const n = 3 + (h % 6); // 3..8 members
  const out: ClusterMember[] = [];
  for (let i = 0; i < n; i += 1) {
    const role: ClusterMember["role"] =
      i === 0 ? "funder" : i % 2 === 0 ? "sibling" : "child";
    const holdings: ClusterMember["active_holdings"] = [];
    const tn = 1 + ((h + i) % 3);
    for (let j = 0; j < tn; j += 1) {
      holdings.push({
        ticker: TICKERS[(h + i + j) % TICKERS.length],
        amount: 1_000 + ((h + i * 31 + j * 7) % 80_000),
      });
    }
    out.push({
      addr: deriveAddr(addr, i + 1),
      role,
      funded_via: i === 0 ? "—" : deriveAddr(addr, 0),
      age_h: (h + i * 13) % 240,
      active_holdings: holdings,
    });
  }
  return out;
}

export function WalletCluster() {
  const [seed, setSeed] = useState("9b1cae42aa01");
  const [members, setMembers] = useState<ClusterMember[]>(() =>
    deriveCluster("9b1cae42aa01"),
  );

  const run = () => {
    if (!seed.trim()) return;
    setMembers(deriveCluster(seed.trim()));
  };

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Wallet cluster
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          24h funding graph · siblings + children + root funder
        </p>
      </header>
      <div className="flex flex-col gap-2 px-3 py-2 text-[11px]">
        <div className="flex gap-2">
          <input
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") run();
            }}
            className="flex-1 rounded border border-border bg-bg/40 px-2 py-1 font-mono text-[11px] text-slate-200 focus:border-accent focus:outline-none"
            placeholder="paste wallet address"
          />
          <button
            type="button"
            onClick={run}
            className="rounded border border-accent/40 bg-accent/10 px-2 py-1 text-[10px] uppercase tracking-wider text-accent hover:bg-accent/20"
          >
            Trace
          </button>
        </div>
      </div>
      <ul className="flex-1 divide-y divide-border/40 overflow-auto">
        {members.map((m) => (
          <li
            key={m.addr}
            className="px-3 py-2 font-mono text-[11px] text-slate-300"
          >
            <div className="flex items-baseline justify-between">
              <span className="font-semibold text-slate-200">{m.addr}</span>
              <span
                className={`rounded border px-1.5 py-0.5 text-[10px] uppercase ${
                  m.role === "funder"
                    ? "border-violet-500/40 bg-violet-500/10 text-violet-300"
                    : m.role === "child"
                      ? "border-sky-500/40 bg-sky-500/10 text-sky-300"
                      : "border-slate-500/40 bg-slate-500/10 text-slate-300"
                }`}
              >
                {m.role}
              </span>
            </div>
            <div className="mt-0.5 flex flex-wrap items-baseline gap-x-3 gap-y-0.5 text-[10px] text-slate-500">
              <span>via {m.funded_via}</span>
              <span>{m.age_h}h ago</span>
              {m.active_holdings.map((h) => (
                <span key={h.ticker} className="text-slate-400">
                  {h.ticker}{" "}
                  <span className="text-slate-300">
                    {h.amount.toLocaleString()}
                  </span>
                </span>
              ))}
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
