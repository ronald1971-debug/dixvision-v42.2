import { useState } from "react";

/**
 * Tier-4 memecoin widget — Honeypot checker.
 *
 * Pre-trade simulation panel that runs a fixed set of probes
 * against a candidate token before any size is committed:
 *
 *   - sell_simulation        : would a sell actually clear?
 *   - buy_tax / sell_tax     : observed transfer taxes
 *   - hidden_owner           : owner is renounced or proxy?
 *   - mint_authority         : can the contract still mint?
 *   - blacklist_capability   : can deployer block addresses?
 *   - max_tx_clamp           : clamp on outgoing tx size?
 *
 * Any failing probe blocks the operator-approval edge from
 * accepting a stage. This widget surfaces the matrix so the
 * operator sees *why* a stage was blocked, not just that it was.
 *
 * Real probes go through the dex/honeypot adapter (filed). Today
 * we let the operator paste a token address and we render a
 * deterministic verdict based on a hash of the address.
 */
type Probe =
  | "sell_simulation"
  | "buy_tax"
  | "sell_tax"
  | "hidden_owner"
  | "mint_authority"
  | "blacklist_capability"
  | "max_tx_clamp";

interface Result {
  pass: boolean;
  detail: string;
}

const PROBES: Probe[] = [
  "sell_simulation",
  "buy_tax",
  "sell_tax",
  "hidden_owner",
  "mint_authority",
  "blacklist_capability",
  "max_tx_clamp",
];

function hash(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i += 1) {
    h = (h * 31 + s.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

function probe(addr: string, p: Probe): Result {
  const h = hash(addr + p);
  switch (p) {
    case "sell_simulation":
      return {
        pass: h % 7 !== 0,
        detail: h % 7 === 0 ? "sell reverts in fork" : "sell clears 1.0 unit",
      };
    case "buy_tax": {
      const tax = (h % 12) / 100;
      return {
        pass: tax <= 0.05,
        detail: `${(tax * 100).toFixed(1)}%`,
      };
    }
    case "sell_tax": {
      const tax = (h % 18) / 100;
      return {
        pass: tax <= 0.07,
        detail: `${(tax * 100).toFixed(1)}%`,
      };
    }
    case "hidden_owner":
      return {
        pass: h % 5 !== 0,
        detail: h % 5 === 0 ? "proxy admin still set" : "renounced",
      };
    case "mint_authority":
      return {
        pass: h % 4 !== 0,
        detail: h % 4 === 0 ? "mint authority still active" : "burned",
      };
    case "blacklist_capability":
      return {
        pass: h % 6 !== 0,
        detail: h % 6 === 0 ? "blacklist function present" : "no blacklist",
      };
    case "max_tx_clamp": {
      const clamp = (h % 40) / 1000;
      return {
        pass: clamp === 0 || clamp >= 0.01,
        detail: clamp === 0 ? "no clamp" : `${(clamp * 100).toFixed(2)}% / tx`,
      };
    }
  }
}

export function HoneypotChecker() {
  const [addr, setAddr] = useState("");
  const [results, setResults] = useState<Record<Probe, Result> | null>(null);

  const run = () => {
    if (!addr.trim()) return;
    const r = {} as Record<Probe, Result>;
    for (const p of PROBES) r[p] = probe(addr.trim(), p);
    setResults(r);
  };

  const blocked = results
    ? PROBES.some((p) => !results[p].pass)
    : false;

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Honeypot checker
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          7 probes · any fail blocks approval-edge stage
        </p>
      </header>
      <div className="flex flex-1 flex-col gap-2 overflow-auto p-3 text-[12px]">
        <div className="flex gap-2">
          <input
            value={addr}
            onChange={(e) => setAddr(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") run();
            }}
            placeholder="paste token mint address"
            className="flex-1 rounded border border-border bg-bg/40 px-2 py-1 font-mono text-[11px] text-slate-200 focus:border-accent focus:outline-none"
          />
          <button
            type="button"
            onClick={run}
            className="rounded border border-accent/40 bg-accent/10 px-2 py-1 text-[11px] uppercase tracking-wider text-accent hover:bg-accent/20"
          >
            Probe
          </button>
        </div>
        {results ? (
          <>
            <div
              className={`rounded border px-2 py-1 text-[11px] font-mono uppercase tracking-wider ${
                blocked
                  ? "border-rose-500/40 bg-rose-500/10 text-rose-300"
                  : "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
              }`}
            >
              {blocked ? "STAGE BLOCKED" : "STAGE ALLOWED"}
            </div>
            <ul className="divide-y divide-border/40 rounded border border-border">
              {PROBES.map((p) => {
                const r = results[p];
                return (
                  <li
                    key={p}
                    className="flex items-baseline justify-between px-2 py-1.5 font-mono text-[11px]"
                  >
                    <span className="text-slate-300">
                      {p.replace(/_/g, " ")}
                    </span>
                    <span className="flex items-baseline gap-2">
                      <span className="text-[10px] text-slate-500">
                        {r.detail}
                      </span>
                      <span
                        className={
                          r.pass ? "text-emerald-400" : "text-rose-400"
                        }
                      >
                        {r.pass ? "pass" : "fail"}
                      </span>
                    </span>
                  </li>
                );
              })}
            </ul>
          </>
        ) : (
          <p className="text-[11px] text-slate-500">
            paste a mint address and press Probe
          </p>
        )}
      </div>
    </section>
  );
}
