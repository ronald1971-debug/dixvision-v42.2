import { useEffect, useRef, useState } from "react";

/**
 * Tier-4 memecoin widget — Launch firehose.
 *
 * Live tape of new launches across Pump.fun + Raydium + Moonshot.
 * Each row surfaces age, deployer, initial liquidity, and a quick
 * "open" / "stage snipe" pair of actions. The "stage snipe" path
 * routes through the operator-approval edge (INV-72) — never
 * directly to `execution_engine`.
 *
 * Real WS wiring lives in the dex/pump adapter (filed for Wave-DEX);
 * the local mock generator below produces deterministic-feeling
 * launches at a steady cadence so the surface is alive today.
 */
interface Launch {
  id: string;
  ts: number;
  source: "PUMP" | "RAYDIUM" | "MOONSHOT";
  ticker: string;
  name: string;
  deployer: string;
  initLiqUsd: number;
  age_s: number;
  flags: string[];
}

const NAMES = [
  ["WIFCAT", "Wif's Cat"],
  ["BONKER", "Bonker"],
  ["TURBOX", "Turbo X"],
  ["GIGAFROG", "Giga Frog"],
  ["MOONOPUS", "Moonopus"],
  ["ZIGZAG", "ZigZag Inu"],
  ["DOGOFC", "Dog of Crypto"],
  ["SPIN", "Spinning Top"],
  ["NIBBLE", "Nibble"],
  ["BURN", "BurnBurn"],
];

const SOURCES: Launch["source"][] = ["PUMP", "RAYDIUM", "MOONSHOT"];

function fakeAddr(seed: number): string {
  const hex = ((seed * 9301 + 49297) % 233280).toString(16).padStart(4, "0");
  return `${hex.slice(0, 4)}…${(seed % 10000).toString(16).padStart(4, "0")}`;
}

function newLaunch(seq: number, now: number): Launch {
  const [ticker, name] = NAMES[seq % NAMES.length];
  const source = SOURCES[seq % SOURCES.length];
  const liq = 800 + (seq % 12) * 1_400;
  const flags: string[] = [];
  if (liq < 2_500) flags.push("low-liq");
  if (seq % 7 === 0) flags.push("bundled");
  if (seq % 11 === 0) flags.push("renounced");
  if (seq % 13 === 0) flags.push("locked");
  return {
    id: `${source}-${seq}`,
    ts: now,
    source,
    ticker,
    name,
    deployer: fakeAddr(seq * 17),
    initLiqUsd: liq,
    age_s: 0,
    flags,
  };
}

export function LaunchFirehose() {
  const [feed, setFeed] = useState<Launch[]>([]);
  const seq = useRef(0);

  useEffect(() => {
    const tick = () => {
      seq.current += 1;
      const now = Date.now();
      setFeed((prev) => {
        const aged = prev.map((l) => ({
          ...l,
          age_s: Math.floor((now - l.ts) / 1000),
        }));
        return [newLaunch(seq.current, now), ...aged].slice(0, 14);
      });
    };
    // Seed a handful so the panel isn't empty on mount.
    for (let i = 0; i < 5; i += 1) tick();
    const id = setInterval(tick, 2_500);
    return () => clearInterval(id);
  }, []);

  const stage = (l: Launch) => {
    // Approval-edge stage. Real wiring goes through
    // `/api/cognitive/proposals` (INV-72). Here we just ack.
    console.info("[snipe-stage]", l.ticker, l.deployer, l.initLiqUsd);
  };

  return (
    <section
      className="flex h-full flex-col rounded border border-border bg-surface"
      data-testid="launch-firehose"
    >
      <header className="border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Launch firehose
        </h3>
        <p className="mt-0.5 text-[11px] text-slate-500">
          Pump.fun · Raydium · Moonshot — stage snipe routes through
          approval edge
        </p>
      </header>
      <ul className="flex-1 divide-y divide-border/40 overflow-auto">
        {feed.map((l) => (
          <li
            key={l.id}
            className="grid grid-cols-[auto_1fr_auto] items-baseline gap-2 px-3 py-1.5 font-mono text-[11px] text-slate-300"
          >
            <span
              className={`rounded border px-1.5 py-0.5 text-[10px] uppercase ${
                l.source === "PUMP"
                  ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
                  : l.source === "RAYDIUM"
                    ? "border-sky-500/40 bg-sky-500/10 text-sky-300"
                    : "border-violet-500/40 bg-violet-500/10 text-violet-300"
              }`}
            >
              {l.source}
            </span>
            <div className="min-w-0">
              <div className="flex items-baseline gap-2">
                <span className="font-semibold text-slate-200">
                  {l.ticker}
                </span>
                <span className="truncate text-[10px] text-slate-500">
                  {l.name}
                </span>
              </div>
              <div className="flex flex-wrap items-baseline gap-2 text-[10px] text-slate-500">
                <span>{l.deployer}</span>
                <span>liq ${l.initLiqUsd.toLocaleString()}</span>
                <span>age {l.age_s}s</span>
                {l.flags.map((f) => (
                  <span
                    key={f}
                    className={`rounded px-1 py-0.5 text-[9px] uppercase ${
                      f === "low-liq" || f === "bundled"
                        ? "bg-rose-500/15 text-rose-300"
                        : "bg-emerald-500/15 text-emerald-300"
                    }`}
                  >
                    {f}
                  </span>
                ))}
              </div>
            </div>
            <button
              type="button"
              onClick={() => stage(l)}
              className="rounded border border-accent/40 bg-accent/10 px-2 py-0.5 text-[10px] uppercase tracking-wider text-accent hover:bg-accent/20"
            >
              stage
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
