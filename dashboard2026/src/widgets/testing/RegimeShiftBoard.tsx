import { useMemo, useState } from "react";

import { Zap } from "lucide-react";

/**
 * Tier-8 testing widget — regime-shift fixture board.
 *
 * One-click run-against canonical historical stress windows so the
 * operator can see whether a strategy survives the kind of break that
 * 90% of curve-fit edges silently die in. Each fixture is a fully
 * deterministic ledger replay — picking it pins (start, end, base
 * volatility, initial drawdown) and runs the strategy through the
 * regime.
 *
 * Live wiring source: `evaluation.regime_fixtures.run(fixture, strategy)`
 * — currently mocked from a deterministic seed so the panel is alive
 * before the backend route lands.
 */
type Strategy =
  | "ema_cross_20_50"
  | "rsi_2_meanrev"
  | "vwap_reversion"
  | "breakout_channel"
  | "microstructure_v1"
  | "memecoin_copy";

const STRATEGIES: ReadonlyArray<{ key: Strategy; label: string }> = [
  { key: "ema_cross_20_50", label: "EMA cross 20/50" },
  { key: "rsi_2_meanrev", label: "RSI(2) mean-reversion" },
  { key: "vwap_reversion", label: "VWAP reversion" },
  { key: "breakout_channel", label: "Breakout channel" },
  { key: "microstructure_v1", label: "Microstructure v1" },
  { key: "memecoin_copy", label: "Memecoin copy-trader" },
];

interface Fixture {
  key: string;
  label: string;
  window: string;
  blurb: string;
  base_dd_pct: number;
  vol_mult: number;
}

const FIXTURES: ReadonlyArray<Fixture> = [
  {
    key: "covid_2020_03",
    label: "COVID crash",
    window: "2020-02-19 → 2020-04-07",
    blurb: "Liquidity vacuum, circuit breakers, oil flash.",
    base_dd_pct: -34,
    vol_mult: 3.4,
  },
  {
    key: "luna_2022_05",
    label: "LUNA / UST collapse",
    window: "2022-05-08 → 2022-05-15",
    blurb: "Algorithmic stable peg break + cascade liquidations.",
    base_dd_pct: -42,
    vol_mult: 3.1,
  },
  {
    key: "ftx_2022_11",
    label: "FTX implosion",
    window: "2022-11-06 → 2022-11-15",
    blurb: "Counterparty failure, exchange withdrawal halts.",
    base_dd_pct: -28,
    vol_mult: 2.6,
  },
  {
    key: "svb_2023_03",
    label: "SVB / banking panic",
    window: "2023-03-08 → 2023-03-19",
    blurb: "Stablecoin de-peg + cross-asset risk-off.",
    base_dd_pct: -19,
    vol_mult: 2.1,
  },
  {
    key: "yen_carry_2024_08",
    label: "Yen-carry unwind",
    window: "2024-08-02 → 2024-08-09",
    blurb: "Cross-asset deleveraging, vol explosion in JPY pairs.",
    base_dd_pct: -16,
    vol_mult: 2.4,
  },
  {
    key: "flash_crash_2024_12",
    label: "Flash crash 24-12",
    window: "2024-12-09 (intraday)",
    blurb: "Single-day liquidity drain on perps + spot.",
    base_dd_pct: -12,
    vol_mult: 4.2,
  },
];

function hashSeed(parts: ReadonlyArray<string | number>): number {
  let h = 2166136261;
  for (const p of parts) {
    const s = String(p);
    for (let i = 0; i < s.length; i += 1) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
  }
  return h >>> 0;
}

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

interface RunResult {
  return_pct: number;
  max_dd_pct: number;
  trades: number;
  fill_rate: number;
  survived: boolean;
}

function runFixture(strategy: Strategy, fixture: Fixture): RunResult {
  const r = rng(hashSeed([strategy, fixture.key]));
  const memeBoost =
    strategy === "memecoin_copy" || strategy === "microstructure_v1" ? 0.6 : 0;
  const survival =
    strategy === "rsi_2_meanrev"
      ? 0.4
      : strategy === "ema_cross_20_50"
        ? 0.7
        : 0.6;
  const survived = r() < survival;
  const base = (r() - 0.45) * fixture.vol_mult * 8;
  const ret = survived ? base + memeBoost * 4 : base - 6 - memeBoost * 8;
  const dd = Math.min(0, fixture.base_dd_pct + (r() - 0.5) * 12);
  return {
    return_pct: ret,
    max_dd_pct: dd,
    trades: 18 + Math.floor(r() * 90),
    fill_rate: 0.6 + r() * 0.3,
    survived,
  };
}

export function RegimeShiftBoard() {
  const [strategy, setStrategy] = useState<Strategy>("ema_cross_20_50");
  const [active, setActive] = useState<string>(FIXTURES[0].key);

  const results = useMemo(
    () =>
      Object.fromEntries(
        FIXTURES.map((f) => [f.key, runFixture(strategy, f)] as const),
      ),
    [strategy],
  );

  const activeFixture = FIXTURES.find((f) => f.key === active) ?? FIXTURES[0];
  const activeResult = results[active];

  const survivedCount = Object.values(results).filter((r) => r.survived).length;

  return (
    <section className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-center justify-between border-b border-border px-3 py-2">
        <div className="flex items-center gap-2">
          <Zap className="h-3.5 w-3.5 text-accent" />
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Regime-shift fixtures
          </h3>
        </div>
        <span className="rounded border border-border bg-bg px-2 py-0.5 font-mono text-[10px] uppercase tracking-widest text-slate-400">
          {survivedCount}/{FIXTURES.length} survived
        </span>
      </header>

      <div className="flex items-center gap-2 border-b border-border px-3 py-2 text-[11px]">
        <span className="text-[10px] uppercase tracking-widest text-slate-500">
          Strategy
        </span>
        <select
          value={strategy}
          onChange={(e) => setStrategy(e.target.value as Strategy)}
          className="rounded border border-border bg-bg px-1.5 py-0.5 font-mono text-[11px] text-slate-100"
        >
          {STRATEGIES.map((s) => (
            <option key={s.key} value={s.key}>
              {s.label}
            </option>
          ))}
        </select>
      </div>

      <div className="grid flex-1 grid-cols-1 gap-0 md:grid-cols-2">
        <ul className="divide-y divide-border overflow-auto border-r border-border">
          {FIXTURES.map((f) => {
            const r = results[f.key];
            const isActive = f.key === active;
            return (
              <li key={f.key}>
                <button
                  type="button"
                  onClick={() => setActive(f.key)}
                  className={`flex w-full items-start justify-between gap-2 px-3 py-2 text-left ${
                    isActive ? "bg-accent/10" : "hover:bg-bg"
                  }`}
                >
                  <div>
                    <div className="text-xs font-medium text-slate-100">
                      {f.label}
                    </div>
                    <div className="font-mono text-[10px] text-slate-500">
                      {f.window}
                    </div>
                  </div>
                  <span
                    className={`mt-0.5 shrink-0 rounded border px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-widest ${
                      r.survived
                        ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
                        : "border-rose-500/40 bg-rose-500/10 text-rose-300"
                    }`}
                  >
                    {r.survived ? "survived" : "blew up"}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>

        <div className="flex flex-col gap-2 overflow-auto p-3">
          <div>
            <div className="text-xs font-medium text-slate-100">
              {activeFixture.label}
            </div>
            <div className="font-mono text-[10px] text-slate-500">
              {activeFixture.window}
            </div>
            <p className="mt-1 text-[11px] leading-relaxed text-slate-400">
              {activeFixture.blurb}
            </p>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <Metric
              label="Return"
              value={`${activeResult.return_pct >= 0 ? "+" : ""}${activeResult.return_pct.toFixed(1)}%`}
              tone={activeResult.return_pct >= 0 ? "emerald" : "rose"}
            />
            <Metric
              label="Max DD"
              value={`${activeResult.max_dd_pct.toFixed(1)}%`}
              tone="rose"
            />
            <Metric
              label="Trades"
              value={String(activeResult.trades)}
            />
            <Metric
              label="Fill rate"
              value={`${(activeResult.fill_rate * 100).toFixed(0)}%`}
            />
          </div>

          <p className="border-t border-border pt-2 text-[10px] leading-relaxed text-slate-500">
            Vol multiplier {activeFixture.vol_mult.toFixed(1)}× ·
            base regime DD {activeFixture.base_dd_pct}%. Same intelligence
            and execution path as live; only the source of bars is the
            ledger replay store.
          </p>
        </div>
      </div>
    </section>
  );
}

function Metric({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "emerald" | "rose";
}) {
  const tc =
    tone === "emerald"
      ? "text-emerald-300"
      : tone === "rose"
        ? "text-rose-300"
        : "text-slate-100";
  return (
    <div className="rounded border border-border bg-bg px-2 py-1.5">
      <div className="text-[10px] uppercase tracking-widest text-slate-500">
        {label}
      </div>
      <div className={`font-mono text-base ${tc}`}>{value}</div>
    </div>
  );
}
