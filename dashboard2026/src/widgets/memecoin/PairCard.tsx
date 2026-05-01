/**
 * Memecoin Pair Card (PR-#2 spec §3.4).
 *
 * Mirrors Photon / Axiom / GMGN 2026 terminal cards: price, MC, FDV,
 * 5m / 1h / 6h / 24h change, buys vs sells, unique wallets. Inline
 * safety badges (mint revoked, freeze revoked, LP burned, top-10
 * holders share) cross-referenced against the rug-score.
 */
interface PairSnapshot {
  symbol: string;
  chain: "solana" | "ethereum" | "base";
  price: number;
  mc: number;
  fdv: number;
  d5m: number;
  d1h: number;
  d6h: number;
  d24h: number;
  buys: number;
  sells: number;
  unique_wallets: number;
  badges: { mint_revoked: boolean; freeze_revoked: boolean; lp_burned: boolean };
}

const MOCK: PairSnapshot = {
  symbol: "BONK / SOL",
  chain: "solana",
  price: 0.0000218,
  mc: 1_580_000_000,
  fdv: 2_100_000_000,
  d5m: 1.4,
  d1h: 5.8,
  d6h: -2.1,
  d24h: 12.3,
  buys: 1842,
  sells: 1109,
  unique_wallets: 2611,
  badges: { mint_revoked: true, freeze_revoked: true, lp_burned: true },
};

export function PairCard() {
  const p = MOCK;
  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            {p.symbol}
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            chain {p.chain} · MC {fmt(p.mc)} · FDV {fmt(p.fdv)}
          </p>
        </div>
        <span className="rounded border border-accent/40 bg-accent/10 px-1.5 py-0.5 font-mono text-[10px] text-accent">
          live
        </span>
      </header>
      <div className="flex-1 space-y-3 overflow-auto p-3">
        <div className="flex items-baseline gap-2 font-mono">
          <span className="text-2xl text-slate-100">
            ${p.price.toFixed(7)}
          </span>
          <span className={p.d24h >= 0 ? "text-emerald-300" : "text-red-300"}>
            {p.d24h >= 0 ? "+" : ""}
            {p.d24h.toFixed(2)}% 24h
          </span>
        </div>
        <div className="grid grid-cols-4 gap-2 font-mono text-[11px]">
          {(
            [
              ["5m", p.d5m],
              ["1h", p.d1h],
              ["6h", p.d6h],
              ["24h", p.d24h],
            ] as const
          ).map(([k, v]) => (
            <div
              key={k}
              className="rounded border border-border bg-bg/40 px-2 py-1 text-center"
            >
              <div className="text-[10px] uppercase tracking-wider text-slate-500">
                {k}
              </div>
              <div
                className={
                  v >= 0 ? "text-emerald-300" : "text-red-300"
                }
              >
                {v >= 0 ? "+" : ""}
                {v.toFixed(2)}%
              </div>
            </div>
          ))}
        </div>
        <div className="grid grid-cols-3 gap-2 font-mono text-[11px]">
          <Stat label="buys" value={p.buys} tone="ok" />
          <Stat label="sells" value={p.sells} tone="danger" />
          <Stat label="wallets" value={p.unique_wallets} tone="info" />
        </div>
        <div className="flex flex-wrap gap-1 text-[10px] uppercase tracking-wider">
          <Badge ok={p.badges.mint_revoked} label="mint revoked" />
          <Badge ok={p.badges.freeze_revoked} label="freeze revoked" />
          <Badge ok={p.badges.lp_burned} label="LP burned" />
        </div>
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone: "ok" | "danger" | "info";
}) {
  const cls =
    tone === "ok"
      ? "text-emerald-300"
      : tone === "danger"
        ? "text-red-300"
        : "text-slate-200";
  return (
    <div className="rounded border border-border bg-bg/40 px-2 py-1 text-center">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">
        {label}
      </div>
      <div className={cls}>{value.toLocaleString()}</div>
    </div>
  );
}

function Badge({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span
      className={`rounded border px-1.5 py-0.5 font-mono ${
        ok
          ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
          : "border-red-500/40 bg-red-500/10 text-red-300"
      }`}
    >
      {ok ? "✓" : "✗"} {label}
    </span>
  );
}

function fmt(n: number): string {
  if (n >= 1e9) return `$${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `$${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `$${(n / 1e3).toFixed(2)}K`;
  return `$${n.toFixed(2)}`;
}
