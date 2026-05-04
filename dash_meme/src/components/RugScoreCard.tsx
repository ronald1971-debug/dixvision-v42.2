import { useQuery } from "@tanstack/react-query";

import { fetchMemecoinSummary } from "@/api/feeds";

import { Panel } from "./Panel";
import { StatusPill } from "./StatusPill";

/**
 * RugScore is a backend-aggregated score (0..100, lower is safer).
 * Today the harness exposes it via /api/dashboard/memecoin's `rug_score`
 * field on the latest pair. If the field is absent we degrade to a
 * neutral display rather than fabricating a number.
 */

function pickNum(o: Record<string, unknown>, ...keys: string[]): number | null {
  for (const k of keys) {
    const v = o[k];
    if (typeof v === "number" && Number.isFinite(v)) return v;
  }
  return null;
}

function pickList(
  o: Record<string, unknown>,
  ...keys: string[]
): ReadonlyArray<string> {
  for (const k of keys) {
    const v = o[k];
    if (Array.isArray(v)) return v.map((x) => String(x));
  }
  return [];
}

export function RugScoreCard({ symbol }: { symbol: string }) {
  const q = useQuery({
    queryKey: ["memecoin", "summary"],
    queryFn: fetchMemecoinSummary,
    refetchInterval: 4_000,
  });
  const meme = (q.data?.memecoin ?? {}) as Record<string, unknown>;
  const score = pickNum(meme, "rug_score", "rugscore", "rug");
  const liq = pickNum(meme, "liq_usd", "liquidity_usd");
  const holders = pickNum(meme, "holders");
  const devPct = pickNum(meme, "dev_pct", "dev_supply_pct");
  const flags = pickList(meme, "flags", "warnings");

  let tone: "ok" | "warn" | "danger" | "neutral" = "neutral";
  let label = "—";
  if (score != null) {
    if (score < 30) {
      tone = "ok";
      label = "LOW";
    } else if (score < 70) {
      tone = "warn";
      label = "ELEVATED";
    } else {
      tone = "danger";
      label = "HIGH";
    }
  }

  return (
    <Panel
      title={`RugScore · ${symbol}`}
      right={<StatusPill tone={tone}>{label}</StatusPill>}
    >
      <div className="space-y-2 p-3 text-xs">
        <div className="flex items-baseline gap-2">
          <span
            className={`font-mono text-3xl tabular-nums ${
              tone === "ok"
                ? "text-ok"
                : tone === "warn"
                  ? "text-warn"
                  : tone === "danger"
                    ? "text-danger"
                    : "text-text-secondary"
            }`}
          >
            {score == null ? "—" : score.toFixed(0)}
          </span>
          <span className="text-text-disabled">/ 100</span>
        </div>
        <dl className="grid grid-cols-2 gap-1 text-[11px]">
          <dt className="text-text-secondary">Liquidity</dt>
          <dd className="text-right font-mono">
            {liq == null ? "—" : `$${liq.toLocaleString()}`}
          </dd>
          <dt className="text-text-secondary">Holders</dt>
          <dd className="text-right font-mono">
            {holders == null ? "—" : holders.toLocaleString()}
          </dd>
          <dt className="text-text-secondary">Dev supply</dt>
          <dd className="text-right font-mono">
            {devPct == null ? "—" : `${devPct.toFixed(1)}%`}
          </dd>
        </dl>
        {flags.length > 0 && (
          <div>
            <div className="mb-1 text-[10px] uppercase tracking-wide text-text-disabled">
              Flags
            </div>
            <ul className="space-y-0.5">
              {flags.map((f, i) => (
                <li key={i} className="text-warn">
                  • {f}
                </li>
              ))}
            </ul>
          </div>
        )}
        {q.isError && (
          <div className="text-danger">
            Backend error: {(q.error as Error).message}
          </div>
        )}
      </div>
    </Panel>
  );
}
