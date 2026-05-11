/**
 * DEX widget — gas/fee estimator.
 *
 * Helius-style p50 / p75 / p90 priority-fee distribution +
 * base-fee + MEV-protected RPC indicator. Polls
 * ``/api/dashboard/dex/gas`` every 2 s; falls back to a
 * deterministic skeleton when the route 404s.
 */
import { useQuery } from "@tanstack/react-query";

import { apiUrl } from "@/api/base";
import { WidgetStatusChip } from "@/components/WidgetStatusChip";

interface GasSnapshot {
  base_fee_lamports: number;
  p50_tip_lamports: number;
  p75_tip_lamports: number;
  p90_tip_lamports: number;
  mev_protected_rpc: string;
  ts_iso: string;
}

const FALLBACK: GasSnapshot = {
  base_fee_lamports: 5_000,
  p50_tip_lamports: 12_400,
  p75_tip_lamports: 24_800,
  p90_tip_lamports: 78_000,
  mev_protected_rpc: "Jito Block-Engine (mock)",
  ts_iso: new Date().toISOString(),
};

async function fetchGas(
  signal?: AbortSignal,
): Promise<{ snap: GasSnapshot; live: boolean }> {
  try {
    const res = await fetch(apiUrl("/api/dashboard/dex/gas"), { signal });
    if (!res.ok) throw new Error(`status ${res.status}`);
    return { snap: (await res.json()) as GasSnapshot, live: true };
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw err;
    }
    return { snap: FALLBACK, live: false };
  }
}

export function GasEstimator() {
  const { data } = useQuery({
    queryKey: ["dashboard", "dex", "gas"],
    queryFn: ({ signal }) => fetchGas(signal),
    refetchInterval: 2_000,
    initialData: { snap: FALLBACK, live: false },
  });
  const { snap, live } = data;
  const total_p50 = snap.base_fee_lamports + snap.p50_tip_lamports;
  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Gas Estimator
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            Helius p50/p75/p90 · base-fee + tip · MEV-protected RPC
          </p>
        </div>
        <WidgetStatusChip mode={live ? "live" : "mock"} />
      </header>
      <div className="flex-1 overflow-auto p-3 font-mono text-[12px]">
        <Row k="base fee" v={`${snap.base_fee_lamports.toLocaleString()} λ`} />
        <Row
          k="tip p50"
          v={`${snap.p50_tip_lamports.toLocaleString()} λ`}
          tone="text-emerald-300"
        />
        <Row k="tip p75" v={`${snap.p75_tip_lamports.toLocaleString()} λ`} />
        <Row
          k="tip p90"
          v={`${snap.p90_tip_lamports.toLocaleString()} λ`}
          tone="text-rose-300"
        />
        <hr className="my-2 border-border" />
        <Row k="total p50" v={`${total_p50.toLocaleString()} λ`} />
        <Row k="MEV-protected" v={snap.mev_protected_rpc} mono={false} />
      </div>
    </div>
  );
}

function Row({
  k,
  v,
  tone = "text-slate-200",
  mono = true,
}: {
  k: string;
  v: string;
  tone?: string;
  mono?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between py-0.5 text-[11px]">
      <span className="text-slate-500">{k}</span>
      <span className={`${tone} ${mono ? "font-mono" : ""}`}>{v}</span>
    </div>
  );
}
