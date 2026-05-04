import { useQuery } from "@tanstack/react-query";

import { fetchMemecoinSummary } from "@/api/feeds";

import { Panel } from "./Panel";

type Holder = {
  address: string;
  amount: string;
  pct: number | null;
  label?: string;
};

function pickHolders(o: Record<string, unknown>): ReadonlyArray<Holder> {
  const raw =
    (o.top_holders as unknown) ??
    (o.holders_top as unknown) ??
    (o.holders_list as unknown) ??
    null;
  if (!Array.isArray(raw)) return [];
  return raw.slice(0, 25).map((row): Holder => {
    const r = row as Record<string, unknown>;
    const pct =
      typeof r.pct === "number"
        ? r.pct
        : typeof r.percent === "number"
          ? r.percent
          : typeof r.share === "number"
            ? r.share * 100
            : null;
    return {
      address: String(r.address ?? r.addr ?? r.wallet ?? "?"),
      amount: String(r.amount ?? r.balance ?? "?"),
      pct,
      label: typeof r.label === "string" ? r.label : undefined,
    };
  });
}

export function HoldersPanel() {
  const q = useQuery({
    queryKey: ["memecoin", "summary"],
    queryFn: fetchMemecoinSummary,
    refetchInterval: 6_000,
  });
  const holders = pickHolders(
    (q.data?.memecoin ?? {}) as Record<string, unknown>,
  );

  return (
    <Panel title="Top holders">
      <table className="w-full font-mono text-[11px] tabular-nums">
        <thead className="sticky top-0 bg-surface text-text-secondary">
          <tr className="border-b border-hairline">
            <th className="px-2 py-1 text-left">#</th>
            <th className="px-2 py-1 text-left">Address</th>
            <th className="px-2 py-1 text-right">Amount</th>
            <th className="px-2 py-1 text-right">%</th>
          </tr>
        </thead>
        <tbody>
          {holders.length === 0 && (
            <tr>
              <td
                colSpan={4}
                className="px-2 py-3 text-center text-text-disabled"
              >
                No holder data exposed by /api/dashboard/memecoin.
              </td>
            </tr>
          )}
          {holders.map((h, i) => (
            <tr key={h.address + i} className="dex-row">
              <td className="px-2 py-0.5 text-text-disabled">{i + 1}</td>
              <td className="truncate px-2 py-0.5">
                {h.label ? `${h.label} · ` : ""}
                <span className="text-text-secondary">
                  {h.address.slice(0, 6)}…{h.address.slice(-4)}
                </span>
              </td>
              <td className="px-2 py-0.5 text-right">{h.amount}</td>
              <td className="px-2 py-0.5 text-right">
                {h.pct == null ? "—" : `${h.pct.toFixed(2)}%`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}
