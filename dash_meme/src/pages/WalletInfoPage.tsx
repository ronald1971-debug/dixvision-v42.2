import { useQuery } from "@tanstack/react-query";

import { apiGet } from "@/api/base";
import { Panel } from "@/components/Panel";
import { StatusPill } from "@/components/StatusPill";

type DashboardSnapshot = Record<string, unknown>;

export function WalletInfoPage() {
  const q = useQuery({
    queryKey: ["wallet", "info"],
    queryFn: () => apiGet<DashboardSnapshot>("/api/dashboard"),
    refetchInterval: 5_000,
  });

  const positions = (q.data?.positions ?? q.data?.position ?? []) as ReadonlyArray<
    Record<string, unknown>
  >;
  const balances =
    (q.data?.balances ?? q.data?.wallet ?? {}) as Record<string, unknown>;

  return (
    <div className="grid h-full grid-cols-12 gap-2 p-2">
      <div className="col-span-5 min-h-0">
        <Panel
          title="Balances"
          right={
            <StatusPill tone={q.isError ? "danger" : "info"}>
              {q.isError ? "ERR" : "LIVE"}
            </StatusPill>
          }
          bodyClassName="p-3"
        >
          {Object.keys(balances).length === 0 ? (
            <p className="text-xs text-text-disabled">
              No balance data exposed by{" "}
              <span className="font-mono">/api/dashboard</span>.
            </p>
          ) : (
            <dl className="grid grid-cols-2 gap-1 text-xs">
              {Object.entries(balances).map(([k, v]) => (
                <div key={k} className="contents">
                  <dt className="text-text-secondary">{k}</dt>
                  <dd className="text-right font-mono">{String(v)}</dd>
                </div>
              ))}
            </dl>
          )}
        </Panel>
      </div>
      <div className="col-span-7 min-h-0">
        <Panel title="Open positions" bodyClassName="p-0">
          <table className="w-full font-mono text-xs tabular-nums">
            <thead className="bg-surface text-text-secondary">
              <tr className="border-b border-hairline">
                <th className="px-2 py-1 text-left">Symbol</th>
                <th className="px-2 py-1 text-right">Size</th>
                <th className="px-2 py-1 text-right">Avg</th>
                <th className="px-2 py-1 text-right">PnL</th>
              </tr>
            </thead>
            <tbody>
              {positions.length === 0 && (
                <tr>
                  <td
                    colSpan={4}
                    className="px-2 py-3 text-center text-text-disabled"
                  >
                    No open positions.
                  </td>
                </tr>
              )}
              {positions.map((p, i) => (
                <tr key={i} className="dex-row">
                  <td className="px-2 py-0.5">{String(p.symbol ?? "?")}</td>
                  <td className="px-2 py-0.5 text-right">
                    {String(p.size ?? p.qty ?? "—")}
                  </td>
                  <td className="px-2 py-0.5 text-right">
                    {String(p.avg ?? p.entry ?? "—")}
                  </td>
                  <td className="px-2 py-0.5 text-right">
                    {String(p.pnl ?? p.unrealized ?? "—")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      </div>
    </div>
  );
}
