import { useQuery } from "@tanstack/react-query";

import { fetchCredentialsStatus } from "@/api/credentials";
import { StateBadge } from "@/components/StateBadge";
import type { CredentialItem } from "@/types/generated/api";

function CredentialRow({ item }: { item: CredentialItem }) {
  return (
    <tr className="border-t border-border align-top">
      <td className="px-3 py-2 font-mono text-xs text-slate-400 whitespace-nowrap">
        {item.source_id}
      </td>
      <td className="px-3 py-2">
        <div className="text-sm">{item.source_name}</div>
        <div className="text-xs text-slate-500">
          {item.category} · {item.provider}
        </div>
      </td>
      <td className="px-3 py-2 font-mono text-xs">
        <ul className="space-y-0.5">
          {item.env_vars.map((name, idx) => (
            <li key={name}>
              <span className="text-slate-200">{name}</span>
              <span className="ml-2 text-slate-500">
                {item.env_vars_present[idx] ? "set" : "unset"}
              </span>
            </li>
          ))}
        </ul>
      </td>
      <td className="px-3 py-2">
        <StateBadge state={item.state} />
      </td>
      <td className="px-3 py-2 text-xs">
        {item.signup_url ? (
          <a
            className="text-accent hover:underline"
            href={item.signup_url}
            target="_blank"
            rel="noreferrer noopener"
          >
            sign up
          </a>
        ) : (
          <span className="text-slate-600">—</span>
        )}
        {item.free_tier && (
          <span className="ml-2 rounded border border-border px-1.5 py-0.5 text-[10px] text-slate-400">
            free tier
          </span>
        )}
      </td>
    </tr>
  );
}

export function CredentialsPage() {
  const { data, isPending, isError, error, refetch, isFetching } = useQuery({
    queryKey: ["credentials", "status"],
    queryFn: ({ signal }) => fetchCredentialsStatus(signal),
    // Live polling at 5 s — slow enough for the credential matrix
    // (10 providers, mostly slow-changing) but fast enough that flipping
    // a key on disk shows up in the UI without a manual refresh.
    refetchInterval: 5_000,
  });

  return (
    <section className="max-w-6xl mx-auto">
      <div className="flex items-end justify-between mb-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">
            Credential discovery
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Every <code className="text-xs">auth: required</code> registry
            row, with the env var(s) that must be set and a per-row
            present / partial / missing state. Read-only here; storage
            happens via the <code className="text-xs">/credentials</code>{" "}
            page or, in Devin sessions, the operator's{" "}
            <code className="text-xs">secrets</code> tool.
          </p>
        </div>
        <button
          type="button"
          onClick={() => refetch()}
          className="rounded border border-border bg-surface px-3 py-1.5 text-xs hover:border-accent disabled:opacity-50"
          disabled={isFetching}
        >
          {isFetching ? "refreshing…" : "refresh"}
        </button>
      </div>

      {isPending && (
        <p className="text-sm text-slate-400">Loading…</p>
      )}

      {isError && (
        <div className="rounded border border-danger/40 bg-danger/10 p-3 text-sm text-danger">
          Failed to load credential status: {(error as Error).message}
        </div>
      )}

      {data && (
        <>
          <div className="mb-4 grid grid-cols-4 gap-3 font-mono text-xs">
            <SummaryTile label="total" value={data.summary.total} />
            <SummaryTile
              label="present"
              value={data.summary.present}
              tone="ok"
            />
            <SummaryTile
              label="partial"
              value={data.summary.partial}
              tone="warn"
            />
            <SummaryTile
              label="missing"
              value={data.summary.missing}
              tone="danger"
            />
          </div>
          {!data.writable && (
            <div className="mb-3 rounded border border-border bg-surface p-2 text-xs text-slate-400">
              Devin session: storage endpoint is read-only. Use the{" "}
              <code>secrets</code> tool to provision keys.
            </div>
          )}
          <table className="w-full text-left">
            <thead className="text-xs uppercase text-slate-500">
              <tr>
                <th className="px-3 py-2">id</th>
                <th className="px-3 py-2">source</th>
                <th className="px-3 py-2">env vars</th>
                <th className="px-3 py-2">state</th>
                <th className="px-3 py-2">signup</th>
              </tr>
            </thead>
            <tbody>
              {data.items.map((item) => (
                <CredentialRow key={item.source_id} item={item} />
              ))}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}

function SummaryTile({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone?: "ok" | "warn" | "danger";
}) {
  const toneCls =
    tone === "ok"
      ? "text-ok"
      : tone === "warn"
        ? "text-warn"
        : tone === "danger"
          ? "text-danger"
          : "text-slate-200";
  return (
    <div className="rounded border border-border bg-surface px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">
        {label}
      </div>
      <div className={`text-lg ${toneCls}`}>{value}</div>
    </div>
  );
}
