import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import {
  fetchPlugins,
  setPluginLifecycle,
  type PluginRecord,
} from "@/api/plugins";

const LIFECYCLE_TONE: Record<string, string> = {
  ACTIVE: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
  SHADOW: "border-amber-500/40 bg-amber-500/10 text-amber-300",
  DISABLED: "border-slate-600/60 bg-slate-700/30 text-slate-400",
};

function LifecycleBadge({ lifecycle }: { lifecycle: string }) {
  const tone = LIFECYCLE_TONE[lifecycle] ?? LIFECYCLE_TONE.DISABLED;
  return (
    <span
      className={`inline-flex items-center rounded border px-2 py-0.5 text-[11px] font-mono uppercase tracking-wide ${tone}`}
      data-testid={`plugin-lifecycle-${lifecycle}`}
    >
      {lifecycle}
    </span>
  );
}

function CategoryPill({ category }: { category: string }) {
  return (
    <span className="rounded border border-border bg-bg/40 px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-slate-500">
      {category}
    </span>
  );
}

function PluginRow({
  plugin,
  onChange,
  pendingLifecycle,
}: {
  plugin: PluginRecord;
  onChange: (lifecycle: string) => void;
  pendingLifecycle: string | null;
}) {
  const current = pendingLifecycle ?? plugin.lifecycle;
  return (
    <tr
      className="border-t border-border align-top"
      data-testid={`plugin-row-${plugin.id}`}
    >
      <td className="px-3 py-2 font-mono text-xs text-slate-300 whitespace-nowrap">
        {plugin.id}
      </td>
      <td className="px-3 py-2">
        <div className="flex items-center gap-2">
          <CategoryPill category={plugin.category} />
          <span className="font-mono text-[10px] text-slate-500">
            v{plugin.version}
          </span>
        </div>
        <p className="mt-1 max-w-prose text-xs leading-snug text-slate-400">
          {plugin.description}
        </p>
      </td>
      <td className="px-3 py-2">
        <LifecycleBadge lifecycle={current} />
      </td>
      <td className="px-3 py-2">
        <div className="flex flex-wrap gap-1.5">
          {plugin.lifecycle_options.map((opt) => {
            const active = opt === current;
            return (
              <button
                key={opt}
                type="button"
                onClick={() => {
                  if (!active) onChange(opt);
                }}
                disabled={active || pendingLifecycle !== null}
                className={`rounded border px-2 py-1 text-[11px] font-mono uppercase tracking-wide transition-colors ${
                  active
                    ? "border-accent bg-accent/10 text-accent"
                    : "border-border bg-bg/40 text-slate-300 hover:border-accent/60 hover:text-accent disabled:opacity-50"
                }`}
                data-testid={`plugin-toggle-${plugin.id}-${opt}`}
              >
                {opt}
              </button>
            );
          })}
        </div>
        <div className="mt-1 text-[10px] text-slate-500 font-mono">
          ledger: {plugin.ledger_kind}
        </div>
      </td>
    </tr>
  );
}

export function PluginsPage() {
  const queryClient = useQueryClient();
  const { data, isPending, isError, error, refetch, isFetching } = useQuery({
    queryKey: ["plugins", "list"],
    queryFn: ({ signal }) => fetchPlugins(signal),
    refetchInterval: 5_000,
  });

  const [pending, setPending] = useState<Record<string, string>>({});
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const mutation = useMutation({
    mutationFn: ({
      pluginId,
      lifecycle,
    }: {
      pluginId: string;
      lifecycle: string;
    }) => setPluginLifecycle(pluginId, lifecycle),
    onMutate: ({ pluginId, lifecycle }) => {
      setErrorMsg(null);
      setPending((p) => ({ ...p, [pluginId]: lifecycle }));
    },
    onSettled: (_data, _err, vars) => {
      setPending((p) => {
        const next = { ...p };
        delete next[vars.pluginId];
        return next;
      });
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["plugins", "list"] });
    },
    onError: (err: unknown) => {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    },
  });

  return (
    <section className="max-w-6xl mx-auto">
      <div className="flex items-end justify-between mb-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">
            Plugin manager
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Toggle every hot-swappable plugin in the runtime without
            restarting. Microstructure plugins carry the tri-state
            <code className="mx-1 text-xs">DISABLED / SHADOW / ACTIVE</code>
            lifecycle; SHADOW emits signals that the Execution Engine
            refuses to fill (so you can shadow-trade a strategy before
            promoting it). Cognitive chat is binary — DISABLED returns
            503 from <code className="text-xs">/api/cognitive/chat/*</code>{" "}
            and the chat page; ACTIVE serves it. Every successful flip
            writes a <code className="text-xs">PLUGIN_LIFECYCLE</code>{" "}
            row to the authority ledger.
          </p>
        </div>
        <button
          type="button"
          onClick={() => void refetch()}
          className="rounded border border-border bg-bg/40 px-3 py-1 text-xs text-slate-300 hover:border-accent hover:text-accent"
          disabled={isFetching}
        >
          {isFetching ? "refreshing…" : "refresh"}
        </button>
      </div>

      {errorMsg && (
        <div
          role="alert"
          className="mb-3 rounded border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-xs text-rose-300"
        >
          {errorMsg}
        </div>
      )}

      {isPending && (
        <div className="rounded border border-border bg-surface px-3 py-2 text-xs text-slate-400">
          Loading plugins…
        </div>
      )}

      {isError && (
        <div className="rounded border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-xs text-rose-300">
          Failed to load plugins: {error instanceof Error ? error.message : String(error)}
        </div>
      )}

      {data && (
        <div className="overflow-hidden rounded border border-border bg-surface">
          <table className="w-full text-left">
            <thead className="bg-bg/40 text-[10px] uppercase tracking-wider text-slate-500">
              <tr>
                <th className="px-3 py-2 font-semibold">id</th>
                <th className="px-3 py-2 font-semibold">plugin</th>
                <th className="px-3 py-2 font-semibold">current</th>
                <th className="px-3 py-2 font-semibold">switch</th>
              </tr>
            </thead>
            <tbody>
              {data.plugins.map((p) => (
                <PluginRow
                  key={p.id}
                  plugin={p}
                  pendingLifecycle={pending[p.id] ?? null}
                  onChange={(lifecycle) =>
                    mutation.mutate({ pluginId: p.id, lifecycle })
                  }
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
