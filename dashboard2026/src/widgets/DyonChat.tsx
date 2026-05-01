import { useState } from "react";

import { apiUrl } from "@/api/base";

/**
 * Dyon chat (Widget #8 — Coding & Configuration) per PR-#2 spec §0.
 *
 * Interface for assigning tasks to Dyon (the self-coding nucleus).
 * Examples:
 *   - "Add new venue adapter for ByBit"
 *   - "Update API keys for Helius"
 *   - "Create custom strategy plugin from this Pine script"
 *   - "Modify risk rule for memecoin sniper"
 *
 * Every patch lands behind the sandbox patch pipeline (authority_lint
 * + tests + dep scan + shadow + canary) and a two-person operator
 * click for any change touching the fast path.
 */
interface PatchMessage {
  id: string;
  role: "operator" | "dyon" | "sandbox";
  text: string;
  ts_iso: string;
  patch_id?: string;
}

const SUGGESTIONS = [
  "Add a Hyperliquid HIP-3 builder adapter",
  "Update Helius API key (rotate)",
  "Wrap this Pine script as a sandbox strategy",
  "Tighten the rug-trip SL threshold to 25%",
];

export function DyonChat() {
  const [history, setHistory] = useState<PatchMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);

  async function send(text: string) {
    const trimmed = text.trim();
    if (!trimmed || pending) return;
    const op: PatchMessage = {
      id: `op-${Date.now()}`,
      role: "operator",
      text: trimmed,
      ts_iso: new Date().toISOString(),
    };
    setHistory((h) => [...h, op]);
    setDraft("");
    setPending(true);
    try {
      const res = await fetch(apiUrl("/api/cognitive/chat/turn"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role: "dyon", text: trimmed }),
      });
      const body = (await res.json()) as {
        text?: string;
        patch_id?: string;
        sandbox_state?: string;
      };
      setHistory((h) => [
        ...h,
        {
          id: `dy-${Date.now()}`,
          role: "dyon",
          text:
            body.text ??
            "(no response — backend may not be wired; the sandbox pipeline still owns merge gating)",
          ts_iso: new Date().toISOString(),
          patch_id: body.patch_id,
        },
      ]);
      if (body.sandbox_state) {
        setHistory((h) => [
          ...h,
          {
            id: `sb-${Date.now()}`,
            role: "sandbox",
            text: `Sandbox: ${body.sandbox_state}`,
            ts_iso: new Date().toISOString(),
            patch_id: body.patch_id,
          },
        ]);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setHistory((h) => [
        ...h,
        {
          id: `err-${Date.now()}`,
          role: "dyon",
          text: `(network error — ${msg})`,
          ts_iso: new Date().toISOString(),
        },
      ]);
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="flex h-full flex-col rounded border border-border bg-surface text-sm">
      <header className="flex items-baseline justify-between border-b border-border px-3 py-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Dyon · Coding &amp; Configuration
          </h3>
          <p className="mt-0.5 text-[11px] text-slate-500">
            #8 · sandbox patch pipeline · two-person gate for fast-path
          </p>
        </div>
        <span className="rounded border border-emerald-500/40 bg-emerald-500/10 px-1.5 py-0.5 font-mono text-[10px] text-emerald-300">
          PR-#2 §0
        </span>
      </header>
      <div className="flex-1 space-y-2 overflow-auto p-3">
        {history.length === 0 && (
          <div className="grid h-full place-items-center text-center text-xs text-slate-500">
            <div className="space-y-2">
              <div className="font-mono uppercase tracking-wider">
                ask Dyon
              </div>
              <div className="flex flex-col gap-1">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => send(s)}
                    className="rounded border border-border bg-bg px-2 py-1 text-left text-[11px] text-slate-300 hover:text-emerald-300"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
        {history.map((m) => (
          <PatchBubble key={m.id} message={m} />
        ))}
      </div>
      <form
        className="flex items-center gap-1 border-t border-border p-2"
        onSubmit={(e) => {
          e.preventDefault();
          void send(draft);
        }}
      >
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Describe a code/config patch for Dyon…"
          className="flex-1 rounded border border-border bg-bg px-2 py-1 text-[12px] focus:border-emerald-500 focus:outline-none"
        />
        <button
          type="submit"
          disabled={pending}
          className="rounded border border-emerald-500/60 bg-emerald-500/15 px-3 py-1 font-mono text-[11px] uppercase tracking-wider text-emerald-300 disabled:opacity-50"
        >
          patch
        </button>
      </form>
    </div>
  );
}

function PatchBubble({ message }: { message: PatchMessage }) {
  const tone =
    message.role === "operator"
      ? "border-accent/40 bg-accent/10 text-slate-100"
      : message.role === "dyon"
        ? "border-emerald-500/40 bg-emerald-500/10 text-slate-100"
        : "border-amber-500/40 bg-amber-500/10 text-amber-200";
  return (
    <div className={`rounded border px-2 py-1.5 ${tone}`}>
      <div className="flex items-baseline justify-between font-mono text-[10px] uppercase tracking-wider opacity-70">
        <span>{message.role}</span>
        <span>
          {new Date(message.ts_iso).toLocaleTimeString()}
          {message.patch_id ? ` · patch ${message.patch_id.slice(0, 8)}` : ""}
        </span>
      </div>
      <div className="whitespace-pre-wrap text-[12px]">{message.text}</div>
    </div>
  );
}
