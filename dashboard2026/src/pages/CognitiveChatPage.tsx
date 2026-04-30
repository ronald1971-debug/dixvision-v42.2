import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";

import { fetchChatStatus, postChatTurn } from "@/api/cognitive_chat";
import { ApprovalPanel } from "@/components/ApprovalPanel";
import type {
  ChatMessageApi,
  ChatTurnResponse,
} from "@/types/generated/api";

function newThreadId(): string {
  // RFC4122-ish without dashes; 16 random bytes is plenty for the
  // checkpoint scope. crypto.randomUUID is widely available in
  // evergreen browsers and Node 18+.
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID().replace(/-/g, "");
  }
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export function CognitiveChatPage() {
  const status = useQuery({
    queryKey: ["cognitive", "chat", "status"],
    queryFn: ({ signal }) => fetchChatStatus(signal),
    refetchInterval: 10_000,
  });

  const [threadId, setThreadId] = useState<string>(() => newThreadId());
  const [messages, setMessages] = useState<ChatMessageApi[]>([]);
  const [input, setInput] = useState("");
  const [errorDetail, setErrorDetail] = useState<string | null>(null);
  const [lastProposalId, setLastProposalId] = useState<string>("");
  const transcriptRef = useRef<HTMLDivElement | null>(null);
  const queryClient = useQueryClient();

  const turn = useMutation({
    mutationFn: (next: ChatMessageApi[]) =>
      postChatTurn({ thread_id: threadId, messages: next }),
    onSuccess: (resp: ChatTurnResponse) => {
      setMessages((prev) => [...prev, resp.reply]);
      setErrorDetail(null);
      const nextProposalId = resp.proposal_id ?? "";
      setLastProposalId(nextProposalId);
      if (nextProposalId !== "") {
        // The chat reply just queued a proposal — kick the approval-
        // panel cache so it refreshes immediately rather than waiting
        // for the 2s poll tick.
        void queryClient.invalidateQueries({
          queryKey: ["cognitive", "chat", "approvals"],
        });
      }
    },
    onError: (err: Error) => {
      setErrorDetail(err.message);
    },
  });

  useEffect(() => {
    transcriptRef.current?.scrollTo({
      top: transcriptRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages.length, turn.isPending]);

  const lastProvider = useMemo(() => {
    const data = turn.data;
    return data?.provider_id ?? "";
  }, [turn.data]);

  function send() {
    const trimmed = input.trim();
    if (!trimmed || turn.isPending) return;
    if (!status.data?.enabled) return;
    const userMsg: ChatMessageApi = { role: "user", content: trimmed };
    // Only the *new* user message goes on the wire — the server
    // replays prior turns from the LangGraph checkpoint keyed by
    // ``thread_id``. Sending the full local transcript would
    // double-append every prior message via the ``add_messages``
    // reducer (Devin Review BUG_0001 on PR #85).
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    turn.mutate([userMsg]);
  }

  function reset() {
    setThreadId(newThreadId());
    setMessages([]);
    setErrorDetail(null);
    // Clear the highlight pointer too — otherwise the ApprovalPanel
    // keeps highlighting a row from the previous thread until the
    // operator submits a new proposal in the fresh thread.
    setLastProposalId("");
  }

  const banner = (() => {
    if (status.isPending) {
      return (
        <p className="text-xs text-slate-400 font-mono">checking status…</p>
      );
    }
    if (status.isError) {
      return (
        <p className="text-xs text-red-400 font-mono">
          status check failed: {(status.error as Error).message}
        </p>
      );
    }
    const data = status.data;
    if (!data) return null;
    if (!data.enabled) {
      return (
        <p className="text-xs text-amber-300 font-mono">
          cognitive chat is OFF — set{" "}
          <code className="text-amber-200">{data.feature_flag_env_var}</code>{" "}
          to a truthy value (1 / true / yes / on) and restart the server
          to enable.
        </p>
      );
    }
    if (data.eligible_providers.length === 0) {
      return (
        <p className="text-xs text-amber-300 font-mono">
          flag is ON but no eligible providers are registered for
          INDIRA_REASONING — every turn will return 502.
        </p>
      );
    }
    return (
      <p className="text-xs text-slate-400 font-mono">
        cognitive chat is ON · {data.eligible_providers.length} eligible
        provider(s):{" "}
        <span className="text-accent">
          {data.eligible_providers.join(", ")}
        </span>
      </p>
    );
  })();

  return (
    <section className="max-w-4xl mx-auto space-y-4">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">
            Cognitive chat
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Wave-03 PR-5 — operator-approval edge.  Chat replies that
            contain a structured ``propose`` block queue an approval
            request that the operator must approve before any
            ``SignalEvent`` reaches the bus.  Approve / reject from the
            panel below.
          </p>
        </div>
        <button
          type="button"
          onClick={reset}
          className="rounded border border-border bg-surface px-3 py-1.5 text-xs hover:border-accent disabled:opacity-50"
          disabled={turn.isPending}
        >
          new thread
        </button>
      </div>

      {banner}

      <div
        ref={transcriptRef}
        className="rounded border border-border bg-surface min-h-[260px] max-h-[480px] overflow-y-auto p-3 space-y-2 font-mono text-sm"
        data-testid="chat-transcript"
      >
        {messages.length === 0 && !turn.isPending ? (
          <p className="text-slate-500 italic">
            no messages yet — say something below.
          </p>
        ) : null}
        {messages.map((m, i) => (
          <div
            key={`${m.role}-${i}`}
            className={
              m.role === "user"
                ? "text-slate-200"
                : m.role === "assistant"
                  ? "text-accent"
                  : "text-amber-300"
            }
          >
            <span className="text-slate-500 mr-2">{m.role}:</span>
            <span className="whitespace-pre-wrap">{m.content}</span>
          </div>
        ))}
        {turn.isPending ? (
          <div className="text-slate-500 italic">assistant is thinking…</div>
        ) : null}
      </div>

      {errorDetail ? (
        <p
          className="text-xs text-red-400 font-mono"
          data-testid="chat-error"
        >
          {errorDetail}
        </p>
      ) : null}

      <form
        onSubmit={(ev) => {
          ev.preventDefault();
          send();
        }}
        className="flex gap-2"
      >
        <input
          type="text"
          className="flex-1 rounded border border-border bg-bg px-3 py-2 text-sm font-mono outline-none focus:border-accent disabled:opacity-50"
          placeholder={
            status.data?.enabled ? "type a message…" : "feature disabled"
          }
          value={input}
          onChange={(ev) => setInput(ev.target.value)}
          disabled={!status.data?.enabled || turn.isPending}
          data-testid="chat-input"
        />
        <button
          type="submit"
          className="rounded border border-accent bg-accent/10 px-4 py-2 text-xs text-accent hover:bg-accent/20 disabled:opacity-50"
          disabled={
            !status.data?.enabled || turn.isPending || input.trim() === ""
          }
          data-testid="chat-send"
        >
          {turn.isPending ? "sending…" : "send"}
        </button>
      </form>

      <div className="flex justify-between text-xs text-slate-500 font-mono">
        <span>
          thread: <span className="text-slate-300">{threadId}</span>
        </span>
        <span>
          {lastProvider ? (
            <>
              last provider:{" "}
              <span className="text-slate-300">{lastProvider}</span>
            </>
          ) : null}
        </span>
      </div>

      <ApprovalPanel highlightProposalId={lastProposalId} />
    </section>
  );
}
