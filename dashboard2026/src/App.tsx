import { LiveStatusPill } from "@/components/LiveStatusPill";
import { CognitiveChatPage } from "@/pages/CognitiveChatPage";
import { CredentialsPage } from "@/pages/CredentialsPage";
import { OperatorPage } from "@/pages/OperatorPage";
import { useHashRoute, type Route } from "@/router";

const TABS: Array<{ key: Route; href: string; label: string }> = [
  { key: "credentials", href: "#/credentials", label: "credentials" },
  { key: "operator", href: "#/operator", label: "operator" },
  { key: "chat", href: "#/chat", label: "chat" },
];

export function App() {
  const route = useHashRoute();
  return (
    <div className="min-h-full flex flex-col">
      <header className="border-b border-border bg-surface px-6 py-3">
        <div className="flex items-baseline gap-3">
          <span className="text-lg font-semibold tracking-tight">
            DIX VISION
          </span>
          <span className="text-xs text-slate-400 font-mono">
            wave-02 · operator console
          </span>
          <span className="ml-auto self-center">
            <LiveStatusPill />
          </span>
        </div>
        <nav className="mt-2 flex gap-2 font-mono text-xs">
          {TABS.map((tab) => {
            const isActive = tab.key === route;
            return (
              <a
                key={tab.key}
                href={tab.href}
                className={
                  isActive
                    ? "rounded border border-accent bg-accent/10 px-2 py-1 text-accent"
                    : "rounded border border-border bg-bg px-2 py-1 text-slate-400 hover:border-accent hover:text-accent"
                }
              >
                {tab.label}
              </a>
            );
          })}
        </nav>
      </header>
      <main className="flex-1 px-6 py-6">
        {route === "operator" ? (
          <OperatorPage />
        ) : route === "chat" ? (
          <CognitiveChatPage />
        ) : (
          <CredentialsPage />
        )}
      </main>
      <footer className="border-t border-border bg-surface px-6 py-2 text-xs text-slate-500">
        Vanilla pages remain the canonical reference at{" "}
        <a className="text-accent hover:underline" href="/credentials">
          /credentials
        </a>
        ,{" "}
        <a className="text-accent hover:underline" href="/operator">
          /operator
        </a>
        , and friends. This wave-02 surface is additive.
      </footer>
    </div>
  );
}
