import { CredentialsPage } from "@/pages/CredentialsPage";

export function App() {
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
        </div>
      </header>
      <main className="flex-1 px-6 py-6">
        <CredentialsPage />
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
