import { dismissToast, useToasts } from "@/state/toast";

const TONE_CLASS: Record<string, string> = {
  ok: "border-ok text-ok",
  warn: "border-warn text-warn",
  danger: "border-danger text-danger",
  info: "border-accent text-accent",
};

export function ToastHost() {
  const toasts = useToasts();
  if (toasts.length === 0) return null;
  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex flex-col gap-2">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={`pointer-events-auto min-w-[260px] max-w-[420px] rounded border bg-surface-overlay px-3 py-2 text-xs shadow-lg ${
            TONE_CLASS[t.tone] ?? TONE_CLASS.info
          }`}
          onClick={() => dismissToast(t.id)}
        >
          <div className="font-medium">{t.message}</div>
          {t.hint && (
            <div className="mt-1 text-[11px] text-text-secondary">{t.hint}</div>
          )}
        </div>
      ))}
    </div>
  );
}
