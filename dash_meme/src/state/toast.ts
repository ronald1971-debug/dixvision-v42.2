import { useEffect, useState } from "react";

export type ToastTone = "ok" | "warn" | "danger" | "info";

export type Toast = {
  id: number;
  tone: ToastTone;
  message: string;
  hint?: string;
  ts: number;
};

const listeners = new Set<(toasts: ReadonlyArray<Toast>) => void>();
let toasts: Toast[] = [];
let nextId = 1;

const TOAST_TTL_MS = 4_500;

export function pushToast(
  message: string,
  opts: { tone?: ToastTone; hint?: string } = {},
): number {
  const id = nextId++;
  const t: Toast = {
    id,
    tone: opts.tone ?? "info",
    message,
    hint: opts.hint,
    ts: Date.now(),
  };
  toasts = [...toasts, t];
  notify();
  setTimeout(() => dismissToast(id), TOAST_TTL_MS);
  return id;
}

export function dismissToast(id: number) {
  const before = toasts.length;
  toasts = toasts.filter((t) => t.id !== id);
  if (toasts.length !== before) notify();
}

function notify() {
  const snapshot = [...toasts];
  listeners.forEach((fn) => fn(snapshot));
}

export function useToasts(): ReadonlyArray<Toast> {
  const [snap, setSnap] = useState<ReadonlyArray<Toast>>(toasts);
  useEffect(() => {
    const fn = (t: ReadonlyArray<Toast>) => setSnap(t);
    listeners.add(fn);
    return () => {
      listeners.delete(fn);
    };
  }, []);
  return snap;
}
