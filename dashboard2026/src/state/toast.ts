import { useSyncExternalStore } from "react";

/**
 * J-track toast store — small global notification queue.
 *
 * Replaces ad-hoc ``alert()`` / ``console.log`` calls scattered across
 * the cockpit. Toasts are addressed by id so a caller can dismiss
 * its own toast before the auto-expire kicks in. The store is a
 * plain external store so any component can push without context.
 */
export type ToastTone = "info" | "success" | "warn" | "danger";

export interface Toast {
  id: string;
  tone: ToastTone;
  message: string;
  hint?: string;
  expires_at: number;
}

const DEFAULT_TTL_MS = 4500;

let queue: Toast[] = [];
const listeners = new Set<() => void>();
let counter = 0;

function emit() {
  for (const fn of listeners) fn();
}

function subscribe(fn: () => void): () => void {
  listeners.add(fn);
  return () => {
    listeners.delete(fn);
  };
}

function snapshot(): readonly Toast[] {
  return queue;
}

export function pushToast(
  message: string,
  opts: { tone?: ToastTone; hint?: string; ttl_ms?: number } = {},
): string {
  counter += 1;
  const id = `toast-${Date.now().toString(36)}-${counter}`;
  const ttl = opts.ttl_ms ?? DEFAULT_TTL_MS;
  const t: Toast = {
    id,
    tone: opts.tone ?? "info",
    message,
    hint: opts.hint,
    expires_at: Date.now() + ttl,
  };
  queue = [...queue, t];
  emit();
  if (typeof window !== "undefined") {
    window.setTimeout(() => dismissToast(id), ttl);
  }
  return id;
}

export function dismissToast(id: string) {
  const before = queue.length;
  queue = queue.filter((t) => t.id !== id);
  if (queue.length !== before) emit();
}

export function useToasts(): readonly Toast[] {
  return useSyncExternalStore(subscribe, snapshot, snapshot);
}
