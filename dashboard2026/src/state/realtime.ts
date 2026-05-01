import { useEffect, useRef, useState } from "react";

import { apiUrl } from "@/api/base";

/**
 * Real-time bridge.
 *
 * The canonical event bus is exposed over Server-Sent Events at
 * `/api/dashboard/stream`. Every widget that wants live updates
 * subscribes through `useEventStream(channel)` instead of polling.
 *
 * If the SSE endpoint is not reachable (older backend, dev box, no
 * provider keys), the bridge falls back to a deterministic mock
 * generator so the dashboard still renders end-to-end.
 */
export interface StreamEvent<T = unknown> {
  channel: string;
  ts_iso: string;
  payload: T;
}

type Listener<T> = (event: StreamEvent<T>) => void;

const channelListeners = new Map<string, Set<Listener<unknown>>>();
let source: EventSource | null = null;
let mockTimer: ReturnType<typeof setInterval> | null = null;
let connectionState: "idle" | "live" | "mock" | "error" = "idle";
const stateListeners = new Set<(state: typeof connectionState) => void>();

function ensureConnected() {
  if (source || mockTimer) return;
  if (typeof window === "undefined" || typeof EventSource === "undefined") {
    startMock();
    return;
  }
  try {
    source = new EventSource(apiUrl("/api/dashboard/stream"));
    source.onopen = () => setConnectionState("live");
    source.onerror = () => {
      setConnectionState("error");
      // Drop SSE and fall back to mock so widgets keep rendering.
      source?.close();
      source = null;
      startMock();
    };
    source.onmessage = (e) => {
      try {
        const parsed = JSON.parse(e.data) as StreamEvent;
        dispatch(parsed);
      } catch {
        // ignore malformed payload
      }
    };
  } catch {
    startMock();
  }
}

function startMock() {
  if (mockTimer) return;
  setConnectionState("mock");
  let tick = 0;
  mockTimer = setInterval(() => {
    tick += 1;
    const now = new Date().toISOString();
    const trade = {
      side: tick % 2 === 0 ? "BUY" : "SELL",
      price: 100 + Math.sin(tick / 7) * 2,
      size: Math.round(50 + Math.random() * 200),
      venue: ["binance", "hyperliquid", "drift"][tick % 3],
    };
    dispatch({ channel: "ticks", ts_iso: now, payload: trade });
    dispatch({
      channel: "depth",
      ts_iso: now,
      payload: generateDepth(100 + Math.sin(tick / 7) * 2),
    });
    if (tick % 5 === 0) {
      dispatch({
        channel: "news",
        ts_iso: now,
        payload: {
          source: "coindesk",
          title: `Mock headline #${tick}`,
          sentiment: Math.sin(tick / 11) * 0.6,
        },
      });
    }
  }, 800);
}

function dispatch(event: StreamEvent) {
  const listeners = channelListeners.get(event.channel);
  if (!listeners) return;
  for (const listener of listeners) {
    try {
      listener(event);
    } catch {
      // never let one widget crash break the bus
    }
  }
}

function setConnectionState(state: typeof connectionState) {
  if (state === connectionState) return;
  connectionState = state;
  for (const listener of stateListeners) listener(state);
}

function generateDepth(mid: number) {
  const bids = Array.from({ length: 12 }, (_, i) => ({
    price: mid - (i + 1) * 0.05,
    size: Math.round(100 + Math.random() * 1000),
  }));
  const asks = Array.from({ length: 12 }, (_, i) => ({
    price: mid + (i + 1) * 0.05,
    size: Math.round(100 + Math.random() * 1000),
  }));
  return { bids, asks, mid };
}

export function useEventStream<T = unknown>(
  channel: string,
  initial: T[] = [],
  cap = 200,
): T[] {
  const [events, setEvents] = useState<T[]>(initial);
  const capRef = useRef(cap);
  capRef.current = cap;
  useEffect(() => {
    ensureConnected();
    const listener: Listener<unknown> = (event) => {
      setEvents((prev) => {
        const next = [...prev, event.payload as T];
        return next.length > capRef.current
          ? next.slice(next.length - capRef.current)
          : next;
      });
    };
    let listeners = channelListeners.get(channel);
    if (!listeners) {
      listeners = new Set();
      channelListeners.set(channel, listeners);
    }
    listeners.add(listener as Listener<unknown>);
    return () => {
      listeners?.delete(listener as Listener<unknown>);
    };
  }, [channel]);
  return events;
}

export function useLatestEvent<T = unknown>(channel: string): T | null {
  const events = useEventStream<T>(channel, [], 1);
  return events.length > 0 ? events[events.length - 1] : null;
}

export function useStreamState(): typeof connectionState {
  const [state, setState] = useState(connectionState);
  useEffect(() => {
    const listener = (s: typeof connectionState) => setState(s);
    stateListeners.add(listener);
    ensureConnected();
    return () => {
      stateListeners.delete(listener);
    };
  }, []);
  return state;
}
