/* service-worker.js \u2014 offline shell only.
 * Caches the static SPA shell so the app opens even offline; all /api/*
 * calls always go to the network (no stale trading data). */
const CACHE = "dix-vision-shell-v1";
const SHELL = [
  "/",
  "/static/app.js",
  "/static/i18n.json",
  "/static/manifest.webmanifest",
  "/static/icon-192.png",
  "/static/icon-512.png",
];

self.addEventListener("install", (ev) => {
  ev.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL).catch(() => {})));
  self.skipWaiting();
});

self.addEventListener("activate", (ev) => {
  ev.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))),
    ),
  );
  self.clients.claim();
});

self.addEventListener("fetch", (ev) => {
  const url = new URL(ev.request.url);
  if (url.pathname.startsWith("/api/") || url.pathname === "/health") {
    ev.respondWith(fetch(ev.request));
    return;
  }
  ev.respondWith(
    caches.match(ev.request).then(r =>
      r || fetch(ev.request).then(resp => {
        if (resp.ok && ev.request.method === "GET") {
          const clone = resp.clone();
          caches.open(CACHE).then(c => c.put(ev.request, clone)).catch(() => {});
        }
        return resp;
      }).catch(() => caches.match("/"))),
  );
});
