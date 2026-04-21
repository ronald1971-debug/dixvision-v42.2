# Mobile dashboard (iPhone + Android)

The mobile dashboard is **the same cockpit SPA** -- identical HTML/JS,
identical endpoints, same charters, same safety gates. No feature cut.

There are two install paths: a PWA today (zero friction, works on every
modern phone), and native Capacitor wrappers later (push notifications,
biometric gates, secure-enclave key storage).

---

## Path A -- PWA (Progressive Web App)

Works on iOS 16+ Safari and Android 8+ Chromium. No app store required.

1. On your phone, open the cockpit URL (see "Connection modes" below).
2. **iOS**: Safari -> Share -> **Add to Home Screen**.
3. **Android**: Chromium pops a native "Install app" banner; tap Install.
4. The app opens full-screen, same layout as desktop (mobile nav at top,
   same panels).

What the PWA supports today:
* Full chat with INDIRA / DYON / GOVERNANCE / DEVIN voices (auto-detects
  the phone's language).
* Live status + ledger + strategy + wallet + safety panels.
* Offline shell: if the cockpit is unreachable, cached HTML/CSS/JS loads
  immediately; `/api/*` always goes to network (never stale trade data).
* Heartbeat -- the app pings `/api/dead-man/heartbeat` every 30 s; if you
  force-quit the app while a live trade is open, the dead-man switch
  trips automatically and INDIRA pauses.

Pairing: open `/pair` on the phone, scan the QR shown in the desktop
cockpit (or paste the token). The phone stores a bearer token in
`localStorage` and redirects to the full cockpit URL.

---

## Path B -- Native wrappers (deferred)

Capacitor wraps the same HTML/JS codebase into:

* `mobile/ios/` -> Xcode project -> `.ipa` (TestFlight / App Store).
* `mobile/android/` -> Gradle project -> `.apk` / `.aab` (Play Store).

Native features we will use on top of the PWA:

* Push notifications for hazards (wallet budget hit, kill-switch armed,
  dead-man tripped).
* Biometric unlock (Face ID / fingerprint) gating "approve live signing"
  and "raise daily cap".
* Secrets in iOS Keychain / Android StrongBox -- never in localStorage.

These land in a later PR. The PWA path covers everything else today.

---

## Connection modes

The cockpit binds to loopback by default. Pick one of three ways for your
phone to reach it:

### 1. LAN mode (simplest at home)

On the desktop, set `DIX_BIND_HOST=0.0.0.0` and open
`http://<desktop-LAN-ip>:8765/` on the phone. Only use this on a trusted
network; the bearer token still guards every `/api/*` call.

### 2. Tailscale / WireGuard (recommended anywhere)

Install Tailscale on the desktop and the phone, both join the same
tailnet, then open `http://<desktop-tailscale-ip>:8765/` on the phone.
Your cockpit is reachable from anywhere with no inbound ports opened to
the public internet.

### 3. Reverse tunnel (Cloudflare Tunnel / ngrok)

Run `cloudflared tunnel --url http://127.0.0.1:8765` on the desktop to
get a public HTTPS URL. Convenient for travel; highest attack surface so
it is off by default. Rotate the cockpit token after every trip.

### 4. Cloud deploy (fourth option, zero LAN setup)

If the cockpit runs on Fly/Render/Railway/your VPS per
[CLOUD.md](CLOUD.md), the phone just opens the public HTTPS URL -- no
LAN, no Tailscale, no tunnel.

---

## Security on the phone

* Pairing tokens are one-time, 15 min TTL, revokable per-device from the
  cockpit Phone tab.
* Each successful pairing emits `SECURITY/PAIRING_CLAIMED` with the
  device user-agent -- audit trail if a device is lost.
* Biometric gate (native path) required for live signing approval + cap
  changes.
* The phone never sees raw wallet key material. All signing decisions
  flow through INDIRA's fast path on the host; the phone only *approves*
  or *rejects*.
