# Installing DIX VISION v42.2

Three installation paths, in order from simplest to most flexible. Pick
whichever matches your machine.

---

## 1. Portable `DIX-VISION.exe` (Windows, zero-install)

The simplest path: **one file, double-click, no admin, no Python, no install
wizard**.

1. Grab `DIX-VISION.exe` from the [latest release][releases] (~35 MB).
2. Double-click it.
3. Your browser opens on `http://127.0.0.1:8765/` with the cockpit already
   authenticated.

That's the whole installation. On first run it creates
`%LOCALAPPDATA%\DIX VISION\data\` to hold:

* `cockpit_token.txt` (per-device bearer token)
* `ledger.sqlite` (event-sourced audit log)
* `wallet_policy.sqlite` (birth-clock + $100/day cap state)
* `pairing.sqlite` (phone pairing tokens)

**Uninstall** = delete `%LOCALAPPDATA%\DIX VISION\` and the `.exe`. No
registry, no services, no shortcuts, no traces.

**Keep it running 24/7** (optional): drop a shortcut to `DIX-VISION.exe` into
`shell:startup` (paste that into the Windows Run dialog to open the Startup
folder). It will launch silently every login.

[releases]: https://github.com/your-org/dix-vision/releases/latest

---

## 2. Cloud one-click deploy (zero local footprint)

Runs on someone else's machine, 24/7. You bookmark a URL and you're done.
Phone and laptop both use the same URL.

### Fly.io (~$5/month, recommended)

```bash
# One-time CLI install
curl -L https://fly.io/install.sh | sh

# From the repo root:
fly launch --copy-config --name dix-vision-$(whoami)
fly volumes create dix_data --size 1
fly deploy
```

Your cockpit is live at `https://dix-vision-<you>.fly.dev/`. Print the
pairing QR in the cockpit "Phone" tab, scan from iPhone or Android.

### Render.com (free tier possible for the worker, starter tier for web)

Push this repo, then click "New Blueprint" and point at
[`cloud/render.yaml`](../cloud/render.yaml). Web service + worker service
are provisioned automatically.

### Railway

Push this repo, then "New Project > Deploy from GitHub", Railway detects
[`cloud/railway.json`](../cloud/railway.json). Storage persists on a default
volume.

### Self-hosted (docker-compose, any VPS, ~$4/month)

```bash
git clone https://github.com/your-org/dix-vision
cd dix-vision
cp .env.example .env
export DIX_HOSTNAME=cockpit.yourdomain.tld   # points your DNS A record here
docker compose --profile tls up -d
```

Caddy provisions Let's Encrypt TLS automatically. Cockpit + worker + reverse
proxy all running.

### Kubernetes

```bash
kubectl apply -f cloud/k8s/deployment.yaml
```

---

## 3. From source (developer path)

```bash
git clone https://github.com/your-org/dix-vision
cd dix-vision
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m cockpit --mode desktop
```

The cockpit launches on `http://127.0.0.1:8765`. Token is auto-generated at
`data/cockpit_token.txt`.

### Run modes

| Mode      | Bind        | HTTP? | Purpose                                |
|-----------|-------------|-------|----------------------------------------|
| `desktop` | 127.0.0.1   | yes   | Single operator on one machine         |
| `cloud`   | 0.0.0.0     | yes   | Publicly reachable via TLS proxy       |
| `worker`  | (no socket) | no    | Headless 24/7 ingest / learning only   |

```bash
python -m cockpit --mode worker            # just learn/source, no HTTP
python -m cockpit --mode cloud             # bind 0.0.0.0, serve cockpit
DIX_MODE=cloud python -m cockpit           # same, via env
```

---

## Pairing a phone

Once the cockpit is running:

1. Open **Phone** tab in the cockpit (or POST to `/api/pair/new`).
2. A QR code appears.
3. On the phone: open Safari/Chrome, scan the QR.
4. Confirm "Pair" -- the phone stores a bearer token in
   `localStorage` and redirects to the full cockpit.
5. On iOS Safari: tap **Share -> Add to Home Screen** to install as an app.
   On Android Chrome: the install banner appears automatically.

Heartbeats from the phone feed the dead-man switch; force-quitting the app
while a live trade is in flight will trip the switch and pause INDIRA.

---

## Security posture

* `desktop` mode binds to loopback only; nothing on your LAN or the internet
  can see the cockpit.
* `cloud` mode is gated by a bearer token on every `/api/*` call; the HTML
  shell, `/health`, and `/pair` are the only public paths.
* Pairing tokens are one-time, short-lived (15 min default), and revokable
  per-device.
* Wallet signing follows the birth-clock policy (30-day warmup, then
  $100/day supervised, then operator-configurable). Changing phases or caps
  requires a governance-approved event.
* Every patch (DEVIN-drafted, DYON-proposed, operator-submitted) must
  traverse the sandbox pipeline before it may be merged or promoted. See
  [SANDBOX.md](SANDBOX.md).

---

## Troubleshooting

* **Browser doesn't open on Windows** -- the `.exe` still started; navigate
  manually to `http://127.0.0.1:8765/`. The token is in
  `%LOCALAPPDATA%\DIX VISION\data\cockpit_token.txt`.
* **`Address already in use`** -- set `DIX_PORT=8766` (or any free port).
* **Phone can't reach the cockpit** -- see [MOBILE.md](MOBILE.md) for LAN,
  Tailscale, and reverse-tunnel recipes.
* **Fly deploy fails on volume** -- run `fly volumes create dix_data --size
  1` before `fly deploy`.
