# Cloud deployment (24/7 learning + sourcing)

DIX VISION can run on your laptop, on a VPS, or as a managed cloud app.
The same Docker image serves all three. This doc covers the cloud paths so
the learning loops, API sniffer, and trader KB keep ingesting around the
clock without relying on your machine being awake.

---

## What runs in the cloud

| Service  | Process                            | Persistence              | Ports |
|----------|------------------------------------|--------------------------|-------|
| cockpit  | `python -m cockpit --mode cloud`   | `/data` volume           | 8765  |
| worker   | `python -m cockpit --mode worker`  | shares cockpit volume    | -     |
| caddy    | TLS reverse proxy (optional)       | `/data`, `/config`       | 80/443|

The **worker** is the 24/7 brain: it drives `arb.refresh_decay`, the dead-man
heartbeat, the API sniffer, the news/market/on-chain providers, and every
bounded learning loop. It writes `SYSTEM/WORKER_TICK` events to the shared
ledger every `DIX_WORKER_INTERVAL_SEC` (60 s default).

The **cockpit** is the same FastAPI HTTP surface you run locally; the phone
and your laptop both point at its public URL.

---

## Fly.io -- ~$5/month, recommended

```bash
curl -L https://fly.io/install.sh | sh        # one time
fly launch --copy-config --name dix-vision-$(whoami)
fly volumes create dix_data --size 1
fly deploy
```

Fly reads `cloud/fly.toml` (shared-CPU, 512 MB, 1 GB persistent volume,
`/health` probe, HTTPS forced). Deploy costs roughly $1.94/mo for the
machine + $0.15/mo per GB of volume.

To also run the worker:
```bash
fly machine run --name dix-vision-worker \
  --region iad --volume dix_data:/data \
  dix-vision:42.2 python -m cockpit --mode worker
```

## Render.com -- push `cloud/render.yaml`

Provisions two services (web + worker) sharing a 1 GB disk. Free tier sleeps
after 15 minutes of idle HTTP traffic; worker stays up.

## Railway -- push `cloud/railway.json`

Single-service deploy. Storage persists on default volume.

## Hetzner / DigitalOcean / any VPS with Docker

```bash
ssh root@your.vps
apt update && apt install -y docker.io docker-compose-plugin
git clone https://github.com/your-org/dix-vision
cd dix-vision
echo "DIX_HOSTNAME=cockpit.yourdomain.tld" > .env
docker compose --profile tls up -d
```

Point a DNS A-record at the VPS, wait 30 s for Caddy to provision Let's
Encrypt certs, and the cockpit is live at HTTPS.

## Kubernetes

```bash
kubectl apply -f cloud/k8s/deployment.yaml
kubectl -n dix-vision port-forward svc/dix-vision 8765:80
```

PVC + Deployment + Service included; add an Ingress for your cluster's
cert-manager setup.

---

## 24/7 learning cadence

All polling is bounded in the cold path (WAL SQLite + LRU + ring buffers).
Default cadences:

| Lane              | Poll rate           | Bound                           |
|-------------------|---------------------|---------------------------------|
| News (RSS/REST)   | 60 -- 300 s         | ring buffer, content hash dedup |
| CEX snapshots     | 5 -- 30 s           | in-RAM, dropped at rollover     |
| On-chain          | 15 s -- 5 min       | RPC rate-limited                |
| EDGAR 13F / Form4 | 24 h                | lazy materialize on mention     |
| Code search       | 1 h (trending), 24 h (deep) | score-gated                  |
| Coding docs (PEP/PyPI/arXiv/StackOverflow) | 6 -- 24 h | LRU cache               |
| API sniffer       | on-demand           | 1x per URL                      |

Nothing runs in the hot path. `fast_execute_trade` remains zero-DB,
zero-allocation, <5 ms.

---

## Environment reference

| Variable                  | Default                       | Purpose                                 |
|---------------------------|-------------------------------|-----------------------------------------|
| `DIX_MODE`                | `desktop`                     | `desktop` / `cloud` / `worker`          |
| `DIX_BIND_HOST`           | 127.0.0.1 desktop, 0.0.0.0 cloud | HTTP bind                            |
| `DIX_PORT`                | 8765                          | HTTP port                               |
| `DIX_PUBLIC_URL`          | (empty)                       | Used to build pairing-QR URLs           |
| `DIX_ALLOWED_ORIGINS`     | (empty)                       | CORS allowlist, comma-separated         |
| `DIX_COCKPIT_TOKEN`       | (auto-generated)              | Bearer token for all `/api/*`           |
| `DIX_COCKPIT_TOKEN_FILE`  | `data/cockpit_token.txt`      | Persisted token location                |
| `DIX_WORKER_INTERVAL_SEC` | 60                            | Worker tick cadence                     |
| `DIX_LEDGER_DB`           | `data/ledger.sqlite`          | Event-sourced audit log                 |
| `DIX_EPISODIC_DB`         | `data/episodes.sqlite`        | Episodic memory (ring, 50k rows max)    |
| `DIX_WALLET_POLICY_DB`    | `data/wallet_policy.sqlite`   | Birth-clock + $100/day caps             |
| `DIX_PAIRING_DB`          | `data/pairing.sqlite`         | Device pairings                         |
| `DIX_LOCALE`              | OS default                    | Force a specific UI/chat language       |
| `DIX_TRANSLATOR`          | `none`                        | `none` / `deepl` / `google` / `openai` / `local:nllb200` |

---

## Safety posture in cloud

* The wallet birth-clock persists on `/data`; redeploying never shortens
  the warmup.
* Every `/api/*` call requires a bearer token.
* Pairing tokens are one-time and short-lived.
* Patches land only through the sandbox pipeline; the CI gate refuses
  merges without a `PATCH_SANDBOX_PASS` ledger event.
* TLS is terminated by Caddy (self-hosted) or the platform (Fly/Render/
  Railway); the cockpit container itself never handles ACME.
