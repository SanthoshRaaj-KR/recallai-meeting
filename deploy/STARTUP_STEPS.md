# Startup steps — this deployment

Concrete steps for *this* deployment (real IPs, real domains), not the generic
template. See `README.md` for the general phase-by-phase rationale, `SERVICE_MAP.md`
for the full route map.

- **VM-A** (bot-service + agent + Caddy): `104.43.112.6`
- **VM-B** (confluence-service + org-service + Caddy): `20.6.9.169`
- **No purchased domain** — both VMs use [nip.io](https://nip.io) (free wildcard
  DNS: `<anything>.<ip-with-dashes>.nip.io` resolves straight back to that IP),
  which lets Caddy get real Let's Encrypt TLS certs with zero DNS setup:
  - VM-A: `bot.104-43-112-6.nip.io`
  - VM-B: `api.20-6-9-169.nip.io`

Both VMs already have their `.env` filled in (co-located `deploy/vm-a/.env` /
`deploy/vm-b/.env` — never the repo-root `.env` or the local `.env.local` files).

---

## VM-A — start first

SSH into `104.43.112.6`, then from the repo root there:

```bash
docker build -t my-agent:latest ./my-agent
docker compose -f deploy/vm-a/docker-compose.yml up -d
docker compose -f deploy/vm-a/docker-compose.yml logs -f caddy   # watch cert issuance
```

Verify:
```bash
curl https://bot.104-43-112-6.nip.io/health
# → {"service":"bot-service", ..., "livekit_configured": true}
```

Then register the Recall webhook at `https://bot.104-43-112-6.nip.io/recall-webhook`.

---

## VM-B — start second

SSH into `20.6.9.169`, then from the repo root there:

```bash
docker build -t my-agent:latest    ./my-agent
docker build -t jarvis-org:latest  ./user_service
docker compose -f deploy/vm-b/docker-compose.yml up -d
docker compose -f deploy/vm-b/docker-compose.yml logs -f caddy   # watch cert issuance
```

Verify:
```bash
curl https://api.20-6-9-169.nip.io/conf-api/health
curl https://api.20-6-9-169.nip.io/org-api/health
```

SSE endurance check (this is *why* VM-B isn't on Azure Container Apps — ACA's
240s ingress cap would kill this):
```bash
curl -N https://api.20-6-9-169.nip.io/conf-api/review/pipeline/<job-id>/stream
# must survive past 5 minutes uninterrupted
```

---

## Frontend (`sync-sage-bot/.env`)

Already points at both VMs:
```
VITE_API_BASE_URL=https://bot.104-43-112-6.nip.io
VITE_CONF_BASE_URL=https://api.20-6-9-169.nip.io/conf-api
VITE_ORG_BASE_URL=https://api.20-6-9-169.nip.io/org-api
VITE_BACKEND_KIND=recall_bridge
```

**Not yet decided:** where this actually gets hosted (no Vercel project /
Netlify config / CI exists yet). `.env` is gitignored on purpose — whatever
host you pick needs these same key/values entered in its own env var settings,
copying this file won't reach it automatically.

**Landmine if you ever build locally:** `sync-sage-bot/.env.local` overrides
`.env` and still points at local dev-proxy paths (`/api`, `/org-api`). Delete
or rename it before running `npm run build` from this machine, or the wrong
URLs get baked into the bundle silently.

---

## Things that must stay in sync

- **`JWT_SECRET`** — must be byte-identical in `deploy/vm-a/.env` and
  `deploy/vm-b/.env` (check the files directly — not reproduced here).
  Org-service (VM-B) issues knowledge-base-write JWTs; bot-service (VM-A)
  verifies them. If you ever rotate it, update both files together.
- **`BRIDGE_SERVER_URL` / `BOT_SERVICE_URL`** — both set to
  `https://bot.104-43-112-6.nip.io` in both `.env` files (VM-B's org-service
  needs this for the admin kick proxy back to VM-A's bot-service).

## Still open

- `CORS_ORIGINS=*` is wildcard-open on both VMs. Fine for initial testing —
  lock it to the real frontend origin once that's hosted somewhere.
- No frontend hosting platform chosen yet (see above).
