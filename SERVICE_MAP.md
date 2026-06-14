# Jarvis Backend — Service / Image / Route / Port Map (canonical)

> **Purpose:** the single source of truth for *what image runs where, on which port, doing which task, serving which routes.* Verified directly from route decorators in `my-agent/src/` and `user_service/routes/` on 2026-06-13. Use with `MIGRATION_PLAN.md` (the *how*) and `HOSTING_ANALYSIS.md` (the *why*).
> ⚠️ Ignore the stale `API_ROUTES.md` / `CLAUDE.md` / `.planning/` — they describe a non-existent `confluence_logic/` app. This file is authoritative for the current code.

---

## 1. Images — what you build vs pull

| Image | Build? | Source | Runs as (command) | Roles |
|---|---|---|---|---|
| **`my-agent`** | **Build** | `my-agent/Dockerfile` (uv, py3.13, `uv.lock`) | 3 commands ↓ | agent worker · bot-service(8000) · confluence-service(8001) |
| **`jarvis-org`** | **Build** | new slim `user_service/Dockerfile` (py3.12-slim, `requirements.txt`) | `uvicorn user_service.main:app --port 8003` | org/user-service(8003) |
| `livekit/livekit-server` | Pull | LiveKit official | from `livekit/generate` compose | SFU / media + TURN |
| `caddy` | Pull | official | from generate / VM-B compose | TLS + reverse proxy |
| `redis` | Pull | official | from `livekit/generate` compose | LiveKit coordination |

**`my-agent` — one image, three commands:**
| Role | Command | Port |
|---|---|---|
| Agent worker | `uv run src/agent.py start` | — (dials out to LiveKit) |
| Bot service | `uv run uvicorn src.bot_service:app --host 0.0.0.0 --port 8000` | 8000 |
| Confluence service | `uv run uvicorn src.recall_bridge:app --host 0.0.0.0 --port 8001` | 8001 |

---

## 2. Port → service → placement map

| Port | Service | Image · command | VM | Public? | Scaling |
|---|---|---|---|---|---|
| — | **Agent worker** | `my-agent` · `agent.py start` | **VM-A** | no (outbound to LiveKit) | 1 worker; N job procs (1/meeting) |
| 7880/7881/3478/443/50000–60000 | **LiveKit server** | `livekit/livekit-server` (stock) | **VM-A** | **yes** (WSS + UDP media) | single node (Redis-backed) |
| **8000** | **Bot service** | `my-agent` · `bot_service` | **VM-A** | **yes** (Recall bot-page + webhooks) | **single replica** (in-proc caches) |
| **8001** | **Confluence service** | `my-agent` · `recall_bridge` | **VM-B** | yes (frontend SSE) | **single replica** (20-min in-proc job) |
| **8003** | **Org/User service** | `jarvis-org` | **VM-B** | yes (frontend) | stateless / elastic |
| 443 | Caddy (TLS) | `caddy` (stock) | VM-A + VM-B | yes | — |

- **VM-A** = "nice" realtime box (B2s, 2 vCPU/4 GB+): LiveKit + agent + bot-service + caddy + redis. Static IP + domain. Heavy residents = LiveKit + agent.
- **VM-B** = "cheap" box (B1ms, 1 vCPU/2 GB): confluence + org + caddy. Low-CPU but 8001 needs ~2 GB for a pipeline run.
- Both VMs read/write the **same Supabase** (shared `session_store`). No server↔server calls except **agent → bot-service over localhost**.

---

## 3. Internal wiring (who calls whom)

| Caller | Callee | Transport | Notes |
|---|---|---|---|
| Agent worker | bot-service `/livekit-transcript/{id}` | **localhost** `BRIDGE_INTERNAL_URL=http://127.0.0.1:8000` (`agent.py:102,471`) | fire-and-forget, 2 s timeout, no retry → **co-locate on VM-A** |
| Agent worker | Pinecone | HTTPS | in-meeting Confluence RAG (`confluence_rag.py`) — **direct, not via 8001** |
| Bot-service | LiveKit server | HTTPS/WS (`LIVEKIT_URL`) | mint token + `agent_dispatch.create_dispatch` |
| Bot-service | Recall.ai | HTTPS | create/stop bot; Recall calls back to `BRIDGE_SERVER_URL` |
| Recall browser bot | LiveKit server | WSS + UDP | publishes mixed meeting audio |
| confluence-service | Cerebras/OpenAI, Atlassian, Pinecone | HTTPS | proposal pipeline |
| All services | Supabase | HTTPS REST | shared session + org state |
| Frontend | 8000 / 8001 / 8003 | HTTPS | base-URL split (see §6) |

**Two URL vars — do not conflate:**
- `BRIDGE_SERVER_URL` = **public HTTPS** of VM-A (Recall loads `bot.html` + posts webhooks). Must start `https://`.
- `BRIDGE_INTERNAL_URL` = **localhost** (`http://127.0.0.1:8000`) agent→bot transcript POST.

---

## 4. Routes — Bot service (port 8000, VM-A)
**Task:** Recall bot lifecycle · LiveKit token mint + agent dispatch · transcript ingest · session status/history · RAG sync trigger · serves `bot.html`.

| Method | Route | Task |
|---|---|---|
| POST | `/bot/start` | Create Recall bot + dispatch agent into a LiveKit room |
| GET | `/bot-page` | Serve `bot.html` (Recall camera webpage) |
| GET | `/bot/status` | Global idle status (no session) |
| GET | `/sessions/{session_id}/bot/status` | Live status (polls Recall, self-heals) |
| POST | `/sessions/{session_id}/bot/stop` | Remove bot from call, mark ended |
| POST | `/sessions/{session_id}/jarvis-call/token` | Mint LiveKit token for 1:1 Jarvis voice call + dispatch |
| POST | `/recall-webhook` | Recall status webhooks (`bot.status_change`/`bot.done`) |
| POST | `/livekit-transcript/{session_id}` | **Agent posts transcript turns here (localhost)** |
| GET | `/sessions/{session_id}/review/transcript` | Read stored transcript ⚠️ *(review path served by 8000, see §6)* |
| GET | `/history` | List all meeting history |
| POST | `/rag/sync` | Start incremental Confluence→Pinecone re-index job |
| GET | `/rag/sync/latest` | Most recent sync job |
| GET | `/rag/sync/{job_id}` | Poll a sync job |
| GET | `/health` | Health check |

---

## 5. Routes — Confluence service (port 8001, VM-B)
**Task:** post-meeting intelligence — proposal pipeline (fact-extract → RAG → draft → verify), Confluence writes, summary/MOM, chat, regenerate. **`/review/pipeline/{job_id}/stream` is the 20-min SSE** (the reason this is on a VM, not ACA).

| Method | Route | Task |
|---|---|---|
| GET | `/sessions/{session_id}/review/changes` | List proposed Confluence changes |
| POST | `/sessions/{session_id}/review/changes/propose` | Generate proposals from transcript |
| POST | `/sessions/{session_id}/review/changes/{change_id}/reject` | Reject a proposal |
| POST | `/sessions/{session_id}/review/execute` | Apply selected proposals to Confluence |
| POST | `/sessions/{session_id}/review/regenerate/{proposal_id}` | Re-draft a stale proposal vs live page |
| GET | `/sessions/{session_id}/review/summary` | Executive summary / MOM |
| POST | `/sessions/{session_id}/review/chat` | Q&A over the meeting |
| POST | `/review/pipeline/start` | Start async pipeline (returns job_id) |
| POST | `/review/pipeline/start-with-transcript` | Start pipeline with supplied transcript |
| GET | `/review/pipeline/{job_id}/stream` | **SSE progress stream (up to ~20 min)** |
| GET | `/health` | Health check |

---

## 6. ⚠️ Routing gotcha — `/sessions/{id}/review/*` is split across TWO services
The same path prefix is served by both boxes:
- `GET /sessions/{id}/review/**transcript**` → **8000 (VM-A)**
- all other `/sessions/{id}/review/*` (changes, execute, summary, chat, regenerate, reject) → **8001 (VM-B)**

**Implication:** you **cannot** route all `/sessions/*` to one upstream. Keep the **frontend's existing base-URL split** (`api.ts` already calls bot routes via `/api` and review/pipeline routes via `/conf-api`). Map at the edge:

| Frontend path | → upstream |
|---|---|
| `/api/*` | VM-A bot-service :8000 |
| `/conf-api/*` | VM-B confluence :8001 |
| `/org-api/*` | VM-B org :8003 |

Do **not** collapse these to a single host-based path router unless you also special-case `/sessions/{id}/review/transcript` → 8000.

---

## 7. Routes — Org/User service (port 8003, VM-B)
**Task:** auth (JWT + Google exchange) · users · teams · org hierarchy · RBAC · bot assignment · invites · analytics. State = Supabase only. *(`POST /admin/reset` exists but must stay disabled in prod — `ALLOW_DB_RESET` unset.)*

| Prefix | Method · Route | Task |
|---|---|---|
| `/auth` | POST `/register`, `/login`, `/refresh`, `/google-exchange`, `/logout` | Auth + token lifecycle |
| `/users` | GET `` (list), `/me`, `/{id}`, `/by-email/{email}`, `/{id}/reports`, `/{id}/manager`, `/me/stats`; PATCH `/{id}` | User profiles, reports, stats |
| `/teams` | GET ``, `/{id}`, `/{id}/members`; POST ``, `/{id}/members`, `/{id}/invite`; PATCH `/{id}`; DELETE `/{id}`, `/{id}/members/{uid}` | Team CRUD + membership |
| `/org` | GET `/hierarchy` | Full org tree (CEO) |
| `/invites` | GET `/{code}`; POST `/{code}/accept` | Team invitations |
| `/analytics` | GET `/usage/me`, `/usage/teams/{id}`, `/usage/org` | Meeting usage stats |
| (no prefix) | GET `/teams/{id}/bot`, `/teams/{id}/meetings`; POST `/teams/{id}/bot` | Bot assignment + team meetings |
| (root) | GET `/health`; POST `/admin/reset` (dev only) | Health / dev reset |

---

## 8. Env vars by service (what to inject where)

| Service (VM) | Required env |
|---|---|
| **Agent** (VM-A) | `LIVEKIT_URL/API_KEY/API_SECRET` (self-hosted), `BRIDGE_INTERNAL_URL=http://127.0.0.1:8000`, `CEREBRAS_API_KEY`, `ASSEMBLYAI_API_KEY`, `CARTESIA_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `PINECONE_API_KEY`, Atlassian (for RAG), optional `GITHUB_TOKEN` |
| **Bot-service** (VM-A) | `LIVEKIT_URL/API_KEY/API_SECRET`, `RECALL_API_KEY`, `RECALL_API_REGION`, **`BRIDGE_SERVER_URL=https://<vm-a-domain>`**, `BOT_NAME`, `AGENT_NAME=my-agent`, `SUPABASE_*`, `PINECONE_API_KEY`, `CORS_ORIGINS` |
| **Confluence-service** (VM-B) | `CEREBRAS_API_KEY`, `OPENAI_API_KEY`, `ATLASSIAN_USER_EMAIL/API_TOKEN/DOMAIN/SPACE_KEY`, `SUPABASE_*`, `PINECONE_API_KEY`, `MY_AGENT_REVIEW_MODEL`, `MY_AGENT_PIPELINE_MAX_PAGES`, `CORS_ORIGINS`, optional `ROVO_MCP_*` |
| **Org-service** (VM-B) | `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `JWT_SECRET`, `JWT_ACCESS_TTL_MINUTES`, `JWT_REFRESH_TTL_DAYS`, `CORS_ORIGINS`, `ALLOW_DB_RESET=false` |
| **LiveKit server** (VM-A) | generated `livekit.yaml` keys (own API key/secret), domains for Caddy TLS |

---

## 9. One-glance summary

```
VM-A (nice, B2s 4GB, static IP + domain, UDP 50000-60000 open)
  caddy:443 ──/api/*──────▶ bot-service:8000  (my-agent)  ── localhost ◀── agent worker (my-agent: agent.py start)
  livekit-server:7880/7881/3478/50000-60000 (stock) ◀─ WSS+UDP ─ Recall bot / agent
  redis (stock)
        │ Supabase (shared)        │ Cerebras · AssemblyAI · Cartesia · Pinecone

VM-B (cheap, B1ms 2GB, domain)
  caddy:443 ──/conf-api/*─▶ confluence-service:8001 (my-agent: recall_bridge)  ← 20-min SSE
            ──/org-api/*──▶ org-service:8003       (jarvis-org)
        │ Supabase (shared) · Cerebras/OpenAI · Atlassian · Pinecone

Static Web Apps (frontend) ──/api→VM-A · /conf-api,/org-api→VM-B ; Supabase for Google OAuth
```

**Build: 2 images (`my-agent`, `jarvis-org`). Pull: 3 stock (`livekit-server`, `caddy`, `redis`).**
