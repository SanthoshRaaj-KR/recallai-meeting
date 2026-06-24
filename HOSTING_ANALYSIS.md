# Jarvis Backend — Hosting & Containerization Analysis

> **Audience:** humans and AI models that will deploy this backend.
> **Generated:** 2026-06-13. Verified against source in `Confluence/` (not against the stale `.planning/` docs — see §11).
> **Scope:** the `Confluence/` backend (the "bot" + "Confluence" + "org" services + the LiveKit voice agent). The `sync-sage-bot/` frontend is covered for completeness.

---

## 1. Executive summary & recommendation

The backend is **not one app**. It is **four runtime components** with very different shapes, plus a static frontend. Almost all the heavy/expensive machinery (real-time media, LLM inference, STT, vector DB, Postgres) is **already off-box on managed SaaS** (LiveKit Cloud, Supabase, Pinecone, OpenAI/Cerebras, AssemblyAI, Recall.ai). So self-hosting is only about a handful of **stateless-ish Python HTTP services** + **one persistent voice-agent worker**.

**Recommended hosting (lowest cost + lowest ops for your situation):**

| Component | Host on | Why |
|---|---|---|
| **LiveKit voice agent** (`agent.py`) | **LiveKit Cloud** (already provisioned — `livekit.toml`) | It uses LiveKit *Inference* + LiveKit Cloud noise-cancellation (`ai_coustics`). Self-hosting forces you to rip those out. Cloud has a usage-based free tier and removes your single biggest hosting burden. |
| **bot-service (8000)**, **confluence-service (8001)**, **org/user-service (8003)** | **Azure Container Apps (ACA)** | You already pay for Azure; ACA gives serverless containers, scale-to-zero, **managed HTTPS FQDN out of the box** (this kills the `ngrok` requirement for Recall webhooks), and a monthly free grant that likely covers low traffic. |
| **sync-sage-bot frontend** | **Azure Static Web Apps** (free tier) | Static Vite/React build + global CDN + free TLS. |

**Why Azure over the AWS free tier here:** your AWS account is a *time-boxed free tier* (see §6.3 — likely the new **6-month / $200-credit** model, not 12 months of a free VM). Your Azure account is a **standing paid account**. For something you want to "host off" durably, building on the account that doesn't expire — and that offers scale-to-zero + free managed TLS — is cheaper in total and far less work than babysitting an EC2 micro with hand-rolled nginx/Let's Encrypt. AWS free tier is the right pick *only* if your goal is "$0 cash for a few months of a demo" and you accept the ops burden (§6.3).

> If you want a single number: on ACA with scale-to-zero + LiveKit Cloud + the SaaS free tiers, a low-traffic deployment is plausibly **~$0–15/month** beyond per-use LLM/STT spend. See §9.

---

## 2. The real service inventory

All paths are under `Confluence/`. Ports are the documented defaults.

### 2.1 Bot Service — port 8000 — `my-agent/src/bot_service.py`
- **What:** FastAPI HTTP service. Recall.ai bot lifecycle (`/bot/start`, `/sessions/{id}/bot/stop|status`), LiveKit token minting, agent dispatch, transcript ingestion (`/livekit-transcript/{id}`), session history, Recall webhooks (`/recall-webhook`), and **triggering** the Confluence→Pinecone RAG sync (`/rag/sync`). Serves `bot.html` (the Recall "camera webpage").
- **Must be publicly reachable over HTTPS.** Recall.ai loads `/bot-page` as the bot's camera output and POSTs webhooks to it. `_create_recall_bot()` hard-fails if `BRIDGE_SERVER_URL` is not `https://`.
- **State:** session data → **Supabase** (shared, durable). But it also keeps **in-process caches**: `_compactors` (transcript compaction objects, not serializable), `_bot_index` (bot_id→session), `_sync_jobs` (RAG sync progress). These **reset on restart and are NOT shared across replicas.** → *This service must run as a single replica.* (See §4.)

### 2.2 Confluence Service — port 8001 — `my-agent/src/recall_bridge.py`
- **What:** FastAPI HTTP service. All meeting *intelligence*: the multi-agent proposal pipeline (`review_pipeline/pipeline.py`, ~84 KB), Confluence page writes, meeting summary/MOM, post-meeting chat. Streams progress over **SSE**, and a pipeline run "may take up to 20 minutes."
- **LLM:** Cerebras `gpt-oss-120b` (primary) with OpenAI `gpt-4o-mini` fallback. Talks to Atlassian Confluence REST (and optional Rovo MCP), Pinecone (via `review_pipeline/rag.py`), and Supabase.
- **State:** reads/writes shared session state in **Supabase**. But the `_pipelines` job registry is **in-process & ephemeral**, and pipeline runs are in-process `asyncio` tasks. → *A restart or replica swap mid-run loses the job and breaks the SSE stream.* *Run as a single replica; don't scale to zero while a job is live.* (See §4.)

### 2.3 LiveKit Voice Agent — **worker, no HTTP port** — `my-agent/src/agent.py`
- **What:** The real-time voice agent ("Jarvis"). Launched as `uv run src/agent.py start` (this is the **Dockerfile `CMD`**). It does **not** listen for inbound HTTP — it dials **out** to LiveKit Cloud (`AgentServer` + `@server.rtc_session(agent_name="my-agent")`) and waits for dispatched jobs.
- **Pipeline:** AssemblyAI STT, Cerebras LLM via **LiveKit Inference** (`inference.LLM/TTS`), Silero VAD + multilingual turn detector (downloaded at build via `download-files`), LiveKit Cloud noise cancellation (`ai_coustics`), optional GitHub MCP, live Confluence RAG (`confluence_rag.py`).
- **Shape:** long-running, latency-sensitive, loads ML models locally (memory-heavier), scales by **concurrent meetings**. **Strongly coupled to LiveKit Cloud** via Inference + `ai_coustics`. → *Deploy to LiveKit Cloud.*

### 2.4 Org / User Service — port 8003 — `user_service/main.py`
- **What:** FastAPI HTTP service. Auth (JWT via `python-jose`/`passlib`), users, teams, org hierarchy, RBAC, bot assignment, invites, analytics. This is the "single Google-OAuth org hierarchy system."
- **State:** **Supabase** only. Fully stateless process. **Separate, lightweight dependency set** (no LiveKit/Pinecone/OpenAI) and its own `requirements.txt` + venv.
- ⚠️ Ships an `/admin/reset` endpoint that wipes all org tables (gated by `ALLOW_DB_RESET`). Must be **off** in production (§10).

### 2.5 Frontend — `sync-sage-bot/` (Vite + React + shadcn/ui)
- Static SPA. In dev, `vite.config.ts` proxies `→ :8000` (bot), `→ :8001` (confluence), `→ :8003` (org). In prod it uses `VITE_API_BASE_URL` / `VITE_BACKEND_KIND` and talks to Supabase directly for Google sign-in. Build with `vite build` → static `dist/`.

---

## 3. External managed dependencies (already off-box — you do **not** host these)

| Service | Used by | Notes / cost shape |
|---|---|---|
| **LiveKit Cloud** | agent, bot-service (tokens/dispatch) | Subdomain `nani-za2evkkc`, agent `CA_5BB4bmhc8PgR` already provisioned. Usage-based; free tier. |
| **Supabase** | all 3 HTTP services (shared session + org data) | Postgres + REST. Free tier exists; this is the **shared state backbone** that makes the split safe. |
| **Pinecone** | confluence-service + bot RAG sync + agent | Vector store for Confluence RAG. Serverless free tier. |
| **OpenAI** | confluence-service (fallback), agent | Pay per token. |
| **Cerebras** | confluence-service + agent (primary LLM `gpt-oss-120b`) | Pay per token. |
| **AssemblyAI** | agent (STT) | Pay per minute. |
| **Recall.ai** | bot-service (meeting bot) | Pay per bot-hour. **Requires your bot-service to be public HTTPS.** |
| **Atlassian Confluence** | confluence-service (target of writes/reads) | Customer's instance. |

**Implication:** your hosting bill is dominated by these per-use SaaS charges, **not** by the compute you host. Optimizing the compute layer to ~$0 is easy; the variable cost is LLM/STT/bot-minutes.

---

## 4. Runtime constraints that drive the topology (read before deploying)

These come from reading the code, and they **directly constrain** how you containerize/scale:

1. **bot-service is effectively single-instance.** `_compactors`, `_bot_index`, `_sync_jobs` live only in process memory and are not shared. Two replicas → webhook lookups and compaction state diverge. **Set min=max=1.** (To make it horizontally scalable later, move compaction + bot-index into Supabase/Redis.)
2. **confluence-service is effectively single-instance during a run.** `_pipelines` registry + in-process `asyncio` pipeline tasks + SSE streams are pinned to one process. A 20-minute job does not survive a replica swap, restart, or scale-to-zero. **min=1 while jobs can run.**
3. **Shared state is in Supabase** (`session_store.py`). This is the good part: bot-service and confluence-service can be separate containers because they both read/write the same Supabase rows. This is what makes the decoupling safe.
4. **agent worker loads local models** (Silero VAD, turn detector). Memory footprint is the real sizing driver, not CPU. Fits LiveKit Cloud's agent sizing; on a 1 GB VM it would be tight alongside the HTTP services.
5. **bot-service needs stable public HTTPS** for the whole time any meeting is active (Recall renders `bot.html` and fires webhooks). Cold-start-on-first-request (scale-to-zero) risks missed webhooks → **keep min=1**.
6. **org/user-service is the only truly elastic one** — stateless, light deps. It can scale to zero and scale out freely.

---

## 5. Logic / health check findings (quick audit, as requested)

Genuinely useful issues found while mapping — not blockers for hosting, but worth fixing:

- 🟥 **Docs are stale and misleading.** `Confluence/CLAUDE.md`, `Confluence/API_ROUTES.md`, and `Confluence/.planning/codebase/*` describe a `confluence_logic/jarvis_agentic.py` + `review-ui/` (Next.js) architecture **that does not exist on disk**. The real backend is `my-agent/` (bot-service, confluence-service, agent) + `user_service/`. Any model reading those docs will be wrong. → Regenerate docs from the current tree. (This file is the corrected map.)
- 🟧 **Duplicate/legacy package.** `Confluence/local_doc_change/` is an earlier standalone version of the proposal pipeline (it has its own `recall_bridge.py`, `agents_local/`, `rag/`). The live logic is `my-agent/src/review_pipeline/`. → Confirm `local_doc_change/` is dead and exclude it from images (don't ship the `stress_corpus/` either — ~130 fixture docs).
- 🟧 **CORS:** all three services default `CORS_ORIGINS="*"` **with** `allow_credentials=True`. That combination is rejected by browsers and is unsafe for prod. → Set explicit origins in prod.
- 🟥 **`/admin/reset`** in user-service truncates every org table. Ensure `ALLOW_DB_RESET` is unset/false in prod and ideally network-restrict the route.
- 🟧 **Root `requirements.txt` is the legacy/monolith manifest** (pulls `neo4j`, `docling`, `pyaudio`, `ngrok`, `openai-agents`, etc.). The **live** agent/services deps are in `my-agent/pyproject.toml` (+ `uv.lock`) and `user_service/requirements.txt`. Build images from those, **not** root `requirements.txt`, or you'll bloat images and pull a Neo4j driver/`docling` you don't use.
- 🟨 Secrets are loaded from `.env.local` via `load_dotenv`. In cloud, inject via the platform secret store, not a baked-in file.

---

## 6. Hosting options compared

### 6.1 The agent worker → LiveKit Cloud (decided)
The agent uses `inference.LLM`/`inference.TTS` (LiveKit Inference) and the `ai_coustics` Cloud noise-cancellation plugin, and `livekit.toml` already points at a Cloud project/agent. Self-hosting means removing Inference + noise cancellation and running a persistent VM that maintains realtime media — more cost, more ops, worse audio. **Keep it on LiveKit Cloud.** Deploy with `lk agent deploy` (uses the existing `Dockerfile`).

### 6.2 The three HTTP services + frontend → **Azure Container Apps + Static Web Apps** (recommended)
- **Managed HTTPS FQDN** per app → satisfies Recall's public-HTTPS requirement with **no ngrok, no cert management**.
- **Scale-to-zero + consumption billing**; monthly free grant (vCPU-seconds / GiB-seconds / requests) typically covers low traffic.
- **Per-app scaling rules** let you honor §4 (bot & confluence pinned to 1 replica; org elastic).
- Secrets + env via ACA's secret store. Deploy straight from a container image (ACR or any registry).
- You already pay for Azure → one bill, one identity, no second cloud to learn.

### 6.3 AWS free tier (the cheaper-cash, higher-effort alternative)
- ⚠️ **Verify which free tier your account is on.** AWS replaced the old model in mid-2025: **accounts created after ~July 15 2025 get a credit-based "free plan" (~$100, up to $200 with activities, valid ~6 months)** — **not** 12 months of a free `t3.micro`. Older accounts may still have the legacy 12-month 750-hr micro. Check the Billing console.
- **If legacy 12-month tier:** one `t3.micro`/`t2.micro` (1 vCPU, **1 GB RAM**) can run all three HTTP services via `docker compose` for ~$0 for 12 months. But: 1 GB is tight (the confluence pipeline + Python × 3), and **you** own OS patching, a domain, TLS (Caddy/Let's Encrypt or an ALB which costs ~$16/mo and erodes the savings), restarts, and monitoring. Free tier then expires → you pay or migrate.
- **If new credit plan:** it's effectively "free for ~6 months then bill" — at which point you're paying AWS anyway, so you might as well be on the Azure account you already fund.
- **Verdict:** good for a short throwaway demo on $0 cash; **not** the durable, low-effort answer for "host it off."

### 6.4 What about coupling everything onto one cheap VM?
Possible (single VM, 4 processes: agent + 3 uvicorns) and is the absolute floor on cash. But you lose LiveKit Cloud Inference + noise cancellation, you re-introduce ngrok/TLS toil, and the agent's realtime media on a shared micro will be jittery. Only consider if cash is the *only* constraint and quality/ops don't matter.

---

## 7. Recommended target architecture

```
                         ┌────────────────────────────────────────────┐
                         │            Managed SaaS (off-box)           │
                         │  LiveKit Cloud · Supabase · Pinecone        │
                         │  OpenAI · Cerebras · AssemblyAI · Recall.ai │
                         │  Atlassian Confluence                       │
                         └───────▲───────────▲──────────────▲──────────┘
                                 │           │              │
        ┌───────────────┐        │           │              │
        │  Recall.ai    │── webhook/bot-page (HTTPS) ──┐     │
        └───────────────┘                              │     │
                                                        │     │
  Azure Static Web Apps            Azure Container Apps (your compute)
  ┌──────────────────┐   /api      ┌───────────────────────────────┐
  │ sync-sage-bot     │──────────▶ │ bot-service   :8000  min=max=1 │──┐
  │ (static SPA, CDN) │   /conf-api │ confluence-svc:8001  min=1     │  │  shared
  │ + Google OAuth ───┼─▶ Supabase  │ org-service   :8003  min=0..N  │  ├─▶ Supabase
  └──────────────────┘   /org-api   └───────────────────────────────┘  │  (session +
                                                                        │   org state)
  LiveKit Cloud (managed agent runtime)                                 │
  ┌───────────────────────────────────────────────────┐               │
  │ my-agent worker (agent.py start) — dispatched into  │───────────────┘
  │ rooms; STT/LLM/TTS via LiveKit Inference; ai_coustics│
  └───────────────────────────────────────────────────┘
```

---

## 8. Containerization & coupling/decoupling strategy

**Principle: couple by build, decouple by scaling profile.**

### 8.1 Images (2 images total)
1. **`jarvis-agent` image** — build from existing `my-agent/Dockerfile` (uv, Python 3.13, `uv.lock`). This **one image serves three roles** via different commands:
   - `uv run src/agent.py start` → the **LiveKit Cloud** worker (Dockerfile default `CMD`).
   - `uv run uvicorn src.bot_service:app --host 0.0.0.0 --port 8000` → **bot-service**.
   - `uv run uvicorn src.recall_bridge:app --host 0.0.0.0 --port 8001` → **confluence-service**.
   - Add a `.dockerignore` for `stress_corpus/`, `sample_docs/`, `tests/`, `.planning/` to keep it lean.
2. **`jarvis-org` image** — new slim Dockerfile for `user_service/` (`python:3.12-slim`, `pip install -r user_service/requirements.txt`, `uvicorn user_service.main:app --port 8003`). Tiny deps → fast cold start → good scale-to-zero citizen.

> Don't build a single image from root `requirements.txt` (legacy/monolith, §5).

### 8.2 Decoupling (keep separate)
- **org-service is its own image + own container app.** Different deps, different lifecycle, stateless/elastic. No reason to couple it to the agent image.
- **agent worker stays on LiveKit Cloud**, separate from your HTTP plane.

### 8.3 Coupling (deliberately share)
- **bot-service + confluence-service share the `jarvis-agent` image** (same code tree `my-agent/src`, same `.env.local`, same deps). Deploy as **two container apps from that one image with different `CMD`s**. They stay separate *processes/apps* (so a 20-min pipeline on 8001 can't starve webhook handling on 8000, per §4), but you build/version once.
  - Cheapest variant if you're on a single small VM instead of ACA: run both as two `uvicorn` processes in one `docker compose`/Procfile — still two processes, one image.
  - Do **not** merge them into one FastAPI process: their `/sessions/{id}/...` route prefixes overlap and both define `/health`, and §4 wants independent restart/scale behavior.

### 8.4 Replica / scaling rules (ACA)
| App | min | max | Scale-to-zero? | Reason |
|---|---|---|---|---|
| bot-service (8000) | 1 | 1 | ❌ | in-proc caches + must catch Recall webhooks anytime (§4.1, §4.5) |
| confluence-service (8001) | 1 | 1 | ❌ (while jobs run) | in-proc pipeline + SSE pinned to process (§4.2) |
| org-service (8003) | 0 | N | ✅ | stateless, light (§4.6) |
| frontend | — | — | — | Static Web Apps / CDN |

> To unlock scale-to-zero/HA for 8000 & 8001 later: externalize `_compactors`/`_bot_index`/`_sync_jobs`/`_pipelines` into Supabase or Redis, then raise max replicas.

---

## 9. Approximate cost (verify against current pricing pages)

> Figures are **representative, low-traffic estimates** as of 2026-06 — confirm current rates. Variable per-use SaaS (LLM tokens, STT minutes, Recall bot-hours) is **separate** and dominates at scale.

| Layer | Azure (recommended) | AWS free tier (alt) |
|---|---|---|
| Agent runtime | LiveKit Cloud usage-based (free tier for dev) | same (don't self-host) |
| bot + confluence svc | ACA, min=1 each: small always-on consumption; partly offset by free grant → **~$5–15/mo combined** | ~$0 on legacy t3.micro (12 mo), then paid |
| org svc | ACA scale-to-zero → **~$0** idle | shares the VM |
| Frontend | Static Web Apps free | S3+CloudFront ~free-tier |
| TLS / public HTTPS | included (managed FQDN) | DIY Caddy/LE (free) or ALB **~$16/mo** |
| Supabase/Pinecone | free tier | free tier |
| **Your compute total** | **~$5–15/mo** | **~$0 for ≤6–12 mo, then ramps** |

**Takeaway:** Azure costs a few dollars/month but is durable, HTTPS-managed, and near-zero-ops. AWS free tier is $0 cash short-term but time-boxed and ops-heavy.

---

## 10. Deployment plan (recommended path)

**Phase 0 — prep**
1. Generate `uv.lock` in `my-agent/` if not committed (`uv lock`); add `.dockerignore` (exclude `stress_corpus/`, `sample_docs/`, `tests/`, `.planning/`).
2. Write the slim `user_service/Dockerfile`.
3. Replace `.env.local` usage with platform secrets (keep `load_dotenv` as local fallback).
4. Set explicit `CORS_ORIGINS` (frontend prod origin) on all three services; ensure `ALLOW_DB_RESET` is unset in prod.

**Phase 1 — agent on LiveKit Cloud**
5. `lk cloud auth`; from `my-agent/`, `lk agent deploy` (uses the existing Dockerfile + `livekit.toml`). Set agent secrets (Cerebras/OpenAI/AssemblyAI/Atlassian/Supabase/Pinecone/GitHub) in the LiveKit Cloud dashboard.

**Phase 2 — HTTP services on Azure**
6. Build & push `jarvis-agent` and `jarvis-org` images to Azure Container Registry.
7. Create a Container Apps environment. Create three apps:
   - `bot-svc` (image `jarvis-agent`, CMD = bot_service, ingress 8000, min=max=1).
   - `confluence-svc` (image `jarvis-agent`, CMD = recall_bridge, ingress 8001, min=1).
   - `org-svc` (image `jarvis-org`, ingress 8003, min=0).
8. Set `BRIDGE_SERVER_URL` = bot-svc's managed HTTPS FQDN. Register that FQDN's `/recall-webhook` in Recall.ai.
9. Inject all secrets per §12.

**Phase 3 — frontend**
10. Deploy `sync-sage-bot/` to Azure Static Web Apps. Configure prod env to route `/api → bot-svc`, `/conf-api → confluence-svc`, `/org-api → org-svc` (or set `VITE_API_BASE_URL`/`VITE_BACKEND_KIND` + the matching base URLs), and `VITE_SUPABASE_URL`/`VITE_SUPABASE_ANON_KEY`.

**Phase 4 — verify**
11. `GET /health` on all three FQDNs. Start a test meeting → confirm Recall loads `/bot-page`, agent joins, transcript ingests, pipeline streams over SSE, proposal applies to Confluence.

---

## 11. Required environment variables (per deployment target)

**LiveKit Cloud — agent** (`my-agent/.env.example` is the source of truth):
`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `CEREBRAS_API_KEY`, `OPENAI_API_KEY`, `ASSEMBLYAI_API_KEY`, `ATLASSIAN_USER_EMAIL`, `ATLASSIAN_API_TOKEN`, `ATLASSIAN_DOMAIN`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `PINECONE_API_KEY`, optional `GITHUB_TOKEN`/`GITHUB_DEFAULT_REPO`, optional `ROVO_MCP_*`.

**bot-service (8000):** `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `RECALL_API_KEY`, `RECALL_API_REGION`, **`BRIDGE_SERVER_URL`** (= its own public HTTPS FQDN), `BOT_NAME`, `AGENT_NAME` (=`my-agent`), `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `PINECONE_API_KEY` (RAG sync), `CORS_ORIGINS`.

**confluence-service (8001):** `CEREBRAS_API_KEY`, `OPENAI_API_KEY`, `ATLASSIAN_USER_EMAIL`, `ATLASSIAN_API_TOKEN`, `ATLASSIAN_DOMAIN`, `ATLASSIAN_SPACE_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `PINECONE_API_KEY`, `MY_AGENT_REVIEW_MODEL`, `MY_AGENT_PIPELINE_MAX_PAGES`, `CORS_ORIGINS`, optional `ROVO_MCP_*`.

**org-service (8003):** `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `JWT_SECRET`, `JWT_ACCESS_TTL_MINUTES`, `JWT_REFRESH_TTL_DAYS`, `CORS_ORIGINS`, `ALLOW_DB_RESET=false`.

**frontend:** `VITE_API_BASE_URL`/`VITE_BACKEND_KIND` (+ per-service base URLs or proxy rules), `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`.

---

## 12. Production hardening checklist
- [ ] Lock `CORS_ORIGINS` to real origins (no `*` with credentials).
- [ ] `ALLOW_DB_RESET` unset/false; consider removing `/admin/reset` route in prod builds.
- [ ] Secrets via platform store, not committed `.env.local`.
- [ ] Build images from `my-agent/pyproject.toml` + `user_service/requirements.txt` — **not** root `requirements.txt`.
- [ ] `.dockerignore` excludes `stress_corpus/`, `sample_docs/`, `tests/`, `.planning/`, `local_doc_change/`.
- [ ] Confirm `local_doc_change/` is dead and not deployed.
- [ ] Add `/health`-based liveness/readiness probes (all three expose `/health`).
- [ ] Plan to externalize bot/confluence in-process state (Supabase/Redis) before raising replica counts.
- [ ] Regenerate `CLAUDE.md` / `API_ROUTES.md` / `.planning/` to match the real architecture.

---

## 13. ⚠️ Note on stale docs (for AI models reading this repo)
`Confluence/CLAUDE.md`, `Confluence/API_ROUTES.md`, and `Confluence/.planning/codebase/*` describe an **obsolete** architecture (`confluence_logic/jarvis_agentic.py` on port 8000, a Next.js `review-ui/`, `local_office_logic/`, Neo4j graph RAG). **None of that is the current backend.** Trust **this file** and the actual source under `my-agent/src/` and `user_service/`. The current LLM stack is **Cerebras `gpt-oss-120b` + OpenAI fallback** (not "GPT-5"), retrieval is **Pinecone** (Neo4j is legacy), and the frontend is **`sync-sage-bot/` (Vite)** (not `review-ui/`).

---

## 14. AWS vs Azure — head-to-head comparative study

> **Accounts on hand:** AWS = free tier only. Azure = a funded account with **~$150 credit**.
> **What's being compared:** *only* the HTTP plane (bot 8000, confluence 8001, org 8003) + static frontend. The voice agent goes to **LiveKit Cloud regardless of cloud** (§6.1), so it is not a differentiator.

### 14.1 The two constraints that decide this
Before pricing, two hard requirements from the code eliminate most "cheap serverless" options on *both* clouds:

1. **Public HTTPS, always reachable** (bot-service serves `bot.html` to Recall + receives webhooks). Needs managed TLS or you build it.
2. **Long-lived connections** — the confluence pipeline streams **SSE for up to 20 minutes** as an in-process `asyncio` task (§4.2). This **disqualifies**:
   - **AWS Lambda + API Gateway** (API GW caps responses ~30 s; Lambda max 15 min; SSE/streaming is painful). ❌ for confluence-service.
   - **AWS App Runner** — historically short request timeouts; not built for 20-min SSE. ⚠️
   - Anything that scales-to-zero *mid-request* on the single-replica services. 
   The services that *do* fit cleanly: **Azure Container Apps**, **AWS ECS/Fargate**, **AWS EC2**, **AWS Lightsail Containers**, **Azure App Service**.

### 14.2 Service-by-service mapping

| Need | Azure option | AWS option |
|---|---|---|
| bot + confluence (single-replica, always-on, public HTTPS, long SSE) | **Container Apps** (min=1), managed FQDN+TLS free | **ECS Fargate** behind ALB, or **EC2 t3.micro + Caddy**, or **Lightsail container** |
| org-service (stateless, elastic, scale-to-zero) | **Container Apps** (min=0) | **App Runner** or **Fargate** |
| static frontend | **Static Web Apps** (free) | **S3 + CloudFront** |
| managed TLS / public URL | **included per app** | ALB (~$16/mo) **or** DIY Caddy/Let's Encrypt on EC2 |
| secrets | ACA secrets / Key Vault | SSM Parameter Store / Secrets Mgr |

### 14.3 Cost over the realistic horizon (low traffic)

> Representative, **verify current rates**. Per-use SaaS (LLM/STT/Recall) is excluded — identical on both clouds.

**Azure (with ~$150 credit):**
- ACA bot + confluence (2× min=1 small) + org (scale-to-zero) ≈ **$8–15/mo**, partly offset by the free monthly grant.
- Static Web Apps = **$0**. Managed TLS = **$0**.
- ⇒ Your **$150 credit covers roughly 10–18 months** of the entire HTTP plane. **Effective cash cost ≈ $0** for the foreseeable runway, with **zero TLS/OS ops**.

**AWS (free tier):**
- *If legacy 12-mo tier:* one **t3.micro (1 GB)** running all three via `docker compose` = **~$0 for 12 months** — but 1 GB is tight for 3 Python services + a 20-min pipeline, and you DIY the OS, TLS (Caddy), restarts, monitoring. Add an ALB for clean TLS and it's **+~$16/mo** (erasing the savings).
- *If new credit plan (~$200 / 6 mo):* effectively free for ~6 months **then bills** — and you'd be paying AWS for an account you don't otherwise fund.
- *Fargate/ALB "proper" setup* (no free tier on Fargate): **~$25–40/mo** for 2 always-on tasks + ALB.

### 14.4 Weighted decision matrix (1–5, higher = better for this workload)

| Criterion | Weight | Azure (ACA) | AWS free tier |
|---|---|---|---|
| Fits 20-min SSE cleanly | ✦✦✦ | 5 | 3 (only EC2/Fargate; not Lambda/App Runner) |
| Managed public HTTPS (Recall) | ✦✦✦ | 5 | 2 (DIY Caddy, or pay ALB) |
| Cash cost over 12+ mo | ✦✦✦ | 5 (credit ⇒ ~$0) | 4 legacy / 2 new-plan |
| Ops burden | ✦✦ | 5 (no servers/TLS) | 2 (patch OS, certs, restarts) |
| Scale-to-zero for org-svc | ✦ | 5 | 3 |
| RAM headroom for pipeline | ✦✦ | 5 (set per-app) | 2 (1 GB micro is tight) |
| Single bill / no 2nd cloud | ✦ | 5 (already funded) | 3 |
| Latency to Recall/LiveKit/Supabase | ✦ | 4 (irrelevant — webhooks/REST, not realtime) | 4 |
| **Weighted leaning** | | **Clear winner** | Runner-up |

Latency note: the realtime media path lives entirely on **LiveKit Cloud + the agent**, so the HTTP plane's cloud/region is **latency-insensitive** (webhooks + REST only). AWS's "closer to Recall (us-west-2)" gives no meaningful edge here.

### 14.5 Verdict

**Host on Azure (Container Apps + Static Web Apps). Burn the $150 credit.**

Rationale in one breath: the credit makes Azure **effectively $0** for ~1–1.5 years, it's the platform you already fund, it satisfies the **public-HTTPS** and **20-minute-SSE** requirements with **managed TLS and no servers to babysit**, and it lets you size RAM per-app instead of cramming everything into a 1 GB micro. AWS free tier only wins on raw short-term cash *if* you're on the legacy 12-month tier **and** accept the DIY-TLS/ops tax — and even then a clean setup needs a paid ALB. Keep AWS in your pocket as a **fallback / throwaway-demo** option, not the primary home.

> **Hybrid worth noting:** you could put the static frontend on **AWS S3+CloudFront** (deep free tier) and the HTTP services on Azure ACA. Marginal benefit; adds a second console. Not worth it unless you're already living in AWS for other reasons.

### 14.6 One thing to verify before committing
Check the **AWS Billing console** to learn which free-tier model your account is on (legacy 12-month vs new 6-month/$200 credit). It changes AWS's standing from "free for a year" to "free for ~6 months then paid," but it does **not** change the verdict — Azure still wins on ops + TLS + the SSE fit. The check just tells you how good your *fallback* is.

---

## 15. Self-hosting LiveKit + the LLM (dropping the paid SaaS)

> Scenario: you cannot pay **LiveKit Cloud** or **Cerebras** and want to run equivalents on your own instances. This is **feasible for LiveKit** and **partly feasible for the LLM** — but there is a hard hardware wall on the 120B model you must understand first.

### 15.1 What exactly is Cloud-only in this code (so you know what breaks)
Grepped from `my-agent/src/agent.py`:

| Line | Call | Cloud dependency | On self-host you must… |
|---|---|---|---|
| 243, 600 | `cerebras.LLM(model="gpt-oss-120b")` | Cerebras API (paid) | repoint to a self-hosted or free OpenAI-compatible LLM |
| 763, 803 | `inference.TTS(model="cartesia/sonic-3")` | **LiveKit Inference (Cloud-only)** | swap to a TTS *plugin* (Cartesia key / self-hosted Piper / `edge-tts`) |
| 759, 798 | `assemblyai.STT(model="u3-rt-pro")` | AssemblyAI (paid) | keep paying, or self-host Whisper |
| 835 | `ai_coustics.audio_enhancement(...)` | **LiveKit Cloud noise cancellation** | remove it, or replace with another NC |

Plus `recall_bridge.py` (confluence-service) uses Cerebras via `https://api.cerebras.ai/v1` with an OpenAI fallback — same LLM swap applies there.

**Consequence #1:** self-hosting LiveKit means **LiveKit Inference and `ai_coustics` stop existing for you** — you can't "self-host" those; you replace them with plugins/your own models.
**Consequence #2:** the **agent worker comes back onto your infrastructure.** On LiveKit Cloud you'd `lk agent deploy`; self-hosted, *you* run the agent worker process (the Dockerfile `start` command) as a long-lived container pointed at your LiveKit URL. So your hosted footprint grows from 3 services → 4 (+ the LiveKit server itself).

### 15.2 Self-hosting LiveKit — how it works
LiveKit's media server (SFU) is open-source (`livekit/livekit-server`, a Go binary/Docker image). The SDK code barely changes — you just repoint `LIVEKIT_URL/API_KEY/API_SECRET` at your server. Token minting, agent dispatch (`create_dispatch`, `AgentServer`, `@server.rtc_session`) all work against the OSS server.

**What you must stand up:**
- **The server**: `docker run livekit/livekit-server` (or compose). Generate your own `API_KEY`/`API_SECRET` (replaces the Cloud pair).
- **Networking (the real work — media is UDP):**
  - `7880/tcp` — signaling (WebSocket). Must be behind **WSS/TLS** with a public domain (browsers + Recall require secure WS).
  - `7881/tcp` — RTC over TCP fallback.
  - `50000–60000/udp` (or a configured range) — actual audio/video media. **These must be open to the internet.**
  - **TURN server** (LiveKit has embedded TURN; for prod expose TURN/TLS on `443`/`5349`) for clients behind strict NAT.
- **Public IP + domain + TLS cert** (Caddy/Let's Encrypt in front, or LiveKit's built-in TLS).
- **Redis** — only needed for multi-node; a single node is fine for your low participant count.

**Who connects, and why low scale is on your side:** per meeting the only LiveKit participants are the **Recall browser bot** (publishes mixed meeting audio), the **agent** (subscribes to audio, publishes TTS), and occasionally a **user browser** (jarvis-call mode). That's ~2–3 participants per room → a single small/medium CPU VM handles many concurrent meetings. LiveKit SFU is light on CPU; your real cost is **egress bandwidth**.

**Code changes:** point env at your server; replace `inference.TTS(...)` with `cartesia.TTS(...)` (needs a Cartesia key) **or** a self-hosted/free TTS (below); delete the `ai_coustics` noise-cancellation block (or swap it).

> ⚠️ Free-tier reality: a self-hosted LiveKit server wants a **public IP + open UDP range + TLS**. **Azure Container Apps is a poor fit** for the media server (HTTP-centric ingress, no arbitrary UDP). Run the LiveKit server on a **plain VM** (Azure VM / AWS EC2) with a public IP and a wide UDP port range. The HTTP services (§2) can stay on ACA; only the LiveKit server + agent worker need the VM.

### 15.3 "Self-hosting Cerebras" — the hard truth about `gpt-oss-120b`
You cannot host Cerebras's hardware. You *can* self-host the **open-weights model** `gpt-oss-120b` with vLLM / TGI / Ollama (all expose OpenAI-compatible APIs, so the swap is just `base_url` + model name + key). **But the hardware:**

- `gpt-oss-120b` ≈ 120B params; even at MXFP4 quantization it needs **~63 GB of VRAM → an 80 GB GPU (H100/A100-80G)**.
- On-demand an 80 GB GPU is **~$2–3/hour ≈ $1,500–2,200/month** if always-on. That is **far more expensive than just paying Cerebras per token.** Self-hosting the 120B to "save money" **backfires**.
- **Neither free option gives you this GPU:** AWS free tier has **no GPU**; your **$150 Azure credit** evaporates in a few days on a GPU VM.

**So the realistic moves (pick per service, by latency tolerance):**

| Path | Good for | Cost | Quality/latency |
|---|---|---|---|
| **A. Free OpenAI-compatible API** (Groq free tier, Cerebras free tier, OpenRouter free models, Google AI Studio) | **realtime agent** (needs speed) | $0 within limits | Groq/Cerebras free tiers run gpt-oss/Llama *fast* — best "don't pay" option for voice |
| **B. Self-host a *small* model** (gpt-oss-20b ~16 GB, Llama-3.1-8B/Qwen2.5 ~8–14 GB) via **Ollama/vLLM** on a modest GPU (RTX 4090/A10) | **confluence pipeline (8001)** — async, 20-min budget tolerates slower LLM | GPU VM hourly, or your own box | lower quality than 120B, but fine for batch drafting |
| **C. Keep paying Cerebras** just for the agent's realtime turn | realtime, if free tiers too limited | per-token (cheap at low volume) | best |

**Recommended split:** realtime **agent → free fast API (Groq/Cerebras free tier)**; async **pipeline → self-hosted small model (Ollama)** or the same free API. Because the agent is latency-critical and the pipeline is not, you avoid both the GPU bill *and* a sluggish voice experience. The code is OpenAI-compatible throughout, so this is a `base_url`/model swap, not a rewrite.

### 15.4 STT and TTS if you also can't pay AssemblyAI / Cartesia
Removing LiveKit Inference forces a TTS decision anyway; if budget is zero, also reconsider STT:
- **TTS (free/self-host):** `edge-tts` (free, already in your deps — Microsoft voices, cloud-dependent but $0) → simplest. Or self-host **Piper** (fast, CPU-only, OK quality) / **Kokoro** / **XTTS** (GPU, better quality). Wrap as a LiveKit TTS plugin or stream PCM.
- **STT (free/self-host):** **faster-whisper** / `whisper.cpp` streaming on CPU or a small GPU. Latency/accuracy is worse than AssemblyAI `u3-rt-pro`, and you lose `keyterms_prompt` wake-word locking — expect more "Jarvis" mis-triggers; compensate with the existing `_WAKE_PATTERN` fuzzy matching.

### 15.5 Resulting self-hosted topology
```
  Plain VM (public IP, UDP open, TLS)            Your existing host (ACA or VM)
  ┌───────────────────────────────────┐         ┌──────────────────────────────┐
  │ livekit-server (SFU)  7880/7881    │◀── WSS ─│ bot-service 8000 (mint tokens │
  │   + TURN, UDP 50000-60000          │         │   now point at YOUR lk URL)   │
  │ agent worker (agent.py start)      │         │ confluence-service 8001       │
  │   LLM → free API or local Ollama   │         │ org-service 8003              │
  │   STT → faster-whisper             │         └──────────────────────────────┘
  │   TTS → edge-tts / Piper           │              │ shared state → Supabase
  │   (ai_coustics REMOVED)            │         Recall bot ─WSS/UDP▶ livekit-server
  └───────────────────────────────────┘
        (optional) GPU box for Ollama small model ── OpenAI-compat ──▶ agent + pipeline
```

### 15.6 Honest recommendation for "we can't pay them"
- **LiveKit → self-host: yes, do it.** Genuinely free-ish on a CPU VM you already have credit for; the only real work is UDP/TURN/TLS networking. Accept that the **agent worker returns to your infra** and that you must replace `inference.TTS` + `ai_coustics`.
- **Cerebras/`gpt-oss-120b` → do NOT buy a GPU to self-host the 120B.** It costs ~10–100× more than the API. Instead: **realtime agent on a free fast API (Groq/Cerebras free tier)**, **async pipeline on a self-hosted small model (Ollama)** if you want true independence. Same OpenAI-compatible interface → trivial swap.
- **Net:** the cheapest *and* most reliable "don't pay" setup is **self-hosted LiveKit + edge-tts + faster-whisper + free-tier LLM API**, with a small local Ollama model only for the non-realtime pipeline. Reserve a paid GPU only if you have a hard requirement to run the full 120B in-house.

---

## 16. DECISION — self-host LiveKit, keep Cerebras (cheapest compliant host)

> **Locked scope (per the team):** Cerebras stays (its plugin calls the Cerebras API directly and is unaffected by where LiveKit runs). **AssemblyAI STT also stays** (same — direct API). The expensive line item is **LiveKit Cloud**, so we self-host *only the LiveKit server* (+ the agent worker, which self-hosting forces back onto our infra). This section is the actionable plan.

### 16.1 What changes in code when LiveKit goes self-hosted (Cerebras kept)
| Component | Change | Effort |
|---|---|---|
| `LIVEKIT_URL / API_KEY / API_SECRET` | Repoint to your server + your generated key pair | env only |
| `cerebras.LLM("gpt-oss-120b")` (agent 243/600) | **No change** — direct Cerebras API ✅ | none |
| `assemblyai.STT("u3-rt-pro")` (759/798) | **No change** — direct AssemblyAI API ✅ | none |
| `inference.TTS("cartesia/sonic-3")` (763/803) | **BREAKS** — Inference is Cloud-only. Swap to `cartesia.TTS(...)` (Cartesia API key, cheap tier) **or** free `edge-tts` | small |
| `ai_coustics.audio_enhancement(...)` (835) | **BREAKS** — Cloud NC. Remove. **Good news:** it's only in the *standard/direct-browser* branch, **not** the Recall meeting path — so the main flow is unaffected | trivial |
| Agent worker deploy | Stop using `lk agent deploy`; run the Docker image's `start` command yourself, pointed at your LiveKit URL | infra |

So the only *required* code edit for the core meeting flow is **replace `inference.TTS` with a plugin** (and drop the unused-on-this-path `ai_coustics`).

### 16.2 Official self-host requirements (confirmed from LiveKit docs, 2026-06)
- **Ports to open on the VM:** `80/tcp` (cert issuance), `443/tcp+udp` (HTTPS + TURN/TLS), `7881/tcp` (WebRTC/TCP), `3478/udp` (TURN/UDP), **`50000–60000/udp` (WebRTC media)**. (Optional: `7880` HTTP, `6789` metrics.)
- **TLS:** a real domain (e.g. `wss://livekit.yourhost.com`) + CA cert. Caddy auto-provisions Let's Encrypt. Self-signed won't work. TURN needs its own domain/cert.
- **Redis:** not required for a single node; recommended only for multi-node.
- **Scaling bound:** "CPU and bandwidth" (docs recommend 10 Gbps for *large* production). **Your workload is audio-only, ~2–3 participants/room → tiny bandwidth, light CPU on the SFU.**
- **Install:** `docker run --rm -it -v$PWD:/output livekit/generate` produces `caddy.yaml`, `docker-compose.yaml`, `livekit.yaml`, `redis.conf`, and a `cloud-init.yaml` / `init_script.sh`. Paste the cloud-init into the VM's "user data" at launch — Docker + compose + a `livekit-docker` systemd service are set up automatically. Use **host networking** in Docker.

### 16.3 The real sizing driver is the agent worker, not the SFU
The SFU for audio-only/low concurrency is featherweight. The heavier tenant on the VM is the **agent worker**: it loads Silero VAD + the multilingual turn detector (CPU, a few hundred MB each) and spawns a job process **per concurrent meeting**. Size RAM for expected concurrency:
- 1–3 concurrent meetings → **2 vCPU / 4 GB** comfortable.
- This is why **AWS's only free VM (t3.micro, 1 GB) is too small** — it can host the SFU alone but chokes once the agent worker + models + a couple of meetings land on it. On AWS you'd move to t3.small/medium, which are **not** free → you pay either way.

### 16.4 Cheapest compliant host — comparison

| Option | Spec | ~Cost | Fit for this job |
|---|---|---|---|
| **Azure VM, B-series burstable** (B2s / B2as_v2, 2 vCPU/4 GB) | bursts CPU, 100 GB/mo egress free | **~$15–30/mo → ~$0 against your $150 credit for ~5–10 mo** | ✅ Right-sized; uses credit you already hold; one cloud with the HTTP services |
| AWS EC2 free tier (t3.micro, 1 GB) | 750 h/12 mo free | $0 but **too small** for agent worker | ⚠️ SFU-only; need paid t3.small/medium for the agent → not actually free |
| **Oracle Cloud "Always Free"** (Ampere A1, up to 4 vCPU/24 GB ARM) | free **forever**, 10 TB/mo egress | **$0 forever** | ✅✅ True cost-floor *if* you'll use a 3rd provider & can get A1 capacity in a region |
| Hetzner CX22 / CAX11 | 2 vCPU/4 GB, ~20 TB traffic | **~€4/mo** | ✅ Cheapest *paid*, far cheaper egress than hyperscalers |

**Egress note:** hyperscalers bill data-out (~$0.08–0.12/GB) beyond a 100 GB/mo free allowance. Audio-only at your scale stays well under that; if usage ever grows, Hetzner/Oracle's huge egress allowances win decisively.

### 16.5 Verdict
**Primary: run LiveKit server + agent worker on one Azure B-series VM (B2s/B2as_v2, 2 vCPU/4 GB), funded by your $150 credit.**
- Meets every networking requirement (public IP, full UDP range, Caddy TLS) that ACA/Fargate can't.
- AWS free tier is *not* genuinely free here — the only free VM is too small for the agent worker, so you'd pay on AWS anyway. Given equal technical fit, your **Azure credit breaks the tie** and keeps you on one cloud.
- Keep the 3 HTTP services on **Azure Container Apps** (§14) and the LiveKit-VM separate — clean split: HTTP plane on ACA, realtime/UDP plane on the VM.

**If "$0 forever" outranks staying on one cloud:** put the LiveKit VM on **Oracle Cloud Always Free (Ampere A1, 4 vCPU/24 GB)** — genuinely free indefinitely and roomy enough for SFU + agent + many meetings. Trade-off: a third provider/account and occasional A1 capacity hunting. **Hetzner (~€4/mo)** is the no-fuss cheap-paid fallback.

### 16.6 Deploy steps (Azure VM path)
1. Buy/point a domain: `livekit.<you>.com` (+ `turn.<you>.com`) → the VM's public IP.
2. Create an **Azure B2s/B2as_v2 VM** (Ubuntu 22.04). NSG inbound: `80,443/tcp`, `443,3478/udp`, `7881/tcp`, `50000-60000/udp`.
3. On your laptop: `docker run --rm -it -v$PWD:/output livekit/generate` → enter the domains. Paste the generated **cloud-init** into the VM (or copy `init_script.sh` and `sudo ./init_script.sh`). Save the generated **API key/secret**.
4. Verify: `wss://livekit.<you>.com` reachable; `systemctl status livekit-docker`.
5. **Agent worker:** run the `my-agent` image with CMD `uv run src/agent.py start`, env `LIVEKIT_URL/KEY/SECRET` = your server, `CEREBRAS_API_KEY`, `ASSEMBLYAI_API_KEY`, Supabase/Pinecone/Atlassian as before. Run it **on the same VM** (compose alongside LiveKit) to start.
6. **Code:** replace `inference.TTS(...)` → `cartesia.TTS(...)` (add `CARTESIA_API_KEY`) or `edge-tts`; delete the `ai_coustics` noise-cancellation block. Run `uv run pytest` (the project mandates TDD for agent-behavior changes).
7. **bot-service** (still on ACA): no change — it already mints tokens against `LIVEKIT_URL/KEY/SECRET`; just point those at your server. Recall connects to your `wss://` URL via the minted token exactly as before.
8. Smoke test a meeting end-to-end; watch for ICE/UDP failures (→ confirm the 50000–60000 UDP range is actually open and TURN domain resolves).
