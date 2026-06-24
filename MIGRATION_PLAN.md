# Jarvis Backend — Migration & Phase Plan: Self-Hosted LiveKit + Azure

> **Companion to** `Confluence/HOSTING_ANALYSIS.md` (the *what/why/where*). **This doc is the *how*** — a phased, verifiable migration plan to move off LiveKit Cloud onto a self-hosted LiveKit server + Azure, **keeping Cerebras and AssemblyAI**, with each phase independently testable and reversible.
> **Generated:** 2026-06-13. Verified against live source under `my-agent/src/` + `user_service/` and against LiveKit docs (see §4). **Do not trust the stale `.planning/`/`CLAUDE.md`/`API_ROUTES.md`** — see HOSTING_ANALYSIS §13.

---

## 1. Why we're doing this (context from the working session)

LiveKit **Cloud** is the costly recurring line item. LiveKit is open-source, so we self-host the media server (SFU) on a cheap VM to kill that bill. Cerebras (LLM) and AssemblyAI (STT) are **kept** — their plugins call those APIs directly and are unaffected by where LiveKit runs. The only Cloud-*locked* features in our code are **LiveKit Inference** (used for TTS) and **`ai_coustics` noise cancellation**, both of which must be replaced/removed when leaving Cloud.

We also have to land the services on real infrastructure anyway, so this plan folds the **provider swap** and the **deploy/containerization** into one sequenced migration, with verification gates so we never break the working meeting flow blindly.

**Decision recap (from HOSTING_ANALYSIS §14–16):**
- 3 HTTP services → **Azure Container Apps** (managed HTTPS, single-replica where required).
- LiveKit server + agent worker → **one Azure B-series VM** (B2s, 2 vCPU/4 GB) on the ~$150 credit. (Oracle Always Free A1 = $0-forever alternative.)
- Frontend → **Azure Static Web Apps**.
- Keep Cerebras + AssemblyAI; swap TTS to **Cartesia plugin** (cheap, low-effort) or **edge-tts/Piper** (free, more effort); drop `ai_coustics`.

---

## 2. The logic we run today (and why each piece exists)

| Component | File | Why it exists | Talks to |
|---|---|---|---|
| **bot-service** :8000 | `my-agent/src/bot_service.py` | Recall bot lifecycle, mints LiveKit tokens, dispatches the agent, ingests transcripts, serves `bot.html`, triggers RAG sync. Must be **public HTTPS** (Recall loads bot.html + posts webhooks). | Recall.ai, LiveKit (token/dispatch), Supabase, Pinecone |
| **confluence-service** :8001 | `my-agent/src/recall_bridge.py` | Post-meeting intelligence: multi-agent proposal pipeline (`review_pipeline/`), Confluence writes, summary/MOM, chat. Streams **SSE up to ~20 min**. | Cerebras (+OpenAI fallback), Atlassian, Supabase, Pinecone |
| **voice agent** (worker) | `my-agent/src/agent.py` | Realtime "Jarvis": STT→LLM→TTS in the room. Dispatched per meeting. **Today binds to LiveKit Cloud** via `inference.TTS` + `ai_coustics`. | LiveKit, AssemblyAI (STT), Cerebras (LLM), TTS, Pinecone |
| **org/user-service** :8003 | `user_service/main.py` | Auth (JWT/Google), teams, org hierarchy, RBAC, bot assignment, analytics. Stateless. | Supabase |
| **frontend** | `sync-sage-bot/` | Vite/React SPA; proxies to 8000/8001/8003; Supabase for Google sign-in. | the 3 services, Supabase |

Shared state lives in **Supabase** (`session_store.py`) — this is what lets the services be separate processes safely.

---

## 3. Constraints (the hard rules this plan must respect)

**From the code:**
1. **bot-service & confluence-service are single-replica.** In-process caches (`_compactors`, `_bot_index`, `_sync_jobs`, `_pipelines`) + in-process `asyncio` SSE pipeline are not shared/serializable. → `min=max=1`; no scale-to-zero mid-job.
2. **bot-service must be public HTTPS, always up** (Recall bot-page + webhooks). `BRIDGE_SERVER_URL` must be `https://`.
3. **confluence-service SSE runs up to ~20 min in-process** — restarts/replica-swaps kill the job. Avoid disruptive deploys while jobs run; keep `min=1`.
4. **Build from `my-agent/pyproject.toml` + `user_service/requirements.txt`**, NOT root `requirements.txt` (legacy monolith with Neo4j/docling/pyaudio we don't use).
5. **`local_doc_change/` is legacy** — exclude from images. Also exclude `stress_corpus/`, `sample_docs/`, `tests/`, `.planning/`.

**From the hosting decision:**
6. LiveKit self-host needs **open UDP 50000–60000 + 3478/udp + 443 + 7881/tcp + 80/tcp** and a **TLS domain** → **must be a VM**, not ACA/Fargate.
7. Self-hosting LiveKit **moves the agent worker onto our infra** (no `lk agent deploy`).
8. **Do NOT self-host `gpt-oss-120b`** (needs ~80 GB GPU ≈ $1.5–2k/mo). Cerebras API stays.

**Project process:**
9. `my-agent/AGENTS.md` mandates **TDD for agent-behavior changes** (instructions/tools/models) — write/adjust tests first, run `uv run pytest`.

---

## 4. Research re-confirmed (2026-06)

- **TTS swap is a one-liner + dep.** `inference.TTS(model="cartesia/sonic-3", voice="<id>")` → `cartesia.TTS(api_key=os.getenv("CARTESIA_API_KEY"), model="sonic-3", voice="<same id>")`. Add `livekit-plugins-cartesia` to `my-agent/pyproject.toml`. Cartesia `TTS` defaults: `model="sonic-3"`, `encoding="pcm_s16le"`, `sample_rate=24000`. The existing voice ID `9626c31c-…` carries over. ([Cartesia plugin docs](https://docs.livekit.io/agents/integrations/tts/cartesia/))
- **Dispatch/agent framework is identical on self-hosted.** "Running your agents is identical across localhost, self-hosted, and LiveKit Cloud." Explicit dispatch via `agent_name` + `AgentDispatch.createDispatch` works on the OSS server → **no code change to dispatch**, just repoint `LIVEKIT_URL/KEY/SECRET`. ([Agent dispatch docs](https://docs.livekit.io/agents/server/agent-dispatch/))
- **Self-host install path:** `docker run --rm -it -v$PWD:/output livekit/generate` → emits `caddy.yaml`, `docker-compose.yaml`, `livekit.yaml`, `redis.conf`, `cloud-init.yaml`. Caddy auto-issues Let's Encrypt TLS. Redis only needed for multi-node. Ports per §3.6. ([VM deploy guide](https://docs.livekit.io/home/self-hosting/vm/))
- **`ai_coustics` is only used in the non-Recall (direct-browser/console) branch** of `agent.py` (line ~835), **not** the Recall meeting path → removing it does not affect the primary meeting flow.

---

## 5. Target end-state (corrected — two Azure VMs, all on Azure)

> **Correction vs the first draft:** confluence-service (8001) is **NOT** on Azure Container Apps — ACA's ingress kills requests at **240 s**, and the proposal pipeline streams SSE for up to **20 min** (see §8 / concern A). It moves onto a VM behind Caddy, which has no such cap. bot-service (8000) moves **onto the realtime VM** because the agent posts transcripts to it over **localhost** (`BRIDGE_INTERNAL_URL=127.0.0.1:8000`, `agent.py:102/471`).

**Two VMs, by workload character:**

```
 Azure Static Web Apps
 ┌─────────────────┐
 │ sync-sage-bot   │  /api ───────────────┐         ┌────────── /conf-api , /org-api
 │ (static + CDN)  │                       │         │
 │ + Google OAuth ─┼─▶ Supabase            ▼         ▼
 └─────────────────┘        ┌─────────────────────────────┐   ┌──────────────────────────────┐
                            │  VM-A  "nice"  (B2s+ 2/4 GB) │   │  VM-B  "cheap" (B1ms ~2 GB)   │
 Recall.ai ──webhook/────────▶ caddy (TLS, public HTTPS)   │   │ caddy (TLS)                   │
   bot-page (HTTPS)         │  bot-service :8000  ◀─local──┐│   │ confluence-svc :8001 (SSE 20m)│
                            │  agent worker (agent.py start)│   │ org-service    :8003          │
 Recall browser bot ──WSS+UDP─▶ livekit-server (SFU)+TURN  ││   └──────────────┬────────────────┘
                            │  redis (from livekit/generate)││                  │ shared state
                            │  443/7881/3478/50000-60000    │└── tokens/dispatch ▼ Supabase
                            └───────────────┬───────────────┘     (both VMs read/write Supabase)
                                            └── Cerebras LLM · AssemblyAI STT · Cartesia TTS · Pinecone
```

**Why this split is correct:**
- **VM-A is the realtime/heavy box.** Its heavy residents are **livekit-server + the agent worker** (VAD/turn-detector models + one job process per concurrent meeting). bot-service (8000) rides along because the agent needs it on **localhost** — bot-service itself is light, but co-location makes the per-utterance transcript POST a loopback call (fast, and removes the cross-host data-loss path). Needs a "nice" VM: **B2s/B2as_v2 (2 vCPU / 4 GB) or larger**, public static IP + domain, full UDP range open.
- **VM-B is the low-compute box.** org-service (8003) is genuinely trivial CRUD-forwarding to Supabase. confluence-service (8001) is **mostly I/O-bound** (it orchestrates Cerebras/OpenAI/Pinecone calls — the heavy inference is on those APIs), so low CPU — **but it is NOT "pure forwarding":** it holds a 20-min in-process pipeline + SSE per job, so give it **~2 GB RAM headroom** and don't put it on a tier that sleeps/evicts mid-job. A **B1ms (1 vCPU / 2 GB ≈ $15/mo)** is comfortable; a free 1 GB B1s works only for light, low-concurrency use. Neither 8001 nor 8003 is latency-coupled to the live agent, so VM-B can be a separate cheap box with zero voice impact.

### Image inventory (what you actually build vs pull)
**Build (2 images):**
1. **`my-agent`** (from `my-agent/Dockerfile`) — **one image, three roles** via different commands: `agent.py start` (VM-A), `uvicorn src.bot_service:app` :8000 (VM-A), `uvicorn src.recall_bridge:app` :8001 (VM-B). 8001 reuses this same image — just a different command.
2. **`jarvis-org`** (slim, from `user_service/`) — org-service :8003 (VM-B).

**Pull, don't build (stock images):** `livekit/livekit-server`, `caddy`, `redis` — all from the `livekit/generate` output on VM-A; plus a stock `caddy` on VM-B for TLS.

So your "two images, one for LiveKit + one for the other" is right with one nuance: **LiveKit is a stock image (you don't build it)** — the only image *you* build for the realtime box is `my-agent` (run as two containers: bot-service + agent). On VM-B: `my-agent` (as confluence-service) + `jarvis-org`.

### Verdict on your plan
| Your statement | Verdict |
|---|---|
| 8001 + 8003 = separate images on one cheap VM | ✅ Right — both low-CPU, not latency-coupled. ⚠️ Caveat: 8001 is **not** pure forwarding (20-min SSE pipeline) → ~2 GB RAM, no sleepy/evicting tier. |
| 8000 needs its own nice VM | ◑ Direction right, reason off — the nice VM is for **LiveKit + agent**; 8000 rides along (localhost coupling), it isn't the heavy part. So it's "**LiveKit + agent + 8000**" on the nice VM, not "8000 alone." |
| 2 images: LiveKit + the other | ✅ Effectively — but LiveKit is a **stock** image; the only one you build for VM-A is `my-agent`. |

---

## 6. Phase-by-phase plan

> **Golden rule:** LiveKit Cloud stays live and is the rollback target until the Phase 6 cutover. Every phase has an explicit **exit gate**; do not proceed until it's green.

### Phase 0 — Baseline & safety net  *(no changes)*
**Goal:** know exactly what "working" looks like before touching anything.
- Record current working config: `.env.local`, `livekit.toml` (Cloud subdomain/agent id), and confirm a meeting works end-to-end on Cloud today.
- `cd my-agent && uv sync && uv run pytest` → capture the green baseline.
- Commit `uv.lock` if not tracked.
- Snapshot/export current secrets to the Azure Key Vault (or a secure note) — you'll reuse them.
**Exit:** baseline meeting verified on Cloud; pytest green; secrets inventoried.

### Phase 1 — Containerization & build hygiene  *(no behavior change, still on Cloud)*
**Goal:** clean, reproducible images for all four roles; prove they run unchanged against LiveKit Cloud.
- Add `my-agent/.dockerignore`: exclude `stress_corpus/`, `sample_docs/`, `tests/`, `.planning/`, `local_doc_change/`, `*.md`.
- Confirm the **single `my-agent` image, three CMDs**: `agent.py start` | `uvicorn src.bot_service:app --port 8000` | `uvicorn src.recall_bridge:app --port 8001`.
- Write a slim **`user_service/Dockerfile`** (`python:3.12-slim`, `pip install -r requirements.txt`, `uvicorn user_service.main:app --port 8003`).
- Local `docker-compose.dev.yaml`: run bot(8000) + confluence(8001) + org(8003), env still pointing at **LiveKit Cloud**.
**Exit:** all images build; `GET /health` green on 8000/8001/8003; a meeting still works via the containerized bot-service against Cloud.

### Phase 2 — Provider decoupling in code  *(TDD, still on Cloud — de-risks the cutover)*
**Goal:** remove Cloud-only model deps *while still on Cloud*, so the later infra flip is pure infra.
- **TTS:** add `livekit-plugins-cartesia`; replace both `inference.TTS(...)` calls (agent lines ~763, ~803) with `cartesia.TTS(...)`; add `CARTESIA_API_KEY`. *(Cartesia works on Cloud too → verify here.)* If going zero-cost instead, wrap `edge-tts`/Piper as a TTS adapter (more effort — see §7).
- **Noise cancellation:** remove/guard the `ai_coustics.audio_enhancement(...)` block (line ~835). Behind an env flag is fine; it's not on the Recall path.
- Drop the `ai_coustics` dependency from `pyproject.toml` once removed.
- **TDD:** add/adjust tests for the TTS path per `AGENTS.md`; `uv run pytest`.
**Exit:** agent runs on **Cloud** with Cartesia TTS and no `ai_coustics`; console + a real meeting sound correct; pytest green. *(Now the code is provider-portable.)*

### Phase 3 — Stand up self-hosted LiveKit (parallel, non-disruptive)
**Goal:** a working self-hosted LiveKit server, independent of production.
- Provision **Azure B2s VM** (Ubuntu 22.04). Open NSG: `80,443,7881/tcp`, `443,3478/udp`, `50000-60000/udp`.
- DNS: `livekit.<you>.com` (+ `turn.<you>.com`) → VM IP.
- `docker run --rm -it -v$PWD:/output livekit/generate` → paste cloud-init into the VM (or run `init_script.sh`). Save the generated **API key/secret**.
- Verify `wss://livekit.<you>.com` reachable; `systemctl status livekit-docker`; Caddy issued TLS.
- Smoke test with `lk` CLI or a throwaway token + the agents console.
**Exit:** a test client + test agent connect through the self-hosted server; **media flows over UDP** (no ICE failures).

### Phase 4 — Put the agent worker + bot-service (8000) on VM-A against self-hosted LiveKit
**Goal:** the realtime box complete — LiveKit + agent + bot-service co-located, talking over localhost.
- Extend the `livekit/generate` compose on **VM-A** with two `my-agent` containers + **host networking** (so localhost wiring works):
  - `agent` → CMD `uv run src/agent.py start`; env `LIVEKIT_URL/KEY/SECRET` = **self-hosted**, `BRIDGE_INTERNAL_URL=http://127.0.0.1:8000`, plus `CEREBRAS_API_KEY`, `ASSEMBLYAI_API_KEY`, `CARTESIA_API_KEY`, Supabase/Pinecone/Atlassian.
  - `bot-service` → CMD `uvicorn src.bot_service:app --port 8000`; env `LIVEKIT_URL/KEY/SECRET` = self-hosted, `RECALL_API_KEY`, **`BRIDGE_SERVER_URL=https://<vm-a-domain>`** (public, for Recall to load `bot.html`), Supabase/Pinecone, explicit `CORS_ORIGINS`.
- ⚠️ **Two different URL vars — don't conflate:** `BRIDGE_SERVER_URL` = **public** HTTPS (Recall loads `bot.html` + posts webhooks here); `BRIDGE_INTERNAL_URL` = **localhost** (agent → bot-service transcript POST). Pin a **static public IP + domain** for VM-A so `BRIDGE_SERVER_URL` never changes.
- Caddy on VM-A: route public `:443` → bot-service `:8000` for `/bot-page`, `/recall-webhook`, `/sessions/*`, etc.
- `agent.py download-files` already runs in the image build → VAD/turn-detector models present.
- Test a **jarvis-call / console** session end-to-end on the self-hosted stack.
**Exit:** agent registers as `agent_name="my-agent"`, accepts a dispatched job, does STT→Cerebras→Cartesia TTS via self-hosted LiveKit; bot-service reachable at `https://<vm-a-domain>/health`; transcript POSTs hit localhost.

### Phase 5 — Deploy confluence-service (8001) + org-service (8003) on VM-B  *(NOT ACA)*
**Goal:** the low-compute HTTP plane on a cheap VM, dodging ACA's 240 s SSE cap.
- Provision **VM-B** (B1ms, 1 vCPU / 2 GB; static IP + domain `api.<you>.com`). One `docker-compose` + Caddy:
  - `confluence-svc` → `my-agent` image, CMD `uvicorn src.recall_bridge:app --port 8001`; env: `CEREBRAS_API_KEY`, `OPENAI_API_KEY`, Atlassian, Supabase, Pinecone, `CORS_ORIGINS`.
  - `org-svc` → `jarvis-org` image, :8003; env: Supabase, `JWT_SECRET`, `CORS_ORIGINS`, `ALLOW_DB_RESET` **unset**.
- Caddy routes `/conf-api/*`→8001, `/org-api/*`→8003 (or use subdomains). Confirm Caddy's proxy read timeout is unbounded/long so the 20-min SSE survives (it is by default; the code already sends a 30 s `ping`).
- Both services read/write the **same Supabase** as VM-A → no server-to-server calls needed.
**Exit:** `/health` green on 8001 + 8003 via `https://api.<you>.com`; a manual `/review/pipeline/start` + SSE stream runs past 5 minutes without being cut.

### Phase 6 — End-to-end cutover (Recall + frontend)
**Goal:** real meeting works fully on the new stack.
- Register Recall webhook → `https://<vm-a-domain>/recall-webhook`.
- Deploy frontend to **Static Web Apps**; set prod routes (`/api`→VM-A bot-service, `/conf-api`→VM-B 8001, `/org-api`→VM-B 8003) + `VITE_SUPABASE_*`.
- Run a **live meeting**: Recall bot joins the **self-hosted** room (verify UDP/ICE), agent transcribes + responds, transcript ingests (localhost), end meeting → VM-B pipeline streams SSE → apply a proposal to Confluence.
**Exit:** full meeting → summary → proposal → apply succeeds. **Rollback** = repoint `LIVEKIT_URL/KEY/SECRET` back to Cloud.

### Phase 7 — Harden, monitor, decommission Cloud (realize the savings)
**Goal:** safe steady-state + the cost win banked.
- TLS auto-renew confirmed on both VMs; security groups least-privilege; per-service health checks + container `restart: always`.
- Log hygiene (no secrets/tokens in logs); CPU/RAM monitoring on both VMs; **alarm on agent worker OOM** under concurrency (VM-A) and on confluence-svc RAM during pipeline runs (VM-B).
- Back up `livekit.yaml` keys + both compose files; AMI/VM snapshot; document restart/upgrade runbook.
- **Cancel/downgrade LiveKit Cloud** — the ~$50 recurring charge stops here.
- *(Optional, only if you later need >1 replica/HA)* externalize bot/confluence in-process state to Supabase/Redis, then scale out.
**Exit:** monitored, documented, Cloud bill stopped.

---

## 7. Key decisions (resolved + still open)
**Resolved this round:**
- **No ACA for confluence-service.** ACA's 240 s ingress timeout can't carry a 20-min SSE → confluence-service (8001) goes on **VM-B**.
- **Topology = two Azure VMs.** VM-A (nice: LiveKit + agent + bot-service) and VM-B (cheap: 8001 + 8003). Driven by the agent↔bot-service **localhost** coupling and the SSE timeout.
- **Agent worker is co-located with bot-service + LiveKit on VM-A** (the code's default `BRIDGE_INTERNAL_URL=127.0.0.1:8000` assumes it). Split to its own VM only if concurrency later saturates VM-A.

**Still open:**
- **TTS provider:** Cartesia plugin (cheap, ~1-line swap, recommended) **vs** edge-tts/Piper (free, needs a custom adapter + more testing). *Default: Cartesia for the cutover.*
- **VM-A size & host:** B2s (2 vCPU/4 GB) is the floor; size up if concurrent meetings grow. Azure (on the credit) vs Oracle Always Free A1 ($0 forever, separate account).
- **VM-B tier:** B1ms (2 GB, ~$15/mo, safe) vs a free 1 GB B1s (only for light, low-concurrency use — pipeline RAM risk).
- **Could collapse to one VM** (everything in one compose) for absolute lowest cost — accept the bigger single-point-of-failure blast radius. Two VMs isolate realtime from the pipeline; one VM is cheaper.

## 8. Risk register
| Risk | Likelihood | Mitigation |
|---|---|---|
| UDP/ICE failures on self-hosted media | High (the classic) | Open full 50000–60000/udp + 3478; TURN/TLS domain resolves; test in Phase 3 before any cutover |
| Agent worker OOM under concurrent meetings (VM-A) | Med | Size VM-A for concurrency (4 GB+); monitor; split agent to its own VM if needed |
| confluence-svc RAM during a pipeline run on a 1 GB box | Med | Use VM-B ≥ 2 GB (B1ms); avoid sleepy/evicting free tiers; monitor |
| Transcript POST is fire-and-forget (2 s, no retry, `agent.py:471`) | Med | Co-location makes it a localhost call (low risk); add a small retry/queue later for zero loss |
| Missed Recall **status** webhook during a VM-A restart | Low | Not data loss — status self-heals via the Recall live-poll in `session_bot_status`; `restart: always` + stable domain |
| 20-min pipeline killed mid-run by a redeploy (VM-B, single instance) | Med | Deploy 8001 during idle windows; don't auto-restart mid-job |
| Single VM = single point of failure | Med | `restart: always`, VM snapshot/AMI, documented rebuild runbook; two-VM split limits blast radius |
| Stale docs mislead future work | High | Trust HOSTING_ANALYSIS.md + this file; regenerate `CLAUDE.md`/`API_ROUTES.md` |
| Recall can't reach bot.html/LiveKit | Med | bot-svc public HTTPS; LiveKit wss public + valid cert; verify in Phase 6 |

## 9. Sources
- LiveKit self-hosting (VM): https://docs.livekit.io/home/self-hosting/vm/ and deployment requirements https://docs.livekit.io/home/self-hosting/deployment/
- Cartesia TTS plugin: https://docs.livekit.io/agents/integrations/tts/cartesia/
- Agent dispatch (self-hosted parity): https://docs.livekit.io/agents/server/agent-dispatch/
