# Jarvis Backend — Execution Plan (containerize → test → self-host → cutover)

> **The actionable buildbook.** Consolidates `HOSTING_ANALYSIS.md` (why), `SERVICE_MAP.md` (what runs where), and supersedes `MIGRATION_PLAN.md §6` with concrete artifacts, commands, and a **test gate per phase**.
> **Order of operations:** containerize + test *everything first, still on LiveKit Cloud* → swap the one Cloud-locked dependency (TTS) with tests → stand up self-hosted LiveKit in parallel → cut over → harden. **LiveKit Cloud stays live as the rollback target until Phase 8.**

## Ground rules (apply to every phase)
- **Build images from `my-agent/pyproject.toml`+`uv.lock` and `user_service/requirements.txt`** — never the legacy root `requirements.txt`.
- **Keep Cerebras + AssemblyAI untouched** (direct APIs, unaffected by LiveKit hosting).
- **TDD for any agent-behavior change** (`my-agent/AGENTS.md`): write/adjust tests first, `uv run pytest` must stay green.
- **Single-replica** for bot-service(8000) and confluence-service(8001) — never run two.
- **Do not proceed past a phase until its Exit Gate is green.**
- Target topology (from SERVICE_MAP §2): **VM-A** = LiveKit + agent + bot-service(8000); **VM-B** = confluence(8001) + org(8003); frontend on Static Web Apps.

---

## Phase 0 — Baseline & safety net  *(no changes)*
**Goal:** capture a known-good state to compare against and roll back to.

**Steps**
1. Confirm a real meeting works end-to-end **today** on LiveKit Cloud (bot joins → Jarvis answers → proposal applies).
2. `cd my-agent && uv sync && uv run pytest` — record pass/fail counts.
3. `cd ../user_service` and run its tests if present; otherwise note none.
4. Commit `my-agent/uv.lock` if untracked. Inventory all secrets (LiveKit, Recall, Cerebras, AssemblyAI, Supabase, Pinecone, Atlassian, GitHub) into Azure Key Vault or a secure note.
5. Tag the repo: `git tag pre-migration-baseline`.

**Exit Gate:** baseline meeting verified on Cloud; pytest baseline recorded; `uv.lock` committed; secrets inventoried; tag created.

---

## Phase 1 — Containerize the `my-agent` image (3 roles) + test each in isolation  *(still on Cloud)*
**Goal:** one reproducible `my-agent` image that runs the agent, bot-service(8000), and confluence-service(8001) — proven against LiveKit Cloud.

**Artifacts to create**
- `my-agent/.dockerignore`:
  ```
  stress_corpus/
  sample_docs/
  tests/
  .planning/
  local_doc_change/
  **/__pycache__/
  *.md
  .env.local
  ```
- Confirm `my-agent/Dockerfile` builds (it already targets uv + `agent.py download-files`). No CMD change — we override per role at run time.

**Steps & tests**
1. `docker build -t my-agent:dev ./my-agent` → image builds, models pre-downloaded.
2. **Bot-service role** — run with Cloud env:
   `docker run --rm -p 8000:8000 --env-file my-agent/.env.local my-agent:dev uv run uvicorn src.bot_service:app --host 0.0.0.0 --port 8000`
   Test: `curl localhost:8000/health` → `{"status":"ok","service":"bot-service",...}`.
3. **Confluence-service role** — `... uv run uvicorn src.recall_bridge:app --port 8001`; `curl localhost:8001/health`.
4. **Agent role** — `... uv run src/agent.py start` with Cloud `LIVEKIT_URL/KEY/SECRET` → logs show worker registered as `agent_name="my-agent"`.
5. `uv run pytest` inside the image (or locally) → still green.

**Exit Gate:** image builds clean; all three roles start; `/health` green on 8000 & 8001; agent registers against Cloud; pytest green.

---

## Phase 2 — Containerize the `jarvis-org` image (8003) + test  *(independent)*
**Goal:** slim image for the stateless org/user service.

**Artifacts to create**
- `user_service/Dockerfile`:
  ```dockerfile
  FROM python:3.12-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . /app/user_service
  ENV PYTHONUNBUFFERED=1
  CMD ["uvicorn", "user_service.main:app", "--host", "0.0.0.0", "--port", "8003"]
  ```
  *(adjust COPY layout so `user_service` is importable as a package)*
- `user_service/.dockerignore` (exclude `__pycache__`, `.env`, tests).

**Steps & tests**
1. `docker build -t jarvis-org:dev ./user_service`.
2. Run with Supabase + `JWT_SECRET` env; `curl localhost:8003/health` → `{"status":"ok","service":"org-user-service"}`.
3. Smoke one real route: `POST /auth/login` (or `/auth/google-exchange`) against the dev Supabase → token returned.
4. Confirm `ALLOW_DB_RESET` is **unset** (so `/admin/reset` is 403).

**Exit Gate:** image builds; `/health` green; one auth route works; reset disabled.

---

## Phase 3 — Local integration via docker-compose  *(still on Cloud — prove the wiring)*
**Goal:** run the whole backend containerized, exactly mirroring the VM-A / VM-B split, before changing any provider.

**Artifacts to create**
- `deploy/vm-a/docker-compose.yml` — bot-service(8000) + agent (Cloud LiveKit for now), `network_mode: host` (or shared network) so agent→`127.0.0.1:8000` works; env `BRIDGE_INTERNAL_URL=http://127.0.0.1:8000`, `BRIDGE_SERVER_URL=<your ngrok/https for now>`.
- `deploy/vm-b/docker-compose.yml` — confluence(8001) + org(8003) + caddy.
- `deploy/.env.example` documenting every var from SERVICE_MAP §8.

**Steps & tests**
1. `docker compose -f deploy/vm-a/docker-compose.yml up` → bot + agent healthy; agent registered on Cloud.
2. `docker compose -f deploy/vm-b/docker-compose.yml up` → 8001 + 8003 healthy.
3. **Transcript loopback test:** `POST /livekit-transcript/<sid>` to 8000 → `GET /sessions/<sid>/review/transcript` returns it (proves localhost ingest path).
4. **SSE smoke (long-connection behavior):** `curl -N localhost:8001/review/pipeline/<job>/stream` for a started job → events stream, `ping` every ~30s.
5. **Routing-split check (SERVICE_MAP §6):** confirm `/sessions/{id}/review/transcript`→8000 vs `/sessions/{id}/review/changes`→8001 are reachable on the right ports.

**Exit Gate:** both composes run; transcript loopback works; SSE streams; routing split verified; a meeting still works through the containerized bot-service against Cloud.

---

## Phase 4 — Swap the one Cloud-locked dependency (TTS) + drop `ai_coustics`  *(TDD, still on Cloud)*
**Goal:** make the code provider-portable so the later infra flip is pure infra. (Cerebras/AssemblyAI untouched.)

> ℹ️ **TTS = a cloud API call, NOT a model on our VM.** STT (AssemblyAI), LLM (Cerebras), and TTS (Cartesia/edge-tts) are all external cloud APIs the agent phones out to — nothing AI runs on our hardware. Self-hosting LiveKit only removes the LiveKit *Inference* gateway that the TTS call rode on, so we point the call at the provider directly. No GPU, no TTS model, no extra server. Free option: `edge-tts`.

**Code changes**
- Add `livekit-plugins-cartesia` to `my-agent/pyproject.toml`; re-lock (`uv lock`).
- Replace both `inference.TTS(model="cartesia/sonic-3", voice="9626c31c-…")` (agent.py ~763, ~803) with
  `cartesia.TTS(api_key=os.getenv("CARTESIA_API_KEY"), model="sonic-3", voice="9626c31c-…")`.
- Remove the `ai_coustics.audio_enhancement(...)` block (agent.py ~835, non-Recall branch only) and drop `ai_coustics` from deps. *(Optional: gate behind an env flag instead of deleting.)*
- Add `CARTESIA_API_KEY` to env docs.

**Tests**
1. Write/adjust a TTS-path test per `AGENTS.md` (assert the session builds with the new TTS node; mock provider).
2. `uv run pytest` green.
3. Rebuild `my-agent:dev`; run agent against **Cloud**; do a `console`/jarvis-call session → Jarvis speaks with Cartesia voice; non-Recall path no longer references `ai_coustics`.

**Exit Gate:** Cartesia TTS works on Cloud; `ai_coustics` removed; pytest green; image rebuilt. **Code is now portable.**

---

## Phase 5 — Stand up self-hosted LiveKit on VM-A  *(parallel, non-disruptive)*
**Goal:** a working self-hosted LiveKit server, independent of production.

**Steps**
1. Provision **VM-A** (Azure B2s, Ubuntu 22.04) with a **static public IP**. DNS: `livekit.<you>.com` (+ `turn.<you>.com`) → IP.
2. NSG inbound: `80,443,7881/tcp`, `443,3478/udp`, `50000-60000/udp`.
3. On your laptop: `docker run --rm -it -v$PWD:/output livekit/generate` → enter domains → emits `caddy.yaml`, `docker-compose.yaml`, `livekit.yaml`, `redis.conf`, `cloud-init.yaml`. **Save the generated API key/secret.** **Run the generated compose as-is (Redis included).**
4. Apply cloud-init on the VM (or `sudo ./init_script.sh`). `systemctl status livekit-docker`.

**Tests**
- `wss://livekit.<you>.com` reachable; Caddy issued a valid Let's Encrypt cert.
- Smoke with `lk` CLI or a throwaway token + the agents console → **media flows over UDP** (no ICE failures).

**Exit Gate:** self-hosted LiveKit reachable over WSS with valid TLS; a test client connects and media flows over UDP.

---

## Phase 6 — Bring up the full VM-A stack against self-hosted LiveKit
**Goal:** LiveKit + agent + bot-service(8000) co-located, talking over localhost, on self-hosted.

**Steps**
1. Extend the LiveKit compose on VM-A with two `my-agent` containers (`network_mode: host`):
   - `agent` → `agent.py start`; env `LIVEKIT_URL/KEY/SECRET=self-hosted`, `BRIDGE_INTERNAL_URL=http://127.0.0.1:8000`, Cerebras/AssemblyAI/Cartesia/Supabase/Pinecone/Atlassian.
   - `bot-service` → `uvicorn src.bot_service:app --port 8000`; env LiveKit self-hosted, `RECALL_API_KEY`, **`BRIDGE_SERVER_URL=https://<vm-a-domain>`**, `AGENT_NAME=my-agent`, Supabase/Pinecone, explicit `CORS_ORIGINS`.
2. Caddy on VM-A: public `:443` → `:8000` for `/bot-page`, `/recall-webhook`, `/bot/*`, `/sessions/*`, `/history`, `/rag/*`.
3. **Startup refinement (carry into all compose CMDs):** the image's default `uv run …` **rebuilds the package at container start (~20 s)**. In prod compose, invoke the prebuilt venv directly to start instantly — e.g. `uv run --no-sync uvicorn …` or `/app/.venv/bin/uvicorn …` (and `/app/.venv/bin/python src/agent.py start` for the agent). Deps are already installed at image build (`uv sync --locked`), so no runtime sync is needed.

**Tests**
- `https://<vm-a-domain>/health` (bot-service) green.
- Trigger `jarvis-call/token` → join the room → STT→Cerebras→**edge-tts** TTS works on self-hosted LiveKit.
- Transcript POST hits `127.0.0.1:8000` (check bot-service logs); `GET /sessions/{id}/review/transcript` returns lines.

**Exit Gate:** agent + bot-service healthy on VM-A against self-hosted LiveKit; a voice session completes; localhost transcript ingest confirmed.

---

## Phase 7 — Deploy VM-B (confluence 8001 + org 8003)
**Goal:** the low-compute HTTP plane on the cheap box, dodging ACA's 240s SSE cap.

**Steps**
1. Provision **VM-B** (Azure B1ms, 2 GB), static IP + domain `api.<you>.com`.
2. `deploy/vm-b/docker-compose.yml`: `confluence-svc` (`my-agent` image → `recall_bridge:app :8001`) + `org-svc` (`jarvis-org` → :8003) + `caddy`.
3. Caddy routes `/conf-api/*`→8001, `/org-api/*`→8003. Ensure Caddy proxy read-timeout is long/unbounded (default is fine) so 20-min SSE survives.
4. Both services point at the **same Supabase** as VM-A.

**Tests**
- `https://api.<you>.com/conf-api/health` and `/org-api/health` green.
- **SSE endurance:** start a pipeline job, `curl -N .../review/pipeline/<job>/stream` runs **past 5 minutes** uninterrupted (proves no 240s cap).
- One org route (`/users/me`) returns with a valid JWT.

**Exit Gate:** 8001 + 8003 reachable via `api.<you>.com`; SSE survives >5 min; org auth works.

---

## Phase 8 — Frontend + routing cutover (Static Web Apps)
**Goal:** the SPA talks to the new endpoints with the correct split.

**Steps**
1. Deploy `sync-sage-bot/` to **Azure Static Web Apps** (`vite build` → `dist/`).
2. Configure prod routing (SERVICE_MAP §6):
   - `/api/*` → `https://<vm-a-domain>` (bot-service)
   - `/conf-api/*` → `https://api.<you>.com` (confluence)
   - `/org-api/*` → `https://api.<you>.com` (org)
   - ⚠️ Honor the split: `/sessions/{id}/review/transcript`→VM-A, other `/sessions/{id}/review/*`→VM-B (the frontend `api.ts` base-URL split already does this — verify).
3. Set `VITE_SUPABASE_URL` / `VITE_SUPABASE_ANON_KEY`; set `CORS_ORIGINS` on all services to the SWA origin.

**Exit Gate:** frontend loads; Google sign-in works; UI reaches all three services; CORS clean.

---

## Phase 9 — End-to-end cutover (Recall + live meeting)
**Goal:** a real meeting runs fully on the self-hosted stack.

**Steps**
1. Register Recall webhook → `https://<vm-a-domain>/recall-webhook`.
2. Run a live meeting: Recall bot joins the **self-hosted** room (verify UDP/ICE), Jarvis answers, transcript ingests (localhost), end → VM-B pipeline streams SSE → apply a proposal to Confluence.

**Exit Gate:** full meeting → summary → proposal → **applied to Confluence** on the new stack.
**Rollback:** repoint `LIVEKIT_URL/KEY/SECRET` (bot-service + agent) back to Cloud; redeploy. (Cloud still live until Phase 10.)

---

## Phase 10 — Harden, monitor, decommission Cloud (bank the savings)
**Goal:** safe steady-state, cost win realized.

**Steps**
- TLS auto-renew confirmed on both VMs; NSG least-privilege; container `restart: always`; `/health` checks.
- Log hygiene (no secrets/tokens); CPU/RAM monitoring; **alarm on agent OOM (VM-A)** and **confluence-svc RAM during pipeline runs (VM-B)**.
- Back up `livekit.yaml` keys + both compose files; VM snapshots; write the restart/upgrade runbook.
- Lock `CORS_ORIGINS` to prod origins; confirm `ALLOW_DB_RESET` unset.
- **Cancel/downgrade LiveKit Cloud — the ~$50 recurring charge stops.**
- *(Optional, only for >1 replica/HA)* externalize bot/confluence in-process state to Supabase/Redis, then scale out.

**Exit Gate:** monitored, documented, Cloud bill stopped.

---

## Phase dependency / sequencing
```
0 baseline
└─1 my-agent image ──┐
  2 jarvis-org image ─┼─3 local compose (Cloud) ─4 TTS swap (TDD, Cloud)
                      │                                   │
                      └───────────── 5 self-host LiveKit (parallel) ─6 VM-A full ─7 VM-B ─8 frontend ─9 E2E cutover ─10 harden+cancel
```
Phases 1–4 are **provider-agnostic** (still on Cloud) — do them first and you de-risk everything. The infra flip is 5→9. Cancel Cloud only at 10.

## Test gates at a glance
| Phase | Hard gate |
|---|---|
| 0 | Cloud meeting works; pytest baseline; tag |
| 1 | 3 roles start; 8000/8001 `/health`; agent registers; pytest green |
| 2 | 8003 `/health`; auth route works; reset disabled |
| 3 | composes run; transcript loopback; SSE streams; routing split verified |
| 4 | Cartesia TTS on Cloud; `ai_coustics` gone; pytest green |
| 5 | self-hosted WSS+TLS; UDP media flows |
| 6 | VM-A voice session on self-hosted; localhost transcript |
| 7 | 8001/8003 up; **SSE >5 min**; org auth |
| 8 | frontend reaches all 3; OAuth; CORS clean |
| 9 | full meeting → proposal applied; rollback rehearsed |
| 10 | monitored; Cloud cancelled |
