# Deploy runbook — self-hosted LiveKit + Azure VMs

Operator steps for **Phase 5 (self-host LiveKit)**, **Phase 6 (VM-A: agent + bot-service)**, and **Phase 7 (VM-B: confluence + org)**. Strategy/rationale lives in `../EXECUTION_PLAN.md`; service/port/route map in `../SERVICE_MAP.md`.

> These phases require a real VM + domain, so they run **on your cloud**, not in CI. Each phase lists its **exit gate** ("what green looks like"). Cerebras + AssemblyAI are unchanged; TTS is **edge-tts** (free, no model on the VM).

## Images
Build once and push to a registry both VMs can pull (e.g. Azure Container Registry), or build on each VM:
```bash
docker build -t my-agent:latest   ./my-agent       # bot-service, confluence-service, agent
docker build -t jarvis-org:latest  ./user_service   # org-service (8003)
```
`my-agent` is one image run in 3 roles via different commands (see compose files). Verified locally: all roles `/health` green; `--no-sync` start ≈ 3s.

---

## Phase 5 — Self-host LiveKit on VM-A
1. **VM-A**: Azure B2s (2 vCPU / 4 GB), Ubuntu 22.04, **static public IP**.
2. **DNS**: `livekit.<you>.com` (+ `turn.<you>.com`) → VM-A IP.
3. **Firewall / NSG inbound**: `80,443,7881/tcp`, `443,3478/udp`, `50000-60000/udp`.
4. **Generate LiveKit config** (on your laptop):
   ```bash
   docker run --rm -it -v$PWD:/output livekit/generate
   ```
   Enter the domains. It emits `caddy.yaml`, `docker-compose.yaml`, `livekit.yaml`, `redis.conf`, `cloud-init.yaml`. **Save the generated API key/secret.** Run the generated compose **as-is (Redis included)**.
5. Apply `cloud-init.yaml` at VM launch (or copy `init_script.sh` and `sudo ./init_script.sh`). Then `systemctl status livekit-docker`.

**Exit gate:** `wss://livekit.<you>.com` reachable with valid TLS; a test client/agent connects and **media flows over UDP** (no ICE errors). Quick check: `lk room list` (with the generated keys) works.

---

## Phase 6 — VM-A app stack (agent + bot-service) against self-hosted LiveKit
Co-located so the agent → bot-service transcript POST is localhost (`BRIDGE_INTERNAL_URL`).

1. Copy `deploy/vm-a/docker-compose.yml` + `deploy/.env.example` → `.env` on VM-A. Fill `.env`:
   - `LIVEKIT_URL=wss://livekit.<you>.com`, `LIVEKIT_API_KEY/SECRET` = **generated in Phase 5**
   - `BRIDGE_SERVER_URL=https://<vm-a-domain>` (public; Recall loads bot.html here)
   - `RECALL_API_KEY`, `CEREBRAS_API_KEY`, `ASSEMBLYAI_API_KEY`, Supabase, Pinecone, Atlassian
   - `JARVIS_EDGE_TTS_VOICE` (optional; default en-US-AriaNeural)
2. **Networking choice:**
   - Simple: keep the bridge network (compose default) — `BRIDGE_INTERNAL_URL=http://bot-service:8000`.
   - Lowest-latency / matches LiveKit host net: add `network_mode: host` to both services and set `BRIDGE_INTERNAL_URL=http://127.0.0.1:8000`.
3. Point public `:443` at bot-service `:8000` via the VM's Caddy (extend the Phase-5 `caddy.yaml`: route `/bot-page`, `/recall-webhook`, `/bot/*`, `/sessions/*`, `/history`, `/rag/*` → `bot-service:8000`).
4. Bring up:
   ```bash
   docker compose -f deploy/vm-a/docker-compose.yml up -d
   docker compose -f deploy/vm-a/docker-compose.yml logs -f agent   # watch it register
   ```

**Exit gate:**
- `https://<vm-a-domain>/health` → `{"service":"bot-service",...,"livekit_configured":true}`.
- Agent logs show it registered as `agent_name="my-agent"` against your LiveKit.
- Trigger a `jarvis-call` (or `uv run --no-sync src/agent.py console` on the box) → **STT (AssemblyAI) → Cerebras → edge-tts** produces audio. *(This is edge-tts's first real audio test — confirm Jarvis speaks.)*
- A transcript POST appears in bot-service logs at `127.0.0.1:8000`.

---

## Phase 7 — VM-B (confluence-service + org-service)
1. **VM-B**: Azure B1ms (1 vCPU / 2 GB), static IP, DNS `api.<you>.com`.
2. Copy `deploy/vm-b/docker-compose.yml` + `Caddyfile` + `.env`. Set the Caddyfile domain to `api.<you>.com`. Fill `.env` (Cerebras/OpenAI, Atlassian, Supabase, Pinecone, JWT_SECRET, `ALLOW_DB_RESET` unset, `CORS_ORIGINS`=frontend origin).
3. Bring up:
   ```bash
   docker compose -f deploy/vm-b/docker-compose.yml up -d
   ```

**Exit gate:**
- `https://api.<you>.com/conf-api/health` and `/org-api/health` green.
- **SSE endurance:** start a pipeline job and `curl -N https://api.<you>.com/conf-api/review/pipeline/<job>/stream` runs **past 5 min** uninterrupted (proves no ACA 240s cap — this is why 8001 is on a VM).
- One org route (`/org-api/users/me`) returns with a valid JWT.

---

## After 5–7: cutover (Phase 9)
- Register Recall webhook → `https://<vm-a-domain>/recall-webhook`.
- Point the frontend at the new endpoints (`/api`→VM-A, `/conf-api`+`/org-api`→VM-B).
- Run a real meeting end-to-end. **Rollback** at any point: set `LIVEKIT_URL/KEY/SECRET` back to LiveKit Cloud and redeploy (keep Cloud until Phase 10).

## Health cheat-sheet
| URL | Expect |
|---|---|
| `https://<vm-a-domain>/health` | `service: bot-service`, `livekit_configured: true` |
| `wss://livekit.<you>.com` | TLS handshake OK; agent registers |
| `https://api.<you>.com/conf-api/health` | `service: confluence-service` |
| `https://api.<you>.com/org-api/health` | `service: org-user-service` |
