# Deployment Process Log — Self-Host on Azure (VM-A)

Running log of the LiveKit self-host migration. Tick boxes as you go. **No secrets in this file** (they live in `deploy/vm-a/.env`, gitignored).

## Goal
Move off LiveKit **Cloud** → self-hosted LiveKit + agent + bot-service on one Azure VM (**VM-A**). Keep Recall (ap-northeast-1), Cerebras (LLM), AssemblyAI (STT). TTS = **edge-tts** (free, no key).

## VM-A facts (fill/confirm)
| Thing | Value |
|---|---|
| Azure region | Southeast Asia |
| Public IP (static) | `104.43.112.6` |
| LiveKit domain | `vm-a-livekit-jarvis.southeastasia.cloudapp.azure.com` → livekit `:7880` |
| TURN domain | `livekit-turn.104-43-112-6.nip.io` → `:5349` |
| bot-service domain | `bot.104-43-112-6.nip.io` → bot-service `:8000` |
| Repo on VM | `~/recallai-meeting` |
| LiveKit config on VM | `~/livekit/vm-a-livekit-jarvis.southeastasia.cloudapp.azure.com/` |
| LiveKit API key/secret | saved in `deploy/vm-a/.env` (NOT here) |

## Progress
- [x] VM-A created (Standard_B2s, Ubuntu 22.04)
- [x] Inbound ports opened (TCP `80,443,7881,8443`; UDP `443,3478,50000-60000`)
- [x] Public IP set **Static** + Azure DNS name label
- [x] Docker installed
- [x] Repo cloned → `~/recallai-meeting`
- [x] `livekit/generate` run → LiveKit + Caddy + Redis up; `https://<livekit-domain>` responds with valid TLS
- [x] LiveKit API key/secret saved
- [ ] **`deploy/vm-a/.env` complete** — add missing `LIVEKIT_URL` as first line
- [ ] bot + agent started → `curl localhost:8000/health` shows `livekit_configured: true`
- [ ] Caddy updated with bot-service route + restarted
- [ ] `https://bot.104-43-112-6.nip.io/health` works from laptop
- [ ] Agent registers (check `logs agent`)
- [ ] VM-B (confluence + org) — later
- [ ] Recall webhook → `https://bot.104-43-112-6.nip.io/recall-webhook`
- [ ] Frontend (Static Web Apps)

---

## Remaining steps

### STEP 1 — finish the env + start bot/agent
`~/recallai-meeting/deploy/vm-a/.env` must have **`LIVEKIT_URL` as the first line** (this was the missing piece):
```
LIVEKIT_URL=wss://vm-a-livekit-jarvis.southeastasia.cloudapp.azure.com
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
BRIDGE_SERVER_URL=https://bot.104-43-112-6.nip.io
AGENT_NAME=my-agent
CORS_ORIGINS=*
RECALL_API_KEY=...
RECALL_API_REGION=ap-northeast-1
CEREBRAS_API_KEY=...
ASSEMBLYAI_API_KEY=...
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...
PINECONE_API_KEY=...
```
Then:
```bash
cd ~/recallai-meeting
docker compose -f deploy/vm-a/docker-compose.yml up -d --build
curl http://localhost:8000/health        # expect "livekit_configured": true
```

### STEP 2 — add bot-service route to Caddy
```bash
sudo truncate -s 0 ~/livekit/vm-a-livekit-jarvis.southeastasia.cloudapp.azure.com/caddy.yaml
sudo nano ~/livekit/vm-a-livekit-jarvis.southeastasia.cloudapp.azure.com/caddy.yaml
```
Paste this (first line `logging:` at the left edge, no leading spaces):
```yaml
logging:
  logs:
    default:
      level: INFO
storage:
  module: file_system
  root: /data
apps:
  tls:
    certificates:
      automate:
        - vm-a-livekit-jarvis.southeastasia.cloudapp.azure.com
        - livekit-turn.104-43-112-6.nip.io
        - bot.104-43-112-6.nip.io
  layer4:
    servers:
      main:
        listen: [":443"]
        routes:
          - match:
              - tls:
                  sni:
                    - "livekit-turn.104-43-112-6.nip.io"
            handle:
              - handler: tls
              - handler: proxy
                upstreams:
                  - dial: ["localhost:5349"]
          - match:
              - tls:
                  sni:
                    - "vm-a-livekit-jarvis.southeastasia.cloudapp.azure.com"
            handle:
              - handler: tls
                connection_policies:
                  - alpn: ["http/1.1"]
              - handler: proxy
                upstreams:
                  - dial: ["localhost:7880"]
          - match:
              - tls:
                  sni:
                    - "bot.104-43-112-6.nip.io"
            handle:
              - handler: tls
                connection_policies:
                  - alpn: ["http/1.1"]
              - handler: proxy
                upstreams:
                  - dial: ["localhost:8000"]
```
Save (Ctrl+O, Enter, Ctrl+X), verify, then restart Caddy:
```bash
cat ~/livekit/vm-a-livekit-jarvis.southeastasia.cloudapp.azure.com/caddy.yaml   # sanity check
cd ~/livekit/vm-a-livekit-jarvis.southeastasia.cloudapp.azure.com && docker compose restart caddy
```

### STEP 3 — verify end-to-end
- From laptop: open `https://bot.104-43-112-6.nip.io/health` → bot-service JSON, valid padlock.
- Agent registered:
  ```bash
  cd ~/recallai-meeting && docker compose -f deploy/vm-a/docker-compose.yml logs agent | tail -30
  ```

## Security TODO (after it works)
Rotate the keys that were pasted in chat (Recall, Cerebras, AssemblyAI, **Supabase service-role**, Pinecone, LiveKit) in each provider dashboard, then update `deploy/vm-a/.env`.

## Known watch-points
- **Hairpin:** agent connects to LiveKit via the public `wss://` URL from inside the same VM. If the agent log shows it can't reach LiveKit, we map the LiveKit FQDN → 127.0.0.1 for the agent (2-line fix).
- **AGENT_NAME** must equal the name the agent registers under (`my-agent` in the code). Mismatch = agent never joins.

---

## ✅ VM-A STATUS: WORKING (validated with a live meeting, 2026-06-15)
Fixes that were applied to get it green:
1. **`LIVEKIT_URL` added** as the first line of `deploy/vm-a/.env` (was missing → `livekit_configured:false`).
2. **Hairpin fix** — added to `bot-service` AND `agent` in `deploy/vm-a/docker-compose.yml`:
   ```yaml
   extra_hosts:
     - "vm-a-livekit-jarvis.southeastasia.cloudapp.azure.com:host-gateway"
   ```
   (containers couldn't reach the VM's own public IP:443; this resolves the LiveKit name to the host gateway → local Caddy).
3. **Turn-detector model fix** — added `ENV HF_HOME=/app/.cache/huggingface` in `my-agent/Dockerfile` (after `UV_COMPILE_BYTECODE`) so `download-files` models ship in the image (was failing on `model_q8.onnx`). Rebuild required.
4. **Caddy** serves 3 SNI routes (livekit `:7880`, turn `:5349`, bot-service `:8000`).

---

## Day-to-day operations

### Stop everything
**VM (SSH):**
```bash
cd ~/recallai-meeting && docker compose -f deploy/vm-a/docker-compose.yml down
cd ~/livekit/vm-a-livekit-jarvis.southeastasia.cloudapp.azure.com && docker compose down
docker ps   # confirm empty
```
**Laptop:**
```bash
cd C:\Users\santh\.vscode\Programming\GenReal\Product2\Confluence
docker compose -f docker-compose.local.yml down
# then Ctrl+C the `npm run dev` (frontend) terminal
```

### Start everything
**VM (SSH)** — LiveKit first, then bot+agent:
```bash
cd ~/livekit/vm-a-livekit-jarvis.southeastasia.cloudapp.azure.com && docker compose up -d
cd ~/recallai-meeting && docker compose -f deploy/vm-a/docker-compose.yml up -d
```
**Laptop:**
```bash
cd C:\Users\santh\.vscode\Programming\GenReal\Product2\Confluence
docker compose -f docker-compose.local.yml up -d confluence-service org-service
cd sync-sage-bot && npm run dev
```

### Pause Azure billing when idle (compute keeps charging even if containers are down)
Azure portal → VM `vm-a-livekit` → **Stop** (deallocate), or:
```bash
az vm deallocate -g jarvis-rg -n vm-a-livekit
```
Keeps the static IP + disk (pennies). After **Start**, re-run the VM start commands above.

### Verify
```bash
curl https://bot.104-43-112-6.nip.io/health          # livekit_configured:true
docker compose -f deploy/vm-a/docker-compose.yml logs agent | tail -20   # "registered worker"
```

### Frontend env (sync-sage-bot/.env.local) — points local UI at VM bot + local services
```
VITE_API_BASE_URL=/api
VITE_API_PROXY_TARGET=https://bot.104-43-112-6.nip.io
VITE_CONFLUENCE_PROXY_TARGET=http://localhost:8001
VITE_ORG_API_PROXY_TARGET=http://localhost:8003
VITE_SUPABASE_URL=<...>
VITE_SUPABASE_ANON_KEY=<...>
```
