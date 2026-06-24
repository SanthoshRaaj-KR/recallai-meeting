# Self-Host Setup Guide (beginner, end-to-end)

Goal: run the whole backend on your own Azure VMs, off LiveKit Cloud. Follow top to bottom.
🤝 = a step we'll do together (it has machine-specific output I should look at).

## What you're building
- **VM-A (realtime):** LiveKit server + agent + bot-service. Domains: `livekit.YOURDOMAIN`, `turn.YOURDOMAIN`, `vm-a.YOURDOMAIN`.
- **VM-B (services):** confluence-service + org-service. Domain: `api.YOURDOMAIN`.
- Frontend stays on Static Web Apps (later).

Replace `YOURDOMAIN` everywhere with your real domain (e.g. `jarvis.example.com`).

## Before you start
- Azure account with the ~$150 credit.
- A **domain name** you control (required — LiveKit's TLS needs it).
- An **SSH key** on your laptop:
  ```powershell
  ssh-keygen -t ed25519 -C "jarvis"      # Enter through prompts
  Get-Content $HOME\.ssh\id_ed25519.pub  # copy this line for Azure
  ```
- Your backend code in a **git repo you can clone** (the one containing `my-agent/`, `user_service/`, `deploy/`).
- Your credentials handy: `RECALL_API_KEY`, `CEREBRAS_API_KEY`, `ASSEMBLYAI_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `PINECONE_API_KEY`, `ATLASSIAN_*`, `JWT_SECRET`. (TTS = edge-tts = free, no key.)

---

# PART 1 — VM-A (LiveKit + agent + bot)

## 1.1 Create the VM
Portal → **Virtual machines** → **+ Create** → Azure virtual machine:
- Resource group: **Create new** → `jarvis-rg`
- Name `vm-a-livekit` · Region near you · Image **Ubuntu Server 22.04 LTS (Gen2)** · Size **Standard_B2s**
- Authentication **SSH public key** · Username `azureuser` · paste your public key
- Inbound ports: allow **SSH(22), HTTP(80), HTTPS(443)**
- Disks: **Standard SSD**, 30 GiB · → **Review + create** → **Create**
- After deploy → **Go to resource** → note the **Public IP**.

## 1.2 Open the LiveKit ports
VM → **Networking** → **Network settings** → add 3 inbound rules (Source `Any`, Action `Allow`, increasing priority):

| Name | Protocol | Destination ports |
|---|---|---|
| livekit-tcp | TCP | `80,443,7881` |
| livekit-udp | UDP | `443,3478` |
| livekit-media | UDP | `50000-60000` |

## 1.3 Lock the IP (static)
VM → Networking → click the **Public IP** → **Configuration** → **Assignment: Static** → **Save**.

## 1.4 DNS
At your domain provider, add **A records** → VM-A's IP:
- `livekit` → IP   ·   `turn` → IP   ·   `vm-a` → IP

Verify from your laptop: `nslookup livekit.YOURDOMAIN` returns the IP.

## 1.5 Log in + install Docker
```powershell
ssh azureuser@<VM-A IP>      # type "yes" first time
```
On the VM:
```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker
docker --version            # confirms Docker works
```

## 1.6 Get the code onto the VM
```bash
git clone <YOUR_REPO_URL> jarvis
cd jarvis/Confluence         # the folder with my-agent/, user_service/, deploy/
```

## 1.7 🤝 Stand up the LiveKit server
```bash
mkdir ~/livekit && cd ~/livekit
docker run --rm -it -v $PWD:/output livekit/generate
```
It asks for your domain(s) — enter `livekit.YOURDOMAIN` and `turn.YOURDOMAIN`. It writes `caddy.yaml`, `docker-compose.yaml`, `livekit.yaml`, `redis.conf`, and prints an **API key + secret — SAVE THESE** (they become `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET`). Then:
```bash
docker compose up -d
docker compose logs --tail 30
```
✅ Check from your laptop browser: `https://livekit.YOURDOMAIN` should respond (and TLS valid). *Ping me here — I'll confirm the generated files + the Caddy edit in 1.9.*

## 1.8 Configure the app env
```bash
cd ~/jarvis/Confluence
cp deploy/.env.example deploy/vm-a/.env
nano deploy/vm-a/.env
```
Set (at minimum):
```
LIVEKIT_URL=wss://livekit.YOURDOMAIN
LIVEKIT_API_KEY=<from 1.7>
LIVEKIT_API_SECRET=<from 1.7>
BRIDGE_SERVER_URL=https://vm-a.YOURDOMAIN
RECALL_API_KEY=...
CEREBRAS_API_KEY=...
ASSEMBLYAI_API_KEY=...
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...
PINECONE_API_KEY=...
ATLASSIAN_USER_EMAIL=...   ATLASSIAN_API_TOKEN=...   ATLASSIAN_DOMAIN=...   ATLASSIAN_SPACE_KEY=...
CORS_ORIGINS=*
AGENT_NAME=my-agent
```
(`BRIDGE_INTERNAL_URL` is set by the compose — leave it out.)

## 1.9 🤝 Expose bot-service publicly (Caddy route for `vm-a.YOURDOMAIN`)
Recall must reach bot-service over HTTPS. We add one site to the LiveKit Caddy:
`vm-a.YOURDOMAIN` → `bot-service` (port 8000). The exact edit depends on the generated `~/livekit/caddy.yaml` format — **paste me that file and I'll give you the precise lines**, then `docker compose -f ~/livekit/docker-compose.yaml restart caddy`.

## 1.10 Bring up bot + agent
```bash
cd ~/jarvis/Confluence
docker compose -f deploy/vm-a/docker-compose.yml up -d --build   # builds my-agent (~3-5 min first time)
docker compose -f deploy/vm-a/docker-compose.yml logs -f agent    # watch it register
```
✅ Checks:
- `curl https://vm-a.YOURDOMAIN/health` → `{"service":"bot-service",...,"livekit_configured":true}`
- agent log shows it registered as `agent_name="my-agent"`.

---

# PART 2 — VM-B (confluence + org)

## 2.1 Create the VM
Same as 1.1 but: name `vm-b-services`, size **Standard_B1ms** (2 GB). Allow only **SSH(22), HTTP(80), HTTPS(443)** (no UDP needed).

## 2.2 Static IP + DNS
Make its IP static (like 1.3). Add A record `api` → VM-B's IP.

## 2.3 Docker + code
SSH in → install Docker (1.5) → `git clone <YOUR_REPO_URL> jarvis && cd jarvis/Confluence`.

## 2.4 Env + Caddy domain
```bash
cp deploy/.env.example deploy/vm-b/.env
nano deploy/vm-b/.env
```
Set: `CEREBRAS_API_KEY`, `OPENAI_API_KEY`, `ATLASSIAN_*`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `PINECONE_API_KEY`, `JWT_SECRET`, `CORS_ORIGINS=<frontend origin>`, `ALLOW_DB_RESET` unset.
Then edit `deploy/vm-b/Caddyfile` → change `api.your-domain.com` to your real `api.YOURDOMAIN`.

## 2.5 Bring up
```bash
docker compose -f deploy/vm-b/docker-compose.yml up -d --build
```
✅ Checks:
- `curl https://api.YOURDOMAIN/conf-api/health` → confluence-service
- `curl https://api.YOURDOMAIN/org-api/health` → org-user-service
- SSE endurance: start a pipeline job and `curl -N https://api.YOURDOMAIN/conf-api/review/pipeline/<job>/stream` runs **past 5 min** uninterrupted.

---

# PART 3 — Wire it together
1. **Recall webhook:** in Recall settings, point the webhook to `https://vm-a.YOURDOMAIN/recall-webhook`.
2. **Frontend:** deploy `sync-sage-bot` to Static Web Apps; route `/api`→`https://vm-a.YOURDOMAIN`, `/conf-api` & `/org-api`→`https://api.YOURDOMAIN`; set `VITE_SUPABASE_*`.
3. **Smoke test:** start a real meeting → bot joins the **self-hosted** LiveKit room (audio over UDP), Jarvis answers (edge-tts), end meeting → proposal pipeline streams → applies to Confluence.

**Rollback** anytime: set `LIVEKIT_URL/KEY/SECRET` back to LiveKit Cloud in `deploy/vm-a/.env` and `up -d` again. Keep Cloud active until Part 3 fully passes, then cancel it.

## Save money
`Stop (deallocate)` a VM from its Overview when not testing — compute billing pauses (you keep the static IP/disk for pennies).
