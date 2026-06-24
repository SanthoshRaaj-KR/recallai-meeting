# Local development & testing

How to run the full stack on your machine with all services talking to each
other. There are two paths — start with **Path A** (org features only, no Docker)
since it's fastest; use **Path B** when you also want to test live meetings.

All services share the **same cloud Supabase** (from `Confluence/.env` /
`user_service/.env`), so org data, sessions, and analytics are consistent across
local and prod.

---

## One-time setup

1. **Supabase redirect URL** (needed for Google login locally):
   Supabase Dashboard → **Authentication → URL Configuration** → add
   `http://localhost:3000` to **Site URL** and **Redirect URLs**.
   (The app redirects OAuth to `window.location.origin`, i.e. `localhost:3000`.)

2. **Frontend env** is already wired: `sync-sage-bot/.env.local` overrides the
   prod-pointing `.env` and proxies `/api`→:8000, `/conf-api`/review→:8001,
   `/org-api`→:8003 on localhost. Delete `.env.local` to point back at prod.

3. **Migration 004** (ADMIN role) is already applied to your Supabase. ✅

---

## How the pieces connect

```
 sync-sage-bot (Vite dev, :3000)
   │  same-origin fetch to /api, /org-api  →  Vite proxy (no CORS)
   ├── /org-api/*                 → org-service       :8003   (auth, teams, invites, usage, hierarchy)
   ├── /api/review|/api/sessions/.../review/* (≠transcript) → confluence-service :8001
   └── /api/*  (everything else)  → bot-service       :8000   (bot start/stop, transcript, history)
                                        │
                                        └─ agent worker → LiveKit + bot-service (localhost)
 Google sign-in → Supabase Auth → POST /org-api/auth/google-exchange → org JWT
```

Because the browser calls **relative paths** that Vite proxies server-side, there
are **no CORS issues** in dev.

---

## Path A — Org features (teams / invites / usage / hierarchy)

Only needs the org-service + the frontend.

**Terminal 1 — org-service (:8003)**
```bash
cd Confluence
pip install -r user_service/requirements.txt        # first time only
python -m uvicorn user_service.main:app --reload --port 8003
curl http://localhost:8003/health                   # {"status":"ok","service":"org-user-service"}
```

**Terminal 2 — frontend (:3000)**
```bash
cd sync-sage-bot
npm install        # first time only
npm run dev        # open http://localhost:3000
```

**Log in** (either works):
- **Google** as a real org user (e.g. `genreal.ai@gmail.com`, `santhoshraajkr.17@gmail.com`).
- **Email/password** for the seeded accounts — password `Jarvis@2024!`
  (works for CEO + the 3 managers; `santhoshraaj1710@gmail.com` is Google-only).

**What to test**
- As **CEO** (`genreal.ai@gmail.com`) or **ADMIN** (`santhoshraajkr.17@gmail.com`):
  Org Settings → **Members** (change roles, activate/deactivate), **Teams**
  (create a team, add an existing user — no admin secret anymore), **Hierarchy**,
  **Usage**.
- As a **manager** (`santhoshraaj1710@gmail.com` via Google): Teams tab → **Invite**
  a colleague by email (the invite code shows in the modal; see email note below).

---

## Path B — Full stack incl. live meetings (bot joins → Jarvis replies)

Adds bot-service, confluence-service, and the agent worker. Runs the `my-agent`
image (its deps live in `uv`, not your conda env — so use Docker here).

### How a meeting actually works (so the setup makes sense)
```
 Frontend → bot-service /bot/start (with team_id)
   bot-service → Recall.ai: "send a browser bot into this Google Meet"
   bot-service → LiveKit:   dispatch the agent worker into a room
 Recall bot (in the Meet) ⇄ LiveKit:  publishes meeting audio, plays Jarvis audio back
 agent worker: subscribes audio → AssemblyAI (STT) → Cerebras (LLM) → edge-tts (TTS) → LiveKit
 Recall loads the bot page + posts status webhooks to  BRIDGE_SERVER_URL  (must be PUBLIC)
 On meeting end → bot-service writes user_meeting_activity → Usage dashboard updates
```

### Prerequisites (one-time)
1. **Cerebras key** (Jarvis's LLM): sign up free at https://cloud.cerebras.ai →
   add to `Confluence/.env`:  `CEREBRAS_API_KEY=csk-...`
2. **Public tunnel for Recall** (Recall can't reach `localhost`). In a terminal:
   ```bash
   ngrok http 8000
   ```
   Copy the `https://<id>.ngrok-free.app` URL and set in `Confluence/.env`:
   ```
   BRIDGE_SERVER_URL=https://<id>.ngrok-free.app
   ```
   (bot-service reads this at startup and refuses to start a bot without it.)

### Start order
```bash
cd Confluence
# 1. ngrok already running (step 2 above) and BRIDGE_SERVER_URL set in .env
# 2. build + start all backend services + the agent worker
docker compose -f docker-compose.local.yml up -d --build
curl http://localhost:8000/health   # bot-service  → livekit_configured:true
curl http://localhost:8001/health   # confluence-service
curl http://localhost:8003/health   # org-service
docker compose -f docker-compose.local.yml logs -f agent   # wait for "registered worker" against LiveKit
```
Then run the frontend (Terminal from Path A: `cd sync-sage-bot && npm run dev`).

> If you change `BRIDGE_SERVER_URL` (e.g. ngrok restarts with a new URL), restart
> bot-service: `docker compose -f docker-compose.local.yml up -d bot-service`.

### Run the end-to-end test
1. Open `http://localhost:3000`, sign in, start a real **Google Meet** in another tab.
2. Paste the Meet link → pick the **team** (drives usage stats) → **Dispatch Jarvis**.
3. Jarvis (the Recall bot) joins the Meet; speak — it transcribes, thinks (Cerebras),
   and **replies with voice** (edge-tts) back into the meeting.
4. End the meeting → Org Settings → **Usage**: the session + per-member attendance
   now appear for that team (this is the real-meeting→analytics wiring).

### Meeting troubleshooting
- **Bot never joins / 400 on dispatch**: `BRIDGE_SERVER_URL` unset or not `https://`. Check `curl localhost:8000/health` → `server_url`.
- **Bot joins but Jarvis is silent**: missing `CEREBRAS_API_KEY` (LLM) — check `docker compose logs agent` for auth errors; also confirm the agent logged "registered worker".
- **Recall webhooks not arriving**: the ngrok URL changed — update `BRIDGE_SERVER_URL` and restart bot-service.
- **Agent not dispatched**: agent registers as `my-agent`; bot-service dispatches `AGENT_NAME` (default `my-agent`). Leave `AGENT_NAME` unset or set it to `my-agent`.

---

## Email invites (optional)

Invites always generate a code shown in the UI. To actually **email** it, set in
`user_service/.env` (or `Confluence/.env`):
```
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=you@gmail.com
SMTP_PASS=<gmail app password>
FROM_EMAIL=you@gmail.com
APP_URL=http://localhost:3000     # invite links point here in dev
```
Without these, the invite still works — the code/link is shown in the modal to share manually.

---

## Quick health cheat-sheet

| URL | Expect |
|---|---|
| `http://localhost:8003/health` | `{"service":"org-user-service"}` |
| `http://localhost:8000/health` | `{"service":"bot-service",...}` |
| `http://localhost:8001/health` | `{"service":"confluence-service"}` |
| `http://localhost:3000` | the app; Google login lands you in Org Settings |

## Gotchas
- **Login redirects to prod / blank**: add `http://localhost:3000` to Supabase redirect URLs.
- **Bot routes hit the prod VM**: ensure `sync-sage-bot/.env.local` exists (it sets `VITE_API_PROXY_TARGET=http://localhost:8000`). Restart `npm run dev` after env changes.
- **`/teams` feels slow on first call**: it does a few Supabase round-trips to enrich member counts — normal, not an error.
