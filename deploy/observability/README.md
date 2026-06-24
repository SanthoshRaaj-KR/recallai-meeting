# Observability — Prometheus metrics + OpenTelemetry traces → Grafana Cloud

How Jarvis is monitored across **VM-A** and **VM-B**: per-VM metrics, distributed
traces, and dashboards. This document explains *what each piece does, how data
flows, how to turn it on, and how to verify it.*

> **TL;DR** — Each VM runs one tiny **Grafana Alloy** agent. Alloy (1) scrapes
> host + app metrics like Prometheus, and (2) receives OpenTelemetry traces from
> the apps. It ships both to **Grafana Cloud** (free tier), where you view them
> in Grafana. The VMs store nothing and run no Grafana/Prometheus server — the
> 2 GB VM-B can't spare the RAM, so the heavy parts are hosted.

---

## 1. Why this design

| Option considered | Verdict |
|---|---|
| Self-host Prometheus + Grafana + Tempo on the VMs | ❌ ~1–1.5 GB RAM. VM-B (2 GB) already needs ~2 GB during a pipeline run → OOM risk. |
| Separate monitoring VM | ❌ Extra cost + infra to babysit. |
| **Grafana Cloud free tier + Alloy agent per VM** | ✅ Agent is ~50–100 MB. Storage/Grafana/Tempo hosted free (10k series, 50 GB traces, 14-day retention). **Chosen.** |

"A Prometheus setup on each VM" is realised as **Alloy's Prometheus scraper +
node exporter on each VM**. The time-series database (the part that's normally
called "Prometheus") is the hosted one in Grafana Cloud — same query language
(PromQL), same dashboards, none of the RAM cost.

---

## 2. Architecture

```
            VM-A (LiveKit box)                          VM-B (low-compute box)
 ┌─────────────────────────────────────┐     ┌─────────────────────────────────────┐
 │ bot-service :8000  ──/metrics──┐     │     │ confluence-service :8001 ──/metrics─┐│
 │ agent worker :9464 ──/metrics──┤     │     │ org-service :8003        ──/metrics─┤│
 │      │  (OTLP traces)          │     │     │      │  (OTLP traces)              ││
 │      ▼                         ▼     │     │      ▼                            ▼ ││
 │  ┌──────────── Alloy ───────────┐    │     │  ┌──────────── Alloy ──────────────┐│
 │  │ • node/unix exporter (host)  │    │     │  │ • node/unix exporter (host)     ││
 │  │ • prometheus.scrape (apps)   │    │     │  │ • prometheus.scrape (apps)      ││
 │  │ • otelcol OTLP receiver :4318│    │     │  │ • otelcol OTLP receiver :4318   ││
 │  └──────────────┬───────────────┘    │     │  └───────────────┬─────────────────┘│
 └─────────────────┼────────────────────┘     └──────────────────┼──────────────────┘
                   │  remote_write (metrics)  +  OTLP (traces)    │
                   └───────────────────────┬──────────────────────┘
                                           ▼
                                  ┌──────────────────┐
                                  │  Grafana Cloud   │  Mimir (metrics) · Tempo (traces)
                                  │   + Grafana UI   │  ← you view dashboards here
                                  └──────────────────┘
```

Each Alloy carries an `external_labels { vm = "vm-a" | "vm-b" }`, so every metric
is tagged by which VM it came from.

---

## 3. What each component does

### App instrumentation (Python)
- **`my-agent/src/observability.py`** and **`user_service/observability.py`** —
  one small helper per image. Called once at app startup:
  - `bot_service.py`, `recall_bridge.py`, `user_service/main.py` →
    `setup_fastapi_observability(app, "<service>")`
  - `agent.py` (`__main__`) → `setup_worker_observability("agent-worker")`
- **What it adds:**
  - A Prometheus **`/metrics`** endpoint on each FastAPI service (request rate,
    latency histogram, status codes, in-progress) via
    `prometheus-fastapi-instrumentator`. The agent worker exposes the same on
    **`:9464`** via `prometheus_client`.
  - **OpenTelemetry traces** for every HTTP request and every outbound
    `requests` call, exported over **OTLP/HTTP** to the local Alloy. This is what
    stitches a single trace across `agent → bot-service`, `frontend → confluence`,
    etc.
- **Fail-safe:** the helper is a **no-op** unless the deps are present *and*
  `OTEL_EXPORTER_OTLP_ENDPOINT` (or `OTEL_ENABLED=true`) is set. Local dev and the
  test suite are unaffected; if anything errors it logs and continues — it can
  never take a service down.

### Dependencies (isolated, locks untouched)
OTel/Prometheus libs are **not** in `uv.lock` / `requirements.lock`. They install
in a dedicated Docker layer from `requirements-otel.txt` in each image, so
`uv sync --locked` and the pinned pip build keep working unchanged. Pure-Python
OTLP/HTTP exporter only — **no `grpcio`** — to keep images light for VM-B.

### Grafana Alloy (per VM)
Single container defined in each `deploy/vm-*/docker-compose.yml`. Config in
`alloy/config-vm-a.alloy` / `config-vm-b.alloy`:
- `prometheus.exporter.unix` — host CPU/RAM/disk/net (host `/`, `/proc`, `/sys`
  mounted read-only).
- `prometheus.scrape` — pulls each app's `/metrics`.
- `prometheus.remote_write` — pushes all metrics to Grafana Cloud.
- `otelcol.receiver.otlp` (`:4317` gRPC, `:4318` HTTP) → `batch` →
  `otelcol.exporter.otlphttp` — forwards traces to Grafana Cloud Tempo.

---

## 4. Setup (one-time)

### Step 1 — Create a Grafana Cloud stack
Sign up at <https://grafana.com> (free tier). You get a Grafana URL plus hosted
Prometheus and Tempo.

### Step 2 — Fill credentials
On **each VM**, copy the template and fill it in:
```bash
cp deploy/observability/alloy.env.example deploy/observability/alloy.env
# edit alloy.env  (see inline comments for where each value lives in Grafana Cloud)
```
`alloy.env` is gitignored. The same file works on both VMs.

### Step 3 — Deploy
The apps and Alloy come up together — no separate steps:
```bash
# VM-A
docker compose -f deploy/vm-a/docker-compose.yml up -d --build
# VM-B
docker compose -f deploy/vm-b/docker-compose.yml up -d --build
```
> Rebuild (`--build`) is needed once so the new observability Docker layer is baked in.

### Step 4 — Import the dashboard
In Grafana Cloud → **Dashboards → Import** → upload
`deploy/observability/dashboards/jarvis-overview.json` → pick your Prometheus
data source. You'll see request rate / errors / p95 per service and CPU/RAM per VM.

For traces: Grafana Cloud → **Explore → Tempo** → "Search" to find traces, or
click a span from a slow request.

---

## 5. Verify it's working

```bash
# 1. App metrics endpoints respond (run on the VM):
curl -s localhost:8000/metrics | head        # VM-A bot-service
curl -s localhost:9464/metrics | head        # VM-A agent worker
curl -s localhost:8001/metrics | head        # VM-B confluence
curl -s localhost:8003/metrics | head        # VM-B org

# 2. Alloy is healthy and shipping (debug UI, localhost only):
#    SSH-tunnel  ssh -L 12345:localhost:12345 <vm>  then open http://localhost:12345
#    → Components page: every component should be "Healthy".

# 3. In Grafana Cloud → Explore → Prometheus:
up{job="jarvis-apps"}          # 1 per service
node_uname_info                # one series per vm label
#    → Explore → Tempo: search for recent traces.
```
Generate some traffic (hit `/health`, run a meeting) and the dashboard panels
fill in within ~30 s.

---

## 6. Networking note (bridge vs host)

The compose files use the **bridge network** (default), so apps reach Alloy at
`http://alloy:4318` and Alloy scrapes `bot-service:8000` etc. by service name.

If you switch VM-A to **`network_mode: host`** (the low-latency option in
`deploy/README.md` Phase 6), service DNS no longer resolves. Then:
- set each app's `OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318`, and
- change the Alloy scrape `__address__` targets to `127.0.0.1:<port>`.

---

## 7. Turn it off / cost

- **Disable per service:** unset `OTEL_EXPORTER_OTLP_ENDPOINT` (tracing off) and
  set `METRICS_ENABLED=false` (no `/metrics`). The helper no-ops.
- **Disable entirely:** remove the `alloy` service from the compose files.
- **Cost:** Grafana Cloud free tier (10k active series, 50 GB traces/logs,
  14-day metrics retention) is comfortably enough for these two VMs. The Alloy
  agent adds ~50–100 MB RAM per VM — safe even on VM-B.

---

## 8. File map

| Path | Purpose |
|---|---|
| `my-agent/src/observability.py` | Instrumentation helper (FastAPI + worker) |
| `user_service/observability.py` | Instrumentation helper (org service) |
| `*/requirements-otel.txt` | Isolated OTel/Prometheus deps (locks stay pristine) |
| `deploy/observability/alloy/config-vm-a.alloy` | Alloy config for VM-A |
| `deploy/observability/alloy/config-vm-b.alloy` | Alloy config for VM-B |
| `deploy/observability/alloy.env.example` | Grafana Cloud creds template |
| `deploy/observability/dashboards/jarvis-overview.json` | Importable Grafana dashboard |
| `deploy/vm-*/docker-compose.yml` | `alloy` service + `OTEL_*` env on the apps |
