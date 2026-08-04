# Observability Implementation Plan — Metrics, Logs, Traces

> **Audience:** humans and AI models implementing this.
> **Generated:** 2026-07-28.
> **Scope:** the `Confluence/` backend (bot-service :8000, confluence-service :8001,
> org-service :8003, the LiveKit agent worker). Builds on the topology in
> `HOSTING_ANALYSIS.md` — read that first if you haven't (single-replica
> constraints on bot/confluence, ACA vs LiveKit-VM split matter here).

---

## 1. Recommendation

Deploy **Grafana Alloy** as the single collector (replaces the old
Prometheus-exporter + Promtail + separate-tracing-agent pattern — Promtail is
being sunset by Grafana Labs), feeding **Prometheus** (metrics), **Loki**
(logs), and **Tempo** (traces), visualized in **Grafana**. Build it in this
order:

**Metrics → Logs → Traces → Correlation → Alerting → Production rollout.**

Metrics first because it's zero app-code-risk (just a `/metrics` endpoint) and
gives RED-method dashboards fastest. Traces are last of the three signals
because they require touching every service's request path.

---

## 2. What's already scaffolded (Phase 0 — done)

| File | Purpose |
|---|---|
| `docker-compose.observability.yml` | Alloy + Prometheus + Grafana, additive to `docker-compose.local.yml`. Loki/Tempo services present but commented out. |
| `observability/alloy/config.alloy` | Alloy pipeline. Phase 1 (host + app metrics → Prometheus) active; Phase 2 (logs) and Phase 3 (traces) blocks present but commented. |
| `observability/prometheus/prometheus.yml` | Minimal — Prometheus only stores/queries; all ingestion is via Alloy's `remote_write`. |
| `observability/grafana/provisioning/datasources/datasources.yml` | Prometheus datasource auto-provisioned; Loki/Tempo entries commented for later phases. |

Run it:

```bash
cd Confluence
docker compose -f docker-compose.local.yml -f docker-compose.observability.yml up -d
```

- Grafana: http://localhost:3000 (default admin/admin, change on first login)
- Prometheus: http://localhost:9090
- Alloy UI (component graph + live debugging): http://localhost:12345

At this point Grafana/Prometheus will be up but **empty** — the three FastAPI
services don't expose `/metrics` yet. That's Phase 1, step 2 below.

---

## 3. Phase 1 — Metrics

**Goal:** RED-method dashboard (Rate/Errors/Duration) for all three HTTP
services, plus host CPU/mem/disk.

1. ✅ Alloy scrapes host metrics (`prometheus.exporter.unix`) and is wired to
   scrape `bot-service:8000/metrics`, `confluence-service:8001/metrics`,
   `org-service:8003/metrics` — already in `config.alloy`.
2. **App task (not yet applied — do this next):** add
   [`prometheus-fastapi-instrumentator`](https://github.com/trallnag/prometheus-fastapi-instrumentator)
   to each FastAPI app. It auto-tracks request count, latency histograms, and
   in-progress requests — exactly what RED needs, no manual metric wiring.

   `my-agent/src/bot_service.py` and `my-agent/src/recall_bridge.py` (add to
   `my-agent/pyproject.toml` deps, then `uv lock`):
   ```python
   from prometheus_fastapi_instrumentator import Instrumentator

   app = FastAPI(...)  # existing app object
   Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
   ```

   `user_service/main.py` (add `prometheus-fastapi-instrumentator` to
   `user_service/requirements.txt`):
   ```python
   from prometheus_fastapi_instrumentator import Instrumentator

   app = FastAPI(...)  # existing app object
   Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
   ```

   ⚠️ `/metrics` should **not** be publicly reachable in prod (same class of
   issue as `/admin/reset` in `HOSTING_ANALYSIS.md` §5) — network-restrict it
   or put it behind the platform's internal-only ingress once on ACA (§7).

3. **Custom business metrics** (optional, add once the baseline works):
   queue length of `_sync_jobs`/`_pipelines`, active `_bot_index` count,
   pipeline duration histogram. Use `prometheus_client` directly (`Counter`,
   `Gauge`, `Histogram`) inside `bot_service.py`/`recall_bridge.py` — these are
   the numbers that matter most given the single-replica, in-process-state
   constraints from `HOSTING_ANALYSIS.md` §4.
4. Verify: `docker compose ... up -d --build`, hit each service, confirm
   targets are `UP` in Prometheus (http://localhost:9090/targets) and build a
   first Grafana dashboard: request rate, error rate (5xx), p95 latency per
   service, host CPU/mem.

---

## 4. Phase 2 — Logs

**Goal:** all container stdout searchable in Loki, filterable by `service`
label, pivotable from a metric spike straight to the log lines.

1. Uncomment the `loki` service in `docker-compose.observability.yml` and the
   Loki datasource block in `datasources.yml`.
2. Uncomment the Phase 2 block in `config.alloy` (`discovery.docker` +
   `loki.source.docker` + `loki.write`) — Alloy tails all container logs via
   the Docker socket automatically, no per-service config needed.
3. **App task:** switch from default `print`/basic logging to **structured
   JSON logs to stdout** (e.g. Python's `logging` with a JSON formatter, or
   `structlog`). Structured logs let Loki/Grafana filter by fields
   (`level`, `session_id`, `bot_id`) instead of regexing raw text. Do this
   incrementally — plain stdout logs already work with Loki, structure is a
   quality upgrade, not a blocker.
4. Verify: trigger a request, find the log line in Grafana Explore filtered by
   `{service="bot-service"}`.

---

## 5. Phase 3 — Traces

**Goal:** a request through bot-service → Supabase/Recall, or a confluence
pipeline run → Cerebras/Pinecone/Confluence, shows up as one trace with per-
call spans and latency breakdown.

1. Uncomment the `tempo` service and the Tempo datasource block.
2. Uncomment the Phase 3 OTLP receiver/exporter block in `config.alloy`.
3. **App task:** add OpenTelemetry auto-instrumentation to each FastAPI
   service — `opentelemetry-instrumentation-fastapi` plus
   `opentelemetry-instrumentation-httpx` (covers calls to Supabase, Pinecone,
   Cerebras, OpenAI, Atlassian — all done via `httpx`/`requests` per
   `HOSTING_ANALYSIS.md` §2). Point the OTLP exporter at `alloy:4317` (service
   name, not `localhost` — these run in Docker).
4. This is the highest-effort phase and the one most worth scoping down
   first: start with **bot-service** and **confluence-service** only (they're
   the ones with the 20-minute pipeline and multi-hop external calls worth
   seeing broken down — see `HOSTING_ANALYSIS.md` §2.2). org-service is
   simple CRUD; add it later if useful.
5. Verify: run a meeting/pipeline end-to-end, find the trace in Tempo,
   confirm spans for each external call with realistic durations.

---

## 6. Phase 4 — Correlation

Once all three signals exist:

- Add `exemplars` to the Prometheus scrape config so a latency spike on a
  dashboard graph links directly to a real trace.
- Confirm the Tempo datasource's `tracesToLogsV2` (already stubbed in
  `datasources.yml`) pivots from a trace span to the matching Loki log line.
- Standardize a `service` label across all three signals (already consistent
  in the scaffold: `bot-service`, `confluence-service`, `org-service`) so
  Grafana's service-map / drill-down works without relabeling.

---

## 7. Phase 5 — Alerting (RED method)

Base alerts on the **RED method**, not raw host thresholds — this matters
more than usual here because of the single-replica constraint:

| Alert | Condition | Why it matters for this codebase |
|---|---|---|
| bot-service down | `up{job="bot-service"} == 0` for >1m | It's pinned to 1 replica and must catch Recall webhooks anytime (`HOSTING_ANALYSIS.md` §4.1/§4.5) — a crash has no failover. |
| confluence-service error rate | 5xx rate > 2% over 5m | A 20-min pipeline run failing mid-stream breaks the SSE client silently otherwise. |
| p95 latency breach | per-service baseline, tune after Phase 1 data exists | Standard RED duration signal. |
| Request rate → 0 unexpectedly | on any of the 3 services | Usually means the container died or ingress broke, not "no traffic." |

Implement with Prometheus alerting rules + Alertmanager (or Grafana's built-in
alerting, simpler for a single-Grafana-instance setup — recommended here
given the small scale). Route to email/Slack — decide the channel before
building this phase.

---

## 8. Phase 6 — Production rollout

Two decisions to make before touching prod, both flagged from the earlier
discussion:

### 8.1 Self-host the storage backends, or use Grafana Cloud?

Running Prometheus + Loki + Tempo + Grafana yourself is a fourth thing
competing for RAM/disk alongside the self-hosted LiveKit VM
(`HOSTING_ANALYSIS.md` §16 — a 2 vCPU/4 GB Azure B-series box already carries
the LiveKit SFU + the agent worker's ML models). Options:

| Option | Where it runs | Trade-off |
|---|---|---|
| Self-host everything | The LiveKit VM, or a new small VM | No extra bill beyond compute, but adds real memory pressure to a box already sized for the agent worker + SFU. |
| Grafana Cloud free tier | Managed (10k metric series, 50 GB logs, 50 GB traces) | Self-host **only Alloy** on your infra (or as a sidecar per ACA app); push everywhere else. Removes the RAM contention entirely; free tier is plausibly enough at this traffic level. |

**Leaning:** Grafana Cloud free tier for storage/query, self-hosted Alloy only
— cheapest on ops and doesn't compete with the LiveKit VM's memory budget.
Revisit if traffic outgrows the free tier.

### 8.2 ACA has no host-level metrics

Unlike the local Docker setup, Azure Container Apps doesn't give you a VM to
run a host-level Alloy against — `prometheus.exporter.unix` has nothing to
scrape there. On ACA, Alloy (or a lighter OTel Collector) runs as a sidecar
per app scraping just that app's `/metrics` + receiving its OTLP traces; host
metrics for the **LiveKit VM** specifically still make sense there since it's
a real VM. Azure Monitor already covers ACA-level infra metrics (CPU/mem per
revision) if you want that separately — don't try to reinvent it with Alloy.

### 8.3 Rollout steps (once 8.1/8.2 are decided)

1. Provision the chosen backend (Grafana Cloud stack, or VM-hosted
   Prometheus/Loki/Tempo/Grafana).
2. Deploy Alloy alongside each ACA app (sidecar container in the same
   Container App) and on the LiveKit VM; point `remote_write`/`loki.write`/
   OTLP exporters at the prod backend instead of `localhost`.
3. Secure the endpoints — auth tokens on remote_write/OTLP push (Grafana
   Cloud requires this by default; if self-hosting, don't expose
   Prometheus/Grafana ports publicly, same class of fix as the `CORS_ORIGINS`
   hardening in `HOSTING_ANALYSIS.md` §12).
4. Re-point the Phase 5 alert routing at real on-call channels.
5. Tune retention/cost (Prometheus retention, Loki/Tempo storage) once real
   traffic volume is known.

---

## 9. Open decisions to confirm before proceeding

- [ ] Phase 1 app task: approve adding `prometheus-fastapi-instrumentator` to
      `my-agent/pyproject.toml` (requires `uv lock` + rebuild) and
      `user_service/requirements.txt`.
- [ ] Alerting channel for Phase 5 (email/Slack/other).
- [ ] Grafana Cloud free tier vs full self-host for Phase 6 (§8.1).
- [ ] Whether org-service gets traced in Phase 3 or is skipped as low-value.
