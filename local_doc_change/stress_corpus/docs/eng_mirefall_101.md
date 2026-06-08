# Mirefall Platform Engineering — SLA & Performance Report Q3 FY2024

This document is the authoritative record of service reliability, incident activity, and system performance for Mirefall's production platform during Q3 FY2024 (1 July – 30 September 2024). It is published by the Platform Engineering team within fifteen business days of quarter end. All SLA calculations use calendar-month windows aligned to the contractual commitments in Mirefall's Enterprise Service Agreement. Figures marked with (†) are derived from Datadog synthetic monitors and may differ by up to 0.01% from provider-reported uptime due to probe location variance.

---

## 1. Executive Summary

Q3 was the most reliable quarter in Mirefall platform history across five of six tracked services. The API Gateway and Commerce Checkout services both achieved **100.00% uptime** for the first time since the platform was consolidated onto the current stack in Q1 FY2023. The sole regression was the ML Inference service, which experienced a 47-minute outage on 6 August caused by a CUDA driver incompatibility introduced during a routine node pool upgrade. Overall weighted platform availability for Q3 was **99.94%**, exceeding the contractual SLA threshold of 99.90%.

Median p99 API latency improved by 18ms quarter-over-quarter, driven by query plan optimisations in the Commerce read path and CDN cache-hit rate improvements following edge node expansion in July.

---

## 2. Service Availability — Monthly Breakdown

Target SLA for all Tier-1 services is **99.90%** (≤ 43.8 minutes downtime per month). Tier-2 services carry a **99.50%** target.

| Service | Tier | July Uptime (%) | Aug Uptime (%) | Sep Uptime (%) | Q3 Uptime (%) | SLA Target (%) | SLA Met? |
|---|---|---:|---:|---:|---:|---:|---|
| API Gateway | T1 | 100.00 | 100.00 | 100.00 | 100.00 | 99.90 | Yes |
| Commerce Checkout | T1 | 100.00 | 100.00 | 100.00 | 100.00 | 99.90 | Yes |
| Auth & Identity | T1 | 99.98 | 99.97 | 99.99 | 99.98 | 99.90 | Yes |
| Data Ingestion Pipeline | T1 | 99.95 | 99.93 | 99.96 | 99.95 | 99.90 | Yes |
| ML Inference Service | T1 | 100.00 | 99.89 | 100.00 | 99.96 | 99.90 | Yes† |
| Internal Admin Portal | T2 | 99.71 | 99.68 | 99.80 | 99.73 | 99.50 | Yes |
| Event Streaming (Kafka) | T2 | 99.94 | 99.91 | 99.96 | 99.94 | 99.50 | Yes |
| Analytics Dashboard | T2 | 99.62 | 99.55 | 99.74 | 99.64 | 99.50 | Yes |

† ML Inference met the quarterly SLA despite the August incident because the 47-minute outage fell within the single-month allowance of 43.8 minutes when measured across the rolling quarter window per §4.2 of the Enterprise Service Agreement.

---

## 3. Incident Summary

A total of **23 incidents** were recorded in Q3, down from 31 in Q2. Of these, 8 were Severity-1 (customer-impacting), 9 were Severity-2 (degraded performance), and 6 were Severity-3 (internal tooling only).

| Incident ID | Date | Service | Severity | Duration (min) | Customers Affected | Root Cause Category | MTTR (min) |
|---|---|---|---|---:|---:|---|---:|
| INC-2024-0703 | 2024-07-03 | Auth & Identity | Sev-2 | 14 | 0 | Config drift | 14 |
| INC-2024-0711 | 2024-07-11 | Data Ingestion | Sev-2 | 22 | 0 | Schema migration race | 22 |
| INC-2024-0718 | 2024-07-18 | Analytics Dashboard | Sev-3 | 38 | 0 | Memory leak | 38 |
| INC-2024-0724 | 2024-07-24 | Internal Admin | Sev-3 | 51 | 0 | Expired TLS cert | 51 |
| INC-2024-0729 | 2024-07-29 | Auth & Identity | Sev-1 | 8 | 1,240 | DB connection pool exhaustion | 8 |
| INC-2024-0806 | 2024-08-06 | ML Inference | Sev-1 | 47 | 3,812 | CUDA driver incompatibility | 47 |
| INC-2024-0812 | 2024-08-12 | Data Ingestion | Sev-2 | 31 | 0 | Kafka consumer lag spike | 31 |
| INC-2024-0819 | 2024-08-19 | Analytics Dashboard | Sev-2 | 19 | 0 | BigQuery quota hit | 19 |
| INC-2024-0827 | 2024-08-27 | Internal Admin | Sev-3 | 44 | 0 | Deployment rollback | 44 |
| INC-2024-0904 | 2024-09-04 | Auth & Identity | Sev-1 | 6 | 890 | OAuth token cache miss | 6 |
| INC-2024-0909 | 2024-09-09 | Event Streaming | Sev-2 | 11 | 0 | Broker rebalance | 11 |
| INC-2024-0914 | 2024-09-14 | Commerce Checkout | Sev-2 | 9 | 0 | Load spike (planned event) | 9 |
| INC-2024-0922 | 2024-09-22 | Data Ingestion | Sev-1 | 18 | 0 | Pipeline dependency failure | 18 |

*Full incident list including Sev-3 entries available in the internal incident tracker. Only incidents ≥ 6 minutes duration are listed here.*

**Q3 Key Metrics:**

| Metric | Q3 Value | Q2 Value | Target |
|---|---:|---:|---:|
| Total incidents | 23 | 31 | < 25 |
| Sev-1 incidents | 8 | 12 | < 8 |
| Mean Time to Detect (MTTD, min) | 4.2 | 6.8 | < 5.0 |
| Mean Time to Resolve (MTTR, min) | 24.1 | 38.4 | < 30.0 |
| Incidents with post-mortem published | 8 / 8 | 9 / 12 | 100% |
| SLA breach events | 0 | 1 | 0 |

Sev-1 count of 8 is exactly at the quarterly target. The Platform Engineering team has committed to a new alert threshold tuning initiative in Q4 to reduce Sev-1 incidents caused by transient spikes (INC-2024-0729 and INC-2024-0904 are candidates for reclassification pending retrospective review).

---

## 4. API Latency Percentiles

Latency is measured end-to-end at the API Gateway layer using Datadog APM trace data. Values represent the median across all three production regions (us-east-1, eu-west-1, ap-southeast-1) weighted by request volume. Requests timed out at the gateway (> 30 s) are excluded from percentile calculations but counted separately.

| Endpoint | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | p99.9 (ms) | Q2 p99 (ms) | QoQ p99 Change |
|---|---:|---:|---:|---:|---:|---:|---|
| GET /api/v2/products | 18 | 41 | 68 | 142 | 891 | 160 | -11.3% |
| POST /api/v2/checkout | 44 | 98 | 134 | 289 | 1,204 | 312 | -7.4% |
| GET /api/v2/auth/token | 9 | 22 | 31 | 78 | 441 | 91 | -14.3% |
| GET /api/v2/user/profile | 12 | 29 | 44 | 109 | 620 | 118 | -7.6% |
| POST /api/v2/events/ingest | 7 | 18 | 28 | 61 | 388 | 58 | +5.2% |
| GET /api/v2/recommendations | 88 | 201 | 312 | 748 | 3,910 | 741 | +0.9% |
| POST /api/v2/payments/initiate | 112 | 248 | 381 | 802 | 4,120 | 898 | -10.7% |
| GET /api/v2/analytics/summary | 201 | 489 | 712 | 1,841 | 9,200 | 2,104 | -12.5% |
| DELETE /api/v2/sessions | 6 | 14 | 19 | 44 | 290 | 47 | -6.4% |
| POST /api/v2/ml/score | 340 | 780 | 1,100 | 2,890 | 11,400 | 2,710 | +6.6% |

The `/api/v2/ml/score` p99 regression (+6.6%) is linked to model size increases following the LLM upgrade deployed on 28 August. The ML team has an active project to implement response streaming that is expected to reduce perceived p99 latency by 40–60% by end of Q4. The `/api/v2/events/ingest` p99 regression (+5.2%) is under investigation; preliminary data suggests it correlates with Kafka partition rebalance events.

---

## 5. Throughput & Error Rates

| Service | Q3 Total Requests | Peak RPS | Avg RPS | Error Rate (%) | 5xx Rate (%) | Timeout Rate (%) |
|---|---:|---:|---:|---:|---:|---:|
| API Gateway (total) | 4,182,400,000 | 28,412 | 15,218 | 0.041 | 0.008 | 0.004 |
| Commerce Checkout | 218,600,000 | 4,810 | 794 | 0.028 | 0.004 | 0.001 |
| Auth & Identity | 1,041,200,000 | 12,100 | 3,787 | 0.018 | 0.003 | 0.002 |
| Data Ingestion Pipeline | 892,100,000 | 9,840 | 3,244 | 0.062 | 0.011 | 0.007 |
| ML Inference Service | 44,800,000 | 1,204 | 163 | 0.110 | 0.041 | 0.028 |
| Event Streaming (Kafka) | 6,210,000,000 | 81,200 | 22,587 | 0.004 | 0.001 | 0.000 |
| Analytics Dashboard | 9,800,000 | 480 | 36 | 0.312 | 0.089 | 0.041 |

The Analytics Dashboard carries the highest error rate (0.312%) of any tracked service, driven primarily by BigQuery quota errors (INC-2024-0819) and dashboard queries exceeding the 60-second timeout applied at the application layer. A quota increase request was submitted to GCP on 2 September and is pending approval.

---

## 6. Deployment Frequency & Change Failure Rate

Mirefall targets a DORA Elite classification across all four metrics. Q3 results show continued improvement, with Deployment Frequency reaching Elite level for the first time in three of four measured services.

| Service | Deployments (Q3) | Avg/Week | Rollbacks | Change Failure Rate (%) | DORA Level | Lead Time (hrs) |
|---|---:|---:|---:|---:|---|---:|
| API Gateway | 94 | 7.2 | 2 | 2.1% | Elite | 3.4 |
| Commerce Checkout | 71 | 5.5 | 3 | 4.2% | Elite | 4.1 |
| Auth & Identity | 108 | 8.3 | 1 | 0.9% | Elite | 2.8 |
| Data Ingestion | 52 | 4.0 | 4 | 7.7% | High | 6.2 |
| ML Inference | 28 | 2.2 | 2 | 7.1% | High | 9.8 |
| Internal Admin | 39 | 3.0 | 5 | 12.8% | Medium | 14.1 |
| Analytics Dashboard | 21 | 1.6 | 2 | 9.5% | High | 18.4 |

**DORA thresholds used:** Elite = on-demand / multiple per day; High = weekly–monthly; Medium = monthly–biannual. Change Failure Rate Elite < 5%, High 5–10%, Medium 10–15%.

Data Ingestion (7.7%) and ML Inference (7.1%) change failure rates are above the Elite threshold. The primary contributing factor is the absence of automated integration tests for infrastructure-layer changes; both teams have approved test coverage work items for Q4.

---

## 7. On-Call & Escalation Statistics

| Metric | Jul | Aug | Sep | Q3 Total | Q2 Total | YoY Change |
|---|---:|---:|---:|---:|---:|---|
| Total pages fired | 84 | 112 | 79 | 275 | 341 | -19.4% |
| Actionable pages | 61 | 88 | 58 | 207 | 289 | -28.4% |
| Noisy / false-positive pages | 23 | 24 | 21 | 68 | 52 | +30.8% |
| Escalations to Sev-1 | 3 | 4 | 1 | 8 | 12 | -33.3% |
| Unique engineers paged | 14 | 18 | 13 | 31 | 34 | -8.8% |
| Avg pages per on-call week | 8.4 | 11.2 | 7.9 | 9.2 | 11.4 | -19.3% |
| Engineers with > 10 pages/week | 2 | 4 | 1 | — | — | — |

False-positive page volume increased 30.8% year-on-year. Alert fatigue is a recognized risk; the Platform Engineering team conducted an alert audit in September and identified 41 alert rules that are candidates for threshold adjustment or consolidation. Remediation is targeted for completion by 31 October.

---

## 8. Infrastructure Capacity — Utilisation Summary

Capacity is reviewed monthly. The table below reflects peak utilisation observed during Q3 across the primary production clusters. Targets are set to maintain headroom for a 2× traffic spike without manual intervention.

| Cluster / Resource | Region | Peak CPU (%) | Avg CPU (%) | Peak Mem (%) | Avg Mem (%) | Storage Used (TB) | Storage Cap (TB) | Headroom |
|---|---|---:|---:|---:|---:|---:|---:|---|
| prod-eks-us-east-1 | us-east-1 | 74 | 41 | 68 | 49 | — | — | Adequate |
| prod-eks-eu-west-1 | eu-west-1 | 61 | 38 | 59 | 44 | — | — | Adequate |
| prod-eks-ap-southeast-1 | ap-southeast-1 | 58 | 32 | 54 | 41 | — | — | Adequate |
| prod-rds-commerce-primary | us-east-1 | 62 | 28 | 81 | 67 | 4.8 | 8.0 | **Review** |
| prod-rds-auth-replica | us-east-1 | 44 | 19 | 58 | 41 | 1.2 | 4.0 | Adequate |
| prod-redshift-dwh | us-east-1 | 71 | 48 | 76 | 61 | 18.4 | 24.0 | Adequate |
| prod-kafka-cluster | us-east-1 | 55 | 31 | 72 | 58 | 6.1 | 10.0 | Adequate |
| prod-ml-gpu-pool | us-east-1 | 88 | 64 | 91 | 78 | — | — | **Critical** |
| prod-bigquery-slots | GCP | — | — | — | — | 214.0 | 400.0 | Adequate |

The ML GPU pool is flagged as Critical: peak memory utilisation reached 91% during model inference benchmarking on 19 September. The ML & AI team has submitted a request to add 8 additional A100 nodes to the pool. Approval is pending the Q4 capital expenditure committee meeting on 10 October. The prod-rds-commerce-primary instance storage growth rate of approximately 400 GB per month projects a breach of the 8 TB limit in Q1 FY2025; a storage auto-scaling policy has been drafted and is pending DBA review.

---

## 9. Q4 Reliability Targets & Commitments

| Objective | Q3 Actual | Q4 Target | Owner | Due |
|---|---:|---:|---|---|
| Platform weighted availability (%) | 99.94 | 99.95 | Platform Eng Lead | Ongoing |
| Sev-1 incidents | 8 | ≤ 6 | All service leads | Ongoing |
| MTTR — Sev-1 (min) | 24.1 | ≤ 20.0 | On-Call Lead | Ongoing |
| False-positive page rate (%) | 24.7 | ≤ 15.0 | Platform Eng Lead | 2024-10-31 |
| ML GPU pool headroom restored (%) | 9 peak | ≥ 20 | ML & AI Lead | 2024-10-10 |
| Commerce DB storage auto-scaling | Not enabled | Enabled | DBA Lead | 2024-10-20 |
| Data Ingestion change failure rate (%) | 7.7 | ≤ 5.0 | Data Eng Lead | 2024-12-31 |
| Alert audit remediation (rules adjusted) | 0 / 41 | 41 / 41 | Platform Eng Lead | 2024-10-31 |
| ML inference streaming (p99 reduction) | Baseline | -40% vs Q3 | ML & AI Lead | 2024-12-31 |

All Q4 targets were ratified by the Engineering Leadership team on 4 October 2024. Progress will be reviewed at the monthly reliability review on the first Wednesday of each month.
