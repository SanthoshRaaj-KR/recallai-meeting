# Thornvale Infrastructure Cost & Budget Report — Q3 FY2024

This document is the official quarterly infrastructure cost report for Thornvale. Finance and Engineering leadership must review this report jointly before the end of each quarter. All figures are in USD unless otherwise stated. Discrepancies between this report and the general ledger must be escalated to the finance controller within five business days of publication. Historical data is retained for a minimum of seven years in accordance with Thornvale's record retention policy.

---

## 1. Executive Summary

Total infrastructure spend for Q3 FY2024 reached **$2,847,412**, representing a **6.3% increase** over Q2 and placing the quarter **$147,412 over the approved budget of $2,700,000**. The primary drivers of overage were a spike in egress costs following the August platform migration and unplanned Elastic compute scaling during the Thornvale Commerce launch event on 14 September. Reserved instance coverage improved from 61% to 74% quarter-over-quarter, partially offsetting on-demand pricing exposure.

The Security and Compliance team consumed the largest share of unplanned spend due to emergency WAF rule expansion in response to CVE-2024-3891. The Shared Services allocation was revised upward mid-quarter following a decision to consolidate three legacy data pipelines onto the central platform.

---

## 2. Cloud Spend by Provider

| Provider | Q2 FY2024 ($) | Q3 FY2024 ($) | QoQ Change ($) | QoQ Change (%) | Budget Q3 ($) | Variance ($) |
|---|---:|---:|---:|---:|---:|---:|
| AWS | 1,521,804 | 1,638,920 | +117,116 | +7.7% | 1,580,000 | +58,920 |
| GCP | 634,210 | 701,488 | +67,278 | +10.6% | 640,000 | +61,488 |
| Azure | 287,330 | 294,004 | +6,674 | +2.3% | 300,000 | -5,996 |
| Cloudflare | 98,450 | 112,000 | +13,550 | +13.8% | 95,000 | +17,000 |
| Fastly CDN | 136,000 | 101,000 | -35,000 | -25.7% | 85,000 | +16,000 |
| **Total** | **2,677,794** | **2,847,412** | **+169,618** | **+6.3%** | **2,700,000** | **+147,412** |

AWS remains the primary provider, accounting for 57.6% of total Q3 spend. GCP growth is attributed to BigQuery slot expansion approved in July to support the Analytics Platform team's transition to real-time dashboards.

---

## 3. Spend by Service Category

| Service Category | Q3 Spend ($) | % of Total | Q2 Spend ($) | QoQ ($) | Reserved Coverage |
|---|---:|---:|---:|---:|---|
| Compute (EC2 / GCE / VMs) | 1,041,200 | 36.6% | 988,100 | +53,100 | 74% |
| Managed Databases (RDS, Cloud SQL, Cosmos) | 528,600 | 18.6% | 501,400 | +27,200 | 81% |
| Object Storage (S3, GCS, Blob) | 214,800 | 7.5% | 198,200 | +16,600 | N/A |
| Data Warehousing (BigQuery, Redshift) | 309,400 | 10.9% | 261,000 | +48,400 | 55% |
| Networking & Egress | 298,712 | 10.5% | 227,114 | +71,598 | N/A |
| Kubernetes / Container Orchestration | 187,300 | 6.6% | 180,900 | +6,400 | 68% |
| Security & Compliance Tools | 142,600 | 5.0% | 98,200 | +44,400 | N/A |
| Monitoring & Observability | 74,800 | 2.6% | 71,100 | +3,700 | N/A |
| CI/CD & Build Infrastructure | 49,200 | 1.7% | 51,780 | -2,580 | N/A |
| Miscellaneous / One-off | 1,000 | 0.0% | 100,000 | -99,000 | N/A |
| **Total** | **2,847,412** | **100%** | **2,677,794** | **+169,618** | — |

Networking & Egress was the fastest-growing category in absolute terms, increasing by $71,598 (31.5%) versus Q2. This was directly caused by cross-region data replication introduced during the August migration and is expected to normalise in Q4 once traffic routing optimisation is complete.

---

## 4. Department Allocation

Each business unit is allocated a share of infrastructure cost based on a combination of direct resource tagging (83% of spend) and a proportional overhead model (17% of spend). Untagged resources are allocated to Shared Services.

| Department | Direct Spend ($) | Allocated Overhead ($) | Total Charge ($) | % of Total | Q2 Total ($) | QoQ Change (%) |
|---|---:|---:|---:|---:|---:|---:|
| Platform Engineering | 712,400 | 48,200 | 760,600 | 26.7% | 698,900 | +2.0% |
| Data & Analytics | 498,100 | 41,800 | 539,900 | 19.0% | 461,200 | +17.0% |
| Commerce & Payments | 441,600 | 37,900 | 479,500 | 16.8% | 454,100 | +5.6% |
| Security & Compliance | 298,200 | 29,100 | 327,300 | 11.5% | 224,800 | +45.6% |
| Customer Success Platform | 211,800 | 24,400 | 236,200 | 8.3% | 228,700 | +3.3% |
| ML & AI Infrastructure | 189,300 | 21,600 | 210,900 | 7.4% | 187,100 | +12.7% |
| Developer Experience | 98,400 | 17,200 | 115,600 | 4.1% | 119,300 | -3.1% |
| Shared Services (untagged) | 104,412 | 73,000 | 177,412 | 6.2% | 303,694 | -41.6% |
| **Total** | **2,554,212** | **293,200** | **2,847,412** | **100%** | **2,677,794** | **+6.3%** |

Security & Compliance showed the largest percentage increase (+45.6%) due to emergency WAF expansion and additional GuardDuty log retention. Data & Analytics growth (+17.0%) reflects the BigQuery slot expansion and a new streaming ingestion pipeline for the Thornvale Events product.

---

## 5. Reserved Instance & Savings Plan Coverage

Thornvale targets a minimum of 70% reserved coverage for compute and database workloads. Q3 marks the first quarter in which the overall compute reserved coverage target was met.

| Resource Type | On-Demand Instances | Reserved / Committed | Coverage Rate | Target | Status |
|---|---:|---:|---:|---:|---|
| AWS EC2 | 412 | 1,148 | 73.6% | 70% | Met |
| AWS RDS | 28 | 119 | 81.0% | 75% | Met |
| GCP Compute Engine | 81 | 204 | 71.6% | 70% | Met |
| GCP Cloud SQL | 14 | 38 | 73.1% | 70% | Met |
| Azure VMs | 22 | 74 | 77.1% | 70% | Met |
| Azure SQL Managed Instance | 4 | 9 | 69.2% | 65% | Met |
| BigQuery Slots | — | 2,000 | 55.0% | 60% | **At Risk** |
| Redshift Nodes | 6 | 12 | 66.7% | 70% | **Below Target** |

BigQuery slot coverage remains below target following the mid-quarter expansion. The Data & Analytics team has committed to purchasing an additional 500 annual slots by 15 October to bring coverage to 62%. Redshift reserved node procurement is scheduled for Q4 budget approval on 22 October.

---

## 6. Top 20 Cost Drivers by Resource

| Rank | Resource Name | Provider | Type | Monthly Avg ($) | Q3 Total ($) | Dept |
|---:|---|---|---|---:|---:|---|
| 1 | prod-analytics-bq-main | GCP | BigQuery | 58,200 | 174,600 | Data & Analytics |
| 2 | prod-eks-cluster-us-east-1 | AWS | EKS / EC2 | 51,400 | 154,200 | Platform Eng |
| 3 | prod-rds-commerce-primary | AWS | RDS Aurora | 42,100 | 126,300 | Commerce |
| 4 | prod-eks-cluster-eu-west-1 | AWS | EKS / EC2 | 38,900 | 116,700 | Platform Eng |
| 5 | prod-redshift-dwh | AWS | Redshift | 34,200 | 102,600 | Data & Analytics |
| 6 | prod-gcs-media-store | GCP | Cloud Storage | 29,800 | 89,400 | Commerce |
| 7 | prod-s3-backup-vault | AWS | S3 | 28,100 | 84,300 | Shared Services |
| 8 | prod-ml-training-gce | GCP | Compute Engine | 26,700 | 80,100 | ML & AI |
| 9 | prod-waf-cloudfront | AWS | CloudFront + WAF | 24,900 | 74,700 | Security |
| 10 | prod-rds-auth-replica | AWS | RDS PostgreSQL | 22,600 | 67,800 | Platform Eng |
| 11 | prod-nat-gateway-us | AWS | NAT Gateway | 21,100 | 63,300 | Platform Eng |
| 12 | prod-cosmos-events | Azure | Cosmos DB | 19,800 | 59,400 | Commerce |
| 13 | prod-dataflow-etl | GCP | Dataflow | 18,400 | 55,200 | Data & Analytics |
| 14 | prod-pubsub-ingest | GCP | Pub/Sub | 17,900 | 53,700 | Data & Analytics |
| 15 | prod-eks-spot-us | AWS | EC2 Spot | 16,200 | 48,600 | Platform Eng |
| 16 | prod-cloudwatch-logs | AWS | CloudWatch | 15,800 | 47,400 | Monitoring |
| 17 | prod-memorystore-session | GCP | Memorystore | 14,700 | 44,100 | Commerce |
| 18 | prod-guard-duty-logs | AWS | GuardDuty | 14,200 | 42,600 | Security |
| 19 | prod-artifact-registry | GCP | Artifact Registry | 13,900 | 41,700 | Dev Experience |
| 20 | prod-azure-aks-emea | Azure | AKS | 12,800 | 38,400 | Platform Eng |
| — | **Top 20 Subtotal** | — | — | — | **1,564,200** | — |
| — | **Remaining Resources** | — | — | — | **1,283,212** | — |

The top 20 resources account for 54.9% of total Q3 spend. The NAT Gateway entry (rank 11) is directly linked to the egress cost increase and is under active optimisation review.

---

## 7. Vendor Contracts & Committed Use

| Vendor | Contract Type | Annual Commit ($) | Q3 Recognised ($) | Remaining Commit ($) | Expiry Date | Owner |
|---|---|---:|---:|---:|---|---|
| AWS Enterprise Agreement | Annual CUD | 5,200,000 | 1,480,000 | 2,340,000 | 2025-06-30 | Head of Infra |
| GCP Committed Use | 3-Year CUD | 1,800,000 | 480,000 | 1,380,000 | 2026-03-31 | Head of Infra |
| Azure MOSA | Annual EA | 900,000 | 228,000 | 456,000 | 2025-01-31 | Head of Infra |
| Cloudflare Enterprise | Annual | 336,000 | 84,000 | 168,000 | 2025-04-15 | Security Lead |
| Datadog | Annual | 480,000 | 120,000 | 240,000 | 2025-02-28 | Platform Eng Lead |
| Snowflake | Capacity | 600,000 | 0 | 600,000 | Not started | Data Lead |
| PagerDuty | Annual | 48,000 | 12,000 | 24,000 | 2025-03-01 | On-Call Lead |
| Elastic Cloud | Monthly | N/A | 74,800 | N/A | Month-to-month | Security Lead |

The Snowflake commitment has not been activated; the Data & Analytics team confirmed the migration from Redshift is now planned for Q1 FY2025. Finance and Legal must be notified if the Snowflake commitment start date is pushed beyond 1 February 2025, as penalty clauses apply after a 120-day activation window.

---

## 8. Anomaly & Overage Events

Three cost anomaly events were detected by the cost alerting system during Q3. All three were investigated and resolved within the stated SLAs.

| Event ID | Date | Service | Spike Amount ($) | Root Cause | Resolution | Time to Resolve |
|---|---|---|---:|---|---|---|
| ANO-2024-0831 | 2024-08-31 | AWS NAT Gateway | 48,200 | Cross-region replication enabled without egress estimate | Routing optimisation applied; replication scoped to same-region replica | 6 days |
| ANO-2024-0914 | 2024-09-14 | AWS EC2 Auto Scaling | 31,700 | Commerce launch traffic 3.1× over load estimate | Post-event scale-in completed; launch sizing model updated | 2 days |
| ANO-2024-0921 | 2024-09-21 | AWS GuardDuty | 14,200 | Emergency WAF expansion increased log ingestion 8× | Log sampling rate adjusted; retention policy revised from 365 to 90 days | 1 day |

Total anomaly-related unplanned spend: **$94,100**, representing 64% of the total Q3 budget variance.

---

## 9. Q4 FY2024 Forecast & Actions

The Q4 approved budget is $2,750,000. Based on current run rates and committed actions, the Finance team forecasts a spend of **$2,680,000–$2,730,000**, within budget assuming the following actions are completed on schedule.

| Action Item | Owner | Due Date | Estimated Saving ($) | Status |
|---|---|---|---:|---|
| Egress routing optimisation (NAT Gateway) | Platform Eng | 2024-10-15 | 35,000–45,000 | In progress |
| Redshift reserved node purchase (12 nodes) | Data & Analytics | 2024-10-22 | 18,000 | Pending budget approval |
| BigQuery slot commitment top-up (500 slots) | Data & Analytics | 2024-10-15 | 12,000 | Pending purchase |
| GuardDuty log retention policy reduction | Security | 2024-10-01 | 9,500 | Completed |
| Decommission 3 legacy Dataflow pipelines | Data & Analytics | 2024-11-01 | 22,000 | Scheduled |
| Migrate Elastic Cloud to reserved tier | Security | 2024-10-31 | 14,000 | In progress |
| Spot instance coverage increase (ML workloads) | ML & AI | 2024-11-15 | 8,000 | Planned |

All action items with a due date before 31 October are owned by department leads who have accepted accountability in the Q4 planning session on 2 October 2024. Finance will review progress at the mid-quarter checkpoint on 15 November.
