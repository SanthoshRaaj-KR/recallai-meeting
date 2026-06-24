# DevOps & CI/CD Pipeline

INFER uses a trunk-based development model with automated quality gates at every stage. All services are containerised and deployed to AWS ECS Fargate via GitHub Actions and ArgoCD.

## Repository Structure


    |  Repository | Language | Description |

    |  infer-api | Python 3.11 | Core FastAPI service (device, alert, compliance APIs) |

    |  infer-streams | Java 21 / Kafka Streams | Telemetry enrichment and aggregation |

    |  infer-flink | Java 21 / Apache Flink | Feature extraction and CEP alert engine |

    |  infer-ml | Python 3.11 / PyTorch | Anomaly detection models and TorchServe config |

    |  infer-policy-engine | Go 1.22 / OPA | Compliance rules and authz policies |

    |  sdedge-gateway | Go 1.22 | SD-EDGE Gateway binary |

    |  infer-frontend | TypeScript / React 18 | Web dashboard |

    |  infer-infra | Terraform / Helm | AWS infrastructure and Kubernetes charts |

    |  infer-platform | Python | Internal tooling: db migrations, seed scripts, load tests |



## GitHub Actions CI Pipeline

yaml 0.5% or p99_latency > 500ms]]>

## Progressive Delivery (Argo Rollouts)

yaml= 0.995"   # 99.5% success rate
      autoRollback:
        enabled: true]]>

## Database Migration Strategy

  - Migrations managed by **Alembic** (Python) and **golang-migrate** (Go services)
  - All migrations must be **backwards-compatible** for at least one release cycle (expand-contract pattern)
  - Column drops require a 2-step process: (1) stop writing in release N, (2) drop in release N+1
  - Large table migrations (ALTER TABLE on >10M rows) run as background jobs using `pg_repack` to avoid table locks
  - Migration smoke test: `alembic upgrade head` runs in CI against a snapshot of production schema

## On-Call & Incident Response


    |  Tier | Rotation | Tools | SLO |

    |  L1 — Platform On-Call | Weekly rotation, 2 engineers | PagerDuty, Grafana, Runbooks | Acknowledge P1 in 5 min |

    |  L2 — Service Owner | Per-service, business hours | Slack #oncall, GitHub Issues | Acknowledge P2 in 30 min |

    |  L3 — Incident Commander | Senior eng, major incidents only | Zoom war room, PagerDuty Escalation | Engaged within 15 min of P1 |
