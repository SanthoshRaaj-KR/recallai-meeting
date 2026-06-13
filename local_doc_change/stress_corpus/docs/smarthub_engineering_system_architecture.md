# Engineering — System Architecture

This page describes the high-level architecture of the INFER™ platform as of v2.5. Audience: Engineering team and technical architects.

## Top-Level Component Map


    |  Component | Technology | Responsibility |

    |  API Gateway | AWS API Gateway + Lambda Authorizer | Auth, rate limiting, request routing |

    |  Core API | Python 3.11 / FastAPI + Uvicorn | REST + WebSocket endpoints for UI and integrations |

    |  Device Telemetry Pipeline | Apache Kafka + Kafka Streams | Ingest, normalize, and stream device telemetry at scale |

    |  ML Inference Service | Python / PyTorch + TorchServe | Real-time anomaly scoring, behavioral baselines |

    |  Policy Engine | Go 1.22 / Open Policy Agent (OPA) | Evaluate device compliance rules in <10ms |

    |  Graph Store | Neo4j AuraDB | Device relationship graph: subnets, groups, policies |

    |  Time-Series Store | InfluxDB Cloud | Telemetry history, SPS trend, health metrics |

    |  Relational DB | PostgreSQL 15 (AWS RDS Multi-AZ) | Tenant data, device inventory, user accounts |

    |  Search / RAG | Pinecone (vector) + Elasticsearch | Natural language device queries, log search |

    |  Frontend | React 18 + TypeScript + Vite | Web dashboard, device map, alert console |

    |  SD-EDGE Gateway | Go 1.22 / gRPC + MQTT broker (EMQX) | Edge protocol translation, local policy enforcement |

    |  Auth | Auth0 (SAML 2.0 / OIDC / MFA) | SSO, user lifecycle, per-tenant isolation |



## Data Flow: Device Telemetry

  - SD-EDGE Gateway collects telemetry (SNMP polls, MQTT subscribe, passive network capture)
  - Normalizes to canonical JSON telemetry schema and publishes to Kafka topic `device.telemetry.raw`
  - Kafka Streams job enriches with device metadata from PostgreSQL cache, publishes to `device.telemetry.enriched`
  - ML Inference Service consumes enriched stream, computes anomaly score, publishes to `device.anomaly.scores`
  - Policy Engine evaluates SPS rules against anomaly scores + compliance state
  - Alert Service fans out to SIEM (Splunk/Sentinel), PagerDuty, Slack, or email
  - InfluxDB ingests enriched telemetry for dashboards and trend analysis

## Multi-Tenancy Model

INFER uses a **schema-per-tenant** model in PostgreSQL for strong data isolation. Each tenant has:

  - Dedicated PostgreSQL schema (e.g., `tenant_acme`)
  - Tenant-scoped JWT claims enforced at API Gateway and Policy Engine
  - Separate Kafka consumer group per tenant for telemetry processing
  - Isolated Pinecone namespace for RAG queries

## SLAs & Reliability Targets


    |  Service | Availability Target | RTO | RPO |

    |  Core API | 99.9% | 15 min | 5 min |

    |  Telemetry Pipeline | 99.5% | 30 min | 1 min |

    |  SD-EDGE Gateway (local) | 99.99% (offline capable) | N/A (self-healing) | N/A |
