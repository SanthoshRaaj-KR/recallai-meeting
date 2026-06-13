# Kafka & Stream Processing Architecture

The INFER telemetry backbone is built on Apache Kafka (MSK Serverless) with Kafka Streams for stateful processing and Flink for complex event processing. This page covers the full topology.

## Kafka Cluster Configuration


    |  Parameter | Value | Rationale |

    |  Kafka version | 3.6.1 | KRaft mode (no Zookeeper) |

    |  Deployment | AWS MSK Serverless | Auto-scaling; no broker management |

    |  Max throughput | 1 GB/s ingress, 2 GB/s egress | MSK Serverless limits |

    |  Message size limit | 1 MB (default) / 10 MB (netflow topic) | Avoids broker memory pressure |

    |  Compression | lz4 (producer-side) | ~60% size reduction on JSON telemetry |

    |  Producer acks | acks=all | No data loss on leader failure |

    |  Consumer isolation | read_committed | Exactly-once semantics for compliance events |



## Stream Processing Topology

bash100 events/min — throttled)

device.telemetry.enriched
    │
    ▼ [Feature Extraction — Flink job, 5-min tumbling windows]
    │  - Compute 364 feature dimensions per device
    │  - Windowed aggregations: sum, mean, p95, entropy
    │
    ├──► device.features.5m               (feature vectors for ML)
    │
    └──► device.stats.hourly              (rolled-up stats → InfluxDB sink)

device.features.5m
    │
    ▼ [Anomaly Scoring — TorchServe HTTP sink via Kafka Connect]
    │  - Batch inference: 500 vectors per request
    │
    └──► device.anomaly.scores

device.anomaly.scores + device.security.events
    │
    ▼ [Alert Engine — Flink CEP (Complex Event Processing)]
    │  - Correlate anomaly scores with auth events
    │  - Detect multi-stage attack patterns (e.g., port scan → auth fail → high anomaly)
    │
    └──► alert.outbound                   (fan-out to SIEM, PagerDuty, email)]]>

## Flink CEP Pattern — Multi-Stage Attack Detection

java attackPattern = Pattern
    .begin("reconnaissance")
        .where(e -> e.getEventType().equals("security.port_scan"))
    .followedByAny("credential_attack")
        .where(e -> e.getEventType().equals("security.auth_attempt")
                 && e.getPayload().getInt("failed_count") > 5)
        .within(Time.minutes(30))
    .followedByAny("anomaly_spike")
        .where(e -> e.getEventType().equals("anomaly.score")
                 && e.getPayload().getDouble("score") > 0.85)
        .within(Time.minutes(60));

// When pattern matches → emit HIGH severity alert with kill-chain evidence]]>

## Kafka Connect Sink Connectors


    |  Sink | Topics Consumed | Connector | Batch Size |

    |  InfluxDB Cloud | device.stats.hourly, device.anomaly.scores | Custom (HTTP sink) | 5,000 points/req |

    |  PostgreSQL (INFER DB) | device.lifecycle, alert.outbound | JDBC Sink Connector | 1,000 rows/req |

    |  Neo4j AuraDB | device.telemetry.enriched (subset) | Neo4j Kafka Connector | 500 nodes/req |

    |  S3 (data lake) | All topics | Confluent S3 Sink | 128 MB Parquet files |

    |  Elasticsearch | device.security.events, alert.outbound | Elasticsearch Sink 3.x | 500 docs/req |



## Consumer Group Lag SLOs


    |  Consumer Group | Max Lag (records) | Alert Threshold |

    |  enrichment-streams | 10,000 | 50,000 → PagerDuty P2 |

    |  anomaly-scorer | 50,000 | 200,000 → PagerDuty P1 |

    |  alert-engine-flink | 5,000 | 20,000 → PagerDuty P1 |

    |  influxdb-sink | 100,000 | 500,000 → PagerDuty P2 |

    |  s3-archive-sink | No SLO | 10M → Slack warning |
