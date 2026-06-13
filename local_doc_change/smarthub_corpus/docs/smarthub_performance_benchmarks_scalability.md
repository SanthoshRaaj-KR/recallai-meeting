# Performance Benchmarks & Scalability

This page documents INFER platform benchmarks as of v2.5, measured in SmartHub's performance lab and on production MSK/ECS infrastructure. All numbers are p99 unless stated otherwise.

## Telemetry Ingestion Throughput


    |  Scenario | Devices | Events/s | Kafka Lag (p99) | CPU (enrichment fleet) |

    |  Small tenant | 1,000 | 1,200 | <100 records | 8% |

    |  Medium tenant | 10,000 | 12,500 | <500 records | 22% |

    |  Large tenant | 100,000 | 118,000 | <2,000 records | 61% |

    |  Peak burst (IoT storm) | 100,000 | 480,000 | <15,000 records | 94% (auto-scaled +8 replicas) |

    |  Platform max (all tenants) | ~2,000,000 | 2,400,000 | <30,000 records | Distributed across 240 vCPU |



## API Latency (REST)


    |  Endpoint | p50 | p95 | p99 | Cache |

    |  GET /devices (10 results) | 12 ms | 28 ms | 45 ms | Redis, 30 s TTL |

    |  GET /devices/{id} | 8 ms | 19 ms | 31 ms | Redis, 60 s TTL |

    |  GET /alerts (open, limit 25) | 15 ms | 34 ms | 52 ms | No cache |

    |  POST /devices/{id}/quarantine | 180 ms | 420 ms | 890 ms | N/A (writes MQTT + DB) |

    |  GET /compliance/posture | 85 ms | 210 ms | 380 ms | Redis, 5 min TTL |

    |  POST /compliance/report (PDF) | 4.2 s | 11 s | 18 s | No cache (async job) |

    |  Gen-AI NLQ query | 1.8 s | 3.2 s | 5.1 s | Semantic cache (Pinecone) |



## ML Anomaly Detection Latency


    |  Stage | p50 | p99 | Batch Size |

    |  Feature extraction (Flink) | 2.1 ms | 8.4 ms | 1 event |

    |  Isolation Forest inference | 0.8 ms | 3.2 ms | 500 vectors |

    |  LSTM autoencoder inference | 18 ms | 45 ms | 100 sequences |

    |  XGBoost classifier | 0.4 ms | 1.8 ms | 500 vectors |

    |  End-to-end (event → alert) | 28 s | 4.2 min | N/A (includes 5-min window) |



## SD-EDGE Gateway Performance


    |  Metric | Value | Hardware |

    |  Max concurrent managed devices | 5,000 | 4 vCPU / 8 GB RAM |

    |  SNMP poll throughput | 8,000 OIDs/s | Same |

    |  Netflow processing | 500,000 flows/min | Same |

    |  MQTT message throughput | 50,000 msg/s | Same |

    |  Memory per managed device | ~80 KB | — |

    |  OTA simultaneous transfers | 10 (configurable to 25) | — |



## Database Performance


    |  Query | p50 | p99 | Table Size (1M devices) |

    |  Device lookup by IP | 0.3 ms | 1.2 ms | devices: 2.4 GB |

    |  Open alerts for site | 2.1 ms | 8.7 ms | alerts: 18 GB (1 year) |

    |  Compliance posture aggregation | 45 ms | 180 ms | compliance_checks: 42 GB |

    |  Neo4j blast-radius traversal (depth 2) | 12 ms | 85 ms | 10M nodes, 80M edges |



## Auto-Scaling Configuration (ECS Fargate)

yaml 65% for 2 min
    scale_in_threshold:  cpu_avg  50000 records
    scale_in_threshold:  kafka_consumer_lag  200000 OR p99_inference_latency > 80ms]]>
