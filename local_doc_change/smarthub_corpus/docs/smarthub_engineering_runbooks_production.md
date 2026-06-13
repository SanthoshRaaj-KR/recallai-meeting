# Engineering Runbooks — Production Operations

Standard operating procedures for the INFER platform on-call engineer. Last reviewed: November 2024. Owner: Platform Engineering.

## Runbook Index


    |  Incident Type | Severity | Jump To |

    |  API elevated error rate (>1%) | P1/P2 | Section: API Error Rate |

    |  Kafka consumer lag >200k records | P1 | Section: Kafka Lag |

    |  Anomaly scorer down | P1 | Section: ML Scorer Outage |

    |  Gateway mass disconnect | P1 | Section: Gateway Disconnect |

    |  PostgreSQL replica lag >30s | P2 | Section: DB Replication Lag |

    |  SIEM connector failures | P2 | Section: SIEM Failures |

    |  OTA campaign stuck | P3 | Section: OTA Stuck Campaign |



## API Elevated Error Rate

bash

## Kafka Consumer Lag — Emergency Scale-Out

bash/dev/null   | awk '$5 > 10000 {print $1, $2, $5}' | sort -k3 -rn | head -20

# Scale out enrichment-streams ECS service immediately
aws ecs update-service --cluster infer-prod   --service infer-enrichment-streams   --desired-count 20

# If Flink job is behind, restart with higher parallelism
flink run -m yarn-cluster -p 32   -c com.smarthub.flink.FeatureExtractionJob   infer-flink-jobs.jar   --kafka.bootstrap.servers $MSK_BROKER   --parallelism 32

# Monitor recovery
watch -n 10 'kafka-consumer-groups.sh --bootstrap-server $MSK_BROKER   --describe --group enrichment-streams-prod | tail -5']]>

## Gateway Mass Disconnect

bash/dev/null   | openssl x509 -noout -dates

# If EMQX overloaded — check connection rate
curl -s https://mqtt.smarthub.ai/api/v5/metrics   -H "Authorization: Bearer $EMQX_API_KEY"   | jq '.["connections.count"], .["messages.received.rate"]'

# Force reconnect: bump the MQTT gateway config version
# (gateways poll for config changes every 60s and reconnect on version mismatch)
aws ssm put-parameter --name /infer/prod/mqtt/config_version   --value "$(date +%s)" --overwrite]]>

## OTA Campaign Stuck

bash

## Key Dashboards & Runbook Links


    |  Dashboard | URL | When to Use |

    |  API Overview | grafana.smarthub.internal/d/infer-api | Error rate, latency, traffic spikes |

    |  Kafka Lag | grafana.smarthub.internal/d/kafka-lag | Consumer group lag, topic throughput |

    |  ML Pipeline | grafana.smarthub.internal/d/ml-pipeline | Inference latency, model drift, scorer health |

    |  Gateway Fleet | grafana.smarthub.internal/d/gw-fleet | Connected gateways, disconnect events, memory |

    |  RDS / PgBouncer | grafana.smarthub.internal/d/rds | Query latency, connections, replication lag |
