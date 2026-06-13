# ML Anomaly Detection — Model Architecture

INFER Secure uses a multi-stage ML pipeline to detect behavioral anomalies in IoT/OT device telemetry without requiring labelled attack data. This page covers the model architecture, feature engineering, training cadence, and serving infrastructure.

## Pipeline Overview


    |  Stage | Model | Input | Output | Latency (p99) |

    |  Feature Extraction | Rule-based + statistical | Raw telemetry stream | 364-dim feature vector | 8 ms |

    |  Short-Term Anomaly | Isolation Forest (per device class) | 5-min rolling window features | anomaly_score_short (0–1) | 12 ms |

    |  Long-Term Drift | Autoencoder (LSTM, 2-layer) | 24-h sliding window | reconstruction_error | 45 ms |

    |  Threat Classifier | Gradient Boosted Trees (XGBoost) | Both anomaly scores + metadata | threat_type, confidence | 6 ms |

    |  Alert Aggregator | Rule engine (OPA) | Threat classifier output | Alert or suppress | 3 ms |



## Feature Engineering — 364 Dimensions


    |  Feature Group | Count | Examples |

    |  Network flow statistics | 48 | bytes_out_5m, unique_dst_ips_1h, dst_port_entropy, new_conn_rate |

    |  Protocol behavior | 64 | http_methods_seen, tls_version_dist, dns_query_rate, snmp_oid_count |

    |  Temporal patterns | 72 | hourly_traffic_profile (24 bins × 3 metrics), weekend_vs_weekday_ratio |

    |  Device health | 32 | cpu_pct_delta, mem_growth_rate, session_count_zscore, temp_anomaly |

    |  Auth & access | 24 | failed_auth_rate, new_src_ip_ratio, privileged_cmd_count |

    |  Peer comparison | 56 | zscore vs. same device_class (18 metrics × 3 aggregations: mean, p95, p99) |

    |  Graph features | 68 | new_neighbors_1h, community_change, centrality_delta, unusual_port_pairs |



## Isolation Forest Configuration

python= current champion on held-out validation set.]]>

## LSTM Autoencoder Architecture

python

## Model Serving Infrastructure

  - **Runtime:** TorchServe 0.9 on AWS ECS Fargate (auto-scaling 2–20 replicas per tenant tier)
  - **Model store:** S3 + MLflow model registry; TorchServe pulls on startup via model-store URI
  - **Warm-up:** First 72 hours after device onboarding — anomaly scores suppressed, baseline being established
  - **Drift detection:** PSI (Population Stability Index) computed daily on feature distributions; PSI > 0.2 triggers forced retraining
  - **Explainability:** SHAP values computed for every score > 0.7; stored in InfluxDB and surfaced in the alert detail panel

## Threat Classification Labels


    |  Label | Description | Typical Signals |

    |  c2_beaconing | Periodic outbound to C2 server | Regular interval dst, low byte variance, unusual dst ASN |

    |  lateral_movement | Scanning/connecting to unexpected internal hosts | New dst IPs spike, ICMP sweep, unusual port access |

    |  data_exfiltration | Unusual outbound data volume | bytes_out zscore > 5, dst outside normal ASNs |

    |  credential_stuffing | Repeated auth failures | failed_auth_rate > 10/min, multiple src IPs |

    |  firmware_tampering | Unexpected firmware hash change | lifecycle.boot with unknown firmware_ver hash |

    |  dos_participation | Device is part of a botnet DDoS | pps spike, syn_flood pattern, single dst_ip |

    |  unknown_anomaly | High reconstruction error, no label match | Autoencoder error > 4σ, no XGBoost match > 0.6 |
