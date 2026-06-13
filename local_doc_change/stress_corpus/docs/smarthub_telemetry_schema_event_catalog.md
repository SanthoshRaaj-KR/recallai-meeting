# Telemetry Schema & Event Catalog

All telemetry emitted by managed devices flows through the INFER pipeline as canonical JSON events. This page is the authoritative reference for every event type, its fields, and Kafka topic routing.

## Canonical Envelope

Every event — regardless of source protocol — is normalized into the following envelope before entering Kafka:

json

## Event Type Registry


    |  event_type | Kafka Topic | Frequency | Description |

    |  telemetry.metric | device.telemetry.enriched | Every 60 s | CPU, memory, uptime, temperature, link state |

    |  telemetry.netflow | device.netflow.raw | Every 30 s (aggregated) | Src/dst IP, port, bytes, packets, protocol — sampled at 1:100 |

    |  security.auth_attempt | device.security.events | On event | Login attempt: username, result (success/fail), source IP |

    |  security.config_change | device.security.events | On event | Config diff, actor (user/automated), before/after hash |

    |  security.port_scan | device.security.events | On detection | Scanner IP, scanned ports, scan pattern classification |

    |  lifecycle.boot | device.lifecycle | On event | Device boot: firmware version, boot reason (cold/watchdog/ota) |

    |  lifecycle.firmware_update | device.lifecycle | On event | OTA result: from_version, to_version, duration_s, success |

    |  lifecycle.cert_expiry | device.lifecycle | Daily check | Certificate CN, expiry_ts, days_remaining |

    |  anomaly.score | device.anomaly.scores | Every 5 min | ML anomaly score (0–1), contributing features, model version |

    |  compliance.check | device.compliance | Every 15 min | Policy ID, control ID, result (pass/fail/warn), evidence |



## Metric Payload Schema (telemetry.metric)

json

## Netflow Payload Schema (telemetry.netflow)

json

## Kafka Topic Configuration


    |  Topic | Partitions | Replication | Retention | Compaction |

    |  device.telemetry.raw | 64 | 3 | 24 h | No |

    |  device.telemetry.enriched | 64 | 3 | 7 days | No |

    |  device.netflow.raw | 128 | 3 | 3 days | No |

    |  device.security.events | 32 | 3 | 30 days | No |

    |  device.anomaly.scores | 32 | 3 | 90 days | Yes (by device_id) |

    |  device.lifecycle | 16 | 3 | 365 days | Yes (by device_id) |

    |  device.compliance | 32 | 3 | 365 days | No |

    |  alert.outbound | 16 | 3 | 90 days | No |



## Schema Evolution Policy

  - Schemas registered in **Confluent Schema Registry** using Avro. All topics enforce `FORWARD_TRANSITIVE` compatibility.
  - Adding optional fields: allowed without version bump. Breaking changes require a new `schema_ver` and a 30-day dual-publish period.
  - Consumers must tolerate unknown fields (ignore-unknown pattern).
