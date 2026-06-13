# SIEM Integration Architecture

INFER Secure forwards enriched IoT threat events to customer SIEM platforms in real time. This page describes supported integrations, event formats, and the forwarding pipeline.

## Supported SIEM Connectors


    |  SIEM | Protocol | Auth | Format | Latency (alert → SIEM) |

    |  Splunk Enterprise / Cloud | HTTP Event Collector (HEC) | HEC token | JSON | <30 s |

    |  Microsoft Sentinel | Azure Monitor Logs Ingestion API (DCR) | Service Principal / Managed Identity | JSON (custom table schema) | <60 s |

    |  IBM QRadar | Syslog TLS (RFC 5424) | TLS client cert | CEF (Common Event Format) | <15 s |

    |  Elastic SIEM / Security | Elasticsearch Bulk API | API key | ECS (Elastic Common Schema) | <30 s |

    |  Sumo Logic | HTTP Source endpoint | URL token | JSON | <30 s |

    |  Chronicle (Google SecOps) | Ingestion API v2 | Service Account JSON | UDM (Unified Data Model) | <60 s |

    |  Generic Webhook | HTTPS POST | Bearer / HMAC-SHA256 signature | JSON (INFER native) | <15 s |



## INFER Native Alert Schema → Splunk HEC Payload

json

## Microsoft Sentinel — DCR Mapping

json

## Alert Deduplication & Rate Control

  - **Dedup window:** Same `(device_id, threat_type)` within 15 minutes → suppressed; only one event forwarded to SIEM
  - **Alert storms:** If a single device generates >20 alerts in 5 minutes, switch to summary mode: one aggregated event per 5-minute window
  - **SIEM rate limit:** Max 1,000 events/minute per tenant per SIEM connector; backpressure via internal queue (Redis Stream), no drops
  - **Retry policy:** Exponential backoff (1s, 2s, 4s, 8s, 16s); after 5 retries, event moved to dead-letter queue + ops alert

## MITRE ATT&CK Mapping


    |  INFER Threat Type | MITRE Technique | Tactic |

    |  c2_beaconing | T1071, T1571, T1132 | Command and Control |

    |  lateral_movement | T1046, T1021, T1570 | Lateral Movement, Discovery |

    |  data_exfiltration | T1041, T1048, T1567 | Exfiltration |

    |  credential_stuffing | T1110, T1078 | Credential Access |

    |  firmware_tampering | T1542, T1601 | Persistence, Defense Evasion |

    |  dos_participation | T1498, T1499 | Impact |
