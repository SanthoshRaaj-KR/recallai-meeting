# Graph Database Schema — Neo4j Device Graph

INFER uses Neo4j AuraDB as a graph store for device relationship modelling, network topology, and Graph RAG queries. This page documents node labels, relationship types, property schemas, and key Cypher query patterns.

## Node Labels


    |  Label | Description | Key Properties |

    |  Device | A managed IoT/OT endpoint | device_id, tenant_id, mac, ip, device_class, sps_score, firmware_ver, site_id, last_seen_ts |

    |  Site | A physical or logical location | site_id, tenant_id, name, city, country, lat, lon, gateway_count |

    |  Gateway | An SD-EDGE gateway instance | gateway_id, tenant_id, site_id, version, status, device_count |

    |  Subnet | IP subnet managed by INFER | cidr, vlan_id, site_id, device_count, zone (IT/OT/DMZ) |

    |  CVE | A known vulnerability | cve_id, cvss_score, cvss_vector, published_ts, affected_firmware_pattern |

    |  PolicyGroup | A compliance/security policy group | group_id, name, framework, rule_count |

    |  AlertEvent | A security alert (denormalised for graph traversal) | alert_id, threat_type, severity, ts, resolved |

    |  Vendor | Device manufacturer | vendor_id, name, oui_prefix, support_url |



## Relationship Types


    |  Relationship | From → To | Properties |

    |  LOCATED_AT | Device → Site | since_ts |

    |  MANAGED_BY | Device → Gateway | first_seen_ts, protocol |

    |  IN_SUBNET | Device → Subnet | assigned_ts, dhcp (bool) |

    |  COMMUNICATES_WITH | Device → Device | bytes_7d, sessions_7d, last_seen_ts, protocol_list |

    |  EXPOSED_TO | Device → CVE | detected_ts, remediation_status |

    |  MEMBER_OF | Device → PolicyGroup | enrolled_ts, compliance_pct |

    |  TRIGGERED | Device → AlertEvent | (none; edge existence is the fact) |

    |  MANUFACTURED_BY | Device → Vendor | (none) |

    |  CONTAINS | Site → Subnet | (none) |

    |  HOSTS | Site → Gateway | (none) |



## Index & Constraint Definitions

cypher

## Key Query Patterns

cypher(c:CVE)
WHERE c.cvss_score >= 9.0 AND d.sps_score 10 unique peers in last 7 days
MATCH (d:Device {tenant_id: $tid})-[r:COMMUNICATES_WITH]->(:Device)
WITH d, count(r) AS peer_count
WHERE peer_count > 10
RETURN d.device_id, d.ip, d.device_class, peer_count
ORDER BY peer_count DESC LIMIT 50;

// 3. Blast radius — what could a compromised camera reach?
MATCH (source:Device {device_id: $did})-[:COMMUNICATES_WITH*1..2]->(reachable:Device)
WHERE reachable.device_class IN ['plc','rtu','historian','workstation']
RETURN DISTINCT reachable.device_id, reachable.device_class, reachable.ip;

// 4. Sites with lowest average SPS score
MATCH (d:Device)-[:LOCATED_AT]->(s:Site {tenant_id: $tid})
WITH s, avg(d.sps_score) AS avg_sps, count(d) AS device_count
WHERE device_count > 5
RETURN s.name, s.city, round(avg_sps, 1) AS avg_sps, device_count
ORDER BY avg_sps ASC LIMIT 10;]]>

## Graph Refresh Cadence


    |  Data | Update Frequency | Source |

    |  Device properties (SPS, firmware, IP) | Every 5 min | Kafka consumer — device.telemetry.enriched |

    |  COMMUNICATES_WITH edges | Hourly (rolling 7-day window) | Aggregated from device.netflow.raw |

    |  EXPOSED_TO (CVE) edges | Daily at 02:00 UTC | NVD feed + firmware fingerprint matching |

    |  AlertEvent nodes | On alert creation | alert.outbound Kafka topic |

    |  Full graph reindex | Weekly (Sunday 03:00 UTC) | Batch reconciliation job |
