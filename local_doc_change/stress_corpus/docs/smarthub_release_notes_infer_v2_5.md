# Release Notes — INFER™ v2.5

**Release Date:** 2024-11-15 | **Type:** Major Feature Release

## Highlights

  - Gen-AI Natural Language Query (NLQ) for device inventory — ask questions in plain English
  - Microsoft Sentinel native connector (no Syslog forwarder required)
  - SD-EDGE Gateway v2.5 with BACnet/IP and DNP3 protocol support
  - Bulk OTA firmware campaigns with staged rollout and automatic rollback
  - New NIST CSF 2.0 compliance dashboard

## New Features


    |  Feature | Component | Description |

    |  Gen-AI NLQ | INFER Discover | Ask "Show me all cameras with firmware older than 6 months in Building 3" — results in <3s powered by RAG over your device inventory |

    |  Sentinel Connector | INFER Secure | Push IoT alerts directly to Microsoft Sentinel workspace via Azure Monitor REST API. No Syslog bridge needed. |

    |  BACnet/IP Support | SD-EDGE Gateway | Discover and monitor building automation controllers via BACnet/IP without a separate BMS integration |

    |  DNP3 Support | SD-EDGE Gateway | Monitor electrical substation and water treatment RTUs using DNP3 Subset Level 2 |

    |  Staged OTA Campaigns | INFER Manage | Roll out firmware to 5% → 25% → 100% of a device group with automatic rollback on failure rate >5% |

    |  NIST CSF 2.0 Dashboard | INFER Comply | New compliance view mapped to all 6 NIST CSF 2.0 functions with drill-down to individual controls |

    |  Device Risk Heatmap | Dashboard | Site-level heatmap visualizing device count vs. average SPS — instantly identify high-risk sites |



## Bug Fixes


    |  Issue ID | Severity | Description |

    |  INF-2341 | High | Fixed: SNMP v3 devices with AES-256 auth would fail fingerprinting and appear as "Unknown" class |

    |  INF-2289 | Medium | Fixed: Alert email notifications would not send if the device label contained special characters (&, <, >) |

    |  INF-2178 | Medium | Fixed: CSV bulk import would silently skip rows with IPv6 addresses instead of reporting an error |

    |  INF-2095 | Low | Fixed: Compliance report PDF generation would time out for tenants with >50,000 devices |



## Breaking Changes

  - **API v1 deprecated:** `/v1/*` endpoints will return HTTP 410 as of 2025-05-15. Migrate to `/v2/*`. See the [API Migration Guide](#).
  - **SD-EDGE Gateway <2.3 EOL:** Gateways running v2.2 or earlier will no longer connect after 2025-02-01. Upgrade via `helm upgrade` or the portal's auto-update flow.

## Known Issues


    |  Issue ID | Description | Workaround |

    |  INF-2412 | NLQ queries that reference more than 3 sites may return incomplete results | Filter by a single site until the fix ships in v2.5.1 (ETA: 2024-12-10) |

    |  INF-2398 | Sentinel Connector may drop events if Sentinel workspace is in "Free" tier with log cap exceeded | Upgrade Sentinel workspace to Pay-as-you-go or configure alert deduplication to reduce volume |
