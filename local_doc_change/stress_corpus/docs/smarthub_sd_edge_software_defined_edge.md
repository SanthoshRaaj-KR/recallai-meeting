# SD-EDGE — Software Defined Edge Solution

SD-EDGE (Software Defined Edge) is SmartHub.ai's infrastructure abstraction layer that sits between the enterprise network core and the heterogeneous sea of edge devices. It virtualizes device management complexity and provides a unified policy plane regardless of underlying hardware or protocol diversity.

## Architecture Overview


    |  Layer | Component | Function |

    |  Cloud Control Plane | INFER™ SaaS Backend | Policy engine, AI analytics, SIEM forwarding, dashboards |

    |  Edge Orchestration | SD-EDGE Gateway | Protocol translation, local buffering, offline mode |

    |  Device Layer | Native device firmware + INFER adapters | Telemetry collection, config enforcement |



## SD-EDGE Gateway

The SD-EDGE Gateway is a lightweight software appliance (available as OVA, Docker container, or Raspberry Pi image) deployed at the network edge — typically per-site or per-subnet. It handles:

  - Protocol translation (Modbus → MQTT, BACnet → REST, etc.)
  - Local enforcement of security policies even when WAN connectivity is lost
  - Compressed telemetry batching to reduce cloud data transfer costs
  - Certificate-based mutual TLS authentication with each managed device

## SD-EDGE vs. Traditional Approaches


    |  Capability | Traditional IT Tools | SD-EDGE |

    |  IoT Device Discovery | Manual inventory / spreadsheets | Automatic fingerprinting in <24h |

    |  Firmware Management | Manual, per-vendor console | Unified OTA across all vendors |

    |  Threat Detection | No visibility beyond IP/MAC | Behavioral anomaly detection per device |

    |  Compliance Reporting | Manual audits, quarterly | Continuous, real-time posture score |

    |  Protocol Support | IP only (TCP/UDP) | IT + OT + IoT (20+ protocols) |



## Partner Integrations (SD-EDGE Ecosystem)

  - **Mocana:** Hardware-rooted identity for IoT devices via X.509 certificates
  - **Microsoft Azure IoT:** Bi-directional device twin sync
  - **AWS IoT Greengrass:** SD-EDGE gateway runs as a Greengrass component
  - **Palo Alto Networks Cortex XSOAR:** Automated playbook triggers on IoT threat events
  - **ServiceNow ITOM:** Discovered assets auto-populated into CMDB
