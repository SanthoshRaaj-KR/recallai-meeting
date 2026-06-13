# INFER™ Platform — Product Overview

INFER™ is SmartHub.ai's flagship AI-enabled product suite for enterprise edge and IoT/OT management. The name stands for **Intelligent Network for Edge Reasoning**. It provides a unified control plane across every connected asset in the enterprise — from IP cameras and industrial sensors to conference room AV equipment and smart building controllers.

## Product Pillars


    |  Pillar | Capability | Key Benefit |

    |  INFER Discover | Agentless device discovery & fingerprinting | Know every device on your network in <24 hours |

    |  INFER Manage | Unified lifecycle management (onboarding → decommission) | Bulk provisioning, OTA updates, templated configs |

    |  INFER Secure | Real-time threat detection, anomaly detection, SIEM integration | Zero-day IoT threat visibility without agents |

    |  INFER Comply | Continuous compliance automation (NIST, CIS, IEC 62443) | Audit-ready posture reports in one click |

    |  INFER Predict | AI-driven predictive maintenance & health scoring | Prevent device failure before it happens |



## Key Features

  - **Agentless Discovery:** No software installed on devices; passive network fingerprinting using ML-based protocol analysis.
  - **Bulk Onboarding:** Onboard thousands of devices via CSV import, DHCP hooks, or API. Preconfigured templates for 2,000+ device types.
  - **AI Alerts & Anomaly Detection:** Behavioral baselines per device class; flags deviations without signature updates.
  - **Gen-AI Adapter SDKs:** LLM-powered natural language queries over your asset inventory (e.g., "Show me all cameras with firmware older than 6 months in Building 3").
  - **SIEM Integration:** Native connectors for Splunk, Microsoft Sentinel, IBM QRadar, and Elastic SIEM. IoT threats surfaced with full context.
  - **OTA Security Updates:** Firmware and configuration updates pushed securely over the air with rollback support.
  - **Identity & Access Management:** Role-based access control (RBAC), SSO via SAML 2.0 / OIDC, and per-device credential vaulting.

## Supported Protocols & Device Classes


    |  Category | Protocols / Standards | Examples |

    |  IP Cameras / Physical Security | ONVIF, RTSP, PSIA | Axis, Hikvision, Bosch, Avigilon |

    |  Building Automation | BACnet, Modbus, KNX, LonWorks | Siemens, Honeywell, Johnson Controls |

    |  Industrial OT | OPC-UA, DNP3, IEC 61850, Profinet | Rockwell, Schneider Electric, ABB |

    |  Network Infrastructure | SNMP, SSH, Netconf/YANG | Cisco, Juniper, Aruba |

    |  AV / Conference Room | HDMI-CEC, Dante, AMX/Crestron APIs | Crestron, Extron, Zoom Rooms hardware |

    |  Smart Sensors / IoT | MQTT, CoAP, Zigbee, Z-Wave, LoRaWAN | Generic sensor nodes, smart meters |



## Deployment Options

  - **SaaS (Recommended):** Multi-tenant cloud hosted on AWS; data residency options in US, EU, APAC.
  - **Private Cloud:** Customer-managed Kubernetes cluster; INFER Helm chart available.
  - **On-Premise:** OVA appliance for air-gapped environments; supports VMware ESXi and KVM.
