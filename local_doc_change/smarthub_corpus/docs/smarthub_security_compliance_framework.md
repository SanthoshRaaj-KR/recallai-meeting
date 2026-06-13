# Security & Compliance Framework

SmartHub.ai's security philosophy is built on continuous, automated posture management rather than point-in-time audits. Every device under INFER management has a live Security Posture Score (SPS) visible in real time.

## Security Posture Score (SPS)

SPS is a 0–100 score computed per device every 5 minutes across four dimensions:


    |  Dimension | Weight | Key Signals |

    |  Firmware Currency | 25% | CVE exposure of installed firmware version |

    |  Configuration Hygiene | 25% | Default credentials, open ports, disabled encryption |

    |  Network Behavior | 30% | Deviation from behavioral baseline (anomaly score) |

    |  Identity & Access | 20% | Certificate validity, stale credentials, over-privileged accounts |



## Compliance Frameworks Supported


    |  Framework | Applicability | INFER Coverage |

    |  NIST CSF 2.0 | All sectors | Identify, Protect, Detect, Respond, Recover |

    |  IEC 62443 | Industrial OT / SCADA | SL-2 controls mapped to device policies |

    |  CIS Controls v8 | All sectors | Controls 1–12 fully automated |

    |  HIPAA (IoMT) | Healthcare IoT | Device encryption, access logging, audit trail |

    |  PCI DSS 4.0 | Payment / Retail | Segmentation validation for IoT zones |

    |  SOC 2 Type II | SaaS platform (SmartHub itself) | Certified; report available under NDA |



## Threat Detection Capabilities

  - **Zero-day IoT Exploits:** Behavioral baselines detect exploitation attempts even before CVE publication
  - **Lateral Movement Detection:** Flags unusual east-west traffic from a device to unexpected subnets
  - **C2 Beaconing:** ML model identifies periodic outbound connections characteristic of command-and-control
  - **Credential Stuffing:** Repeated failed authentication events across multiple devices flagged as coordinated attack
  - **Firmware Tampering:** Hash-based integrity verification detects unauthorized firmware modifications

## Incident Response Integration

INFER integrates with SOAR platforms to automate response actions:

  - Quarantine a compromised device (VLAN reassignment via SD-EDGE Gateway)
  - Revoke device certificates via Mocana CMS
  - Auto-create incident ticket in ServiceNow or Jira
  - Notify on-call via PagerDuty
