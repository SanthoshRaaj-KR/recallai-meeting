# Device Onboarding Guide

This guide explains how to onboard devices into INFER™ — from initial discovery through full policy coverage. Three onboarding paths are supported: Auto-Discovery, Bulk Import, and Manual Registration.

## Prerequisites

  - SD-EDGE Gateway deployed and connected to INFER SaaS (green status in **Sites > Gateways**)
  - Network VLAN(s) accessible from the SD-EDGE Gateway (SNMP read community string or credentials configured)
  - INFER role: `Device Manager` or higher

## Path 1: Auto-Discovery (Recommended)

  - Navigate to **Inventory > Discover**
  - Select the SD-EDGE Gateway for the target site
  - Enter subnet CIDR(s) to scan (e.g., `10.10.5.0/24`)
  - Choose scan profile: *Passive Only* (low-impact), *Active SNMP*, or *Deep Scan* (includes port scan)
  - Click **Start Discovery**. Progress visible in the Jobs panel.
  - Discovered devices appear in **Inventory > Pending Review** for approval
  - Review fingerprinted device class and metadata; click **Approve & Onboard**

## Path 2: Bulk Import (CSV)

Use when you have an existing spreadsheet inventory.


    |  Column | Required | Example |

    |  ip_address | Yes | 10.10.5.42 |

    |  mac_address | No | AA:BB:CC:DD:EE:FF |

    |  hostname | No | cam-lobby-01 |

    |  device_class | No (auto-detected if blank) | ip_camera |

    |  manufacturer | No | Axis |

    |  model | No | P3245-V |

    |  site_id | Yes | site_hq_sf |

    |  group_tags | No (pipe-separated) | security|floor-1|critical |



Upload via **Inventory > Bulk Import** or API: `POST /v2/devices/bulk-import`. Maximum 10,000 rows per file.

## Path 3: Manual Registration

For isolated or air-gapped devices not reachable via SD-EDGE Gateway.

  - Go to **Inventory > Add Device**
  - Enter IP address, MAC, hostname, device class, and site
  - Optionally upload a device certificate for mutual TLS
  - Assign to a policy group
  - Click **Register**

## Post-Onboarding Checklist


    |  Step | Action | Where |

    |  1 | Assign device to a Policy Group | Inventory > Device > Policies tab |

    |  2 | Verify baseline telemetry is flowing | Inventory > Device > Telemetry tab |

    |  3 | Check initial SPS score (target >70 within 24h) | Inventory > Device > Security tab |

    |  4 | Review and resolve any auto-generated alerts | Alerts console |

    |  5 | Configure alert notifications (email, Slack, SIEM) | Settings > Notifications |
