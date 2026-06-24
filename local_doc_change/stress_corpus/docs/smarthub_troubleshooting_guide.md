# Troubleshooting Guide — Common Issues

This guide covers the most frequently reported issues by customers and internal support. Last updated by Engineering — Nov 2024.

## SD-EDGE Gateway Issues


    |  Symptom | Probable Cause | Resolution |

    |  Gateway shows **Disconnected** in portal | Outbound HTTPS blocked by firewall | Allow `gateway.smarthub.ai:443` and `telemetry.smarthub.ai:443` outbound. Verify with `curl -I https://gateway.smarthub.ai/ping` |

    |  Gateway connects but no devices discovered | SNMP community string mismatch, or device VLANs not routed to gateway | Check **Sites > Gateway > Diagnostics > Run Network Reachability Test**. Verify VLAN routing. |

    |  High memory usage (>90%) on gateway VM | Too many concurrent SNMP polls on large subnet | Reduce polling frequency in **Settings > Discovery Profiles** from 60s to 300s. Add more RAM if subnet >5,000 devices. |

    |  BACnet devices not discovered | BACnet/IP broadcast not crossing subnet boundary | Configure BACnet Broadcast Management Device (BBMD) address in SD-EDGE gateway config: `sdedge.bacnet.bbmd_address=10.10.1.1` |



## Device Fingerprinting Issues


    |  Symptom | Probable Cause | Resolution |

    |  Device shows as **Unknown** class after 24h | Device uses non-standard OUI or proprietary protocol | Manually set device class in portal. Submit device details via **Help > Report Unknown Device** so SmartHub can add it to the fingerprint library. |

    |  Firmware version shows **N/A** | Device does not expose firmware version via SNMP/HTTP | Enter firmware version manually or use INFER Manage to trigger a credentials-based config read (if device supports SSH/API) |

    |  Duplicate device entries after discovery | Device has multiple IP addresses (multi-homed) or DHCP lease changed | Merge duplicates via **Inventory > Merge Devices**. Enable MAC-based deduplication in tenant settings. |



## Alert & SIEM Issues


    |  Symptom | Probable Cause | Resolution |

    |  No alerts forwarded to Splunk | Splunk HEC token expired or index permissions changed | Rotate HEC token in Splunk. Update in INFER: **Integrations > Splunk > Edit** |

    |  Alert storm — thousands of low-severity alerts | New device onboarded without baseline established (baselines need 72h) | Suppress alerts for `baseline_learning` tagged devices for 72h. Adjust alert threshold in **Policies > Alert Tuning** |

    |  Sentinel connector shows **Error: 403** | Managed Identity or service principal missing **Monitoring Metrics Publisher** role on DCE | Grant `Monitoring Metrics Publisher` role to the INFER app registration on the target Data Collection Endpoint |



## Escalation Path

  - **Tier 1 (Self-serve):** INFER portal → Help → Diagnostics; this guide
  - **Tier 2 (Customer Success):** [support@smarthub.ai](mailto:support@smarthub.ai) — response within 4 business hours (Standard) / 1 hour (Enterprise)
  - **Tier 3 (Engineering Escalation):** Filed by CS; engineering on-call paged for P1 incidents

## Useful Diagnostic Commands

bash&1 | grep -E "Connected|SSL|HTTP"

# SNMP test from gateway to a device
docker compose exec sdedge-gateway snmpwalk -v2c -c public 10.10.5.42 sysDescr]]>
