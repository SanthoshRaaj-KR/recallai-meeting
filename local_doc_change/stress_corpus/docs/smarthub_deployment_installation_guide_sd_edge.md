# Deployment & Installation Guide — SD-EDGE Gateway

The SD-EDGE Gateway is the edge component that connects your on-premises devices to the INFER™ cloud. This guide covers the three supported deployment methods.

## System Requirements


    |  Component | Minimum | Recommended |

    |  CPU | 2 vCPU (x86_64) | 4 vCPU |

    |  RAM | 4 GB | 8 GB |

    |  Disk | 20 GB | 50 GB SSD |

    |  OS | Ubuntu 20.04 LTS / RHEL 8 | Ubuntu 22.04 LTS |

    |  Network | 1 Gbps NIC with routed access to device VLANs | 2 x 1 Gbps (bonded) |

    |  Outbound HTTPS | Port 443 to `gateway.smarthub.ai` | Same |



## Method 1: Docker Compose (Recommended for PoC)

bash

## Method 2: Kubernetes (Helm Chart)

bash

## Method 3: OVA (VMware ESXi / KVM)

  - Download the OVA from `https://releases.smarthub.ai/sdedge/latest/sdedge-gateway.ova`
  - Deploy via vSphere Client: **File > Deploy OVF Template**
  - Allocate minimum 4 vCPU, 8 GB RAM, 50 GB disk
  - Set network adapter to a trunk port with access to device VLANs
  - Power on; the first-boot wizard will prompt for Tenant ID and Gateway Token

## Post-Deployment Verification


    |  Check | Expected Result |

    |  INFER Portal > Sites > Gateways | Gateway appears with status **Connected** (green) within 5 minutes |

    |  Gateway health API | `curl http://<gateway-ip>:8080/health` returns `{"status":"ok"}` |

    |  Outbound connectivity | `curl -I https://gateway.smarthub.ai/ping` returns HTTP 200 |



## Upgrading the Gateway

Gateway updates are pushed automatically when **Auto-Update** is enabled (default). To manually trigger:

bash
