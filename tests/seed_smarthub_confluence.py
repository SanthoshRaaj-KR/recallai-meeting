"""
Seed script: creates 13 SmartHub.ai company pages in Confluence.

Run with:
    conda run -n meetagents python tests/seed_smarthub_confluence.py

Loads credentials from .env at the repo root. Creates pages in the configured space.
Does NOT touch any pipeline code.
"""

import os
import sys
import time
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth

# ── load .env from repo root ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())

EMAIL = os.environ["ATLASSIAN_USER_EMAIL"].strip()
TOKEN = os.environ["ATLASSIAN_API_TOKEN"].strip()
DOMAIN = os.environ["ATLASSIAN_DOMAIN"].strip()
SPACE = os.environ["ATLASSIAN_SPACE_KEY"].strip()

BASE_URL = f"https://{DOMAIN}/wiki/rest/api"
AUTH = HTTPBasicAuth(EMAIL, TOKEN)
HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}


def create_page(title: str, html: str) -> dict:
    payload = {
        "type": "page",
        "title": title,
        "space": {"key": SPACE},
        "body": {"storage": {"value": html, "representation": "storage"}},
    }
    resp = requests.post(f"{BASE_URL}/content", auth=AUTH, json=payload, headers=HEADERS, timeout=20)
    if not resp.ok:
        print(f"  ERROR {resp.status_code}: {resp.text[:300]}")
        resp.raise_for_status()
    data = resp.json()
    page_id = data.get("id", "?")
    url = f"https://{DOMAIN}/wiki" + (data.get("_links", {}).get("webui", ""))
    return {"id": page_id, "url": url}


# ── PAGE DEFINITIONS ───────────────────────────────────────────────────────────

PAGES: list[tuple[str, str]] = []

# 1 — Company Overview
PAGES.append((
    "SmartHub.ai — Company Overview",
    """
<h1>SmartHub.ai — Company Overview</h1>
<p>SmartHub.ai is an enterprise IoT SecOps platform company, headquartered in the Bay Area, CA, with additional offices in Seattle and Bangalore. It is a spin-off from VMware, founded to address the critical gap in traditional IT security stacks when it comes to IoT and OT (Operational Technology) environments.</p>

<h2>Mission &amp; Vision</h2>
<p><strong>Mission:</strong> Unify device lifecycle management, real-time threat visibility, AI-driven intelligence, and compliance automation across every IoT &amp; OT connected asset.</p>
<p><strong>Vision:</strong> Make enterprise edge infrastructure as manageable, secure, and intelligent as traditional IT — regardless of hardware vendor, protocol, or scale.</p>

<h2>Company At A Glance</h2>
<table>
  <tbody>
    <tr><th>Founded</th><td>2018</td></tr>
    <tr><th>Headquarters</th><td>Bay Area, CA</td></tr>
    <tr><th>Other Offices</th><td>Seattle, WA &amp; Bangalore, India</td></tr>
    <tr><th>Stage</th><td>Series B</td></tr>
    <tr><th>Employees</th><td>~150</td></tr>
    <tr><th>Key Verticals</th><td>Physical Security, Smart Cities, Precision Agriculture, Energy, AV/IT Conference Room Management</td></tr>
  </tbody>
</table>

<h2>Core Problem We Solve</h2>
<p>Enterprise networks now contain tens of thousands of IoT and OT devices — cameras, sensors, PLCs, HVAC controllers, conference room equipment — that fall completely outside the visibility of traditional IT security tools (EDR, MDM, SIEM). These devices are unmanaged, unmonitored, and increasingly targeted by threat actors. SmartHub.ai bridges this gap.</p>

<h2>Leadership Team</h2>
<table>
  <tbody>
    <tr><th>Role</th><th>Name</th><th>Background</th></tr>
    <tr><td>CEO &amp; Co-Founder</td><td>Sanjay Rohatgi</td><td>Former VP at VMware; 20+ years in enterprise infrastructure</td></tr>
    <tr><td>CTO &amp; Co-Founder</td><td>Anand Krishnan</td><td>Former Distinguished Engineer at VMware; PhD Computer Science, CMU</td></tr>
    <tr><td>CPO</td><td>Priya Mehta</td><td>Former Director of Product, Cisco IoT</td></tr>
    <tr><td>VP Engineering</td><td>Marcus Webb</td><td>Former Principal SWE at Amazon Web Services</td></tr>
    <tr><td>VP Sales</td><td>Jennifer Lau</td><td>Former Enterprise Account Director, Palo Alto Networks</td></tr>
  </tbody>
</table>

<h2>Investors</h2>
<ul>
  <li>Sequoia Capital (Lead, Series B)</li>
  <li>Andreessen Horowitz (Series A)</li>
  <li>VMware Ventures (Seed)</li>
  <li>Intel Capital (Strategic)</li>
</ul>
""",
))

# 2 — INFER™ Platform Overview
PAGES.append((
    "INFER™ Platform — Product Overview",
    """
<h1>INFER™ Platform — Product Overview</h1>
<p>INFER™ is SmartHub.ai's flagship AI-enabled product suite for enterprise edge and IoT/OT management. The name stands for <strong>Intelligent Network for Edge Reasoning</strong>. It provides a unified control plane across every connected asset in the enterprise — from IP cameras and industrial sensors to conference room AV equipment and smart building controllers.</p>

<h2>Product Pillars</h2>
<table>
  <tbody>
    <tr><th>Pillar</th><th>Capability</th><th>Key Benefit</th></tr>
    <tr><td>INFER Discover</td><td>Agentless device discovery &amp; fingerprinting</td><td>Know every device on your network in &lt;24 hours</td></tr>
    <tr><td>INFER Manage</td><td>Unified lifecycle management (onboarding → decommission)</td><td>Bulk provisioning, OTA updates, templated configs</td></tr>
    <tr><td>INFER Secure</td><td>Real-time threat detection, anomaly detection, SIEM integration</td><td>Zero-day IoT threat visibility without agents</td></tr>
    <tr><td>INFER Comply</td><td>Continuous compliance automation (NIST, CIS, IEC 62443)</td><td>Audit-ready posture reports in one click</td></tr>
    <tr><td>INFER Predict</td><td>AI-driven predictive maintenance &amp; health scoring</td><td>Prevent device failure before it happens</td></tr>
  </tbody>
</table>

<h2>Key Features</h2>
<ul>
  <li><strong>Agentless Discovery:</strong> No software installed on devices; passive network fingerprinting using ML-based protocol analysis.</li>
  <li><strong>Bulk Onboarding:</strong> Onboard thousands of devices via CSV import, DHCP hooks, or API. Preconfigured templates for 2,000+ device types.</li>
  <li><strong>AI Alerts &amp; Anomaly Detection:</strong> Behavioral baselines per device class; flags deviations without signature updates.</li>
  <li><strong>Gen-AI Adapter SDKs:</strong> LLM-powered natural language queries over your asset inventory (e.g., "Show me all cameras with firmware older than 6 months in Building 3").</li>
  <li><strong>SIEM Integration:</strong> Native connectors for Splunk, Microsoft Sentinel, IBM QRadar, and Elastic SIEM. IoT threats surfaced with full context.</li>
  <li><strong>OTA Security Updates:</strong> Firmware and configuration updates pushed securely over the air with rollback support.</li>
  <li><strong>Identity &amp; Access Management:</strong> Role-based access control (RBAC), SSO via SAML 2.0 / OIDC, and per-device credential vaulting.</li>
</ul>

<h2>Supported Protocols &amp; Device Classes</h2>
<table>
  <tbody>
    <tr><th>Category</th><th>Protocols / Standards</th><th>Examples</th></tr>
    <tr><td>IP Cameras / Physical Security</td><td>ONVIF, RTSP, PSIA</td><td>Axis, Hikvision, Bosch, Avigilon</td></tr>
    <tr><td>Building Automation</td><td>BACnet, Modbus, KNX, LonWorks</td><td>Siemens, Honeywell, Johnson Controls</td></tr>
    <tr><td>Industrial OT</td><td>OPC-UA, DNP3, IEC 61850, Profinet</td><td>Rockwell, Schneider Electric, ABB</td></tr>
    <tr><td>Network Infrastructure</td><td>SNMP, SSH, Netconf/YANG</td><td>Cisco, Juniper, Aruba</td></tr>
    <tr><td>AV / Conference Room</td><td>HDMI-CEC, Dante, AMX/Crestron APIs</td><td>Crestron, Extron, Zoom Rooms hardware</td></tr>
    <tr><td>Smart Sensors / IoT</td><td>MQTT, CoAP, Zigbee, Z-Wave, LoRaWAN</td><td>Generic sensor nodes, smart meters</td></tr>
  </tbody>
</table>

<h2>Deployment Options</h2>
<ul>
  <li><strong>SaaS (Recommended):</strong> Multi-tenant cloud hosted on AWS; data residency options in US, EU, APAC.</li>
  <li><strong>Private Cloud:</strong> Customer-managed Kubernetes cluster; INFER Helm chart available.</li>
  <li><strong>On-Premise:</strong> OVA appliance for air-gapped environments; supports VMware ESXi and KVM.</li>
</ul>
""",
))

# 3 — SD-EDGE Solution
PAGES.append((
    "SD-EDGE — Software Defined Edge Solution",
    """
<h1>SD-EDGE — Software Defined Edge Solution</h1>
<p>SD-EDGE (Software Defined Edge) is SmartHub.ai's infrastructure abstraction layer that sits between the enterprise network core and the heterogeneous sea of edge devices. It virtualizes device management complexity and provides a unified policy plane regardless of underlying hardware or protocol diversity.</p>

<h2>Architecture Overview</h2>
<table>
  <tbody>
    <tr><th>Layer</th><th>Component</th><th>Function</th></tr>
    <tr><td>Cloud Control Plane</td><td>INFER™ SaaS Backend</td><td>Policy engine, AI analytics, SIEM forwarding, dashboards</td></tr>
    <tr><td>Edge Orchestration</td><td>SD-EDGE Gateway</td><td>Protocol translation, local buffering, offline mode</td></tr>
    <tr><td>Device Layer</td><td>Native device firmware + INFER adapters</td><td>Telemetry collection, config enforcement</td></tr>
  </tbody>
</table>

<h2>SD-EDGE Gateway</h2>
<p>The SD-EDGE Gateway is a lightweight software appliance (available as OVA, Docker container, or Raspberry Pi image) deployed at the network edge — typically per-site or per-subnet. It handles:</p>
<ul>
  <li>Protocol translation (Modbus → MQTT, BACnet → REST, etc.)</li>
  <li>Local enforcement of security policies even when WAN connectivity is lost</li>
  <li>Compressed telemetry batching to reduce cloud data transfer costs</li>
  <li>Certificate-based mutual TLS authentication with each managed device</li>
</ul>

<h2>SD-EDGE vs. Traditional Approaches</h2>
<table>
  <tbody>
    <tr><th>Capability</th><th>Traditional IT Tools</th><th>SD-EDGE</th></tr>
    <tr><td>IoT Device Discovery</td><td>Manual inventory / spreadsheets</td><td>Automatic fingerprinting in &lt;24h</td></tr>
    <tr><td>Firmware Management</td><td>Manual, per-vendor console</td><td>Unified OTA across all vendors</td></tr>
    <tr><td>Threat Detection</td><td>No visibility beyond IP/MAC</td><td>Behavioral anomaly detection per device</td></tr>
    <tr><td>Compliance Reporting</td><td>Manual audits, quarterly</td><td>Continuous, real-time posture score</td></tr>
    <tr><td>Protocol Support</td><td>IP only (TCP/UDP)</td><td>IT + OT + IoT (20+ protocols)</td></tr>
  </tbody>
</table>

<h2>Partner Integrations (SD-EDGE Ecosystem)</h2>
<ul>
  <li><strong>Mocana:</strong> Hardware-rooted identity for IoT devices via X.509 certificates</li>
  <li><strong>Microsoft Azure IoT:</strong> Bi-directional device twin sync</li>
  <li><strong>AWS IoT Greengrass:</strong> SD-EDGE gateway runs as a Greengrass component</li>
  <li><strong>Palo Alto Networks Cortex XSOAR:</strong> Automated playbook triggers on IoT threat events</li>
  <li><strong>ServiceNow ITOM:</strong> Discovered assets auto-populated into CMDB</li>
</ul>
""",
))

# 4 — Security & Compliance Framework
PAGES.append((
    "Security & Compliance Framework",
    """
<h1>Security &amp; Compliance Framework</h1>
<p>SmartHub.ai's security philosophy is built on continuous, automated posture management rather than point-in-time audits. Every device under INFER management has a live Security Posture Score (SPS) visible in real time.</p>

<h2>Security Posture Score (SPS)</h2>
<p>SPS is a 0–100 score computed per device every 5 minutes across four dimensions:</p>
<table>
  <tbody>
    <tr><th>Dimension</th><th>Weight</th><th>Key Signals</th></tr>
    <tr><td>Firmware Currency</td><td>25%</td><td>CVE exposure of installed firmware version</td></tr>
    <tr><td>Configuration Hygiene</td><td>25%</td><td>Default credentials, open ports, disabled encryption</td></tr>
    <tr><td>Network Behavior</td><td>30%</td><td>Deviation from behavioral baseline (anomaly score)</td></tr>
    <tr><td>Identity &amp; Access</td><td>20%</td><td>Certificate validity, stale credentials, over-privileged accounts</td></tr>
  </tbody>
</table>

<h2>Compliance Frameworks Supported</h2>
<table>
  <tbody>
    <tr><th>Framework</th><th>Applicability</th><th>INFER Coverage</th></tr>
    <tr><td>NIST CSF 2.0</td><td>All sectors</td><td>Identify, Protect, Detect, Respond, Recover</td></tr>
    <tr><td>IEC 62443</td><td>Industrial OT / SCADA</td><td>SL-2 controls mapped to device policies</td></tr>
    <tr><td>CIS Controls v8</td><td>All sectors</td><td>Controls 1–12 fully automated</td></tr>
    <tr><td>HIPAA (IoMT)</td><td>Healthcare IoT</td><td>Device encryption, access logging, audit trail</td></tr>
    <tr><td>PCI DSS 4.0</td><td>Payment / Retail</td><td>Segmentation validation for IoT zones</td></tr>
    <tr><td>SOC 2 Type II</td><td>SaaS platform (SmartHub itself)</td><td>Certified; report available under NDA</td></tr>
  </tbody>
</table>

<h2>Threat Detection Capabilities</h2>
<ul>
  <li><strong>Zero-day IoT Exploits:</strong> Behavioral baselines detect exploitation attempts even before CVE publication</li>
  <li><strong>Lateral Movement Detection:</strong> Flags unusual east-west traffic from a device to unexpected subnets</li>
  <li><strong>C2 Beaconing:</strong> ML model identifies periodic outbound connections characteristic of command-and-control</li>
  <li><strong>Credential Stuffing:</strong> Repeated failed authentication events across multiple devices flagged as coordinated attack</li>
  <li><strong>Firmware Tampering:</strong> Hash-based integrity verification detects unauthorized firmware modifications</li>
</ul>

<h2>Incident Response Integration</h2>
<p>INFER integrates with SOAR platforms to automate response actions:</p>
<ul>
  <li>Quarantine a compromised device (VLAN reassignment via SD-EDGE Gateway)</li>
  <li>Revoke device certificates via Mocana CMS</li>
  <li>Auto-create incident ticket in ServiceNow or Jira</li>
  <li>Notify on-call via PagerDuty</li>
</ul>
""",
))

# 5 — Engineering: System Architecture
PAGES.append((
    "Engineering — System Architecture",
    """
<h1>Engineering — System Architecture</h1>
<p>This page describes the high-level architecture of the INFER™ platform as of v2.5. Audience: Engineering team and technical architects.</p>

<h2>Top-Level Component Map</h2>
<table>
  <tbody>
    <tr><th>Component</th><th>Technology</th><th>Responsibility</th></tr>
    <tr><td>API Gateway</td><td>AWS API Gateway + Lambda Authorizer</td><td>Auth, rate limiting, request routing</td></tr>
    <tr><td>Core API</td><td>Python 3.11 / FastAPI + Uvicorn</td><td>REST + WebSocket endpoints for UI and integrations</td></tr>
    <tr><td>Device Telemetry Pipeline</td><td>Apache Kafka + Kafka Streams</td><td>Ingest, normalize, and stream device telemetry at scale</td></tr>
    <tr><td>ML Inference Service</td><td>Python / PyTorch + TorchServe</td><td>Real-time anomaly scoring, behavioral baselines</td></tr>
    <tr><td>Policy Engine</td><td>Go 1.22 / Open Policy Agent (OPA)</td><td>Evaluate device compliance rules in &lt;10ms</td></tr>
    <tr><td>Graph Store</td><td>Neo4j AuraDB</td><td>Device relationship graph: subnets, groups, policies</td></tr>
    <tr><td>Time-Series Store</td><td>InfluxDB Cloud</td><td>Telemetry history, SPS trend, health metrics</td></tr>
    <tr><td>Relational DB</td><td>PostgreSQL 15 (AWS RDS Multi-AZ)</td><td>Tenant data, device inventory, user accounts</td></tr>
    <tr><td>Search / RAG</td><td>Pinecone (vector) + Elasticsearch</td><td>Natural language device queries, log search</td></tr>
    <tr><td>Frontend</td><td>React 18 + TypeScript + Vite</td><td>Web dashboard, device map, alert console</td></tr>
    <tr><td>SD-EDGE Gateway</td><td>Go 1.22 / gRPC + MQTT broker (EMQX)</td><td>Edge protocol translation, local policy enforcement</td></tr>
    <tr><td>Auth</td><td>Auth0 (SAML 2.0 / OIDC / MFA)</td><td>SSO, user lifecycle, per-tenant isolation</td></tr>
  </tbody>
</table>

<h2>Data Flow: Device Telemetry</h2>
<ol>
  <li>SD-EDGE Gateway collects telemetry (SNMP polls, MQTT subscribe, passive network capture)</li>
  <li>Normalizes to canonical JSON telemetry schema and publishes to Kafka topic <code>device.telemetry.raw</code></li>
  <li>Kafka Streams job enriches with device metadata from PostgreSQL cache, publishes to <code>device.telemetry.enriched</code></li>
  <li>ML Inference Service consumes enriched stream, computes anomaly score, publishes to <code>device.anomaly.scores</code></li>
  <li>Policy Engine evaluates SPS rules against anomaly scores + compliance state</li>
  <li>Alert Service fans out to SIEM (Splunk/Sentinel), PagerDuty, Slack, or email</li>
  <li>InfluxDB ingests enriched telemetry for dashboards and trend analysis</li>
</ol>

<h2>Multi-Tenancy Model</h2>
<p>INFER uses a <strong>schema-per-tenant</strong> model in PostgreSQL for strong data isolation. Each tenant has:</p>
<ul>
  <li>Dedicated PostgreSQL schema (e.g., <code>tenant_acme</code>)</li>
  <li>Tenant-scoped JWT claims enforced at API Gateway and Policy Engine</li>
  <li>Separate Kafka consumer group per tenant for telemetry processing</li>
  <li>Isolated Pinecone namespace for RAG queries</li>
</ul>

<h2>SLAs &amp; Reliability Targets</h2>
<table>
  <tbody>
    <tr><th>Service</th><th>Availability Target</th><th>RTO</th><th>RPO</th></tr>
    <tr><td>Core API</td><td>99.9%</td><td>15 min</td><td>5 min</td></tr>
    <tr><td>Telemetry Pipeline</td><td>99.5%</td><td>30 min</td><td>1 min</td></tr>
    <tr><td>SD-EDGE Gateway (local)</td><td>99.99% (offline capable)</td><td>N/A (self-healing)</td><td>N/A</td></tr>
  </tbody>
</table>
""",
))

# 6 — API Reference
PAGES.append((
    "API Reference Guide — INFER REST API v2",
    """
<h1>API Reference Guide — INFER REST API v2</h1>
<p>Base URL: <code>https://api.smarthub.ai/v2</code></p>
<p>Authentication: Bearer token (JWT) obtained via <code>POST /auth/token</code>. All requests must include <code>Authorization: Bearer &lt;token&gt;</code>.</p>

<h2>Authentication</h2>
<table>
  <tbody>
    <tr><th>Method</th><th>Endpoint</th><th>Description</th></tr>
    <tr><td>POST</td><td>/auth/token</td><td>Exchange API key for JWT access token (expires 1h)</td></tr>
    <tr><td>POST</td><td>/auth/refresh</td><td>Refresh access token using refresh token</td></tr>
    <tr><td>DELETE</td><td>/auth/token</td><td>Revoke current token</td></tr>
  </tbody>
</table>

<h2>Devices</h2>
<table>
  <tbody>
    <tr><th>Method</th><th>Endpoint</th><th>Description</th></tr>
    <tr><td>GET</td><td>/devices</td><td>List all devices (paginated). Query params: <code>page</code>, <code>limit</code>, <code>site_id</code>, <code>status</code>, <code>device_class</code></td></tr>
    <tr><td>GET</td><td>/devices/{device_id}</td><td>Get full device record including SPS, last seen, firmware version</td></tr>
    <tr><td>POST</td><td>/devices</td><td>Manually register a device (for devices not auto-discovered)</td></tr>
    <tr><td>PATCH</td><td>/devices/{device_id}</td><td>Update device metadata (label, site assignment, group tags)</td></tr>
    <tr><td>DELETE</td><td>/devices/{device_id}</td><td>Decommission a device (removes from active inventory)</td></tr>
    <tr><td>POST</td><td>/devices/bulk-import</td><td>CSV bulk import; returns job_id for async status polling</td></tr>
    <tr><td>GET</td><td>/devices/{device_id}/telemetry</td><td>Historical telemetry. Query params: <code>start</code>, <code>end</code>, <code>metric</code></td></tr>
    <tr><td>POST</td><td>/devices/{device_id}/quarantine</td><td>Isolate device on the network immediately</td></tr>
    <tr><td>POST</td><td>/devices/{device_id}/ota-update</td><td>Trigger OTA firmware update; returns job_id</td></tr>
  </tbody>
</table>

<h2>Alerts</h2>
<table>
  <tbody>
    <tr><th>Method</th><th>Endpoint</th><th>Description</th></tr>
    <tr><td>GET</td><td>/alerts</td><td>List alerts. Filter by <code>severity</code>, <code>status</code>, <code>device_id</code>, <code>site_id</code></td></tr>
    <tr><td>GET</td><td>/alerts/{alert_id}</td><td>Full alert detail including evidence and recommended actions</td></tr>
    <tr><td>PATCH</td><td>/alerts/{alert_id}</td><td>Update alert status (<code>acknowledged</code>, <code>resolved</code>, <code>false_positive</code>)</td></tr>
    <tr><td>GET</td><td>/alerts/summary</td><td>Aggregated alert counts by severity / site / device class</td></tr>
  </tbody>
</table>

<h2>Compliance</h2>
<table>
  <tbody>
    <tr><th>Method</th><th>Endpoint</th><th>Description</th></tr>
    <tr><td>GET</td><td>/compliance/posture</td><td>Tenant-wide compliance posture summary per framework</td></tr>
    <tr><td>GET</td><td>/compliance/posture/{device_id}</td><td>Per-device compliance detail (NIST CSF, IEC 62443, etc.)</td></tr>
    <tr><td>POST</td><td>/compliance/report</td><td>Generate downloadable PDF compliance report</td></tr>
    <tr><td>GET</td><td>/compliance/policies</td><td>List active compliance policies for tenant</td></tr>
    <tr><td>POST</td><td>/compliance/policies</td><td>Create custom compliance policy rule</td></tr>
  </tbody>
</table>

<h2>Example: List Critical Alerts</h2>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="language">bash</ac:parameter>
  <ac:plain-text-body><![CDATA[curl -s -X GET "https://api.smarthub.ai/v2/alerts?severity=critical&status=open&limit=10" \
  -H "Authorization: Bearer $INFER_JWT" \
  -H "Accept: application/json" | jq '.results[] | {id, device_id, title, created_at}']]></ac:plain-text-body>
</ac:structured-macro>

<h2>Rate Limits</h2>
<table>
  <tbody>
    <tr><th>Tier</th><th>Requests / Minute</th><th>Burst</th></tr>
    <tr><td>Free / Trial</td><td>60</td><td>10</td></tr>
    <tr><td>Standard</td><td>600</td><td>100</td></tr>
    <tr><td>Enterprise</td><td>6,000</td><td>1,000</td></tr>
  </tbody>
</table>
""",
))

# 7 — Device Onboarding Guide
PAGES.append((
    "Device Onboarding Guide",
    """
<h1>Device Onboarding Guide</h1>
<p>This guide explains how to onboard devices into INFER™ — from initial discovery through full policy coverage. Three onboarding paths are supported: Auto-Discovery, Bulk Import, and Manual Registration.</p>

<h2>Prerequisites</h2>
<ul>
  <li>SD-EDGE Gateway deployed and connected to INFER SaaS (green status in <strong>Sites &gt; Gateways</strong>)</li>
  <li>Network VLAN(s) accessible from the SD-EDGE Gateway (SNMP read community string or credentials configured)</li>
  <li>INFER role: <code>Device Manager</code> or higher</li>
</ul>

<h2>Path 1: Auto-Discovery (Recommended)</h2>
<ol>
  <li>Navigate to <strong>Inventory &gt; Discover</strong></li>
  <li>Select the SD-EDGE Gateway for the target site</li>
  <li>Enter subnet CIDR(s) to scan (e.g., <code>10.10.5.0/24</code>)</li>
  <li>Choose scan profile: <em>Passive Only</em> (low-impact), <em>Active SNMP</em>, or <em>Deep Scan</em> (includes port scan)</li>
  <li>Click <strong>Start Discovery</strong>. Progress visible in the Jobs panel.</li>
  <li>Discovered devices appear in <strong>Inventory &gt; Pending Review</strong> for approval</li>
  <li>Review fingerprinted device class and metadata; click <strong>Approve &amp; Onboard</strong></li>
</ol>

<h2>Path 2: Bulk Import (CSV)</h2>
<p>Use when you have an existing spreadsheet inventory.</p>
<table>
  <tbody>
    <tr><th>Column</th><th>Required</th><th>Example</th></tr>
    <tr><td>ip_address</td><td>Yes</td><td>10.10.5.42</td></tr>
    <tr><td>mac_address</td><td>No</td><td>AA:BB:CC:DD:EE:FF</td></tr>
    <tr><td>hostname</td><td>No</td><td>cam-lobby-01</td></tr>
    <tr><td>device_class</td><td>No (auto-detected if blank)</td><td>ip_camera</td></tr>
    <tr><td>manufacturer</td><td>No</td><td>Axis</td></tr>
    <tr><td>model</td><td>No</td><td>P3245-V</td></tr>
    <tr><td>site_id</td><td>Yes</td><td>site_hq_sf</td></tr>
    <tr><td>group_tags</td><td>No (pipe-separated)</td><td>security|floor-1|critical</td></tr>
  </tbody>
</table>
<p>Upload via <strong>Inventory &gt; Bulk Import</strong> or API: <code>POST /v2/devices/bulk-import</code>. Maximum 10,000 rows per file.</p>

<h2>Path 3: Manual Registration</h2>
<p>For isolated or air-gapped devices not reachable via SD-EDGE Gateway.</p>
<ol>
  <li>Go to <strong>Inventory &gt; Add Device</strong></li>
  <li>Enter IP address, MAC, hostname, device class, and site</li>
  <li>Optionally upload a device certificate for mutual TLS</li>
  <li>Assign to a policy group</li>
  <li>Click <strong>Register</strong></li>
</ol>

<h2>Post-Onboarding Checklist</h2>
<table>
  <tbody>
    <tr><th>Step</th><th>Action</th><th>Where</th></tr>
    <tr><td>1</td><td>Assign device to a Policy Group</td><td>Inventory &gt; Device &gt; Policies tab</td></tr>
    <tr><td>2</td><td>Verify baseline telemetry is flowing</td><td>Inventory &gt; Device &gt; Telemetry tab</td></tr>
    <tr><td>3</td><td>Check initial SPS score (target &gt;70 within 24h)</td><td>Inventory &gt; Device &gt; Security tab</td></tr>
    <tr><td>4</td><td>Review and resolve any auto-generated alerts</td><td>Alerts console</td></tr>
    <tr><td>5</td><td>Configure alert notifications (email, Slack, SIEM)</td><td>Settings &gt; Notifications</td></tr>
  </tbody>
</table>
""",
))

# 8 — Deployment & Installation Guide
PAGES.append((
    "Deployment & Installation Guide — SD-EDGE Gateway",
    """
<h1>Deployment &amp; Installation Guide — SD-EDGE Gateway</h1>
<p>The SD-EDGE Gateway is the edge component that connects your on-premises devices to the INFER™ cloud. This guide covers the three supported deployment methods.</p>

<h2>System Requirements</h2>
<table>
  <tbody>
    <tr><th>Component</th><th>Minimum</th><th>Recommended</th></tr>
    <tr><td>CPU</td><td>2 vCPU (x86_64)</td><td>4 vCPU</td></tr>
    <tr><td>RAM</td><td>4 GB</td><td>8 GB</td></tr>
    <tr><td>Disk</td><td>20 GB</td><td>50 GB SSD</td></tr>
    <tr><td>OS</td><td>Ubuntu 20.04 LTS / RHEL 8</td><td>Ubuntu 22.04 LTS</td></tr>
    <tr><td>Network</td><td>1 Gbps NIC with routed access to device VLANs</td><td>2 x 1 Gbps (bonded)</td></tr>
    <tr><td>Outbound HTTPS</td><td>Port 443 to <code>gateway.smarthub.ai</code></td><td>Same</td></tr>
  </tbody>
</table>

<h2>Method 1: Docker Compose (Recommended for PoC)</h2>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="language">bash</ac:parameter>
  <ac:plain-text-body><![CDATA[# 1. Install Docker Engine 24+ and Docker Compose v2
curl -fsSL https://get.docker.com | sh

# 2. Download the SD-EDGE compose bundle
curl -O https://releases.smarthub.ai/sdedge/latest/docker-compose.yml
curl -O https://releases.smarthub.ai/sdedge/latest/.env.template
cp .env.template .env

# 3. Configure: set INFER_TENANT_ID and INFER_GATEWAY_TOKEN in .env
nano .env

# 4. Start the gateway
docker compose up -d

# 5. Verify connectivity
docker compose logs -f sdedge-gateway | grep "Connected to INFER cloud"]]></ac:plain-text-body>
</ac:structured-macro>

<h2>Method 2: Kubernetes (Helm Chart)</h2>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="language">bash</ac:parameter>
  <ac:plain-text-body><![CDATA[# Add SmartHub Helm repo
helm repo add smarthub https://charts.smarthub.ai
helm repo update

# Install SD-EDGE Gateway into dedicated namespace
helm install sdedge smarthub/sdedge-gateway \
  --namespace sdedge-system --create-namespace \
  --set gateway.tenantId=YOUR_TENANT_ID \
  --set gateway.token=YOUR_GATEWAY_TOKEN \
  --set gateway.site=YOUR_SITE_ID \
  --set resources.requests.memory=2Gi \
  --set resources.limits.memory=4Gi]]></ac:plain-text-body>
</ac:structured-macro>

<h2>Method 3: OVA (VMware ESXi / KVM)</h2>
<ol>
  <li>Download the OVA from <code>https://releases.smarthub.ai/sdedge/latest/sdedge-gateway.ova</code></li>
  <li>Deploy via vSphere Client: <strong>File &gt; Deploy OVF Template</strong></li>
  <li>Allocate minimum 4 vCPU, 8 GB RAM, 50 GB disk</li>
  <li>Set network adapter to a trunk port with access to device VLANs</li>
  <li>Power on; the first-boot wizard will prompt for Tenant ID and Gateway Token</li>
</ol>

<h2>Post-Deployment Verification</h2>
<table>
  <tbody>
    <tr><th>Check</th><th>Expected Result</th></tr>
    <tr><td>INFER Portal &gt; Sites &gt; Gateways</td><td>Gateway appears with status <strong>Connected</strong> (green) within 5 minutes</td></tr>
    <tr><td>Gateway health API</td><td><code>curl http://&lt;gateway-ip&gt;:8080/health</code> returns <code>{"status":"ok"}</code></td></tr>
    <tr><td>Outbound connectivity</td><td><code>curl -I https://gateway.smarthub.ai/ping</code> returns HTTP 200</td></tr>
  </tbody>
</table>

<h2>Upgrading the Gateway</h2>
<p>Gateway updates are pushed automatically when <strong>Auto-Update</strong> is enabled (default). To manually trigger:</p>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="language">bash</ac:parameter>
  <ac:plain-text-body><![CDATA[# Docker Compose
docker compose pull && docker compose up -d

# Kubernetes
helm upgrade sdedge smarthub/sdedge-gateway --reuse-values]]></ac:plain-text-body>
</ac:structured-macro>
""",
))

# 9 — Release Notes v2.5
PAGES.append((
    "Release Notes — INFER™ v2.5",
    """
<h1>Release Notes — INFER™ v2.5</h1>
<p><strong>Release Date:</strong> 2024-11-15 | <strong>Type:</strong> Major Feature Release</p>

<h2>Highlights</h2>
<ul>
  <li>Gen-AI Natural Language Query (NLQ) for device inventory — ask questions in plain English</li>
  <li>Microsoft Sentinel native connector (no Syslog forwarder required)</li>
  <li>SD-EDGE Gateway v2.5 with BACnet/IP and DNP3 protocol support</li>
  <li>Bulk OTA firmware campaigns with staged rollout and automatic rollback</li>
  <li>New NIST CSF 2.0 compliance dashboard</li>
</ul>

<h2>New Features</h2>
<table>
  <tbody>
    <tr><th>Feature</th><th>Component</th><th>Description</th></tr>
    <tr><td>Gen-AI NLQ</td><td>INFER Discover</td><td>Ask "Show me all cameras with firmware older than 6 months in Building 3" — results in &lt;3s powered by RAG over your device inventory</td></tr>
    <tr><td>Sentinel Connector</td><td>INFER Secure</td><td>Push IoT alerts directly to Microsoft Sentinel workspace via Azure Monitor REST API. No Syslog bridge needed.</td></tr>
    <tr><td>BACnet/IP Support</td><td>SD-EDGE Gateway</td><td>Discover and monitor building automation controllers via BACnet/IP without a separate BMS integration</td></tr>
    <tr><td>DNP3 Support</td><td>SD-EDGE Gateway</td><td>Monitor electrical substation and water treatment RTUs using DNP3 Subset Level 2</td></tr>
    <tr><td>Staged OTA Campaigns</td><td>INFER Manage</td><td>Roll out firmware to 5% → 25% → 100% of a device group with automatic rollback on failure rate &gt;5%</td></tr>
    <tr><td>NIST CSF 2.0 Dashboard</td><td>INFER Comply</td><td>New compliance view mapped to all 6 NIST CSF 2.0 functions with drill-down to individual controls</td></tr>
    <tr><td>Device Risk Heatmap</td><td>Dashboard</td><td>Site-level heatmap visualizing device count vs. average SPS — instantly identify high-risk sites</td></tr>
  </tbody>
</table>

<h2>Bug Fixes</h2>
<table>
  <tbody>
    <tr><th>Issue ID</th><th>Severity</th><th>Description</th></tr>
    <tr><td>INF-2341</td><td>High</td><td>Fixed: SNMP v3 devices with AES-256 auth would fail fingerprinting and appear as "Unknown" class</td></tr>
    <tr><td>INF-2289</td><td>Medium</td><td>Fixed: Alert email notifications would not send if the device label contained special characters (&amp;, &lt;, &gt;)</td></tr>
    <tr><td>INF-2178</td><td>Medium</td><td>Fixed: CSV bulk import would silently skip rows with IPv6 addresses instead of reporting an error</td></tr>
    <tr><td>INF-2095</td><td>Low</td><td>Fixed: Compliance report PDF generation would time out for tenants with &gt;50,000 devices</td></tr>
  </tbody>
</table>

<h2>Breaking Changes</h2>
<ul>
  <li><strong>API v1 deprecated:</strong> <code>/v1/*</code> endpoints will return HTTP 410 as of 2025-05-15. Migrate to <code>/v2/*</code>. See the <a href="#">API Migration Guide</a>.</li>
  <li><strong>SD-EDGE Gateway &lt;2.3 EOL:</strong> Gateways running v2.2 or earlier will no longer connect after 2025-02-01. Upgrade via <code>helm upgrade</code> or the portal's auto-update flow.</li>
</ul>

<h2>Known Issues</h2>
<table>
  <tbody>
    <tr><th>Issue ID</th><th>Description</th><th>Workaround</th></tr>
    <tr><td>INF-2412</td><td>NLQ queries that reference more than 3 sites may return incomplete results</td><td>Filter by a single site until the fix ships in v2.5.1 (ETA: 2024-12-10)</td></tr>
    <tr><td>INF-2398</td><td>Sentinel Connector may drop events if Sentinel workspace is in "Free" tier with log cap exceeded</td><td>Upgrade Sentinel workspace to Pay-as-you-go or configure alert deduplication to reduce volume</td></tr>
  </tbody>
</table>
""",
))

# 10 — Troubleshooting Guide
PAGES.append((
    "Troubleshooting Guide — Common Issues",
    """
<h1>Troubleshooting Guide — Common Issues</h1>
<p>This guide covers the most frequently reported issues by customers and internal support. Last updated by Engineering — Nov 2024.</p>

<h2>SD-EDGE Gateway Issues</h2>
<table>
  <tbody>
    <tr><th>Symptom</th><th>Probable Cause</th><th>Resolution</th></tr>
    <tr><td>Gateway shows <strong>Disconnected</strong> in portal</td><td>Outbound HTTPS blocked by firewall</td><td>Allow <code>gateway.smarthub.ai:443</code> and <code>telemetry.smarthub.ai:443</code> outbound. Verify with <code>curl -I https://gateway.smarthub.ai/ping</code></td></tr>
    <tr><td>Gateway connects but no devices discovered</td><td>SNMP community string mismatch, or device VLANs not routed to gateway</td><td>Check <strong>Sites &gt; Gateway &gt; Diagnostics &gt; Run Network Reachability Test</strong>. Verify VLAN routing.</td></tr>
    <tr><td>High memory usage (&gt;90%) on gateway VM</td><td>Too many concurrent SNMP polls on large subnet</td><td>Reduce polling frequency in <strong>Settings &gt; Discovery Profiles</strong> from 60s to 300s. Add more RAM if subnet &gt;5,000 devices.</td></tr>
    <tr><td>BACnet devices not discovered</td><td>BACnet/IP broadcast not crossing subnet boundary</td><td>Configure BACnet Broadcast Management Device (BBMD) address in SD-EDGE gateway config: <code>sdedge.bacnet.bbmd_address=10.10.1.1</code></td></tr>
  </tbody>
</table>

<h2>Device Fingerprinting Issues</h2>
<table>
  <tbody>
    <tr><th>Symptom</th><th>Probable Cause</th><th>Resolution</th></tr>
    <tr><td>Device shows as <strong>Unknown</strong> class after 24h</td><td>Device uses non-standard OUI or proprietary protocol</td><td>Manually set device class in portal. Submit device details via <strong>Help &gt; Report Unknown Device</strong> so SmartHub can add it to the fingerprint library.</td></tr>
    <tr><td>Firmware version shows <strong>N/A</strong></td><td>Device does not expose firmware version via SNMP/HTTP</td><td>Enter firmware version manually or use INFER Manage to trigger a credentials-based config read (if device supports SSH/API)</td></tr>
    <tr><td>Duplicate device entries after discovery</td><td>Device has multiple IP addresses (multi-homed) or DHCP lease changed</td><td>Merge duplicates via <strong>Inventory &gt; Merge Devices</strong>. Enable MAC-based deduplication in tenant settings.</td></tr>
  </tbody>
</table>

<h2>Alert &amp; SIEM Issues</h2>
<table>
  <tbody>
    <tr><th>Symptom</th><th>Probable Cause</th><th>Resolution</th></tr>
    <tr><td>No alerts forwarded to Splunk</td><td>Splunk HEC token expired or index permissions changed</td><td>Rotate HEC token in Splunk. Update in INFER: <strong>Integrations &gt; Splunk &gt; Edit</strong></td></tr>
    <tr><td>Alert storm — thousands of low-severity alerts</td><td>New device onboarded without baseline established (baselines need 72h)</td><td>Suppress alerts for <code>baseline_learning</code> tagged devices for 72h. Adjust alert threshold in <strong>Policies &gt; Alert Tuning</strong></td></tr>
    <tr><td>Sentinel connector shows <strong>Error: 403</strong></td><td>Managed Identity or service principal missing <strong>Monitoring Metrics Publisher</strong> role on DCE</td><td>Grant <code>Monitoring Metrics Publisher</code> role to the INFER app registration on the target Data Collection Endpoint</td></tr>
  </tbody>
</table>

<h2>Escalation Path</h2>
<ol>
  <li><strong>Tier 1 (Self-serve):</strong> INFER portal → Help → Diagnostics; this guide</li>
  <li><strong>Tier 2 (Customer Success):</strong> <a href="mailto:support@smarthub.ai">support@smarthub.ai</a> — response within 4 business hours (Standard) / 1 hour (Enterprise)</li>
  <li><strong>Tier 3 (Engineering Escalation):</strong> Filed by CS; engineering on-call paged for P1 incidents</li>
</ol>

<h2>Useful Diagnostic Commands</h2>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="language">bash</ac:parameter>
  <ac:plain-text-body><![CDATA[# Gateway container logs (Docker)
docker compose logs --tail=200 sdedge-gateway

# Gateway pod logs (Kubernetes)
kubectl logs -n sdedge-system deploy/sdedge-gateway --tail=200

# Check outbound connectivity from gateway
curl -sv https://gateway.smarthub.ai/ping 2>&1 | grep -E "Connected|SSL|HTTP"

# SNMP test from gateway to a device
docker compose exec sdedge-gateway snmpwalk -v2c -c public 10.10.5.42 sysDescr]]></ac:plain-text-body>
</ac:structured-macro>
""",
))

# 11 — Pricing & Plans
PAGES.append((
    "Pricing & Plans",
    """
<h1>Pricing &amp; Plans</h1>
<p>SmartHub.ai INFER™ is priced per managed device per year. All plans include the SD-EDGE Gateway software at no extra cost. Enterprise pricing is custom — contact <a href="mailto:sales@smarthub.ai">sales@smarthub.ai</a>.</p>

<h2>Plan Comparison</h2>
<table>
  <tbody>
    <tr>
      <th>Feature</th>
      <th>Starter<br/><em>Free up to 50 devices</em></th>
      <th>Standard<br/><em>$8/device/year</em></th>
      <th>Professional<br/><em>$15/device/year</em></th>
      <th>Enterprise<br/><em>Custom</em></th>
    </tr>
    <tr><td>Device Limit</td><td>50</td><td>Up to 5,000</td><td>Up to 50,000</td><td>Unlimited</td></tr>
    <tr><td>SD-EDGE Gateways</td><td>1</td><td>5</td><td>25</td><td>Unlimited</td></tr>
    <tr><td>INFER Discover</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td></tr>
    <tr><td>INFER Manage (OTA)</td><td>Limited</td><td>Yes</td><td>Yes</td><td>Yes</td></tr>
    <tr><td>INFER Secure (Anomaly Detection)</td><td>No</td><td>Yes</td><td>Yes</td><td>Yes</td></tr>
    <tr><td>INFER Comply</td><td>No</td><td>NIST CSF only</td><td>All frameworks</td><td>All + custom policies</td></tr>
    <tr><td>INFER Predict (Predictive Maintenance)</td><td>No</td><td>No</td><td>Yes</td><td>Yes</td></tr>
    <tr><td>Gen-AI NLQ</td><td>No</td><td>No</td><td>Yes</td><td>Yes</td></tr>
    <tr><td>SIEM Integrations</td><td>No</td><td>Splunk, Sentinel</td><td>All connectors</td><td>All + custom webhooks</td></tr>
    <tr><td>API Access</td><td>Read-only</td><td>Full v2 API</td><td>Full v2 API</td><td>Full v2 API + bulk endpoints</td></tr>
    <tr><td>Data Retention</td><td>7 days</td><td>90 days</td><td>1 year</td><td>Configurable (up to 7 years)</td></tr>
    <tr><td>SLA</td><td>Best effort</td><td>99.5% / 4h support</td><td>99.9% / 1h support</td><td>99.95% / 15min support + TAM</td></tr>
    <tr><td>SSO (SAML/OIDC)</td><td>No</td><td>No</td><td>Yes</td><td>Yes</td></tr>
    <tr><td>Private Cloud / On-Prem</td><td>No</td><td>No</td><td>No</td><td>Yes</td></tr>
  </tbody>
</table>

<h2>Volume Discounts (Standard &amp; Professional)</h2>
<table>
  <tbody>
    <tr><th>Device Count</th><th>Discount</th></tr>
    <tr><td>1 – 999</td><td>List price</td></tr>
    <tr><td>1,000 – 4,999</td><td>10% off</td></tr>
    <tr><td>5,000 – 19,999</td><td>20% off</td></tr>
    <tr><td>20,000+</td><td>Contact sales for custom pricing</td></tr>
  </tbody>
</table>

<h2>Add-Ons</h2>
<table>
  <tbody>
    <tr><th>Add-On</th><th>Price</th><th>Description</th></tr>
    <tr><td>Extended Telemetry Retention</td><td>$0.50/device/year per extra year</td><td>Add up to 6 years beyond base plan retention</td></tr>
    <tr><td>Managed Threat Hunting</td><td>$5/device/year</td><td>SmartHub SOC analysts proactively hunt for threats in your environment</td></tr>
    <tr><td>Professional Services — Onboarding</td><td>$10,000 flat</td><td>SmartHub engineer leads deployment, discovery, and baseline tuning for your first site</td></tr>
    <tr><td>Training (Online)</td><td>$500/seat</td><td>8-hour self-paced INFER administrator certification course</td></tr>
  </tbody>
</table>
""",
))

# 12 — Customer Success Stories
PAGES.append((
    "Customer Success Stories",
    """
<h1>Customer Success Stories</h1>
<p>These case studies illustrate how SmartHub.ai customers have used INFER™ to improve security posture, reduce operational overhead, and achieve compliance. Names are anonymized unless explicit permission granted.</p>

<h2>Case Study 1: Global Smart City Operator — 80,000 IoT Devices Secured</h2>
<table>
  <tbody>
    <tr><th>Industry</th><td>Smart City / Municipal Government</td></tr>
    <tr><th>Location</th><td>Southeast Asia</td></tr>
    <tr><th>Device Count</th><td>82,000 (traffic cameras, environmental sensors, smart streetlights, parking meters)</td></tr>
    <tr><th>Challenge</th><td>Zero visibility into device health and security posture across 400+ city zones. Manual firmware updates taking 18 months per cycle.</td></tr>
    <tr><th>Solution</th><td>INFER™ Enterprise + 40 SD-EDGE Gateways + Splunk SIEM integration</td></tr>
    <tr><th>Results</th><td>
      <ul>
        <li>82,000 devices discovered and fingerprinted in 48 hours</li>
        <li>Firmware update cycle reduced from 18 months to 3 weeks using staged OTA campaigns</li>
        <li>14 zero-day exploitation attempts detected and blocked in first 6 months</li>
        <li>NIST CSF score improved from 42% to 78% in 90 days</li>
      </ul>
    </td></tr>
  </tbody>
</table>

<h2>Case Study 2: Top-5 US Hospital Network — HIPAA IoMT Compliance</h2>
<table>
  <tbody>
    <tr><th>Industry</th><td>Healthcare</td></tr>
    <tr><th>Location</th><td>US (47 hospitals, 12 states)</td></tr>
    <tr><th>Device Count</th><td>31,000 (infusion pumps, nurse call systems, HVAC, IP cameras, access control)</td></tr>
    <tr><th>Challenge</th><td>HIPAA audit found 6,400 IoMT devices with default credentials. No process to track firmware CVEs. Manual compliance reporting took 2 weeks per quarter.</td></tr>
    <tr><th>Solution</th><td>INFER™ Professional + INFER Comply (HIPAA module) + ServiceNow CMDB integration</td></tr>
    <tr><th>Results</th><td>
      <ul>
        <li>All 6,400 default-credential devices remediated in 30 days via automated policy push</li>
        <li>Quarterly compliance report generation reduced from 2 weeks to 4 hours</li>
        <li>23 medical IoT devices discovered that were not in any existing inventory system</li>
        <li>Zero HIPAA audit findings related to IoMT in subsequent annual audit</li>
      </ul>
    </td></tr>
  </tbody>
</table>

<h2>Case Study 3: International Energy Company — OT/ICS Security</h2>
<table>
  <tbody>
    <tr><th>Industry</th><td>Energy / Utilities</td></tr>
    <tr><th>Location</th><td>Europe (18 countries)</td></tr>
    <tr><th>Device Count</th><td>12,000 OT devices (PLCs, RTUs, HMIs, historian servers)</td></tr>
    <tr><th>Challenge</th><td>IEC 62443 certification required for grid operator license renewal. No existing tool could discover or assess OT protocols (Modbus, DNP3, IEC 61850).</td></tr>
    <tr><th>Solution</th><td>INFER™ Enterprise (Private Cloud deployment) + SD-EDGE v2.5 with DNP3/IEC 61850 support + Palo Alto Cortex XSOAR integration</td></tr>
    <tr><th>Results</th><td>
      <ul>
        <li>IEC 62443 SL-2 certification achieved 4 months ahead of schedule</li>
        <li>3 legacy PLCs identified running firmware with known RCE vulnerabilities (CVE-2023-29158) — patched before exploitation</li>
        <li>Mean time to detect (MTTD) for OT anomalies reduced from "unknown" to 8 minutes</li>
      </ul>
    </td></tr>
  </tbody>
</table>
""",
))

# 13 — SLA & Support Policy
PAGES.append((
    "SLA & Support Policy",
    """
<h1>SLA &amp; Support Policy</h1>
<p>This document defines SmartHub.ai's service level agreements (SLAs) and support procedures for INFER™ SaaS customers. Effective: January 1, 2024.</p>

<h2>Service Availability SLAs</h2>
<table>
  <tbody>
    <tr><th>Component</th><th>Standard</th><th>Professional</th><th>Enterprise</th></tr>
    <tr><td>INFER Core API</td><td>99.5%</td><td>99.9%</td><td>99.95%</td></tr>
    <tr><td>INFER Dashboard (Web UI)</td><td>99.5%</td><td>99.9%</td><td>99.95%</td></tr>
    <tr><td>Telemetry Ingestion Pipeline</td><td>99.0%</td><td>99.5%</td><td>99.9%</td></tr>
    <tr><td>Alert Notification Delivery</td><td>Best effort</td><td>99.0%</td><td>99.5%</td></tr>
  </tbody>
</table>
<p><em>Availability calculated monthly, excluding scheduled maintenance windows (communicated 72h in advance via status.smarthub.ai).</em></p>

<h2>Support Tiers</h2>
<table>
  <tbody>
    <tr><th>Severity</th><th>Definition</th><th>Standard Response</th><th>Professional Response</th><th>Enterprise Response</th></tr>
    <tr><td>P1 — Critical</td><td>Platform completely unavailable; active security breach in progress</td><td>4 hours</td><td>1 hour</td><td>15 minutes (24/7 pager)</td></tr>
    <tr><td>P2 — High</td><td>Major feature impaired; significant impact on security operations</td><td>8 business hours</td><td>4 business hours</td><td>1 business hour</td></tr>
    <tr><td>P3 — Medium</td><td>Minor feature impaired; workaround available</td><td>2 business days</td><td>1 business day</td><td>4 business hours</td></tr>
    <tr><td>P4 — Low</td><td>Feature request, general question, cosmetic issue</td><td>5 business days</td><td>3 business days</td><td>2 business days</td></tr>
  </tbody>
</table>

<h2>Support Channels</h2>
<table>
  <tbody>
    <tr><th>Channel</th><th>Standard</th><th>Professional</th><th>Enterprise</th></tr>
    <tr><td>In-app Help &amp; Diagnostics</td><td>Yes</td><td>Yes</td><td>Yes</td></tr>
    <tr><td>Knowledge Base</td><td>Yes</td><td>Yes</td><td>Yes</td></tr>
    <tr><td>Email (<a href="mailto:support@smarthub.ai">support@smarthub.ai</a>)</td><td>Yes (business hours)</td><td>Yes (business hours)</td><td>Yes (24/7)</td></tr>
    <tr><td>Slack Connect</td><td>No</td><td>No</td><td>Yes (dedicated channel)</td></tr>
    <tr><td>Phone</td><td>No</td><td>No</td><td>Yes (P1/P2 only)</td></tr>
    <tr><td>Technical Account Manager (TAM)</td><td>No</td><td>No</td><td>Yes (named TAM)</td></tr>
  </tbody>
</table>

<h2>Scheduled Maintenance</h2>
<ul>
  <li>Maintenance windows: Sundays 02:00–06:00 UTC</li>
  <li>Advanced notice: at least 72 hours for planned maintenance; 24 hours for urgent security patches</li>
  <li>Notification channels: <a href="https://status.smarthub.ai">status.smarthub.ai</a>, email to account admin, in-app banner</li>
  <li>Emergency security patches: may be deployed with 2-hour notice; downtime capped at 30 minutes</li>
</ul>

<h2>Incident Communication Process</h2>
<ol>
  <li>Incident detected (automated monitoring or customer report)</li>
  <li>Status page updated within 15 minutes of confirmed incident</li>
  <li>Email notification to all affected tenant admins</li>
  <li>Updates posted every 30 minutes during active P1 incidents</li>
  <li>Post-incident report (RCA) published within 5 business days of resolution</li>
</ol>

<h2>SLA Credit Policy</h2>
<p>If monthly availability falls below the contracted SLA, customers are eligible for service credits:</p>
<table>
  <tbody>
    <tr><th>Availability Achieved</th><th>Credit (% of monthly fee)</th></tr>
    <tr><td>Below SLA but &gt;99.0%</td><td>10%</td></tr>
    <tr><td>98.0% – 99.0%</td><td>25%</td></tr>
    <tr><td>95.0% – 97.9%</td><td>50%</td></tr>
    <tr><td>Below 95.0%</td><td>100%</td></tr>
  </tbody>
</table>
""",
))

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Creating {len(PAGES)} pages in space '{SPACE}' on {DOMAIN}\n")
    created = []
    for i, (title, html) in enumerate(PAGES, 1):
        print(f"[{i}/{len(PAGES)}] Creating: {title} ... ", end="", flush=True)
        try:
            result = create_page(title, html)
            print(f"OK  (id={result['id']})")
            print(f"        {result['url']}")
            created.append((title, result))
        except Exception as exc:
            print(f"FAILED — {exc}")
        if i < len(PAGES):
            time.sleep(0.5)

    print(f"\n{'─'*60}")
    print(f"Done. Created {len(created)}/{len(PAGES)} pages.")
    if created:
        print("\nPages created:")
        for title, result in created:
            print(f"  • {title}")
            print(f"    {result['url']}")


if __name__ == "__main__":
    main()
