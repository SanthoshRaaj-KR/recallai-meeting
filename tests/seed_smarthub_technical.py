"""
Seed script: creates 13 deep-technical SmartHub.ai pages in Confluence.

Run with:
    conda run -n meetagents python tests/seed_smarthub_technical.py
"""

import os
import time
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth

ROOT = Path(__file__).resolve().parent.parent
env = {}
for line in (ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, _, v = line.partition("=")
    env.setdefault(k.strip(), v.strip())

EMAIL  = env["ATLASSIAN_USER_EMAIL"]
TOKEN  = env["ATLASSIAN_API_TOKEN"]
DOMAIN = env["ATLASSIAN_DOMAIN"]
SPACE  = env["ATLASSIAN_SPACE_KEY"]

BASE_URL = f"https://{DOMAIN}/wiki/rest/api"
AUTH     = HTTPBasicAuth(EMAIL, TOKEN)
HEADERS  = {"Content-Type": "application/json", "Accept": "application/json"}


def create_page(title: str, html: str) -> dict:
    payload = {
        "type": "page",
        "title": title,
        "space": {"key": SPACE},
        "body": {"storage": {"value": html, "representation": "storage"}},
    }
    resp = requests.post(f"{BASE_URL}/content", auth=AUTH, json=payload, headers=HEADERS, timeout=30)
    if not resp.ok:
        print(f"\n  ERROR {resp.status_code}: {resp.text[:400]}")
        resp.raise_for_status()
    data = resp.json()
    return {"id": data["id"], "url": f"https://{DOMAIN}/wiki" + data["_links"]["webui"]}


PAGES: list[tuple[str, str]] = []

# ── 1. Telemetry Schema & Event Catalog ───────────────────────────────────────
PAGES.append(("Telemetry Schema & Event Catalog", """
<h1>Telemetry Schema &amp; Event Catalog</h1>
<p>All telemetry emitted by managed devices flows through the INFER pipeline as canonical JSON events. This page is the authoritative reference for every event type, its fields, and Kafka topic routing.</p>

<h2>Canonical Envelope</h2>
<p>Every event — regardless of source protocol — is normalized into the following envelope before entering Kafka:</p>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">json</ac:parameter><ac:plain-text-body><![CDATA[{
  "event_id":    "uuid-v4",
  "tenant_id":   "tnt_acme_corp",
  "site_id":     "site_hq_sf",
  "gateway_id":  "gw_001",
  "device_id":   "dev_a1b2c3",
  "device_mac":  "AA:BB:CC:DD:EE:FF",
  "device_ip":   "10.10.5.42",
  "device_class":"ip_camera",
  "event_type":  "telemetry.metric",
  "schema_ver":  "2.1",
  "ts":          1717200000000,
  "payload":     { ... }
}]]></ac:plain-text-body></ac:structured-macro>

<h2>Event Type Registry</h2>
<table>
  <tbody>
    <tr><th>event_type</th><th>Kafka Topic</th><th>Frequency</th><th>Description</th></tr>
    <tr><td>telemetry.metric</td><td>device.telemetry.enriched</td><td>Every 60 s</td><td>CPU, memory, uptime, temperature, link state</td></tr>
    <tr><td>telemetry.netflow</td><td>device.netflow.raw</td><td>Every 30 s (aggregated)</td><td>Src/dst IP, port, bytes, packets, protocol — sampled at 1:100</td></tr>
    <tr><td>security.auth_attempt</td><td>device.security.events</td><td>On event</td><td>Login attempt: username, result (success/fail), source IP</td></tr>
    <tr><td>security.config_change</td><td>device.security.events</td><td>On event</td><td>Config diff, actor (user/automated), before/after hash</td></tr>
    <tr><td>security.port_scan</td><td>device.security.events</td><td>On detection</td><td>Scanner IP, scanned ports, scan pattern classification</td></tr>
    <tr><td>lifecycle.boot</td><td>device.lifecycle</td><td>On event</td><td>Device boot: firmware version, boot reason (cold/watchdog/ota)</td></tr>
    <tr><td>lifecycle.firmware_update</td><td>device.lifecycle</td><td>On event</td><td>OTA result: from_version, to_version, duration_s, success</td></tr>
    <tr><td>lifecycle.cert_expiry</td><td>device.lifecycle</td><td>Daily check</td><td>Certificate CN, expiry_ts, days_remaining</td></tr>
    <tr><td>anomaly.score</td><td>device.anomaly.scores</td><td>Every 5 min</td><td>ML anomaly score (0–1), contributing features, model version</td></tr>
    <tr><td>compliance.check</td><td>device.compliance</td><td>Every 15 min</td><td>Policy ID, control ID, result (pass/fail/warn), evidence</td></tr>
  </tbody>
</table>

<h2>Metric Payload Schema (telemetry.metric)</h2>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">json</ac:parameter><ac:plain-text-body><![CDATA[{
  "cpu_pct":        23.4,
  "mem_used_mb":    512,
  "mem_total_mb":   1024,
  "uptime_s":       864000,
  "temp_celsius":   48.2,
  "link_state":     "up",
  "link_speed_mbps":1000,
  "firmware_ver":   "9.80.2.6",
  "open_ports":     [22, 80, 443, 554],
  "active_sessions":3
}]]></ac:plain-text-body></ac:structured-macro>

<h2>Netflow Payload Schema (telemetry.netflow)</h2>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">json</ac:parameter><ac:plain-text-body><![CDATA[{
  "flows": [
    {
      "src_ip":    "10.10.5.42",
      "dst_ip":    "93.184.216.34",
      "src_port":  52341,
      "dst_port":  443,
      "protocol":  "TCP",
      "bytes":     18432,
      "packets":   24,
      "start_ts":  1717200000000,
      "end_ts":    1717200030000,
      "tcp_flags": "SYN,ACK,FIN"
    }
  ],
  "sample_rate": 100,
  "total_flows_observed": 2400
}]]></ac:plain-text-body></ac:structured-macro>

<h2>Kafka Topic Configuration</h2>
<table>
  <tbody>
    <tr><th>Topic</th><th>Partitions</th><th>Replication</th><th>Retention</th><th>Compaction</th></tr>
    <tr><td>device.telemetry.raw</td><td>64</td><td>3</td><td>24 h</td><td>No</td></tr>
    <tr><td>device.telemetry.enriched</td><td>64</td><td>3</td><td>7 days</td><td>No</td></tr>
    <tr><td>device.netflow.raw</td><td>128</td><td>3</td><td>3 days</td><td>No</td></tr>
    <tr><td>device.security.events</td><td>32</td><td>3</td><td>30 days</td><td>No</td></tr>
    <tr><td>device.anomaly.scores</td><td>32</td><td>3</td><td>90 days</td><td>Yes (by device_id)</td></tr>
    <tr><td>device.lifecycle</td><td>16</td><td>3</td><td>365 days</td><td>Yes (by device_id)</td></tr>
    <tr><td>device.compliance</td><td>32</td><td>3</td><td>365 days</td><td>No</td></tr>
    <tr><td>alert.outbound</td><td>16</td><td>3</td><td>90 days</td><td>No</td></tr>
  </tbody>
</table>

<h2>Schema Evolution Policy</h2>
<ul>
  <li>Schemas registered in <strong>Confluent Schema Registry</strong> using Avro. All topics enforce <code>FORWARD_TRANSITIVE</code> compatibility.</li>
  <li>Adding optional fields: allowed without version bump. Breaking changes require a new <code>schema_ver</code> and a 30-day dual-publish period.</li>
  <li>Consumers must tolerate unknown fields (ignore-unknown pattern).</li>
</ul>
"""))

# ── 2. ML Anomaly Detection — Model Architecture ──────────────────────────────
PAGES.append(("ML Anomaly Detection — Model Architecture", """
<h1>ML Anomaly Detection — Model Architecture</h1>
<p>INFER Secure uses a multi-stage ML pipeline to detect behavioral anomalies in IoT/OT device telemetry without requiring labelled attack data. This page covers the model architecture, feature engineering, training cadence, and serving infrastructure.</p>

<h2>Pipeline Overview</h2>
<table>
  <tbody>
    <tr><th>Stage</th><th>Model</th><th>Input</th><th>Output</th><th>Latency (p99)</th></tr>
    <tr><td>Feature Extraction</td><td>Rule-based + statistical</td><td>Raw telemetry stream</td><td>364-dim feature vector</td><td>8 ms</td></tr>
    <tr><td>Short-Term Anomaly</td><td>Isolation Forest (per device class)</td><td>5-min rolling window features</td><td>anomaly_score_short (0–1)</td><td>12 ms</td></tr>
    <tr><td>Long-Term Drift</td><td>Autoencoder (LSTM, 2-layer)</td><td>24-h sliding window</td><td>reconstruction_error</td><td>45 ms</td></tr>
    <tr><td>Threat Classifier</td><td>Gradient Boosted Trees (XGBoost)</td><td>Both anomaly scores + metadata</td><td>threat_type, confidence</td><td>6 ms</td></tr>
    <tr><td>Alert Aggregator</td><td>Rule engine (OPA)</td><td>Threat classifier output</td><td>Alert or suppress</td><td>3 ms</td></tr>
  </tbody>
</table>

<h2>Feature Engineering — 364 Dimensions</h2>
<table>
  <tbody>
    <tr><th>Feature Group</th><th>Count</th><th>Examples</th></tr>
    <tr><td>Network flow statistics</td><td>48</td><td>bytes_out_5m, unique_dst_ips_1h, dst_port_entropy, new_conn_rate</td></tr>
    <tr><td>Protocol behavior</td><td>64</td><td>http_methods_seen, tls_version_dist, dns_query_rate, snmp_oid_count</td></tr>
    <tr><td>Temporal patterns</td><td>72</td><td>hourly_traffic_profile (24 bins × 3 metrics), weekend_vs_weekday_ratio</td></tr>
    <tr><td>Device health</td><td>32</td><td>cpu_pct_delta, mem_growth_rate, session_count_zscore, temp_anomaly</td></tr>
    <tr><td>Auth &amp; access</td><td>24</td><td>failed_auth_rate, new_src_ip_ratio, privileged_cmd_count</td></tr>
    <tr><td>Peer comparison</td><td>56</td><td>zscore vs. same device_class (18 metrics × 3 aggregations: mean, p95, p99)</td></tr>
    <tr><td>Graph features</td><td>68</td><td>new_neighbors_1h, community_change, centrality_delta, unusual_port_pairs</td></tr>
  </tbody>
</table>

<h2>Isolation Forest Configuration</h2>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">python</ac:parameter><ac:plain-text-body><![CDATA[IsolationForest(
    n_estimators=200,
    max_samples=512,
    contamination=0.02,     # expected 2% anomaly rate in training data
    max_features=0.8,       # random subspace for diversity
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
)
# One model per device_class (22 classes).
# Retrained nightly on 30-day rolling window per tenant.
# Stored in MLflow with experiment tracking; promoted to "production" if
# precision@0.8recall >= current champion on held-out validation set.]]></ac:plain-text-body></ac:structured-macro>

<h2>LSTM Autoencoder Architecture</h2>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">python</ac:parameter><ac:plain-text-body><![CDATA[# Encoder
LSTM(128, return_sequences=True, dropout=0.2)
LSTM(64,  return_sequences=False, dropout=0.2)
Dense(32, activation='relu')          # bottleneck

# Decoder
RepeatVector(sequence_len=288)        # 24h at 5-min intervals
LSTM(64,  return_sequences=True, dropout=0.2)
LSTM(128, return_sequences=True, dropout=0.2)
TimeDistributed(Dense(n_features))

# Training
optimizer  = Adam(lr=1e-3, clipnorm=1.0)
loss       = MeanSquaredError()
epochs     = 50 (early stopping patience=5)
batch_size = 128
# Anomaly threshold = mean(val_reconstruction_error) + 3*std]]></ac:plain-text-body></ac:structured-macro>

<h2>Model Serving Infrastructure</h2>
<ul>
  <li><strong>Runtime:</strong> TorchServe 0.9 on AWS ECS Fargate (auto-scaling 2–20 replicas per tenant tier)</li>
  <li><strong>Model store:</strong> S3 + MLflow model registry; TorchServe pulls on startup via model-store URI</li>
  <li><strong>Warm-up:</strong> First 72 hours after device onboarding — anomaly scores suppressed, baseline being established</li>
  <li><strong>Drift detection:</strong> PSI (Population Stability Index) computed daily on feature distributions; PSI &gt; 0.2 triggers forced retraining</li>
  <li><strong>Explainability:</strong> SHAP values computed for every score &gt; 0.7; stored in InfluxDB and surfaced in the alert detail panel</li>
</ul>

<h2>Threat Classification Labels</h2>
<table>
  <tbody>
    <tr><th>Label</th><th>Description</th><th>Typical Signals</th></tr>
    <tr><td>c2_beaconing</td><td>Periodic outbound to C2 server</td><td>Regular interval dst, low byte variance, unusual dst ASN</td></tr>
    <tr><td>lateral_movement</td><td>Scanning/connecting to unexpected internal hosts</td><td>New dst IPs spike, ICMP sweep, unusual port access</td></tr>
    <tr><td>data_exfiltration</td><td>Unusual outbound data volume</td><td>bytes_out zscore &gt; 5, dst outside normal ASNs</td></tr>
    <tr><td>credential_stuffing</td><td>Repeated auth failures</td><td>failed_auth_rate &gt; 10/min, multiple src IPs</td></tr>
    <tr><td>firmware_tampering</td><td>Unexpected firmware hash change</td><td>lifecycle.boot with unknown firmware_ver hash</td></tr>
    <tr><td>dos_participation</td><td>Device is part of a botnet DDoS</td><td>pps spike, syn_flood pattern, single dst_ip</td></tr>
    <tr><td>unknown_anomaly</td><td>High reconstruction error, no label match</td><td>Autoencoder error &gt; 4σ, no XGBoost match &gt; 0.6</td></tr>
  </tbody>
</table>
"""))

# ── 3. Graph Database Schema (Neo4j) ─────────────────────────────────────────
PAGES.append(("Graph Database Schema — Neo4j Device Graph", """
<h1>Graph Database Schema — Neo4j Device Graph</h1>
<p>INFER uses Neo4j AuraDB as a graph store for device relationship modelling, network topology, and Graph RAG queries. This page documents node labels, relationship types, property schemas, and key Cypher query patterns.</p>

<h2>Node Labels</h2>
<table>
  <tbody>
    <tr><th>Label</th><th>Description</th><th>Key Properties</th></tr>
    <tr><td>Device</td><td>A managed IoT/OT endpoint</td><td>device_id, tenant_id, mac, ip, device_class, sps_score, firmware_ver, site_id, last_seen_ts</td></tr>
    <tr><td>Site</td><td>A physical or logical location</td><td>site_id, tenant_id, name, city, country, lat, lon, gateway_count</td></tr>
    <tr><td>Gateway</td><td>An SD-EDGE gateway instance</td><td>gateway_id, tenant_id, site_id, version, status, device_count</td></tr>
    <tr><td>Subnet</td><td>IP subnet managed by INFER</td><td>cidr, vlan_id, site_id, device_count, zone (IT/OT/DMZ)</td></tr>
    <tr><td>CVE</td><td>A known vulnerability</td><td>cve_id, cvss_score, cvss_vector, published_ts, affected_firmware_pattern</td></tr>
    <tr><td>PolicyGroup</td><td>A compliance/security policy group</td><td>group_id, name, framework, rule_count</td></tr>
    <tr><td>AlertEvent</td><td>A security alert (denormalised for graph traversal)</td><td>alert_id, threat_type, severity, ts, resolved</td></tr>
    <tr><td>Vendor</td><td>Device manufacturer</td><td>vendor_id, name, oui_prefix, support_url</td></tr>
  </tbody>
</table>

<h2>Relationship Types</h2>
<table>
  <tbody>
    <tr><th>Relationship</th><th>From → To</th><th>Properties</th></tr>
    <tr><td>LOCATED_AT</td><td>Device → Site</td><td>since_ts</td></tr>
    <tr><td>MANAGED_BY</td><td>Device → Gateway</td><td>first_seen_ts, protocol</td></tr>
    <tr><td>IN_SUBNET</td><td>Device → Subnet</td><td>assigned_ts, dhcp (bool)</td></tr>
    <tr><td>COMMUNICATES_WITH</td><td>Device → Device</td><td>bytes_7d, sessions_7d, last_seen_ts, protocol_list</td></tr>
    <tr><td>EXPOSED_TO</td><td>Device → CVE</td><td>detected_ts, remediation_status</td></tr>
    <tr><td>MEMBER_OF</td><td>Device → PolicyGroup</td><td>enrolled_ts, compliance_pct</td></tr>
    <tr><td>TRIGGERED</td><td>Device → AlertEvent</td><td>(none; edge existence is the fact)</td></tr>
    <tr><td>MANUFACTURED_BY</td><td>Device → Vendor</td><td>(none)</td></tr>
    <tr><td>CONTAINS</td><td>Site → Subnet</td><td>(none)</td></tr>
    <tr><td>HOSTS</td><td>Site → Gateway</td><td>(none)</td></tr>
  </tbody>
</table>

<h2>Index &amp; Constraint Definitions</h2>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">cypher</ac:parameter><ac:plain-text-body><![CDATA[// Uniqueness constraints
CREATE CONSTRAINT device_id_unique FOR (d:Device) REQUIRE (d.device_id, d.tenant_id) IS UNIQUE;
CREATE CONSTRAINT site_id_unique   FOR (s:Site)   REQUIRE (s.site_id,   s.tenant_id) IS UNIQUE;
CREATE CONSTRAINT cve_id_unique    FOR (c:CVE)    REQUIRE c.cve_id IS UNIQUE;

// Full-text index for NLQ (used by Gen-AI adapter)
CREATE FULLTEXT INDEX device_search FOR (n:Device) ON EACH [n.hostname, n.device_class, n.ip];

// Range index for time-based queries
CREATE INDEX device_last_seen FOR (d:Device) ON (d.last_seen_ts);
CREATE INDEX alert_ts         FOR (a:AlertEvent) ON (a.ts);

// Composite index for tenant-scoped queries
CREATE INDEX device_tenant_class FOR (d:Device) ON (d.tenant_id, d.device_class);]]></ac:plain-text-body></ac:structured-macro>

<h2>Key Query Patterns</h2>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">cypher</ac:parameter><ac:plain-text-body><![CDATA[// 1. Find all devices exposed to a critical CVE within a tenant
MATCH (d:Device {tenant_id: $tid})-[:EXPOSED_TO]->(c:CVE)
WHERE c.cvss_score >= 9.0 AND d.sps_score < 50
RETURN d.device_id, d.ip, d.device_class, c.cve_id, c.cvss_score
ORDER BY c.cvss_score DESC;

// 2. Lateral movement risk: devices talking to >10 unique peers in last 7 days
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
ORDER BY avg_sps ASC LIMIT 10;]]></ac:plain-text-body></ac:structured-macro>

<h2>Graph Refresh Cadence</h2>
<table>
  <tbody>
    <tr><th>Data</th><th>Update Frequency</th><th>Source</th></tr>
    <tr><td>Device properties (SPS, firmware, IP)</td><td>Every 5 min</td><td>Kafka consumer — device.telemetry.enriched</td></tr>
    <tr><td>COMMUNICATES_WITH edges</td><td>Hourly (rolling 7-day window)</td><td>Aggregated from device.netflow.raw</td></tr>
    <tr><td>EXPOSED_TO (CVE) edges</td><td>Daily at 02:00 UTC</td><td>NVD feed + firmware fingerprint matching</td></tr>
    <tr><td>AlertEvent nodes</td><td>On alert creation</td><td>alert.outbound Kafka topic</td></tr>
    <tr><td>Full graph reindex</td><td>Weekly (Sunday 03:00 UTC)</td><td>Batch reconciliation job</td></tr>
  </tbody>
</table>
"""))

# ── 4. Authentication & Authorization Deep Dive ───────────────────────────────
PAGES.append(("Authentication & Authorization — Technical Deep Dive", """
<h1>Authentication &amp; Authorization — Technical Deep Dive</h1>
<p>INFER uses a layered auth model: Auth0 for identity brokering, short-lived JWTs for API access, Open Policy Agent (OPA) for fine-grained authorization, and per-device mTLS for gateway-to-cloud channels.</p>

<h2>JWT Token Anatomy</h2>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">json</ac:parameter><ac:plain-text-body><![CDATA[// Header
{ "alg": "RS256", "kid": "signing-key-2024-03", "typ": "JWT" }

// Claims
{
  "iss": "https://auth.smarthub.ai",
  "aud": "https://api.smarthub.ai/v2",
  "sub": "user_7f3a9d",
  "tenant_id": "tnt_acme_corp",
  "tenant_tier": "enterprise",
  "roles": ["device_manager", "alert_viewer"],
  "sites": ["site_hq_sf", "site_nyc_01"],   // null = all sites
  "device_classes": null,                    // null = all classes
  "iat": 1717200000,
  "exp": 1717203600,     // 1-hour lifetime
  "jti": "uuid-v4"       // used for token revocation checks
}]]></ac:plain-text-body></ac:structured-macro>

<h2>RBAC Role Definitions</h2>
<table>
  <tbody>
    <tr><th>Role</th><th>Permissions</th><th>Typical Assignee</th></tr>
    <tr><td>org_admin</td><td>All permissions + user management + billing</td><td>IT Director / CISO</td></tr>
    <tr><td>security_analyst</td><td>Read all; acknowledge/resolve alerts; quarantine devices; generate reports</td><td>SOC Analyst</td></tr>
    <tr><td>device_manager</td><td>Read all; onboard/decommission devices; trigger OTA; edit device metadata</td><td>IT Admin / NOC Engineer</td></tr>
    <tr><td>compliance_auditor</td><td>Read-only: devices, compliance, reports; export PDF</td><td>Internal Auditor / GRC</td></tr>
    <tr><td>alert_viewer</td><td>Read-only: alerts, device health, dashboards</td><td>Help Desk / L1 Support</td></tr>
    <tr><td>api_integration</td><td>Scoped API access per integration; no UI login</td><td>SIEM / SOAR service account</td></tr>
  </tbody>
</table>

<h2>OPA Policy Example — Device Quarantine</h2>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">rego</ac:parameter><ac:plain-text-body><![CDATA[package infer.authz

# A user may quarantine a device if:
# 1. They have the security_analyst or org_admin role
# 2. The device's site_id is in their permitted sites (or sites claim is null)
# 3. The tenant_id on the device matches the token's tenant_id

default allow_quarantine = false

allow_quarantine {
    input.action == "device:quarantine"
    has_role({"security_analyst", "org_admin"})
    site_permitted(input.resource.site_id)
    input.token.tenant_id == input.resource.tenant_id
}

has_role(roles) {
    r := input.token.roles[_]
    roles[r]
}

site_permitted(site_id) {
    input.token.sites == null    # null means all sites
}

site_permitted(site_id) {
    input.token.sites[_] == site_id
}]]></ac:plain-text-body></ac:structured-macro>

<h2>mTLS: Gateway-to-Cloud Channel</h2>
<p>Every SD-EDGE Gateway authenticates to the INFER cloud using mutual TLS with a per-gateway X.509 certificate issued by SmartHub's private CA (Mocana CMS-backed).</p>
<table>
  <tbody>
    <tr><th>Parameter</th><th>Value</th></tr>
    <tr><td>CA hierarchy</td><td>Root CA (offline HSM) → Intermediate CA → Gateway leaf cert</td></tr>
    <tr><td>Leaf cert lifetime</td><td>365 days; auto-rotated 30 days before expiry via ACME-like renewal</td></tr>
    <tr><td>Key algorithm</td><td>ECDSA P-256</td></tr>
    <tr><td>TLS version</td><td>TLS 1.3 only; TLS 1.2 disabled</td></tr>
    <tr><td>Cipher suites</td><td>TLS_AES_256_GCM_SHA384, TLS_CHACHA20_POLY1305_SHA256</td></tr>
    <tr><td>Certificate pinning</td><td>Gateway pins intermediate CA SPKI hash; rejects if mismatch</td></tr>
    <tr><td>Revocation</td><td>OCSP Stapling; gateway checks every 4 hours</td></tr>
  </tbody>
</table>

<h2>SAML 2.0 SSO Flow</h2>
<ol>
  <li>User navigates to <code>https://app.smarthub.ai</code></li>
  <li>INFER redirects to Auth0 with <code>connection=saml-&lt;tenant_slug&gt;</code></li>
  <li>Auth0 sends SAML AuthnRequest to customer IdP (Okta / Azure AD / Ping)</li>
  <li>IdP authenticates user; returns signed SAML assertion</li>
  <li>Auth0 validates assertion; maps IdP group claims to INFER roles via attribute mapping config</li>
  <li>Auth0 issues INFER JWT (RS256, 1-hour TTL) + refresh token (rotating, 7-day TTL)</li>
  <li>Frontend stores JWT in memory only (no localStorage); refresh token in httpOnly cookie</li>
</ol>

<h2>Token Revocation</h2>
<p>Token revocation is maintained in Redis (cluster mode, 3 shards). On logout or account suspension:</p>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">python</ac:parameter><ac:plain-text-body><![CDATA[# Revoke a specific JTI
redis.setex(f"revoked_jti:{jti}", ttl=3600, value="1")

# Check at API Gateway Lambda Authorizer (adds ~0.8ms p99)
if redis.exists(f"revoked_jti:{token_claims['jti']}"):
    raise Unauthorized("Token has been revoked")

# Tenant-wide revocation (e.g., on breach response)
# Set revoked_before timestamp; all tokens issued before it are invalid
redis.set(f"tenant_revoked_before:{tenant_id}", value=int(time.time()))]]></ac:plain-text-body></ac:structured-macro>
"""))

# ── 5. OTA Firmware Update Pipeline ──────────────────────────────────────────
PAGES.append(("OTA Firmware Update Pipeline — Technical Spec", """
<h1>OTA Firmware Update Pipeline — Technical Spec</h1>
<p>INFER Manage supports secure over-the-air firmware updates for 2,000+ device types. This page covers the full pipeline from campaign creation through delivery, verification, and rollback.</p>

<h2>Update Campaign State Machine</h2>
<table>
  <tbody>
    <tr><th>State</th><th>Description</th><th>Transitions</th></tr>
    <tr><td>DRAFT</td><td>Campaign created, not yet started</td><td>→ STAGED (on start) | → CANCELLED</td></tr>
    <tr><td>STAGED</td><td>Firmware package uploaded and validated; ring 0 (5%) selected</td><td>→ RING_0_ACTIVE</td></tr>
    <tr><td>RING_0_ACTIVE</td><td>Deploying to 5% of target devices</td><td>→ RING_0_BAKING | → ROLLING_BACK (failure &gt; 5%)</td></tr>
    <tr><td>RING_0_BAKING</td><td>Observing ring 0 for 2 hours post-update</td><td>→ RING_1_ACTIVE | → ROLLING_BACK</td></tr>
    <tr><td>RING_1_ACTIVE</td><td>Deploying to 25% of remaining devices</td><td>→ RING_1_BAKING | → ROLLING_BACK</td></tr>
    <tr><td>RING_1_BAKING</td><td>Bake period: 4 hours</td><td>→ FULL_ROLLOUT | → ROLLING_BACK</td></tr>
    <tr><td>FULL_ROLLOUT</td><td>Deploying to 100% of remaining devices</td><td>→ COMPLETED | → PARTIAL_FAILURE</td></tr>
    <tr><td>COMPLETED</td><td>All devices updated successfully</td><td>Terminal</td></tr>
    <tr><td>ROLLING_BACK</td><td>Reverting ring 0/1 devices to previous firmware</td><td>→ ROLLED_BACK | → ROLLBACK_FAILED</td></tr>
    <tr><td>PARTIAL_FAILURE</td><td>Full rollout complete but &gt;2% devices failed</td><td>→ COMPLETED (after manual override)</td></tr>
  </tbody>
</table>

<h2>Firmware Package Format</h2>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">bash</ac:parameter><ac:plain-text-body><![CDATA[# INFER firmware bundle structure (.ifb file — ZIP archive)
firmware_bundle.ifb
├── manifest.json          # package metadata + SHA-256 hashes
├── firmware.bin           # raw firmware binary
├── firmware.bin.sig       # ECDSA P-384 signature (SmartHub signing key)
├── delta/
│   └── patch_9.80.1.6_to_9.80.2.6.bsdiff  # binary delta for bandwidth savings
└── install_scripts/
    ├── pre_install.sh     # pre-checks (storage space, power state)
    ├── install.sh         # device-class-specific installer
    └── post_install.sh    # verification + health check]]></ac:plain-text-body></ac:structured-macro>

<h2>manifest.json Schema</h2>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">json</ac:parameter><ac:plain-text-body><![CDATA[{
  "package_id":     "pkg_axis_p3245_9.80.2.6",
  "vendor":         "Axis Communications",
  "model_pattern":  "P3245-.*",
  "firmware_ver":   "9.80.2.6",
  "min_from_ver":   "9.60.0.0",
  "size_bytes":     14680064,
  "delta_size_bytes":2097152,
  "sha256_full":    "e3b0c44298fc1c149afb...",
  "sha256_delta":   "a87ff679a2f3e71d9181...",
  "signature":      "MEUCIQD3...",
  "signing_key_id": "ota-signing-2024-01",
  "min_free_storage_mb": 32,
  "reboot_required": true,
  "downtime_estimate_s": 90,
  "rollback_supported": true,
  "release_notes_url": "https://releases.smarthub.ai/axis/p3245/9.80.2.6"
}]]></ac:plain-text-body></ac:structured-macro>

<h2>Delivery Protocol</h2>
<ol>
  <li>Campaign scheduler (Go service) selects target devices for current ring using consistent hashing on <code>device_id</code></li>
  <li>SD-EDGE Gateway receives update job via MQTT topic <code>infer/gw/{gateway_id}/ota/command</code></li>
  <li>Gateway downloads firmware package from pre-signed S3 URL (1-hour TTL) over HTTPS</li>
  <li>Gateway verifies: SHA-256 hash of downloaded file, then ECDSA signature against embedded public key</li>
  <li>Gateway pushes firmware to device using device-class-specific protocol (ONVIF firmware upgrade API, SSH SCP, TFTP, HTTP PUT)</li>
  <li>Device reboots; gateway polls for reconnection (timeout: 10 min)</li>
  <li>Gateway reports result to INFER cloud via MQTT <code>infer/gw/{gateway_id}/ota/result</code></li>
</ol>

<h2>Rollback Logic</h2>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">python</ac:parameter><ac:plain-text-body><![CDATA[ROLLBACK_THRESHOLD = 0.05   # 5% failure rate triggers auto-rollback

def evaluate_ring(campaign_id: str, ring: int) -> RingDecision:
    stats = get_ring_stats(campaign_id, ring)
    failure_rate = stats.failed / stats.attempted

    if failure_rate > ROLLBACK_THRESHOLD:
        logger.warning(
            "Ring %d failure rate %.1f%% exceeds threshold — triggering rollback",
            ring, failure_rate * 100,
        )
        return RingDecision.ROLLBACK

    # Also check health metrics post-update
    health = get_post_update_health(campaign_id, ring)
    if health.avg_sps_delta < -15:   # SPS dropped >15 points
        return RingDecision.ROLLBACK

    return RingDecision.PROCEED]]></ac:plain-text-body></ac:structured-macro>

<h2>Bandwidth Optimization</h2>
<table>
  <tbody>
    <tr><th>Technique</th><th>Saving</th><th>Applicability</th></tr>
    <tr><td>Binary delta patches (bsdiff)</td><td>60–85% reduction</td><td>When previous version is known and delta exists</td></tr>
    <tr><td>SD-EDGE Gateway P2P sharing</td><td>Eliminates redundant S3 downloads within a site</td><td>When multiple devices of same model at same site</td></tr>
    <tr><td>Scheduled maintenance windows</td><td>Avoids business-hours bandwidth impact</td><td>All campaigns</td></tr>
    <tr><td>Parallel update throttle</td><td>Max 10 concurrent updates per gateway</td><td>Prevents gateway CPU saturation</td></tr>
  </tbody>
</table>
"""))

# ── 6. Kafka & Stream Processing Architecture ─────────────────────────────────
PAGES.append(("Kafka & Stream Processing Architecture", """
<h1>Kafka &amp; Stream Processing Architecture</h1>
<p>The INFER telemetry backbone is built on Apache Kafka (MSK Serverless) with Kafka Streams for stateful processing and Flink for complex event processing. This page covers the full topology.</p>

<h2>Kafka Cluster Configuration</h2>
<table>
  <tbody>
    <tr><th>Parameter</th><th>Value</th><th>Rationale</th></tr>
    <tr><td>Kafka version</td><td>3.6.1</td><td>KRaft mode (no Zookeeper)</td></tr>
    <tr><td>Deployment</td><td>AWS MSK Serverless</td><td>Auto-scaling; no broker management</td></tr>
    <tr><td>Max throughput</td><td>1 GB/s ingress, 2 GB/s egress</td><td>MSK Serverless limits</td></tr>
    <tr><td>Message size limit</td><td>1 MB (default) / 10 MB (netflow topic)</td><td>Avoids broker memory pressure</td></tr>
    <tr><td>Compression</td><td>lz4 (producer-side)</td><td>~60% size reduction on JSON telemetry</td></tr>
    <tr><td>Producer acks</td><td>acks=all</td><td>No data loss on leader failure</td></tr>
    <tr><td>Consumer isolation</td><td>read_committed</td><td>Exactly-once semantics for compliance events</td></tr>
  </tbody>
</table>

<h2>Stream Processing Topology</h2>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">bash</ac:parameter><ac:plain-text-body><![CDATA[device.telemetry.raw
    │
    ▼ [Enrichment Streams App — Kafka Streams]
    │  - Join with device metadata (KTable from PostgreSQL CDC)
    │  - Add tenant_tier, site_name, device_class
    │  - Filter invalid events (schema validation)
    │
    ├──► device.telemetry.enriched         (main enriched stream)
    │
    ├──► device.telemetry.dlq             (schema violations / parse errors)
    │
    └──► device.telemetry.highrate        (devices emitting >100 events/min — throttled)

device.telemetry.enriched
    │
    ▼ [Feature Extraction — Flink job, 5-min tumbling windows]
    │  - Compute 364 feature dimensions per device
    │  - Windowed aggregations: sum, mean, p95, entropy
    │
    ├──► device.features.5m               (feature vectors for ML)
    │
    └──► device.stats.hourly              (rolled-up stats → InfluxDB sink)

device.features.5m
    │
    ▼ [Anomaly Scoring — TorchServe HTTP sink via Kafka Connect]
    │  - Batch inference: 500 vectors per request
    │
    └──► device.anomaly.scores

device.anomaly.scores + device.security.events
    │
    ▼ [Alert Engine — Flink CEP (Complex Event Processing)]
    │  - Correlate anomaly scores with auth events
    │  - Detect multi-stage attack patterns (e.g., port scan → auth fail → high anomaly)
    │
    └──► alert.outbound                   (fan-out to SIEM, PagerDuty, email)]]></ac:plain-text-body></ac:structured-macro>

<h2>Flink CEP Pattern — Multi-Stage Attack Detection</h2>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">java</ac:parameter><ac:plain-text-body><![CDATA[Pattern<TelemetryEvent, ?> attackPattern = Pattern
    .<TelemetryEvent>begin("reconnaissance")
        .where(e -> e.getEventType().equals("security.port_scan"))
    .followedByAny("credential_attack")
        .where(e -> e.getEventType().equals("security.auth_attempt")
                 && e.getPayload().getInt("failed_count") > 5)
        .within(Time.minutes(30))
    .followedByAny("anomaly_spike")
        .where(e -> e.getEventType().equals("anomaly.score")
                 && e.getPayload().getDouble("score") > 0.85)
        .within(Time.minutes(60));

// When pattern matches → emit HIGH severity alert with kill-chain evidence]]></ac:plain-text-body></ac:structured-macro>

<h2>Kafka Connect Sink Connectors</h2>
<table>
  <tbody>
    <tr><th>Sink</th><th>Topics Consumed</th><th>Connector</th><th>Batch Size</th></tr>
    <tr><td>InfluxDB Cloud</td><td>device.stats.hourly, device.anomaly.scores</td><td>Custom (HTTP sink)</td><td>5,000 points/req</td></tr>
    <tr><td>PostgreSQL (INFER DB)</td><td>device.lifecycle, alert.outbound</td><td>JDBC Sink Connector</td><td>1,000 rows/req</td></tr>
    <tr><td>Neo4j AuraDB</td><td>device.telemetry.enriched (subset)</td><td>Neo4j Kafka Connector</td><td>500 nodes/req</td></tr>
    <tr><td>S3 (data lake)</td><td>All topics</td><td>Confluent S3 Sink</td><td>128 MB Parquet files</td></tr>
    <tr><td>Elasticsearch</td><td>device.security.events, alert.outbound</td><td>Elasticsearch Sink 3.x</td><td>500 docs/req</td></tr>
  </tbody>
</table>

<h2>Consumer Group Lag SLOs</h2>
<table>
  <tbody>
    <tr><th>Consumer Group</th><th>Max Lag (records)</th><th>Alert Threshold</th></tr>
    <tr><td>enrichment-streams</td><td>10,000</td><td>50,000 → PagerDuty P2</td></tr>
    <tr><td>anomaly-scorer</td><td>50,000</td><td>200,000 → PagerDuty P1</td></tr>
    <tr><td>alert-engine-flink</td><td>5,000</td><td>20,000 → PagerDuty P1</td></tr>
    <tr><td>influxdb-sink</td><td>100,000</td><td>500,000 → PagerDuty P2</td></tr>
    <tr><td>s3-archive-sink</td><td>No SLO</td><td>10M → Slack warning</td></tr>
  </tbody>
</table>
"""))

# ── 7. PostgreSQL Database Schema ─────────────────────────────────────────────
PAGES.append(("PostgreSQL Database Schema — Core Tables", """
<h1>PostgreSQL Database Schema — Core Tables</h1>
<p>INFER uses PostgreSQL 15 (AWS RDS Multi-AZ) as the primary relational store. Tenant isolation uses a schema-per-tenant model. This page documents core tables in the <code>infer_shared</code> schema (cross-tenant) and the per-tenant schema.</p>

<h2>infer_shared — Cross-Tenant Tables</h2>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">sql</ac:parameter><ac:plain-text-body><![CDATA[-- Tenant registry
CREATE TABLE infer_shared.tenants (
    tenant_id        TEXT PRIMARY KEY,               -- e.g. tnt_acme_corp
    name             TEXT NOT NULL,
    tier             TEXT NOT NULL CHECK (tier IN ('starter','standard','professional','enterprise')),
    schema_name      TEXT NOT NULL UNIQUE,           -- e.g. tenant_acme_corp
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    suspended_at     TIMESTAMPTZ,
    max_devices      INT NOT NULL DEFAULT 50,
    max_gateways     INT NOT NULL DEFAULT 1,
    data_region      TEXT NOT NULL DEFAULT 'us-east-1',
    sso_connection   TEXT                            -- Auth0 connection ID
);

-- CVE master table (shared across tenants, NVD-sourced)
CREATE TABLE infer_shared.cve_catalog (
    cve_id           TEXT PRIMARY KEY,               -- e.g. CVE-2024-12345
    cvss_score       NUMERIC(3,1),
    cvss_vector      TEXT,
    description      TEXT,
    published_at     DATE,
    modified_at      DATE,
    affected_vendors TEXT[],                         -- ['axis','hikvision']
    firmware_patterns TEXT[],                        -- regex patterns for matching
    exploit_available BOOLEAN DEFAULT FALSE
);
CREATE INDEX ON infer_shared.cve_catalog (cvss_score DESC) WHERE cvss_score >= 7.0;]]></ac:plain-text-body></ac:structured-macro>

<h2>Per-Tenant Schema — Core Tables</h2>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">sql</ac:parameter><ac:plain-text-body><![CDATA[-- Devices (main inventory table)
CREATE TABLE {tenant_schema}.devices (
    device_id        TEXT PRIMARY KEY,
    mac_address      MACADDR,
    ip_address       INET,
    hostname         TEXT,
    device_class     TEXT NOT NULL,                  -- ip_camera, plc, switch, ...
    vendor_id        TEXT,
    model            TEXT,
    firmware_ver     TEXT,
    firmware_hash    TEXT,                           -- SHA-256 of firmware binary
    site_id          TEXT REFERENCES {tenant_schema}.sites(site_id),
    gateway_id       TEXT,
    sps_score        SMALLINT CHECK (sps_score BETWEEN 0 AND 100),
    status           TEXT DEFAULT 'active' CHECK (status IN ('active','quarantined','decommissioned')),
    first_seen_at    TIMESTAMPTZ DEFAULT NOW(),
    last_seen_at     TIMESTAMPTZ,
    onboarded_by     TEXT,                           -- user_id or 'auto-discovery'
    tags             TEXT[] DEFAULT '{}',
    metadata         JSONB DEFAULT '{}'
);
CREATE INDEX ON {tenant_schema}.devices (site_id);
CREATE INDEX ON {tenant_schema}.devices (device_class);
CREATE INDEX ON {tenant_schema}.devices (sps_score) WHERE status = 'active';
CREATE INDEX ON {tenant_schema}.devices USING GIN (tags);

-- Alerts
CREATE TABLE {tenant_schema}.alerts (
    alert_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id        TEXT REFERENCES {tenant_schema}.devices(device_id),
    site_id          TEXT,
    threat_type      TEXT NOT NULL,
    severity         TEXT NOT NULL CHECK (severity IN ('info','low','medium','high','critical')),
    title            TEXT NOT NULL,
    description      TEXT,
    evidence         JSONB,                          -- SHAP values, raw signals
    status           TEXT DEFAULT 'open' CHECK (status IN ('open','acknowledged','resolved','false_positive')),
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    acknowledged_at  TIMESTAMPTZ,
    resolved_at      TIMESTAMPTZ,
    resolved_by      TEXT,
    siem_forwarded   BOOLEAN DEFAULT FALSE,
    siem_event_id    TEXT
);
CREATE INDEX ON {tenant_schema}.alerts (device_id, created_at DESC);
CREATE INDEX ON {tenant_schema}.alerts (severity, status) WHERE status = 'open';

-- OTA campaigns
CREATE TABLE {tenant_schema}.ota_campaigns (
    campaign_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name             TEXT NOT NULL,
    package_id       TEXT NOT NULL,
    target_class     TEXT,
    target_site_ids  TEXT[],
    target_device_ids TEXT[],
    state            TEXT NOT NULL DEFAULT 'DRAFT',
    ring_0_pct       SMALLINT DEFAULT 5,
    ring_1_pct       SMALLINT DEFAULT 25,
    bake_duration_h  SMALLINT DEFAULT 2,
    created_by       TEXT,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    started_at       TIMESTAMPTZ,
    completed_at     TIMESTAMPTZ,
    stats            JSONB DEFAULT '{}'
);]]></ac:plain-text-body></ac:structured-macro>

<h2>Partitioning Strategy</h2>
<table>
  <tbody>
    <tr><th>Table</th><th>Partition By</th><th>Retention</th><th>Archive</th></tr>
    <tr><td>device_telemetry_hourly</td><td>RANGE on hour (monthly partitions)</td><td>13 months hot</td><td>S3 Parquet via pg_partman + custom archiver</td></tr>
    <tr><td>alerts</td><td>RANGE on created_at (quarterly)</td><td>3 years hot</td><td>S3 after 3 years</td></tr>
    <tr><td>compliance_checks</td><td>RANGE on checked_at (monthly)</td><td>1 year hot</td><td>S3 after 1 year</td></tr>
    <tr><td>audit_log</td><td>RANGE on ts (monthly)</td><td>7 years (immutable)</td><td>Glacier Deep Archive</td></tr>
  </tbody>
</table>

<h2>Connection Pooling</h2>
<p>PgBouncer runs as a sidecar in each API service pod, configured in transaction-mode pooling:</p>
<table>
  <tbody>
    <tr><th>Parameter</th><th>Value</th></tr>
    <tr><td>pool_mode</td><td>transaction</td></tr>
    <tr><td>max_client_conn</td><td>1000 per API replica</td></tr>
    <tr><td>default_pool_size</td><td>25</td></tr>
    <tr><td>reserve_pool_size</td><td>5</td></tr>
    <tr><td>server_idle_timeout</td><td>600 s</td></tr>
    <tr><td>RDS max_connections</td><td>5000 (db.r6g.4xlarge)</td></tr>
  </tbody>
</table>
"""))

# ── 8. SIEM Integration Architecture ─────────────────────────────────────────
PAGES.append(("SIEM Integration Architecture", """
<h1>SIEM Integration Architecture</h1>
<p>INFER Secure forwards enriched IoT threat events to customer SIEM platforms in real time. This page describes supported integrations, event formats, and the forwarding pipeline.</p>

<h2>Supported SIEM Connectors</h2>
<table>
  <tbody>
    <tr><th>SIEM</th><th>Protocol</th><th>Auth</th><th>Format</th><th>Latency (alert → SIEM)</th></tr>
    <tr><td>Splunk Enterprise / Cloud</td><td>HTTP Event Collector (HEC)</td><td>HEC token</td><td>JSON</td><td>&lt;30 s</td></tr>
    <tr><td>Microsoft Sentinel</td><td>Azure Monitor Logs Ingestion API (DCR)</td><td>Service Principal / Managed Identity</td><td>JSON (custom table schema)</td><td>&lt;60 s</td></tr>
    <tr><td>IBM QRadar</td><td>Syslog TLS (RFC 5424)</td><td>TLS client cert</td><td>CEF (Common Event Format)</td><td>&lt;15 s</td></tr>
    <tr><td>Elastic SIEM / Security</td><td>Elasticsearch Bulk API</td><td>API key</td><td>ECS (Elastic Common Schema)</td><td>&lt;30 s</td></tr>
    <tr><td>Sumo Logic</td><td>HTTP Source endpoint</td><td>URL token</td><td>JSON</td><td>&lt;30 s</td></tr>
    <tr><td>Chronicle (Google SecOps)</td><td>Ingestion API v2</td><td>Service Account JSON</td><td>UDM (Unified Data Model)</td><td>&lt;60 s</td></tr>
    <tr><td>Generic Webhook</td><td>HTTPS POST</td><td>Bearer / HMAC-SHA256 signature</td><td>JSON (INFER native)</td><td>&lt;15 s</td></tr>
  </tbody>
</table>

<h2>INFER Native Alert Schema → Splunk HEC Payload</h2>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">json</ac:parameter><ac:plain-text-body><![CDATA[{
  "time": 1717200000.000,
  "host": "infer-cloud",
  "source": "smarthub:infer:alerts",
  "sourcetype": "smarthub:iot:alert",
  "index": "iot_security",
  "event": {
    "alert_id":        "7f3a9d12-...",
    "tenant_id":       "tnt_acme_corp",
    "severity":        "high",
    "threat_type":     "lateral_movement",
    "title":           "IP Camera communicating with 47 new internal hosts",
    "device_id":       "dev_a1b2c3",
    "device_class":    "ip_camera",
    "device_ip":       "10.10.5.42",
    "device_mac":      "AA:BB:CC:DD:EE:FF",
    "device_firmware": "9.80.1.6",
    "site_id":         "site_hq_sf",
    "site_name":       "HQ San Francisco",
    "anomaly_score":   0.94,
    "sps_score":       31,
    "evidence": {
      "new_dst_ips_1h": 47,
      "bytes_out_delta_pct": 840,
      "top_shap_features": ["new_neighbors_1h", "bytes_out_zscore", "dst_port_entropy"]
    },
    "mitre_techniques": ["T1046", "T1571"],
    "recommended_action": "Quarantine device and investigate lateral traffic",
    "infer_alert_url":  "https://app.smarthub.ai/alerts/7f3a9d12"
  }
}]]></ac:plain-text-body></ac:structured-macro>

<h2>Microsoft Sentinel — DCR Mapping</h2>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">json</ac:parameter><ac:plain-text-body><![CDATA[// Data Collection Rule transformation (KQL)
source
| project
    TimeGenerated           = todatetime(alert.time),
    AlertId                 = tostring(alert.alert_id),
    TenantId_CF             = tostring(alert.tenant_id),
    Severity                = tostring(alert.severity),
    ThreatType              = tostring(alert.threat_type),
    Title                   = tostring(alert.title),
    DeviceId                = tostring(alert.device_id),
    DeviceClass             = tostring(alert.device_class),
    DeviceIp                = tostring(alert.device_ip),
    DeviceMac               = tostring(alert.device_mac),
    SiteName                = tostring(alert.site_name),
    AnomalyScore            = todouble(alert.anomaly_score),
    SpsScore                = toint(alert.sps_score),
    MitreTechniques         = tostring(alert.mitre_techniques),
    EvidenceJson            = tostring(alert.evidence)]]></ac:plain-text-body></ac:structured-macro>

<h2>Alert Deduplication &amp; Rate Control</h2>
<ul>
  <li><strong>Dedup window:</strong> Same <code>(device_id, threat_type)</code> within 15 minutes → suppressed; only one event forwarded to SIEM</li>
  <li><strong>Alert storms:</strong> If a single device generates &gt;20 alerts in 5 minutes, switch to summary mode: one aggregated event per 5-minute window</li>
  <li><strong>SIEM rate limit:</strong> Max 1,000 events/minute per tenant per SIEM connector; backpressure via internal queue (Redis Stream), no drops</li>
  <li><strong>Retry policy:</strong> Exponential backoff (1s, 2s, 4s, 8s, 16s); after 5 retries, event moved to dead-letter queue + ops alert</li>
</ul>

<h2>MITRE ATT&amp;CK Mapping</h2>
<table>
  <tbody>
    <tr><th>INFER Threat Type</th><th>MITRE Technique</th><th>Tactic</th></tr>
    <tr><td>c2_beaconing</td><td>T1071, T1571, T1132</td><td>Command and Control</td></tr>
    <tr><td>lateral_movement</td><td>T1046, T1021, T1570</td><td>Lateral Movement, Discovery</td></tr>
    <tr><td>data_exfiltration</td><td>T1041, T1048, T1567</td><td>Exfiltration</td></tr>
    <tr><td>credential_stuffing</td><td>T1110, T1078</td><td>Credential Access</td></tr>
    <tr><td>firmware_tampering</td><td>T1542, T1601</td><td>Persistence, Defense Evasion</td></tr>
    <tr><td>dos_participation</td><td>T1498, T1499</td><td>Impact</td></tr>
  </tbody>
</table>
"""))

# ── 9. Performance Benchmarks & Scalability ───────────────────────────────────
PAGES.append(("Performance Benchmarks & Scalability", """
<h1>Performance Benchmarks &amp; Scalability</h1>
<p>This page documents INFER platform benchmarks as of v2.5, measured in SmartHub's performance lab and on production MSK/ECS infrastructure. All numbers are p99 unless stated otherwise.</p>

<h2>Telemetry Ingestion Throughput</h2>
<table>
  <tbody>
    <tr><th>Scenario</th><th>Devices</th><th>Events/s</th><th>Kafka Lag (p99)</th><th>CPU (enrichment fleet)</th></tr>
    <tr><td>Small tenant</td><td>1,000</td><td>1,200</td><td>&lt;100 records</td><td>8%</td></tr>
    <tr><td>Medium tenant</td><td>10,000</td><td>12,500</td><td>&lt;500 records</td><td>22%</td></tr>
    <tr><td>Large tenant</td><td>100,000</td><td>118,000</td><td>&lt;2,000 records</td><td>61%</td></tr>
    <tr><td>Peak burst (IoT storm)</td><td>100,000</td><td>480,000</td><td>&lt;15,000 records</td><td>94% (auto-scaled +8 replicas)</td></tr>
    <tr><td>Platform max (all tenants)</td><td>~2,000,000</td><td>2,400,000</td><td>&lt;30,000 records</td><td>Distributed across 240 vCPU</td></tr>
  </tbody>
</table>

<h2>API Latency (REST)</h2>
<table>
  <tbody>
    <tr><th>Endpoint</th><th>p50</th><th>p95</th><th>p99</th><th>Cache</th></tr>
    <tr><td>GET /devices (10 results)</td><td>12 ms</td><td>28 ms</td><td>45 ms</td><td>Redis, 30 s TTL</td></tr>
    <tr><td>GET /devices/{id}</td><td>8 ms</td><td>19 ms</td><td>31 ms</td><td>Redis, 60 s TTL</td></tr>
    <tr><td>GET /alerts (open, limit 25)</td><td>15 ms</td><td>34 ms</td><td>52 ms</td><td>No cache</td></tr>
    <tr><td>POST /devices/{id}/quarantine</td><td>180 ms</td><td>420 ms</td><td>890 ms</td><td>N/A (writes MQTT + DB)</td></tr>
    <tr><td>GET /compliance/posture</td><td>85 ms</td><td>210 ms</td><td>380 ms</td><td>Redis, 5 min TTL</td></tr>
    <tr><td>POST /compliance/report (PDF)</td><td>4.2 s</td><td>11 s</td><td>18 s</td><td>No cache (async job)</td></tr>
    <tr><td>Gen-AI NLQ query</td><td>1.8 s</td><td>3.2 s</td><td>5.1 s</td><td>Semantic cache (Pinecone)</td></tr>
  </tbody>
</table>

<h2>ML Anomaly Detection Latency</h2>
<table>
  <tbody>
    <tr><th>Stage</th><th>p50</th><th>p99</th><th>Batch Size</th></tr>
    <tr><td>Feature extraction (Flink)</td><td>2.1 ms</td><td>8.4 ms</td><td>1 event</td></tr>
    <tr><td>Isolation Forest inference</td><td>0.8 ms</td><td>3.2 ms</td><td>500 vectors</td></tr>
    <tr><td>LSTM autoencoder inference</td><td>18 ms</td><td>45 ms</td><td>100 sequences</td></tr>
    <tr><td>XGBoost classifier</td><td>0.4 ms</td><td>1.8 ms</td><td>500 vectors</td></tr>
    <tr><td>End-to-end (event → alert)</td><td>28 s</td><td>4.2 min</td><td>N/A (includes 5-min window)</td></tr>
  </tbody>
</table>

<h2>SD-EDGE Gateway Performance</h2>
<table>
  <tbody>
    <tr><th>Metric</th><th>Value</th><th>Hardware</th></tr>
    <tr><td>Max concurrent managed devices</td><td>5,000</td><td>4 vCPU / 8 GB RAM</td></tr>
    <tr><td>SNMP poll throughput</td><td>8,000 OIDs/s</td><td>Same</td></tr>
    <tr><td>Netflow processing</td><td>500,000 flows/min</td><td>Same</td></tr>
    <tr><td>MQTT message throughput</td><td>50,000 msg/s</td><td>Same</td></tr>
    <tr><td>Memory per managed device</td><td>~80 KB</td><td>—</td></tr>
    <tr><td>OTA simultaneous transfers</td><td>10 (configurable to 25)</td><td>—</td></tr>
  </tbody>
</table>

<h2>Database Performance</h2>
<table>
  <tbody>
    <tr><th>Query</th><th>p50</th><th>p99</th><th>Table Size (1M devices)</th></tr>
    <tr><td>Device lookup by IP</td><td>0.3 ms</td><td>1.2 ms</td><td>devices: 2.4 GB</td></tr>
    <tr><td>Open alerts for site</td><td>2.1 ms</td><td>8.7 ms</td><td>alerts: 18 GB (1 year)</td></tr>
    <tr><td>Compliance posture aggregation</td><td>45 ms</td><td>180 ms</td><td>compliance_checks: 42 GB</td></tr>
    <tr><td>Neo4j blast-radius traversal (depth 2)</td><td>12 ms</td><td>85 ms</td><td>10M nodes, 80M edges</td></tr>
  </tbody>
</table>

<h2>Auto-Scaling Configuration (ECS Fargate)</h2>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">yaml</ac:parameter><ac:plain-text-body><![CDATA[services:
  api:
    min_tasks: 3
    max_tasks: 50
    scale_out_threshold: cpu_avg > 65% for 2 min
    scale_in_threshold:  cpu_avg < 30% for 10 min
    cooldown_out: 60s
    cooldown_in:  300s

  enrichment_streams:
    min_tasks: 4
    max_tasks: 80
    scale_out_threshold: kafka_consumer_lag > 50000 records
    scale_in_threshold:  kafka_consumer_lag < 5000 records for 5 min

  anomaly_scorer:
    min_tasks: 2
    max_tasks: 40
    scale_out_threshold: kafka_consumer_lag > 200000 OR p99_inference_latency > 80ms]]></ac:plain-text-body></ac:structured-macro>
"""))

# ── 10. Encryption & Key Management ──────────────────────────────────────────
PAGES.append(("Encryption & Key Management", """
<h1>Encryption &amp; Key Management</h1>
<p>SmartHub.ai follows a defense-in-depth encryption strategy covering data at rest, data in transit, application-layer secrets, and device credentials. This page is the single authoritative reference for all encryption choices.</p>

<h2>Data at Rest</h2>
<table>
  <tbody>
    <tr><th>Data Store</th><th>Encryption</th><th>Key Management</th></tr>
    <tr><td>PostgreSQL (RDS)</td><td>AES-256 (AWS storage encryption)</td><td>AWS KMS CMK, per-tenant key</td></tr>
    <tr><td>S3 (telemetry archive)</td><td>SSE-KMS (AES-256)</td><td>AWS KMS CMK, per-tenant key</td></tr>
    <tr><td>Redis (ElastiCache)</td><td>AES-256 (encryption at rest enabled)</td><td>AWS KMS managed key</td></tr>
    <tr><td>InfluxDB Cloud</td><td>AES-256 (InfluxData-managed)</td><td>InfluxData KMS</td></tr>
    <tr><td>MSK (Kafka)</td><td>AES-256 (broker-level)</td><td>AWS KMS managed key</td></tr>
    <tr><td>Neo4j AuraDB</td><td>AES-256 (Neo4j-managed)</td><td>Neo4j internal KMS</td></tr>
    <tr><td>Secrets (API keys, tokens)</td><td>AES-256-GCM (application-layer)</td><td>AWS Secrets Manager with automatic 90-day rotation</td></tr>
  </tbody>
</table>

<h2>Data in Transit</h2>
<table>
  <tbody>
    <tr><th>Channel</th><th>Protocol</th><th>Minimum TLS</th><th>Certificate</th></tr>
    <tr><td>Client → API Gateway</td><td>HTTPS</td><td>TLS 1.2 (1.3 preferred)</td><td>Let's Encrypt via ACM</td></tr>
    <tr><td>API Gateway → ECS</td><td>HTTPS</td><td>TLS 1.2</td><td>ACM private CA</td></tr>
    <tr><td>ECS → PostgreSQL</td><td>PostgreSQL TLS</td><td>TLS 1.2</td><td>RDS-managed cert</td></tr>
    <tr><td>ECS → MSK</td><td>Kafka TLS</td><td>TLS 1.2</td><td>ACM private CA</td></tr>
    <tr><td>SD-EDGE Gateway → INFER cloud</td><td>mTLS over HTTPS/MQTT</td><td>TLS 1.3 only</td><td>Per-gateway cert (Mocana CA)</td></tr>
    <tr><td>Gateway → Device (SNMP v3)</td><td>SNMP v3 authPriv</td><td>AES-128 (privacy)</td><td>Pre-shared key</td></tr>
    <tr><td>Gateway → Device (OTA/SSH)</td><td>SSH</td><td>ED25519 host key</td><td>Per-device key stored in Secrets Manager</td></tr>
  </tbody>
</table>

<h2>Application-Layer Sensitive Field Encryption</h2>
<p>Selected columns in PostgreSQL are encrypted at the application layer before write (envelope encryption), in addition to RDS storage encryption:</p>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">python</ac:parameter><ac:plain-text-body><![CDATA[# Fields encrypted at application layer using AES-256-GCM
# Key: fetched from AWS Secrets Manager, cached in-memory 5 min
ENCRYPTED_COLUMNS = {
    "devices":      ["snmp_community_string", "ssh_password", "api_credential"],
    "integrations": ["hec_token", "api_key", "webhook_secret"],
    "users":        ["mfa_secret"],
}

def encrypt_field(plaintext: str, tenant_id: str) -> str:
    dek = get_tenant_dek(tenant_id)         # 256-bit data encryption key
    iv = os.urandom(12)                     # 96-bit IV for GCM
    cipher = AES.new(dek, AES.MODE_GCM, nonce=iv)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode())
    # Store as base64(iv || tag || ciphertext)
    return b64encode(iv + tag + ciphertext).decode()]]></ac:plain-text-body></ac:structured-macro>

<h2>Key Hierarchy</h2>
<table>
  <tbody>
    <tr><th>Level</th><th>Key Type</th><th>Lifetime</th><th>Storage</th></tr>
    <tr><td>Master Key (KMK)</td><td>AWS KMS CMK (RSA-4096)</td><td>Permanent (auto-rotate annually)</td><td>AWS KMS HSM</td></tr>
    <tr><td>Data Encryption Key (DEK)</td><td>AES-256-GCM, per-tenant</td><td>90 days (auto-rotate)</td><td>AWS Secrets Manager (encrypted by KMK)</td></tr>
    <tr><td>Session Keys (TLS)</td><td>ECDHE P-256 (ephemeral)</td><td>Per connection (forward secrecy)</td><td>Memory only</td></tr>
    <tr><td>Gateway Identity Cert</td><td>ECDSA P-256</td><td>365 days (auto-renew at 30d)</td><td>Gateway TPM / Secrets Manager</td></tr>
    <tr><td>JWT Signing Key</td><td>RSA-2048 (Auth0-managed)</td><td>30 days (automatic rotation)</td><td>Auth0 JWKS endpoint</td></tr>
  </tbody>
</table>

<h2>Crypto Deprecation Schedule</h2>
<table>
  <tbody>
    <tr><th>Algorithm</th><th>Status</th><th>Deprecation Date</th><th>Migration Path</th></tr>
    <tr><td>TLS 1.0 / 1.1</td><td>Already disabled</td><td>2022-01-01</td><td>TLS 1.2+</td></tr>
    <tr><td>RSA-1024 device certs</td><td>Already disabled</td><td>2023-06-01</td><td>ECDSA P-256</td></tr>
    <tr><td>SHA-1 signatures</td><td>Already disabled</td><td>2023-01-01</td><td>SHA-256+</td></tr>
    <tr><td>AES-128-CBC (SNMP privacy)</td><td>Deprecating</td><td>2025-01-01</td><td>AES-256-CFB (SNMP v3)</td></tr>
    <tr><td>RSA-2048 JWT signing</td><td>Reviewing</td><td>TBD (post-NIST PQC standards)</td><td>ML-KEM / ML-DSA (post-quantum)</td></tr>
  </tbody>
</table>
"""))

# ── 11. DevOps & CI/CD Pipeline ───────────────────────────────────────────────
PAGES.append(("DevOps & CI/CD Pipeline", """
<h1>DevOps &amp; CI/CD Pipeline</h1>
<p>INFER uses a trunk-based development model with automated quality gates at every stage. All services are containerised and deployed to AWS ECS Fargate via GitHub Actions and ArgoCD.</p>

<h2>Repository Structure</h2>
<table>
  <tbody>
    <tr><th>Repository</th><th>Language</th><th>Description</th></tr>
    <tr><td>infer-api</td><td>Python 3.11</td><td>Core FastAPI service (device, alert, compliance APIs)</td></tr>
    <tr><td>infer-streams</td><td>Java 21 / Kafka Streams</td><td>Telemetry enrichment and aggregation</td></tr>
    <tr><td>infer-flink</td><td>Java 21 / Apache Flink</td><td>Feature extraction and CEP alert engine</td></tr>
    <tr><td>infer-ml</td><td>Python 3.11 / PyTorch</td><td>Anomaly detection models and TorchServe config</td></tr>
    <tr><td>infer-policy-engine</td><td>Go 1.22 / OPA</td><td>Compliance rules and authz policies</td></tr>
    <tr><td>sdedge-gateway</td><td>Go 1.22</td><td>SD-EDGE Gateway binary</td></tr>
    <tr><td>infer-frontend</td><td>TypeScript / React 18</td><td>Web dashboard</td></tr>
    <tr><td>infer-infra</td><td>Terraform / Helm</td><td>AWS infrastructure and Kubernetes charts</td></tr>
    <tr><td>infer-platform</td><td>Python</td><td>Internal tooling: db migrations, seed scripts, load tests</td></tr>
  </tbody>
</table>

<h2>GitHub Actions CI Pipeline</h2>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">yaml</ac:parameter><ac:plain-text-body><![CDATA[# Triggered on: push to main, PRs to main
stages:
  lint-and-typecheck:       # ~45 s
    - ruff check + ruff format --check  (Python)
    - mypy --strict                      (Python)
    - golangci-lint run                  (Go)
    - tsc --noEmit                       (TypeScript)

  unit-tests:               # ~3 min
    - pytest -x --cov=. --cov-fail-under=85
    - go test ./... -race -coverprofile=cover.out
    - vitest run --coverage

  integration-tests:        # ~8 min
    - Spin up: PostgreSQL 15, Redis 7, Kafka 3.6 (testcontainers)
    - pytest tests/integration/ -x
    - go test ./integration/...

  security-scan:            # ~4 min
    - trivy image --exit-code 1 --severity HIGH,CRITICAL
    - bandit -r src/ -ll             (Python SAST)
    - gosec ./...                    (Go SAST)
    - semgrep --config=auto

  build-and-push:           # ~6 min
    - docker buildx build --platform linux/amd64,linux/arm64
    - Push to ECR with tag: sha-{commit_sha}

  deploy-staging:           # ~4 min
    - ArgoCD sync to staging namespace
    - Run smoke tests (30 API health checks)
    - Run k6 load test (500 VU, 5 min)

  deploy-production:        # ~5 min (main branch only, after staging passes)
    - ArgoCD sync to production (progressive delivery via Argo Rollouts)
    - Canary: 5% → 25% → 100% over 30 min
    - Auto-rollback if error_rate > 0.5% or p99_latency > 500ms]]></ac:plain-text-body></ac:structured-macro>

<h2>Progressive Delivery (Argo Rollouts)</h2>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">yaml</ac:parameter><ac:plain-text-body><![CDATA[apiVersion: argoproj.io/v1alpha1
kind: Rollout
spec:
  strategy:
    canary:
      steps:
        - setWeight: 5
        - pause: {duration: 5m}
        - analysis:
            templates: [{templateName: success-rate}]
        - setWeight: 25
        - pause: {duration: 10m}
        - analysis:
            templates: [{templateName: success-rate}, {templateName: latency-p99}]
        - setWeight: 100
      analysis:
        successCondition: "result[0] >= 0.995"   # 99.5% success rate
      autoRollback:
        enabled: true]]></ac:plain-text-body></ac:structured-macro>

<h2>Database Migration Strategy</h2>
<ul>
  <li>Migrations managed by <strong>Alembic</strong> (Python) and <strong>golang-migrate</strong> (Go services)</li>
  <li>All migrations must be <strong>backwards-compatible</strong> for at least one release cycle (expand-contract pattern)</li>
  <li>Column drops require a 2-step process: (1) stop writing in release N, (2) drop in release N+1</li>
  <li>Large table migrations (ALTER TABLE on &gt;10M rows) run as background jobs using <code>pg_repack</code> to avoid table locks</li>
  <li>Migration smoke test: <code>alembic upgrade head</code> runs in CI against a snapshot of production schema</li>
</ul>

<h2>On-Call &amp; Incident Response</h2>
<table>
  <tbody>
    <tr><th>Tier</th><th>Rotation</th><th>Tools</th><th>SLO</th></tr>
    <tr><td>L1 — Platform On-Call</td><td>Weekly rotation, 2 engineers</td><td>PagerDuty, Grafana, Runbooks</td><td>Acknowledge P1 in 5 min</td></tr>
    <tr><td>L2 — Service Owner</td><td>Per-service, business hours</td><td>Slack #oncall, GitHub Issues</td><td>Acknowledge P2 in 30 min</td></tr>
    <tr><td>L3 — Incident Commander</td><td>Senior eng, major incidents only</td><td>Zoom war room, PagerDuty Escalation</td><td>Engaged within 15 min of P1</td></tr>
  </tbody>
</table>
"""))

# ── 12. Gen-AI NLQ — Technical Design ─────────────────────────────────────────
PAGES.append(("Gen-AI Natural Language Query — Technical Design", """
<h1>Gen-AI Natural Language Query — Technical Design</h1>
<p>INFER v2.5 ships a natural language query (NLQ) interface that lets operators ask plain-English questions over their device inventory and security posture. This page covers the full RAG pipeline, prompt engineering, and evaluation approach.</p>

<h2>Architecture</h2>
<table>
  <tbody>
    <tr><th>Component</th><th>Technology</th><th>Role</th></tr>
    <tr><td>Query Parser</td><td>GPT-4o (OpenAI)</td><td>Classify intent + extract structured filters (site, device_class, time range)</td></tr>
    <tr><td>Vector Retrieval</td><td>Pinecone (text-embedding-3-large, 3072-dim)</td><td>Semantic similarity search over device summaries</td></tr>
    <tr><td>Graph Retrieval</td><td>Neo4j (Cypher)</td><td>Relationship-aware context (e.g. blast radius, peer devices)</td></tr>
    <tr><td>Structured Query</td><td>PostgreSQL (generated Cypher/SQL)</td><td>Precise filter queries (exact site, CVE, firmware version)</td></tr>
    <tr><td>Answer Synthesizer</td><td>GPT-4o</td><td>Fuses retrieval results into a natural language answer with citations</td></tr>
    <tr><td>Semantic Cache</td><td>Pinecone (separate namespace)</td><td>Cache answers for semantically similar questions (cosine &gt; 0.97)</td></tr>
  </tbody>
</table>

<h2>Query Processing Flow</h2>
<ol>
  <li><strong>Receive query:</strong> User types "Show me all cameras with firmware older than 6 months in Building 3 with SPS below 50"</li>
  <li><strong>Cache check:</strong> Embed query with text-embedding-3-small; check Pinecone semantic cache (threshold: cosine ≥ 0.97)</li>
  <li><strong>Intent classification:</strong> GPT-4o classifies as <code>device_filter_query</code> and extracts: <code>device_class=ip_camera, location=Building 3, firmware_age_gt=180d, sps_lt=50</code></li>
  <li><strong>Dual retrieval:</strong> Pinecone vector search (top-20 device summaries) + PostgreSQL structured query (exact filter)</li>
  <li><strong>Graph augmentation:</strong> For each retrieved device, fetch Neo4j context (CVE exposure, peer count, recent alerts)</li>
  <li><strong>Reranking:</strong> Cohere Rerank API re-scores the top-20 Pinecone results against the original query</li>
  <li><strong>Answer synthesis:</strong> GPT-4o generates answer with inline citations to device IDs and page links</li>
  <li><strong>Cache write:</strong> Store query embedding + answer in semantic cache (TTL: 10 minutes)</li>
</ol>

<h2>System Prompt — Answer Synthesizer</h2>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">text</ac:parameter><ac:plain-text-body><![CDATA[You are INFER Assistant, an AI security analyst for the IoT management platform INFER™ by SmartHub.ai.
You answer questions about the user's device inventory, security alerts, and compliance posture.

Rules:
1. Answer ONLY from the provided context. Do not hallucinate device details.
2. Cite device IDs inline: "Camera dev_a1b2c3 (10.10.5.42) has firmware 9.60.1.2 (8 months old)."
3. If the context is insufficient, say: "I don't have enough data to answer that. Try narrowing the scope."
4. Format lists as tables when >3 items.
5. End every answer with a recommended action if a security risk is identified.
6. Never reveal tenant data from other tenants.
7. Keep answers under 400 words unless the user explicitly asks for detail.

Context:
{retrieved_chunks}

User question: {user_query}]]></ac:plain-text-body></ac:structured-macro>

<h2>Device Summary Embedding Schema</h2>
<p>Each device in Pinecone is represented as a text chunk embedded every 15 minutes:</p>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">text</ac:parameter><ac:plain-text-body><![CDATA[Device: dev_a1b2c3
MAC: AA:BB:CC:DD:EE:FF | IP: 10.10.5.42 | Hostname: cam-lobby-01
Class: IP Camera | Vendor: Axis | Model: P3245-V
Firmware: 9.60.1.2 (released 2024-01-15, age 152 days)
Site: HQ San Francisco, Building 3, Floor 1
SPS Score: 34/100 (POOR) — failing: firmware_currency, default_credentials
CVEs: CVE-2024-8765 (CVSS 8.1), CVE-2024-2341 (CVSS 6.3)
Last seen: 2024-11-14 09:23 UTC | Status: active
Open alerts: 2 (1 high: lateral_movement, 1 medium: credential_stuffing)
Peer devices communicated (7d): 23 unique IPs including 3 PLCs]]></ac:plain-text-body></ac:structured-macro>

<h2>Evaluation Metrics</h2>
<table>
  <tbody>
    <tr><th>Metric</th><th>Method</th><th>Target</th><th>Current (v2.5)</th></tr>
    <tr><td>Retrieval Recall@10</td><td>Annotated golden set (500 queries)</td><td>&gt;0.90</td><td>0.93</td></tr>
    <tr><td>Answer Faithfulness</td><td>RAGAS faithfulness score (GPT-4 judge)</td><td>&gt;0.92</td><td>0.94</td></tr>
    <tr><td>Answer Relevance</td><td>RAGAS answer relevance</td><td>&gt;0.88</td><td>0.91</td></tr>
    <tr><td>Context Precision</td><td>RAGAS context precision</td><td>&gt;0.85</td><td>0.87</td></tr>
    <tr><td>Hallucination rate</td><td>Fact-check vs. live DB (random 100 answers/day)</td><td>&lt;2%</td><td>1.1%</td></tr>
    <tr><td>Cache hit rate</td><td>Production telemetry</td><td>&gt;25%</td><td>31%</td></tr>
    <tr><td>End-to-end p95 latency</td><td>Production telemetry</td><td>&lt;5 s</td><td>3.2 s</td></tr>
  </tbody>
</table>
"""))

# ── 13. Internal Engineering Runbooks ─────────────────────────────────────────
PAGES.append(("Engineering Runbooks — Production Operations", """
<h1>Engineering Runbooks — Production Operations</h1>
<p>Standard operating procedures for the INFER platform on-call engineer. Last reviewed: November 2024. Owner: Platform Engineering.</p>

<h2>Runbook Index</h2>
<table>
  <tbody>
    <tr><th>Incident Type</th><th>Severity</th><th>Jump To</th></tr>
    <tr><td>API elevated error rate (&gt;1%)</td><td>P1/P2</td><td>Section: API Error Rate</td></tr>
    <tr><td>Kafka consumer lag &gt;200k records</td><td>P1</td><td>Section: Kafka Lag</td></tr>
    <tr><td>Anomaly scorer down</td><td>P1</td><td>Section: ML Scorer Outage</td></tr>
    <tr><td>Gateway mass disconnect</td><td>P1</td><td>Section: Gateway Disconnect</td></tr>
    <tr><td>PostgreSQL replica lag &gt;30s</td><td>P2</td><td>Section: DB Replication Lag</td></tr>
    <tr><td>SIEM connector failures</td><td>P2</td><td>Section: SIEM Failures</td></tr>
    <tr><td>OTA campaign stuck</td><td>P3</td><td>Section: OTA Stuck Campaign</td></tr>
  </tbody>
</table>

<h2>API Elevated Error Rate</h2>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">bash</ac:parameter><ac:plain-text-body><![CDATA[# 1. Check current error rate in Grafana
# Dashboard: infer-api/overview → panel "5xx Rate by Endpoint"

# 2. Check recent deployments
aws ecs describe-services --cluster infer-prod --services infer-api \
  | jq '.services[].deployments[] | {status, runningCount, failedTasks, createdAt}'

# 3. Tail recent error logs
aws logs filter-log-events \
  --log-group-name /ecs/infer-api \
  --start-time $(date -d '10 minutes ago' +%s000) \
  --filter-pattern '"level":"ERROR"' \
  | jq '.events[].message' | tail -20

# 4. If new deployment: rollback
aws ecs update-service --cluster infer-prod --service infer-api \
  --task-definition infer-api:{PREVIOUS_TASK_DEF_ARN}

# 5. If DB issue: check RDS metrics
aws cloudwatch get-metric-statistics --namespace AWS/RDS \
  --metric-name DatabaseConnections --dimensions Name=DBInstanceIdentifier,Value=infer-prod-primary \
  --start-time $(date -d '1 hour ago' -u +%Y-%m-%dT%H:%M:%SZ) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
  --period 60 --statistics Maximum]]></ac:plain-text-body></ac:structured-macro>

<h2>Kafka Consumer Lag — Emergency Scale-Out</h2>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">bash</ac:parameter><ac:plain-text-body><![CDATA[# Check lag per consumer group
kafka-consumer-groups.sh --bootstrap-server $MSK_BROKER \
  --describe --all-groups 2>/dev/null \
  | awk '$5 > 10000 {print $1, $2, $5}' | sort -k3 -rn | head -20

# Scale out enrichment-streams ECS service immediately
aws ecs update-service --cluster infer-prod \
  --service infer-enrichment-streams \
  --desired-count 20

# If Flink job is behind, restart with higher parallelism
flink run -m yarn-cluster -p 32 \
  -c com.smarthub.flink.FeatureExtractionJob \
  infer-flink-jobs.jar \
  --kafka.bootstrap.servers $MSK_BROKER \
  --parallelism 32

# Monitor recovery
watch -n 10 'kafka-consumer-groups.sh --bootstrap-server $MSK_BROKER \
  --describe --group enrichment-streams-prod | tail -5']]></ac:plain-text-body></ac:structured-macro>

<h2>Gateway Mass Disconnect</h2>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">bash</ac:parameter><ac:plain-text-body><![CDATA[# Check MQTT broker (EMQX) cluster status
curl -s https://mqtt.smarthub.ai/api/v5/cluster \
  -H "Authorization: Bearer $EMQX_API_KEY" | jq '.nodes[] | {node, running, connections}'

# Check for TLS cert expiry on MQTT endpoint
echo | openssl s_client -connect mqtt.smarthub.ai:8883 2>/dev/null \
  | openssl x509 -noout -dates

# If EMQX overloaded — check connection rate
curl -s https://mqtt.smarthub.ai/api/v5/metrics \
  -H "Authorization: Bearer $EMQX_API_KEY" \
  | jq '.["connections.count"], .["messages.received.rate"]'

# Force reconnect: bump the MQTT gateway config version
# (gateways poll for config changes every 60s and reconnect on version mismatch)
aws ssm put-parameter --name /infer/prod/mqtt/config_version \
  --value "$(date +%s)" --overwrite]]></ac:plain-text-body></ac:structured-macro>

<h2>OTA Campaign Stuck</h2>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">bash</ac:parameter><ac:plain-text-body><![CDATA[# Check campaign state
curl -s "https://api.smarthub.ai/v2/ota/campaigns/$CAMPAIGN_ID" \
  -H "Authorization: Bearer $INFER_ADMIN_TOKEN" | jq '{state, stats}'

# Check for devices that have not responded
psql $INFER_DB -c "
  SELECT d.device_id, d.ip_address, oj.status, oj.last_attempt_at
  FROM ota_jobs oj JOIN devices d USING (device_id)
  WHERE oj.campaign_id = '$CAMPAIGN_ID'
    AND oj.status NOT IN ('completed','skipped')
    AND oj.last_attempt_at < NOW() - INTERVAL '30 minutes'
  ORDER BY oj.last_attempt_at ASC LIMIT 20;"

# Retry stuck jobs (sets status back to 'pending')
curl -s -X POST \
  "https://api.smarthub.ai/v2/ota/campaigns/$CAMPAIGN_ID/retry-stuck" \
  -H "Authorization: Bearer $INFER_ADMIN_TOKEN" \
  -d '{"older_than_minutes": 30}']]></ac:plain-text-body></ac:structured-macro>

<h2>Key Dashboards &amp; Runbook Links</h2>
<table>
  <tbody>
    <tr><th>Dashboard</th><th>URL</th><th>When to Use</th></tr>
    <tr><td>API Overview</td><td>grafana.smarthub.internal/d/infer-api</td><td>Error rate, latency, traffic spikes</td></tr>
    <tr><td>Kafka Lag</td><td>grafana.smarthub.internal/d/kafka-lag</td><td>Consumer group lag, topic throughput</td></tr>
    <tr><td>ML Pipeline</td><td>grafana.smarthub.internal/d/ml-pipeline</td><td>Inference latency, model drift, scorer health</td></tr>
    <tr><td>Gateway Fleet</td><td>grafana.smarthub.internal/d/gw-fleet</td><td>Connected gateways, disconnect events, memory</td></tr>
    <tr><td>RDS / PgBouncer</td><td>grafana.smarthub.internal/d/rds</td><td>Query latency, connections, replication lag</td></tr>
  </tbody>
</table>
"""))


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Creating {len(PAGES)} technical pages in space '{SPACE}' on {DOMAIN}\n")
    created = []
    for i, (title, html) in enumerate(PAGES, 1):
        print(f"[{i}/{len(PAGES)}] {title} ... ", end="", flush=True)
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


if __name__ == "__main__":
    main()
