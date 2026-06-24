"""
Fetches all SmartHub.ai Confluence pages and saves them as markdown files
in stress_corpus/docs/.
"""

import re
import os
import html
import requests
from requests.auth import HTTPBasicAuth
from pathlib import Path

CONFLUENCE_BASE = "https://akshathr333.atlassian.net"
EMAIL = "akshath.r333@gmail.com"
TOKEN = "ATATT3xFfGF0p1782cWmnUPRWoBgzGmK29GNJPR7SXJ6JU2maGrS6RrlP1S6vyS-v9XYmSYlPIswJYuvpZiyp8_tgZuywSBhXTfOcKhzMZHgDYfKl-Sr2OFNWj5vs5VefSylwu75TfKqpJfXhDpKe7VLC1cfjQ_TiTIN9DwTZV1x4BaeqjqOiws=72218B06"

# SmartHub.ai descriptive pages (non-synthetic corpus pages)
SMARTHUB_PAGE_IDS = [
    ("48660482", "smarthub_ai_company_overview"),
    ("48988161", "smarthub_infer_platform_product_overview"),
    ("49020929", "smarthub_sd_edge_software_defined_edge"),
    ("49020945", "smarthub_customer_success_stories"),
    ("49053697", "smarthub_security_compliance_framework"),
    ("49053713", "smarthub_devops_cicd_pipeline"),
    ("49086465", "smarthub_engineering_system_architecture"),
    ("49119233", "smarthub_api_reference_guide_infer_v2"),
    ("49152001", "smarthub_device_onboarding_guide"),
    ("49184769", "smarthub_release_notes_infer_v2_5"),
    ("49184785", "smarthub_troubleshooting_guide"),
    ("49217537", "smarthub_pricing_and_plans"),
    ("49250305", "smarthub_sla_support_policy"),
    ("49250326", "smarthub_telemetry_schema_event_catalog"),
    ("49283073", "smarthub_ota_firmware_update_pipeline"),
    ("49315841", "smarthub_postgresql_database_schema"),
    ("49348609", "smarthub_siem_integration_architecture"),
    ("49381377", "smarthub_gen_ai_natural_language_query"),
    ("48463881", "smarthub_engineering_runbooks_production"),
    ("48594947", "smarthub_encryption_key_management"),
    ("48758786", "smarthub_deployment_installation_guide_sd_edge"),
    ("48758803", "smarthub_authentication_authorization_deep_dive"),
    ("48824323", "smarthub_ml_anomaly_detection_model_architecture"),
    ("48824340", "smarthub_performance_benchmarks_scalability"),
    ("48889858", "smarthub_graph_database_schema_neo4j"),
    ("48332810", "smarthub_kafka_stream_processing_architecture"),
]

AUTH = HTTPBasicAuth(EMAIL, TOKEN)
OUT_DIR = Path(__file__).parent / "docs"


def html_to_markdown(html_body: str, title: str) -> str:
    """Convert Confluence storage-format HTML to clean markdown."""
    text = html_body

    # Headings
    for level in range(6, 0, -1):
        text = re.sub(rf"<h{level}[^>]*>(.*?)</h{level}>", lambda m, l=level: "\n" + "#" * l + " " + m.group(1) + "\n", text, flags=re.DOTALL)

    # Bold / italic
    text = re.sub(r"<strong[^>]*>(.*?)</strong>", r"**\1**", text, flags=re.DOTALL)
    text = re.sub(r"<b[^>]*>(.*?)</b>", r"**\1**", text, flags=re.DOTALL)
    text = re.sub(r"<em[^>]*>(.*?)</em>", r"*\1*", text, flags=re.DOTALL)
    text = re.sub(r"<i[^>]*>(.*?)</i>", r"*\1*", text, flags=re.DOTALL)

    # Code
    text = re.sub(r"<code[^>]*>(.*?)</code>", r"`\1`", text, flags=re.DOTALL)
    text = re.sub(r"<pre[^>]*>(.*?)</pre>", r"\n```\n\1\n```\n", text, flags=re.DOTALL)

    # Links
    text = re.sub(r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', r"[\2](\1)", text, flags=re.DOTALL)

    # Table rows → pipe-delimited
    text = re.sub(r"<th[^>]*>(.*?)</th>", lambda m: f" {m.group(1).strip()} |", text, flags=re.DOTALL)
    text = re.sub(r"<td[^>]*>(.*?)</td>", lambda m: f" {m.group(1).strip()} |", text, flags=re.DOTALL)
    text = re.sub(r"<tr[^>]*>", "| ", text, flags=re.DOTALL)
    text = re.sub(r"</tr>", "\n", text, flags=re.DOTALL)
    text = re.sub(r"<thead[^>]*>|</thead>|<tbody[^>]*>|</tbody>|<tfoot[^>]*>|</tfoot>", "", text)
    text = re.sub(r"<table[^>]*>|</table>", "\n", text)

    # Lists
    text = re.sub(r"<li[^>]*>(.*?)</li>", r"- \1", text, flags=re.DOTALL)
    text = re.sub(r"<ul[^>]*>|</ul>|<ol[^>]*>|</ol>", "\n", text)

    # Paragraphs and line breaks
    text = re.sub(r"<p[^>]*>(.*?)</p>", r"\1\n\n", text, flags=re.DOTALL)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<hr\s*/?>", "\n---\n", text)

    # Confluence macros / panels — strip wrapper, keep content
    text = re.sub(r"<ac:[^>]+>|</ac:[^>]+>", "", text)
    text = re.sub(r"<ri:[^>]+>|</ri:[^>]+>", "", text)

    # Strip remaining tags
    text = re.sub(r"<[^>]+>", "", text)

    # Decode HTML entities
    text = html.unescape(text)

    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = text.strip()

    return f"# {title}\n\n{text}\n"


def fetch_page(page_id: str) -> dict:
    url = f"{CONFLUENCE_BASE}/wiki/rest/api/content/{page_id}?expand=body.storage"
    resp = requests.get(url, auth=AUTH)
    resp.raise_for_status()
    return resp.json()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = []

    for page_id, slug in SMARTHUB_PAGE_IDS:
        print(f"Fetching {page_id} → {slug}.md ...", end=" ", flush=True)
        try:
            data = fetch_page(page_id)
            title = data["title"]
            body_html = data["body"]["storage"]["value"]
            markdown = html_to_markdown(body_html, title)
            out_path = OUT_DIR / f"{slug}.md"
            out_path.write_text(markdown, encoding="utf-8")
            size = len(markdown)
            print(f"OK ({size:,} chars)")
            saved.append((slug, title, size))
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nSaved {len(saved)} files to {OUT_DIR}")
    for slug, title, size in saved:
        print(f"  {slug}.md  ({size:,} chars)  — {title}")


if __name__ == "__main__":
    main()
