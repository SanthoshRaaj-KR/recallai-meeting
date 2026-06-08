"""Prove the reported-fact extractor generalizes across domains (not hardcoded).

Five messy, report-style transcripts (typos/fillers/self-corrections) in different
domains, each run (a) against a matching doc -> expect a correctly-routed card with
the new value, and (b) against the unrelated stress_corpus -> expect 0 cards.
"""
import asyncio, os, re, sys, shutil, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / "../confluence_logic/.env")
from pipeline.run import run_pipeline, PipelineConfig


def num(s):
    m = re.search(r"\d[\d,\.]*", s or "")
    return m.group(0).replace(",", "") if m else None


CASES = [
    dict(
        name="saas-churn",
        transcript="hey quick update from the growth side, um, our quarterly customer churn "
                   "rate jumped to like 7 percent this quarter, up from before, wanted to flag it.",
        doc="# Growth Metrics\n\n## Customer Retention\nOur quarterly customer churn rate is "
            "currently 4 percent. Retention is tracked monthly by the growth team and reviewed "
            "at the quarterly business review.\n\n## Expansion Revenue\nNet revenue retention "
            "has held steady around 112 percent over the last year.\n",
        section="Customer Retention", value="7",
    ),
    dict(
        name="security-breach",
        transcript="I wanted to note we had a phishing incident last month, it exposed around "
                   "1200 user accounts and it took us ahh 4 days to fully contain it.",
        doc="# Security Incident Register\n\n## Phishing Incidents\nThe last recorded phishing "
            "incident affected 300 user accounts and was contained within 1 day. All affected "
            "users were notified and credentials were reset.\n\n## Malware Incidents\nNo malware "
            "incidents have been recorded in the current quarter.\n",
        section="Phishing Incidents", value="1200",
    ),
    dict(
        name="vendor-cost",
        transcript="oh also the datadog contract, it renewed, the new annual price is 220 "
                   "thousand dollars now, bit of a jump from what we had.",
        doc="# Vendor Contracts\n\n## Observability Tooling\nThe company uses Datadog for "
            "observability and monitoring. The annual contract value is 150 thousand dollars, "
            "renewed each January. The platform team owns this relationship.\n\n## CI/CD Tooling\n"
            "GitHub Actions is used for continuous integration across all repositories.\n",
        section="Observability Tooling", value="220",
    ),
    dict(
        name="compliance-audit",
        transcript="from compliance, the soc 2 audit this cycle came back with, um, 2 high "
                   "severity findings, sorry, 3 high severity findings and a couple of mediums.",
        doc="# Compliance Status\n\n## SOC 2 Audit\nThe most recent SOC 2 Type II audit returned "
            "zero high-severity findings and two low-severity observations. Remediation is "
            "tracked by the security team to closure.\n\n## GDPR\nThe company maintains a record "
            "of processing activities reviewed annually.\n",
        section="SOC 2 Audit", value="3",
    ),
    dict(
        name="headcount",
        transcript="quick people update, we just crossed 340 employees, up from where we were "
                   "at the start of the year.",
        doc="# Company Overview\n\n## Headcount\nThornvale currently employs 280 full-time staff "
            "across engineering, sales, and operations. Headcount is reviewed by People Ops each "
            "quarter.\n\n## Offices\nThe company operates from two offices and supports remote "
            "work for most roles.\n",
        section="Headcount", value="340",
    ),
]


async def run_case(c):
    tmpd = tempfile.mkdtemp(prefix=f"gen_{c['name']}_")
    folder = os.path.join(tmpd, "docs")
    os.makedirs(folder)
    Path(os.path.join(folder, f"{c['name']}.md")).write_text(c["doc"], encoding="utf-8")
    try:
        cfg = PipelineConfig(session_id=c["name"], doc_folder=folder, use_embeddings=True,
                             rerank=False, contextual_retrieval=False)
        ps = await run_pipeline(c["transcript"], cfg)
        on_sec = [p for p in ps if p.source_chunk.section_heading == c["section"]]
        val_ok = any(num(c["value"]) in (num(p.after_content) or "") or c["value"] in (p.after_content or "")
                     for p in on_sec)
        # precision: same transcript vs unrelated stress_corpus
        cfg2 = PipelineConfig(session_id=f"{c['name']}-neg", doc_folder="stress_corpus/docs",
                              use_embeddings=True, rerank=False, contextual_retrieval=False)
        neg = await run_pipeline(c["transcript"], cfg2)
        return len(ps), bool(on_sec), val_ok, len(neg)
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


async def main():
    print(f"{'case':18s} {'cards':5s} {'routed':6s} {'value':5s} {'neg(stress)':11s} verdict")
    print("-" * 64)
    passes = 0
    for c in CASES:
        cards, routed, val_ok, neg = await run_case(c)
        ok = routed and val_ok and neg == 0
        passes += ok
        print(f"{c['name']:18s} {cards:<5d} {('YES' if routed else 'no'):6s} "
              f"{('ok' if val_ok else '--'):5s} {neg:<11d} {'PASS' if ok else 'CHECK'}")
    print("-" * 64)
    print(f"generalization: {passes}/{len(CASES)} domains pass "
          f"(correct card on matching doc + 0 false-positives on unrelated corpus)")


if __name__ == "__main__":
    asyncio.run(main())
