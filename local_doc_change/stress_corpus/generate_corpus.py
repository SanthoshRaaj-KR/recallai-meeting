"""Generate a 100-document, mixed-format stress corpus for the local-doc pipeline.

- 100 documents, 25 each of .md / .txt / .docx / .odt
- 15-80 pages each (~480 words/page), realistic multi-section policy/handbook prose
- Globally-UNIQUE "anchor" facts planted in known sections (distinctive value +
  distinctive topic phrase) so RAG retrieval has an unambiguous correct target
- Ground-truth written to manifest.json: every anchor records the file, format,
  section heading, the value to change, and a ready-made meeting transcript

Deterministic (seeded) so the corpus is reproducible.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HERE = Path(__file__).resolve().parent
DOCS = HERE / "docs"

SEED = 20260605
WORDS_PER_PAGE = 480

# ── Fillers for coherent-ish policy prose ─────────────────────────────────────

ORGS = [
    "Aldermont", "Brightforge", "Cindergate", "Drayveil", "Everwynd", "Fenmark",
    "Glasshollow", "Harrowfield", "Ironbloom", "Junivault", "Kesterline",
    "Larkspire", "Morrowgate", "Netherby", "Oakmeridian", "Pallasvane",
    "Quillhaven", "Rampart Lyle", "Stormfell", "Tindermere", "Umberlane",
    "Vauxhollow", "Westmarch", "Xanthe Reef", "Yarrowdale", "Zephyrion",
]
DOMAINS = [
    ("Information Security Policy", "security", [
        "Access Control", "Data Retention", "Incident Response", "Encryption Standards",
        "Vendor Risk Management", "Endpoint Protection", "Network Segmentation",
        "Vulnerability Management", "Identity and Access", "Logging and Monitoring",
        "Backup and Recovery", "Physical Security", "Secure Development",
        "Threat Intelligence", "Key Management", "Audit and Compliance",
    ]),
    ("People Operations Handbook", "hr", [
        "Paid Time Off", "Sick Leave", "Remote Work", "Expense Reimbursement",
        "Performance Reviews", "Parental Leave", "Benefits Enrollment", "Code of Conduct",
        "Compensation Bands", "Promotion Process", "Grievance Procedure",
        "Learning Budget", "Travel Policy", "Onboarding", "Offboarding", "Relocation",
    ]),
    ("Engineering Handbook", "eng", [
        "Source Control", "Code Review", "Testing Standards", "Continuous Integration",
        "Deployment Process", "Release Cadence", "On-Call Rotation", "Incident Management",
        "Observability", "Technical Documentation", "Secrets Management",
        "Architecture Review", "Performance Budgets", "Dependency Policy",
        "Feature Flags", "Postmortems",
    ]),
    ("Customer Support SOP", "support", [
        "Response SLA", "Ticket Prioritization", "Escalation Procedure",
        "Communication Standards", "Refund and Credit Policy", "Knowledge Base",
        "After-Hours Coverage", "Quality Assurance", "Complaint Handling",
        "Churn Prevention", "Tooling and Macros", "Reporting and Metrics",
    ]),
    ("Product Operations Guide", "product", [
        "Roadmap Governance", "Supported Platforms", "Launch Schedule",
        "Hardware Compatibility", "Upcoming Features", "Deprecation Policy",
        "Beta Program", "Pricing and Packaging", "Localization", "Accessibility",
        "Telemetry and Analytics", "Partner Integrations",
    ]),
    ("Finance and Procurement Policy", "finance", [
        "Purchase Approvals", "Vendor Onboarding", "Expense Categories",
        "Budget Cycle", "Invoice Processing", "Travel Reimbursement",
        "Capital Expenditure", "Petty Cash", "Corporate Cards", "Revenue Recognition",
        "Audit Trail", "Cost Allocation",
    ]),
    ("IT and Infrastructure Policy", "it", [
        "Acceptable Use", "Device Encryption", "Software Approval",
        "Account Provisioning", "Network and Wi-Fi", "Data Backup",
        "Asset Inventory", "Patch Management", "Cloud Governance",
        "Remote Access", "Printer and Peripheral", "Disaster Recovery",
    ]),
    ("Data Governance and Privacy", "data", [
        "Data Classification", "Retention Schedule", "Subject Access Requests",
        "Consent Management", "Data Sharing", "Anonymization", "Breach Notification",
        "Cross-Border Transfer", "Records of Processing", "Data Quality",
        "Lineage and Catalog", "Privacy by Design",
    ]),
]

ROLES = [
    "the security team", "a direct manager", "the on-call engineer", "the data steward",
    "the compliance officer", "the people operations team", "the platform team",
    "the finance controller", "the support lead", "the product manager", "the IT desk",
    "a department head", "the legal team", "the privacy office",
]
SYSTEMS = [
    "the production cluster", "the identity provider", "the artifact registry",
    "the data warehouse", "the ticketing system", "the backup vault",
    "the CI pipeline", "the billing platform", "the document store", "the VPN gateway",
    "the monitoring stack", "the customer portal",
]
PERIODS = ["quarterly", "annually", "every two weeks", "monthly", "each sprint",
           "twice a year", "every business day", "on a rolling basis"]
ADJ = ["documented", "approved", "auditable", "least-privilege", "encrypted",
       "reviewed", "standardized", "automated", "monitored", "compliant"]

SENTENCES = [
    "All requests must be {adj} and approved by {role} before they take effect.",
    "{system_cap} is reviewed {period} to confirm it meets the current standard.",
    "Exceptions require written sign-off from {role} and are logged for audit.",
    "Every change is recorded in {system} so that actions remain attributable.",
    "Staff are expected to follow this procedure without deviation unless {role} grants a waiver.",
    "Where this policy conflicts with a contractual obligation, the stricter requirement applies.",
    "Records related to this section are kept {adj} and made available to {role} on request.",
    "Training on this topic is delivered {period} and tracked to completion.",
    "Non-compliance may result in escalation to {role} and corrective action.",
    "Owners must keep {system} aligned with the thresholds described in this section.",
    "Access to {system} is granted on a {adj} basis and revoked when no longer required.",
    "The process is tested {period} to verify that controls operate as intended.",
    "Any incident affecting {system} is triaged by {role} within the stated window.",
    "Metrics for this area are reported {period} and reviewed by {role}.",
    "Supporting runbooks are maintained alongside this document and kept {adj}.",
]


def cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def sentence(rng: random.Random) -> str:
    t = rng.choice(SENTENCES)
    return t.format(
        adj=rng.choice(ADJ),
        role=rng.choice(ROLES),
        period=rng.choice(PERIODS),
        system=rng.choice(SYSTEMS),
        system_cap=cap(rng.choice(SYSTEMS)),
    )


def paragraph(rng: random.Random, n: int) -> str:
    return " ".join(sentence(rng) for _ in range(n))


def section_body(rng: random.Random, heading: str, anchor: str | None,
                 word_budget: int = 380) -> str:
    """Build a multi-paragraph section body of ~word_budget words.

    Generates paragraphs until the word budget is met, so document length is
    controllable (lets the corpus actually span 15-80 pages). Injects the anchor
    sentence into a middle paragraph if given.
    """
    h = heading.lower()
    intro_templates = [
        f"This section defines how {h} is governed across the organization.",
        f"The following standards apply to {h} and are mandatory for all teams.",
        f"{cap(h)} is managed according to the rules described below.",
        f"This part of the document covers {h} and the controls around it.",
        f"Responsibilities and limits for {h} are set out in this section.",
        f"{cap(h)} follows the practices outlined here and is reviewed regularly.",
        f"Teams handling {h} must comply with the requirements that follow.",
    ]
    intro = rng.choice(intro_templates) + " " + paragraph(rng, rng.randint(3, 5))
    paras = [intro]
    words = len(intro.split())
    while words < word_budget:
        p = paragraph(rng, rng.randint(4, 7))
        paras.append(p)
        words += len(p.split())
    if anchor:
        idx = rng.randint(1, len(paras) - 1)
        paras[idx] = anchor + " " + paras[idx]
    return "\n\n".join(paras)


# ── Anchor specification (ground truth for the RAG needle tests) ──────────────
# Each anchor: distinctive topic phrase + globally-unique value, with a sentence
# template that embeds both. The test transcript references them.

EDIT_ANCHORS = [
    ("cold-storage archival window", "47 days", "30 days",
     "Records moved to cold storage use a {old} archival window before deletion."),
    ("incident bridge auto-timeout", "53 minutes", "20 minutes",
     "The incident bridge has an auto-timeout of {old} when no updates are posted."),
    ("privileged session recording limit", "236 gigabytes", "500 gigabytes",
     "Privileged session recordings are capped at {old} per quarter."),
    ("vendor reassessment interval", "19 months", "12 months",
     "High-risk vendors are reassessed on a {old} interval."),
    ("refund auto-approval ceiling", "418 dollars", "250 dollars",
     "Support agents may auto-approve refunds up to {old} without escalation."),
    ("beta cohort size cap", "1,740 participants", "5,000 participants",
     "Each closed beta cohort is capped at {old}."),
    ("learning stipend", "2,360 dollars", "3,000 dollars",
     "Each employee receives an annual learning stipend of {old}."),
    ("purchase order fast-track threshold", "7,950 dollars", "10,000 dollars",
     "Purchase orders under {old} qualify for the fast-track approval lane."),
    ("warm standby failover budget", "73 seconds", "30 seconds",
     "The warm standby must complete failover within {old}."),
    ("data subject request deadline", "27 days", "30 days",
     "Subject access requests are fulfilled within {old} of verification."),
    ("artifact retention depth", "184 builds", "50 builds",
     "The artifact registry retains the last {old} per service."),
    ("on-call acknowledgement window", "11 minutes", "5 minutes",
     "On-call engineers acknowledge pages within {old} during business hours."),
    ("badge re-enrollment grace", "62 hours", "24 hours",
     "Lost badges must be re-enrolled within a {old} grace period."),
    ("telemetry sampling rate", "3.7 percent", "10 percent",
     "Client telemetry is sampled at {old} of sessions."),
    ("snapshot replication lag ceiling", "94 seconds", "30 seconds",
     "Cross-region snapshot replication lag must stay below {old}."),
    ("contractor access expiry", "208 days", "90 days",
     "Contractor accounts automatically expire after {old}."),
]

REMOVE_NAMED_ANCHORS = [
    ("Legacy Fax Intake Procedure",
     "drop the legacy fax intake procedure section"),
    ("Pager Duty Carbon-Copy Rule",
     "remove the pager duty carbon-copy rule section"),
    ("Floppy Disk Archival Standard",
     "delete the floppy disk archival standard section"),
    ("On-Site Smoking Shelter Policy",
     "get rid of the on-site smoking shelter policy section"),
]

RENAME_BRAND = "Vantcorex Robotics"   # planted across a subset; renamed corpus-wide


def fmt_for(i: int) -> str:
    return ["md", "txt", "docx", "odt"][i % 4]


def build_corpus():
    DOCS.mkdir(parents=True, exist_ok=True)
    for old in DOCS.glob("*"):
        if old.is_file():
            old.unlink()

    manifest = {"documents": [], "anchors": []}
    rng_global = random.Random(SEED)

    # Decide which docs carry which anchors. Spread edit anchors EVENLY across
    # all 4 formats (doc index i -> format i%4) so the needle test covers each
    # format, not just .md.
    edit_doc_ids = [8, 9, 10, 11, 12, 13, 14, 15, 28, 29, 30, 31, 40, 41, 42, 43][:len(EDIT_ANCHORS)]
    remove_doc_ids = [44, 5, 46, 23][:len(REMOVE_NAMED_ANCHORS)]  # md, txt, docx, odt
    rename_doc_ids = set(range(2, 100, 7))  # ~14 docs share the brand

    docs_meta = []
    for i in range(100):
        fmt = fmt_for(i)
        org = ORGS[i % len(ORGS)]
        title_suffix, dom_key, headings_pool = DOMAINS[i % len(DOMAINS)]
        pages = rng_global.randint(15, 80)
        docs_meta.append(dict(i=i, fmt=fmt, org=org, title_suffix=title_suffix,
                              dom_key=dom_key, headings_pool=headings_pool, pages=pages))

    for meta in docs_meta:
        i, fmt = meta["i"], meta["fmt"]
        rng = random.Random(SEED + i * 101)
        org, dom_key = meta["org"], meta["dom_key"]
        title = f"{org} {meta['title_suffix']}"
        target_words = meta["pages"] * WORDS_PER_PAGE

        # Choose section count and a per-section word budget so the rendered doc
        # actually lands near its page target (15-80 pages).
        per_section_budget = rng.randint(360, 520)
        n_sections = max(12, min(72, target_words // per_section_budget))
        base = meta["headings_pool"]
        headings = []
        round_no = 0
        while len(headings) < n_sections:
            for h in base:
                headings.append(h if round_no == 0 else f"{h} (Part {round_no + 1})")
                if len(headings) >= n_sections:
                    break
            round_no += 1

        # Assign anchors for this doc.
        anchor_specs = []  # (heading, anchor_sentence, record)
        if i in edit_doc_ids:
            a = EDIT_ANCHORS[edit_doc_ids.index(i)]
            topic, old, new, tmpl = a
            h = headings[len(headings) // 2]
            sent = tmpl.format(old=old)
            anchor_specs.append((h, sent))
            manifest["anchors"].append(dict(
                kind="edit", file=None, fmt=fmt, doc_index=i, section_heading=h,
                topic=topic, old_value=old, new_value=new,
                transcript=f"For the {org} {meta['title_suffix'].lower()}, change the "
                           f"{topic} from {old} to {new}.",
                transcript_blind=f"We need to change the {topic} from {old} to {new}.",
            ))
        if i in remove_doc_ids:
            rh, instr = REMOVE_NAMED_ANCHORS[remove_doc_ids.index(i)]
            headings.append(rh)  # add the to-be-removed named section at the end
            anchor_specs.append((rh, f"This section documents the {rh.lower()}, retained "
                                     f"only for historical reference."))
            manifest["anchors"].append(dict(
                kind="remove_named", file=None, fmt=fmt, doc_index=i, section_heading=rh,
                topic=rh, old_value=None, new_value=None,
                transcript=f"In the {org} {meta['title_suffix'].lower()}, {instr}.",
            ))
        brand_in_doc = i in rename_doc_ids

        # Render body text.
        sections = []  # (heading, body)
        # Title/intro section first
        intro_anchor = None
        intro_body = (
            f"This document is the official {meta['title_suffix'].lower()} for {org}. "
            + paragraph(rng, 4)
        )
        if brand_in_doc:
            intro_body = (f"{org} operates as a division of {RENAME_BRAND}. " + intro_body)
        sections.append((title, intro_body))

        anchor_map = {h: s for h, s in anchor_specs}
        for h in headings:
            sec_anchor = anchor_map.get(h)
            body = section_body(rng, h, sec_anchor, word_budget=per_section_budget)
            if brand_in_doc and rng.random() < 0.15:
                body = body + f"\n\nThis standard is coordinated with {RENAME_BRAND} group policy."
            sections.append((h, body))

        path = write_document(fmt, meta, title, sections)
        rel = str(path.relative_to(HERE))
        # backfill file path into manifest anchors for this doc
        for a in manifest["anchors"]:
            if a["doc_index"] == i:
                a["file"] = rel
        manifest["documents"].append(dict(
            doc_index=i, file=rel, fmt=fmt, org=org, kind=meta["title_suffix"],
            target_pages=meta["pages"], sections=len(sections),
            has_brand=brand_in_doc,
        ))

    # Resolve every anchor's section_heading to the ACTUAL chunked heading. txt
    # renders headings as "SECTION N. UPPERCASE", so the raw name won't match;
    # chunk each anchor doc once and record the real heading for exact ground truth.
    from rag.chunker import chunk_document  # noqa: PLC0415
    for a in manifest["anchors"]:
        chunks = chunk_document(str(HERE / a["file"]))
        if a["kind"] == "edit":
            sec = next((c for c in chunks if a["old_value"] in c.content), None)
        else:  # remove_named: heading contains the raw name
            sec = next((c for c in chunks
                        if a["section_heading"].lower() in c.section_heading.lower()), None)
        if sec is not None:
            a["section_heading"] = sec.section_heading
        else:
            print(f"  WARN: could not resolve anchor heading in {a['file']}")

    # Rename test record
    manifest["rename_test"] = dict(
        kind="rename", brand=RENAME_BRAND, new_value="Helios Automata",
        doc_count=sum(1 for d in manifest["documents"] if d["has_brand"]),
        transcript=f"Company-wide announcement: we are renaming {RENAME_BRAND} to "
                   f"Helios Automata. Update the documents to reflect the new name.",
    )

    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# ── Format writers ────────────────────────────────────────────────────────────

def write_document(fmt, meta, title, sections):
    i = meta["i"]
    stem = f"{meta['dom_key']}_{meta['org'].split()[0].lower()}_{i:03d}"
    stem = re.sub(r"[^a-z0-9_]+", "", stem)
    if fmt == "md":
        return _write_md(DOCS / f"{stem}.md", title, sections)
    if fmt == "txt":
        return _write_txt(DOCS / f"{stem}.txt", title, sections)
    if fmt == "docx":
        return _write_docx(DOCS / f"{stem}.docx", title, sections)
    if fmt == "odt":
        return _write_odt(DOCS / f"{stem}.odt", title, sections)
    raise ValueError(fmt)


def _write_md(path, title, sections):
    out = []
    for idx, (h, body) in enumerate(sections):
        out.append(f"# {h}\n\n{body}\n")
    path.write_text("\n".join(out), encoding="utf-8")
    return path


def _write_txt(path, title, sections):
    out = []
    for idx, (h, body) in enumerate(sections):
        if idx == 0:
            out.append(h.upper())            # title line (ALL CAPS -> heading)
        else:
            out.append(f"SECTION {idx}. {h.upper()}")
        out.append(body)
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")
    return path


def _write_docx(path, title, sections):
    import docx
    d = docx.Document()
    for idx, (h, body) in enumerate(sections):
        d.add_heading(h, level=0 if idx == 0 else 1)
        for para in body.split("\n\n"):
            d.add_paragraph(para)
    d.save(str(path))
    return path


def _write_odt(path, title, sections):
    from odf.opendocument import OpenDocumentText
    from odf.text import H, P
    doc = OpenDocumentText()
    for idx, (h, body) in enumerate(sections):
        doc.text.addElement(H(outlinelevel=1, text=h))
        for para in body.split("\n\n"):
            doc.text.addElement(P(text=para))
    doc.save(str(path))
    return path


if __name__ == "__main__":
    m = build_corpus()
    docs = m["documents"]
    print(f"generated {len(docs)} documents into {DOCS}")
    from collections import Counter
    print("formats:", dict(Counter(d["fmt"] for d in docs)))
    print("anchors:", len(m["anchors"]), "| rename docs:", m["rename_test"]["doc_count"])
