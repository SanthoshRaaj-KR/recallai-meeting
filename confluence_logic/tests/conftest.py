"""Shared deterministic pytest fixtures for Phase 11 pipeline v3 tests.

All fixtures are credential-free (no OPENAI_API_KEY, NEO4J_URI, PINECONE_API_KEY
required). They mirror the inline-stub style used in
``tests/e2e_proposal_quality_v2_eval.py`` but are extracted here for reuse
across all ``test_*_v3.py`` files.

Fixture shapes:
  fake_section_corpus  — list of (page_id, section_id, heading, text, terms)
                         rows representing the pipeline's SectionCorpus.
  fake_pinecone_store  — object whose .search(query, top_k) returns canned
                         Pinecone-style match dicts with metadata.page_id and
                         metadata.heading (never match.id as page_id — Pitfall 2).
  fake_graph_rows      — Neo4j record.data()-shaped dicts as returned by
                         query_user_confluence_graph.
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# SectionRow — lightweight named tuple modelling a SectionCorpus row
# ---------------------------------------------------------------------------

class SectionRow(NamedTuple):
    page_id: str
    section_id: str
    heading: str
    text: str
    terms: List[str]


# ---------------------------------------------------------------------------
# fake_section_corpus
# ---------------------------------------------------------------------------

_FAKE_SECTIONS: List[SectionRow] = [
    SectionRow(
        page_id="pg-auth",
        section_id="pg-auth::provider",
        heading="Provider",
        text="We use OAuth for all sign-ins.",
        terms=["oauth", "sign-ins", "authentication"],
    ),
    SectionRow(
        page_id="pg-auth",
        section_id="pg-auth::mfa",
        heading="MFA",
        text="Multi-factor authentication is required for admin access.",
        terms=["mfa", "multi-factor", "admin", "authentication"],
    ),
    SectionRow(
        page_id="pg-soc2",
        section_id="pg-soc2::audit-schedule",
        heading="Audit Schedule",
        text="SOC2 Type II audit is scheduled for Q3 2025.",
        terms=["soc2", "audit", "q3", "2025", "schedule"],
    ),
    SectionRow(
        page_id="pg-security-overview",
        section_id="pg-security-overview::audit-schedule",
        heading="Audit Schedule",
        text="Annual SOC2 audit planned for Q3.",
        terms=["soc2", "audit", "q3", "annual", "security"],
    ),
    SectionRow(
        page_id="pg-deployment",
        section_id="pg-deployment::topology",
        heading="Topology",
        text="We deploy directly to bare-metal hosts via Ansible.",
        terms=["ansible", "bare-metal", "deployment", "hosts"],
    ),
    SectionRow(
        page_id="pg-deployment",
        section_id="pg-deployment::rollout",
        heading="Rollout Strategy",
        text="Rolling deploys with canary gates before full rollout.",
        terms=["canary", "rolling", "rollout", "deploy"],
    ),
    SectionRow(
        page_id="pg-runbook",
        section_id="pg-runbook::setup",
        heading="Setup",
        text="Install dependencies via pip. Run the migrations.",
        terms=["setup", "pip", "dependencies", "migrations"],
    ),
    SectionRow(
        page_id="pg-runbook",
        section_id="pg-runbook::oncall",
        heading="On-call",
        text="On-call rotation uses PagerDuty. Escalation path: L1 → L2 → Lead.",
        terms=["oncall", "pagerduty", "escalation", "rotation"],
    ),
    SectionRow(
        page_id="pg-team-roster",
        section_id="pg-team-roster::members",
        heading="Members",
        text="Alice, Bob, Carol are the current team members.",
        terms=["alice", "bob", "carol", "team", "members"],
    ),
    SectionRow(
        page_id="pg-onboarding",
        section_id="pg-onboarding::welcome",
        heading="Welcome",
        text="Welcome to the team. Please complete setup before your first sprint.",
        terms=["welcome", "onboarding", "setup", "sprint"],
    ),
    SectionRow(
        page_id="pg-api-docs",
        section_id="pg-api-docs::auth-header",
        heading="Auth Header",
        text="Pass the Bearer token in the Authorization header.",
        terms=["bearer", "token", "authorization", "header", "auth"],
    ),
    SectionRow(
        page_id="pg-api-docs",
        section_id="pg-api-docs::rate-limits",
        heading="Rate Limits",
        text="The API allows 100 requests per minute per client.",
        terms=["rate", "limits", "100", "requests", "minute"],
    ),
    SectionRow(
        page_id="pg-infra",
        section_id="pg-infra::regions",
        heading="Regions",
        text="Production runs in us-east-1 and eu-west-1.",
        terms=["regions", "us-east-1", "eu-west-1", "production"],
    ),
    SectionRow(
        page_id="pg-incident",
        section_id="pg-incident::severity",
        heading="Severity Levels",
        text="P0: site down. P1: degraded. P2: minor. P3: cosmetic.",
        terms=["severity", "p0", "p1", "p2", "p3", "incident"],
    ),
    SectionRow(
        page_id="pg-db",
        section_id="pg-db::connection",
        heading="Connection Pooling",
        text="Use PgBouncer for connection pooling with max_client_conn=100.",
        terms=["pgbouncer", "connection", "pooling", "postgres"],
    ),
]


@pytest.fixture
def fake_section_corpus() -> List[SectionRow]:
    """Return a deterministic list of SectionRow tuples.

    Suitable for constructing an in-process BM25 index or passing directly
    to retrieval stages under test.  No credentials required.
    """
    return list(_FAKE_SECTIONS)


# ---------------------------------------------------------------------------
# fake_pinecone_store
# ---------------------------------------------------------------------------

class _FakePineconeStore:
    """Minimal stand-in for PineconeStore used by retrieval stages.

    Returns canned Pinecone-style match dicts.  Critically, the page_id is in
    ``metadata.page_id``, NOT ``match.id`` — avoids the Pitfall-2 footgun
    where the vector id (a chunk id) is silently used as the page id.
    """

    def __init__(self, canned_matches: List[Dict[str, Any]]) -> None:
        self._canned = canned_matches

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:  # noqa: ARG002
        """Return up to top_k canned matches regardless of query."""
        return self._canned[:top_k]

    # Async variant for stages that await search results.
    async def asearch(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        return self.search(query, top_k)


_DEFAULT_PINECONE_MATCHES: List[Dict[str, Any]] = [
    {
        "id": "chunk-001",  # chunk id — NOT the page_id
        "score": 0.92,
        "metadata": {
            "page_id": "pg-auth",
            "heading": "Provider",
            "text": "We use OAuth for all sign-ins.",
        },
    },
    {
        "id": "chunk-002",
        "score": 0.85,
        "metadata": {
            "page_id": "pg-soc2",
            "heading": "Audit Schedule",
            "text": "SOC2 Type II audit is scheduled for Q3 2025.",
        },
    },
    {
        "id": "chunk-003",
        "score": 0.78,
        "metadata": {
            "page_id": "pg-security-overview",
            "heading": "Audit Schedule",
            "text": "Annual SOC2 audit planned for Q3.",
        },
    },
    {
        "id": "chunk-004",
        "score": 0.71,
        "metadata": {
            "page_id": "pg-deployment",
            "heading": "Topology",
            "text": "We deploy directly to bare-metal hosts via Ansible.",
        },
    },
    {
        "id": "chunk-005",
        "score": 0.65,
        "metadata": {
            "page_id": "pg-runbook",
            "heading": "Setup",
            "text": "Install dependencies via pip.",
        },
    },
]


@pytest.fixture
def fake_pinecone_store() -> _FakePineconeStore:
    """Return a credential-free fake PineconeStore with canned search results.

    Results include ``metadata.page_id`` and ``metadata.heading`` — the correct
    fields to read (never use ``match['id']`` as the page_id).
    """
    return _FakePineconeStore(_DEFAULT_PINECONE_MATCHES)


# ---------------------------------------------------------------------------
# fake_graph_rows
# ---------------------------------------------------------------------------

_DEFAULT_GRAPH_ROWS: List[Dict[str, Any]] = [
    {
        "page_id": "pg-auth",
        "title": "Authentication",
        "heading": "Provider",
        "text": "We use OAuth for all sign-ins.",
        "terms": ["oauth", "sign-ins", "authentication"],
        "kind": "CfSection",
    },
    {
        "page_id": "pg-auth",
        "title": "Authentication",
        "heading": "MFA",
        "text": "Multi-factor authentication is required for admin access.",
        "terms": ["mfa", "multi-factor", "admin", "authentication"],
        "kind": "CfSection",
    },
    {
        "page_id": "pg-soc2",
        "title": "SOC2 Compliance",
        "heading": "Audit Schedule",
        "text": "SOC2 Type II audit is scheduled for Q3 2025.",
        "terms": ["soc2", "audit", "q3", "2025"],
        "kind": "CfSection",
    },
    {
        "page_id": "pg-security-overview",
        "title": "Security Overview",
        "heading": "Audit Schedule",
        "text": "Annual SOC2 audit planned for Q3.",
        "terms": ["soc2", "audit", "q3", "annual"],
        "kind": "CfSection",
    },
    {
        "page_id": "pg-deployment",
        "title": "Deployment Architecture",
        "heading": "Topology",
        "text": "We deploy directly to bare-metal hosts via Ansible.",
        "terms": ["ansible", "bare-metal", "deployment"],
        "kind": "CfSection",
    },
    {
        "page_id": "pg-runbook",
        "title": "Engineering Runbook",
        "heading": "Setup",
        "text": "Install dependencies via pip.",
        "terms": ["setup", "pip", "dependencies"],
        "kind": "CfSection",
    },
    {
        "page_id": "pg-api-docs",
        "title": "API Documentation",
        "heading": "Auth Header",
        "text": "Pass the Bearer token in the Authorization header.",
        "terms": ["bearer", "token", "authorization"],
        "kind": "CfSection",
    },
    {
        "page_id": "pg-incident",
        "title": "Incident Management",
        "heading": "Severity Levels",
        "text": "P0: site down. P1: degraded.",
        "terms": ["severity", "p0", "p1", "incident"],
        "kind": "CfSection",
    },
]


@pytest.fixture
def fake_graph_rows() -> List[Dict[str, Any]]:
    """Return Neo4j record.data()-shaped dicts as query_user_confluence_graph returns.

    Each row has: page_id, title, heading, text, terms, kind.
    No NEO4J_URI required.
    """
    return list(_DEFAULT_GRAPH_ROWS)
