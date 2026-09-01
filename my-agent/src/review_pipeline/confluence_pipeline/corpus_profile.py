"""Who the indexed documents belong to — the missing half of subject attribution.

The attribution guard already refuses to write one party's value into another party's
section, but it decides that from ``subject_scope``, and the extractor could only ever
read that from the SPEAKER's framing. "Our fee is two and a half percent" is
``internal`` no matter who says it — so when the people in the meeting are not the
people whose documents these are, every value they state about themselves is labelled
as belonging to the document owner and lands squarely on the owner's own equivalent
row. The topic matches, the units match, the table shape matches, and the resulting
edit is immaculate. It is also completely wrong.

Fixing that needs one fact the pipeline never had: the name of the organization the
corpus is about. With it, ``internal`` can mean "internal to the document owner"
rather than "internal to whoever is talking", and every existing gate does the rest.

Cheap and scalable by construction:
  * ``LDOC_CORPUS_OWNER`` short-circuits everything — no calls at all.
  * otherwise one ``gpt-4o-mini`` call over a bounded sample of document titles and
    intros, memoized per (namespace, corpus fingerprint) for ``LDOC_CORPUS_PROFILE_TTL_S``
    (default 1h). A busy process derives this once an hour, not once a meeting.
  * every failure path returns ``None``, which restores the exact prompts the pipeline
    used before this module existed. A corpus whose owner cannot be named degrades to
    the old behaviour rather than misattributing anything.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from typing import Any

from agents import Agent
from pydantic import BaseModel

from .llm_runtime import guarded_run

logger = logging.getLogger(__name__)

# How much of the namespace to look at. The owner of a corpus is obvious from a
# handful of document titles and opening paragraphs — reading all 8,000 chunks of a
# 500-page space would buy nothing and cost a lot of listing calls.
_MAX_LIST_PAGES = int(os.getenv("LDOC_CORPUS_PROFILE_LIST_PAGES", "5"))
_LIST_PAGE_SIZE = int(os.getenv("LDOC_CORPUS_PROFILE_LIST_SIZE", "100"))
_MAX_SAMPLE_DOCS = int(os.getenv("LDOC_CORPUS_PROFILE_MAX_DOCS", "12"))
_INTRO_CHARS = int(os.getenv("LDOC_CORPUS_PROFILE_INTRO_CHARS", "400"))
_TTL_S = float(os.getenv("LDOC_CORPUS_PROFILE_TTL_S", "3600"))
# A FAILURE must not be cached for as long as an answer. Losing the owner reverts
# attribution to speaker-relative — the exact bug this module exists to fix — and
# it does so silently, with the pipeline still reporting clean, high-quality
# proposals. One 429 must therefore cost a minute, not an hour. Same for an empty
# sample, which is what a namespace mid-first-index looks like.
_FAILURE_TTL_S = float(os.getenv("LDOC_CORPUS_PROFILE_FAILURE_TTL_S", "60"))
_MODEL = os.getenv("LDOC_CORPUS_PROFILE_MODEL", "gpt-4o-mini")

INSTRUCTIONS = """\
You are told the titles and opening lines of the documents in one organization's
internal knowledge base. Name the organization those documents BELONG TO — the one
whose own policies, terms, processes, people and figures these pages record.

- owner_name: the organization's name as the documents write it. If the documents are
  about several parties (customers, portfolio companies, vendors, competitors), the
  owner is the one whose PERSPECTIVE the documents are written from — the "we" of the
  corpus — not any party merely described in it.
- aliases: other names the same organization goes by in these documents — a short
  form, an abbreviation, a fund or product name that stands in for it. Omit generic
  words. Empty list is fine.
- description: one short factual sentence — what kind of organization it is and what
  these documents cover.

Return only what the documents support. If you genuinely cannot tell whose documents
these are, return an empty owner_name rather than guessing.
"""


class CorpusProfile(BaseModel):
    """The organization an indexed document corpus belongs to."""

    owner_name: str
    aliases: list[str] = []
    description: str = ""

    def prompt_block(self) -> str:
        """Render the owner as a prompt prefix, or "" when there is nothing to say."""
        name = (self.owner_name or "").strip()
        if not name:
            return ""
        alias_list = [a.strip() for a in self.aliases if (a or "").strip() and a.strip() != name]
        also = f" (also referred to as {', '.join(alias_list)})" if alias_list else ""
        desc = f" {self.description.strip()}" if (self.description or "").strip() else ""
        return (
            f"The documents that may be edited belong to {name}{also}.{desc}\n"
            f"Values stated about {name} are that organization's own; values stated "
            f"about anyone else belong to that other party."
        )


# (namespace, fingerprint) -> (profile, expires_at). Process-local by design: it is a
# cache of a stable fact, and losing it on restart costs one cheap call.
_CACHE: dict[tuple[str, str], tuple[CorpusProfile | None, float]] = {}
# namespace -> (fingerprint, sample, expires_at). Caches the Pinecone sampling itself
# so a cache hit costs no calls at all, not just no LLM call.
_SAMPLE_CACHE: dict[str, tuple[str, list[tuple[str, str]], float]] = {}


def _parse_env_owner() -> CorpusProfile | None:
    name = os.getenv("LDOC_CORPUS_OWNER", "").strip()
    if not name:
        return None
    aliases = [a.strip() for a in os.getenv("LDOC_CORPUS_OWNER_ALIASES", "").split(",") if a.strip()]
    return CorpusProfile(
        owner_name=name,
        aliases=aliases,
        description=os.getenv("LDOC_CORPUS_OWNER_DESCRIPTION", "").strip(),
    )


def _sample_corpus(index: Any) -> list[tuple[str, str]]:
    """Return up to ``_MAX_SAMPLE_DOCS`` (doc_title, intro) pairs from the namespace.

    Reads only the ``:0`` chunk of each page — the document's opening section, which is
    where a corpus says who it belongs to. Listing is bounded to ``_MAX_LIST_PAGES``
    pages so this stays O(1) in corpus size.
    """
    dense = index._index("dense")
    if dense is None:
        return []
    namespace = index.namespace

    head_ids: list[str] = []
    token: str | None = None
    for _ in range(max(1, _MAX_LIST_PAGES)):
        kwargs: dict[str, Any] = {"namespace": namespace, "limit": _LIST_PAGE_SIZE}
        if token:
            kwargs["pagination_token"] = token
        page = dense.list_paginated(**kwargs)
        items = getattr(page, "vectors", None) or []
        for item in items:
            cid = getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None)
            if cid and str(cid).endswith(":0"):
                head_ids.append(str(cid))
        pagination = getattr(page, "pagination", None)
        token = getattr(pagination, "next", None) if pagination else None
        if not token:
            break
    if not head_ids:
        return []

    head_ids = sorted(set(head_ids))[:_MAX_SAMPLE_DOCS]
    result = dense.fetch(ids=head_ids, namespace=namespace)
    if isinstance(result, dict):
        records = result.get("vectors") or result.get("records") or {}
    else:
        records = getattr(result, "vectors", None) or getattr(result, "records", None) or {}

    sample: list[tuple[str, str]] = []
    for cid in head_ids:
        rec = (records or {}).get(cid)
        if not rec:
            continue
        if isinstance(rec, dict):
            fields = rec.get("metadata") or rec.get("fields") or {}
        else:
            fields = getattr(rec, "metadata", None) or getattr(rec, "fields", None) or {}
        title = str((fields or {}).get("doc_title") or "").strip()
        intro = " ".join(str((fields or {}).get("content") or "").split())[:_INTRO_CHARS]
        if title or intro:
            sample.append((title, intro))
    return sample


def _fingerprint(sample: list[tuple[str, str]]) -> str:
    return hashlib.sha256("\n".join(sorted(t for t, _ in sample)).encode()).hexdigest()[:16]


def _render_sample(sample: list[tuple[str, str]]) -> str:
    return "\n\n".join(
        f"Document: {title or '(untitled)'}\nOpening: {intro}" for title, intro in sample
    )


async def get_corpus_profile(index: Any) -> CorpusProfile | None:
    """Resolve who the indexed corpus belongs to. Returns None when it cannot be told.

    Never raises: a corpus whose owner cannot be resolved simply leaves every prompt
    exactly as it was before this module existed.
    """
    env_owner = _parse_env_owner()
    if env_owner:
        return env_owner

    namespace = str(getattr(index, "namespace", "") or "")
    now = time.monotonic()
    try:
        cached = _SAMPLE_CACHE.get(namespace)
        if cached and cached[2] > now:
            fingerprint, sample = cached[0], cached[1]
        else:
            sample = await asyncio.to_thread(_sample_corpus, index)
            fingerprint = _fingerprint(sample)
            ttl = _TTL_S if sample else _FAILURE_TTL_S
            _SAMPLE_CACHE[namespace] = (fingerprint, sample, now + ttl)
    except Exception as exc:  # noqa: BLE001
        logger.warning("corpus profile: sampling %r failed (%s); attribution stays speaker-relative",
                       namespace, str(exc)[:160])
        return None
    if not sample:
        logger.info("corpus profile: namespace %r yielded no sample; owner unknown", namespace)
        return None

    key = (namespace, fingerprint)
    hit = _CACHE.get(key)
    if hit and hit[1] > now:
        return hit[0]

    profile: CorpusProfile | None = None
    derivation_failed = False
    try:
        agent = Agent(
            name="CorpusOwnerProfiler",
            model=_MODEL,
            instructions=INSTRUCTIONS,
            output_type=CorpusProfile,
        )
        result = await guarded_run(agent, _render_sample(sample))
        candidate: CorpusProfile = result.final_output
        if (candidate.owner_name or "").strip():
            profile = candidate
            logger.info(
                "corpus profile: namespace %r belongs to %r (aliases=%s) from %d document(s)",
                namespace, profile.owner_name, profile.aliases, len(sample),
            )
        else:
            logger.info("corpus profile: owner not identifiable from %d document(s)", len(sample))
    except Exception:
        logger.warning("corpus profile: derivation failed; attribution stays speaker-relative "
                       "for the next %.0fs, then retried", _FAILURE_TTL_S, exc_info=True)
        profile = None
        derivation_failed = True

    # A model that ran and could not name an owner is a real answer worth caching for
    # the full TTL; a model that never ran is not.
    _CACHE[key] = (profile, now + (_FAILURE_TTL_S if derivation_failed else _TTL_S))
    return profile


def reset_cache() -> None:
    """Drop the memoized profile and sample. For tests and for a forced re-derive."""
    _CACHE.clear()
    _SAMPLE_CACHE.clear()


# ── Whose meeting is this? ────────────────────────────────────────────────────
#
# Knowing who owns the documents is only half the question. The other half is
# whether those owners are even in the room, and it cannot be answered where the
# extractor works: the transcript is processed one 230-word segment at a time, so
# no segment can tell whether the "we" running through the whole meeting is the
# document owner or a visitor. Per-intent attribution therefore only catches a
# value whose owner is named in the same breath ("Meridian charges 2.5%"). It
# cannot catch a founder pitching an investor who says "we need SOC 2" and "our
# competitors are Otter and FlyFS" — no other party is named, so every gate reads
# those as the document owner's own facts and writes them into the owner's
# compliance and competitive-landscape sections.
#
# One look at the whole transcript settles it, and the answer is reusable for every
# intent in the meeting: which organizations do the speakers actually represent,
# and is the document owner among them?

_SCOPE_HEAD_CHARS = int(os.getenv("LDOC_MEETING_SCOPE_HEAD_CHARS", "7000"))
_SCOPE_TAIL_CHARS = int(os.getenv("LDOC_MEETING_SCOPE_TAIL_CHARS", "2000"))
_SCOPE_MODEL = os.getenv("LDOC_MEETING_SCOPE_MODEL", "gpt-4o-mini")

SCOPE_INSTRUCTIONS = """\
You are told which organization owns a set of internal documents, then given a
meeting transcript. Decide whether that organization is a PARTY to this meeting —
whether the people speaking are its own people discussing its own affairs.

- speaker_organizations: every organization the speakers appear to belong to or
  speak for, as the transcript names them. A meeting between two companies has
  two. Use the names as spoken.
- owner_is_party: true when the document owner is one of those organizations —
  when "we"/"our" in this meeting means the document owner. False when the
  speakers all belong to OTHER organizations, however much the subject matter
  overlaps with what the owner's documents cover.
- reasoning: one sentence, citing what in the transcript decided it.

Judging this correctly matters more than being generous:
- Do NOT infer that the owner is present merely because the meeting discusses the
  same INDUSTRY, the same kind of work, or the same vocabulary as its documents. A
  startup pitching an investor talks about funding, valuation, competitors,
  compliance and team — all of which appear in any investment firm's own manuals.
  Shared subject matter is not shared identity.
- An organization that is only DISCUSSED — a company being evaluated, a customer,
  a vendor, a competitor — is not a party. Being a party means the speakers act and
  speak AS that organization.
- If NO organization can be identified for the speakers at all, set
  speaker_organizations to [] and owner_is_party to true: an unmarked "our" in a
  meeting that names no company is the document owner's own, which is the ordinary
  internal-meeting case and must not be disrupted.
"""


class MeetingScope(BaseModel):
    """Whether the document owner is a party to this meeting, or a bystander."""

    owner_is_party: bool = True
    speaker_organizations: list[str] = []
    reasoning: str = ""

    def outsider_entity(self) -> str:
        """The organization to attribute an unmarked "our" to, when not the owner."""
        return self.speaker_organizations[0] if self.speaker_organizations else ""


def _scope_excerpt(transcript: str) -> str:
    """Head + tail of the transcript — who is in the room is established early and
    confirmed at sign-off, and this keeps the call O(1) in meeting length."""
    text = transcript.strip()
    if len(text) <= _SCOPE_HEAD_CHARS + _SCOPE_TAIL_CHARS:
        return text
    return f"{text[:_SCOPE_HEAD_CHARS]}\n\n[…]\n\n{text[-_SCOPE_TAIL_CHARS:]}"


async def resolve_meeting_scope(
    transcript: str, profile: CorpusProfile | None
) -> MeetingScope:
    """Decide whether the corpus owner is a party to this meeting.

    Fails OPEN — any error, or no known owner, returns ``owner_is_party=True``,
    which is exactly the behaviour the pipeline had before this check existed.
    """
    if not profile or not (profile.owner_name or "").strip() or not transcript.strip():
        return MeetingScope(owner_is_party=True, reasoning="owner unknown; scope not evaluated")
    try:
        agent = Agent(
            name="MeetingScopeResolver",
            model=_SCOPE_MODEL,
            instructions=SCOPE_INSTRUCTIONS,
            output_type=MeetingScope,
        )
        prompt = (
            f"{profile.prompt_block()}\n\n"
            f"Meeting transcript:\n\n{_scope_excerpt(transcript)}"
        )
        result = await guarded_run(agent, prompt)
        scope: MeetingScope = result.final_output
        logger.info(
            "meeting scope: owner %r is%s a party — speakers represent %s (%s)",
            profile.owner_name, "" if scope.owner_is_party else " NOT",
            scope.speaker_organizations or "[no organization identified]",
            (scope.reasoning or "")[:160],
        )
        return scope
    except Exception:
        logger.warning(
            "meeting scope: resolution failed; assuming the owner IS a party "
            "(pre-existing behaviour)", exc_info=True,
        )
        return MeetingScope(owner_is_party=True, reasoning="scope resolution failed")
