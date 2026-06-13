from __future__ import annotations

import asyncio
import datetime as dt
import html
import json
import logging
import os
import re
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, ValidationError

from memory_compaction import add_memory_context

from .confluence import HybridConfluenceClient
from .models import ChangeIntent, EditMode, ExtractedMeeting, PageCandidate, Proposal
from .rag import ConfluenceVectorIndex, VectorSearchHit
from .text_utils import (
    append_new_section,
    extract_sections,
    format_transcript,
    html_to_text,
    insert_html_in_section,
    normalize_for_match,
    normalize_ws,
    replace_task_status_in_storage,
    replace_text_in_storage,
    task_body_from_label,
    transcript_highlights,
)

logger = logging.getLogger(__name__)

EmitFn = Callable[[dict[str, Any]], Awaitable[None]]


# ── Cerebras response schemas ────────────────────────────────────────────────

class _SummaryResponse(BaseModel):
    title: str = ""
    summary: str = ""
    key_topics: list[str] = []


class _DecisionsResponse(BaseModel):
    decisions: list[str] = []


class _ActionItem(BaseModel):
    description: str
    owner: str | None = None
    due: str | None = None


class _ActionItemsResponse(BaseModel):
    action_items: list[_ActionItem] = []


class _MOMEntry(BaseModel):
    topic: str
    summary: str = ""


class _MOMResponse(BaseModel):
    mom: list[_MOMEntry] = []


def _human_only(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip Jarvis's own spoken replies from the transcript.

    Jarvis replies are posted back to the transcript with speaker='Jarvis' so the
    live agent has full context. But those replies embed RAG-retrieved Confluence
    content — including them in summary/proposal generation causes the LLM to treat
    wiki text as meeting discussion.
    """
    return [
        e for e in transcript
        if (e.get("participant") or e.get("speaker") or "").strip().lower() != "jarvis"
    ]


def _extract_context_lines(section_text: str, before_content: str, n: int = 5) -> tuple[str, str]:
    """Return up to n lines before and after before_content within section_text."""
    lines = section_text.splitlines()
    before_lines = before_content.strip().splitlines()
    if not before_lines or not lines:
        return "", ""
    first_line = before_lines[0].strip()
    start_idx = next(
        (i for i, l in enumerate(lines) if first_line in l.strip() or l.strip() in first_line),
        None,
    )
    if start_idx is None:
        return "", ""
    end_idx = start_idx + len(before_lines)
    ctx_before = "\n".join(lines[max(0, start_idx - n):start_idx])
    ctx_after = "\n".join(lines[end_idx:end_idx + n])
    return ctx_before, ctx_after


def _extract_json(raw: str) -> str:
    """Strip markdown code fences so json.loads can handle LLM output reliably."""
    stripped = raw.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.MULTILINE)
    stripped = re.sub(r"\s*```\s*$", "", stripped, flags=re.MULTILINE)
    return stripped.strip()


# ── Cerebras structured-output schemas ───────────────────────────────────────
# Cerebras supports json_schema response_format with strict=True, which uses
# constrained decoding to guarantee schema-valid JSON — no prompt-level JSON
# instructions or post-parse fallbacks needed for these 4 calls.
# Constraints: additionalProperties must be False on every object when strict=True;
# no minItems/maxItems; nullable fields use anyOf.

_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "summary_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "summary": {"type": "string"},
                "key_topics": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "summary", "key_topics"],
            "additionalProperties": False,
        },
    },
}

_DECISIONS_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "decisions_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "decisions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["decisions"],
            "additionalProperties": False,
        },
    },
}

_ACTION_ITEMS_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "action_items_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "action_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            # Cerebras strict mode does not support anyOf/nullable.
                            # Use plain string; empty string means unassigned/unknown.
                            "owner": {"type": "string"},
                            "due": {"type": "string"},
                        },
                        "required": ["description", "owner", "due"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["action_items"],
            "additionalProperties": False,
        },
    },
}

_MOM_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "mom_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "mom": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string"},
                            "summary": {"type": "string"},
                        },
                        "required": ["topic", "summary"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["mom"],
            "additionalProperties": False,
        },
    },
}


class ProposalPipeline:
    """Transcript -> atomic intents -> Confluence page retrieval -> proposal cards.

    This is intentionally self-contained under my-agent. It does not import the
    older confluence_logic package.
    """

    _CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
    _CEREBRAS_MODEL = "gpt-oss-120b"

    def __init__(self) -> None:
        self.model = os.getenv("MY_AGENT_REVIEW_MODEL", os.getenv("JARVIS_REVIEW_MODEL", "gpt-5-mini")).strip()
        self.max_candidate_pages = int(os.getenv("MY_AGENT_PIPELINE_MAX_PAGES", "16"))
        self.max_search_terms = int(os.getenv("MY_AGENT_PIPELINE_MAX_SEARCH_TERMS", "24"))
        self.rag_top_k = int(os.getenv("MY_AGENT_PIPELINE_RAG_TOP_K", "8"))
        self._client: HybridConfluenceClient | None = None
        self._rag: ConfluenceVectorIndex | None = None
        self._openai: OpenAI | None = None
        self._cerebras: OpenAI | None = None
        self.last_diagnostics: list[dict[str, Any]] = []

    async def run(
        self,
        *,
        session_id: str,
        transcript: list[dict[str, Any]],
        query: str | None = None,
        memory_context: str | None = None,
        emit: EmitFn | None = None,
    ) -> tuple[ExtractedMeeting, list[dict[str, Any]]]:
        self.last_diagnostics = []

        async def _emit(event: dict[str, Any]) -> None:
            if emit:
                await emit(event)

        await _emit({"type": "stage_start", "stage": "transcript_source"})
        transcript_text = add_memory_context(format_transcript(transcript), memory_context)
        if not transcript_text:
            meeting = ExtractedMeeting(
                title="Meeting Review",
                summary="No transcript was captured for this session yet.",
            )
            return meeting, []

        # Stage 1: Extract meeting topics and loose intents (no forced old_value).
        await _emit({"type": "stage_start", "stage": "fact_extraction"})
        meeting = await self._extract_meeting(transcript, transcript_text, query=query)
        fallback = self._intents_from_action_items(meeting)
        if fallback:
            meeting.change_intents = self._merge_intents(meeting.change_intents, fallback)[:30]

        if not meeting.change_intents:
            return meeting, []

        # Stage 2: Multi-query RAG + rerank — find the most relevant Confluence sections.
        await _emit({"type": "stage_start", "stage": "rag_retrieval"})
        intent_sections = await self._retrieve_all_sections(meeting.change_intents)
        await _emit({
            "type": "stage_progress",
            "stage": "rag_retrieval",
            "intent_count": len(meeting.change_intents),
            "sections_found": sum(len(hits) for _, hits in intent_sections),
        })

        # Stage 3: Batch-fetch the live page sections identified by RAG.
        await _emit({"type": "stage_start", "stage": "section_fetch"})
        section_cache, page_cache = await self._build_section_cache(intent_sections)
        await _emit({
            "type": "stage_progress",
            "stage": "section_fetch",
            "pages_fetched": len(page_cache),
            "sections_cached": len(section_cache),
        })

        # Stage 4: Grounded proposal drafting — LLM sees transcript evidence + live section.
        await _emit({"type": "stage_start", "stage": "drafting"})
        proposals = await asyncio.to_thread(
            self._draft_grounded_proposals_sync,
            session_id=session_id,
            meeting=meeting,
            intent_sections=intent_sections,
            section_cache=section_cache,
            page_cache=page_cache,
            query=query,
        )

        # Stage 5: Dedup + single adversarial verification pass.
        await _emit({"type": "stage_start", "stage": "verification"})
        verified = self._verify_and_dedupe(proposals)
        verified = await self._adversarial_verify(meeting, verified, transcript_text)
        verified = await self._coverage_audit(meeting, verified, transcript_text)

        if self.last_diagnostics:
            await _emit({
                "type": "stage_progress",
                "stage": "verification",
                "intent_diagnostics": self.last_diagnostics,
            })
        for proposal in verified:
            await _emit({"type": "proposal_ready", **proposal})
        return meeting, verified

    async def propose_custom_new_page(
        self,
        *,
        session_id: str,
        transcript: list[dict[str, Any]],
        query: str,
        memory_context: str | None = None,
    ) -> tuple[ExtractedMeeting, list[dict[str, Any]]]:
        """Draft a create-page proposal from explicit user guidance.

        This is intentionally separate from the normal edit-heavy pipeline. The
        custom dialog's "add new page" mode should create one cohesive page,
        using current meeting context plus style samples from existing pages.
        """
        transcript_text = add_memory_context(format_transcript(transcript), memory_context)
        meeting = await self._extract_meeting(transcript, transcript_text, query=query)
        style_pages = await self._sample_style_pages(query=query, meeting=meeting)
        try:
            proposal = await asyncio.to_thread(
                self._draft_styled_new_page_sync,
                session_id,
                meeting,
                transcript_text,
                query,
                style_pages,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Styled new-page drafting failed; using fallback: %s", exc)
            proposal = self._fallback_styled_new_page(session_id, meeting, query)
        proposals = [proposal.to_dict()]
        for page in style_pages:
            await self._index_page_for_rag(page)
        return meeting, proposals

    async def _sample_style_pages(
        self,
        *,
        query: str,
        meeting: ExtractedMeeting,
        limit: int = 4,
    ) -> list[PageCandidate]:
        pages_by_id: dict[str, PageCandidate] = {}
        terms: list[str] = []
        if query:
            terms.append(normalize_ws(query)[:300])
        for intent in meeting.change_intents[:6]:
            for v in [intent.subject, intent.target_hint]:
                if normalize_ws(v):
                    terms.append(normalize_ws(v)[:300])
        terms.extend(normalize_ws(t)[:300] for t in meeting.key_topics[:4] if normalize_ws(t))
        seen: set[str] = set()
        unique_terms = [t for t in terms if t.lower() not in seen and not seen.add(t.lower())]  # type: ignore[func-returns-value]

        for term in unique_terms[:6]:
            try:
                hits = await asyncio.to_thread(self.rag.search, term, limit * 2)
            except Exception:
                continue
            for hit in hits:
                page_id = hit.page_id
                if not page_id or page_id in pages_by_id:
                    continue
                try:
                    page = await self._fetch_page_for_retrieval(page_id, "style_sample")
                except Exception:
                    continue
                pages_by_id[page_id] = page
                if len(pages_by_id) >= limit:
                    return list(pages_by_id.values())

        if len(pages_by_id) < limit:
            try:
                recent = await self.client.list_pages(limit=limit * 2)
            except Exception as exc:
                logger.debug("Could not list style sample pages: %s", exc)
                recent = []
            for result in recent:
                page_id = str(result.get("page_id") or result.get("id") or "")
                if not page_id or page_id in pages_by_id:
                    continue
                try:
                    page = await self._fetch_page_for_retrieval(page_id, "style_recent_page")
                except Exception:
                    continue
                pages_by_id[page_id] = page
                if len(pages_by_id) >= limit:
                    break

        return list(pages_by_id.values())

    def _draft_styled_new_page_sync(
        self,
        session_id: str,
        meeting: ExtractedMeeting,
        transcript_text: str,
        query: str,
        style_pages: list[PageCandidate],
    ) -> Proposal:
        client = self._get_openai()
        title_hint = self._new_page_title_hint(query, meeting)
        style_payload = [
            {
                "title": page.title,
                "space_key": page.space_key,
                "sections": [
                    {
                        "heading": section.get("heading") or "",
                        "text_excerpt": (section.get("text") or "")[:900],
                        "html_excerpt": (section.get("html") or "")[:1200],
                    }
                    for section in (page.sections or extract_sections(page.html))[:8]
                ],
            }
            for page in style_pages[:4]
        ]
        prompt = (
            "You are drafting a new Confluence page from a meeting transcript.\n\n"
            "Step 1 — Determine the page subject:\n"
            "  • If user_request names a SPECIFIC topic (e.g. 'Nova framework', 'Q3 roadmap', "
            "'onboarding checklist') — that topic is the sole subject of the page.\n"
            "  • If user_request is a GENERIC instruction (e.g. 'create a page from the meeting', "
            "'document what we discussed', 'propose updates') — ignore the instruction wording and "
            "instead derive the subject from the meeting's actual content: its decisions, key topics, "
            "and action items. Do NOT write a page about the act of requesting a page.\n\n"
            "Step 2 — Infer style from same_space_style_samples: heading depth, section order, tone, "
            "bullets vs paragraphs, tables, and how concise the pages are. Write the new page in that style.\n\n"
            "Step 3 — Extract relevant facts from the transcript. Include decisions made, action items "
            "assigned, and key conclusions. Do NOT document process meta-commentary or the discussion "
            "about creating this page.\n\n"
            "Return JSON only with: title, body_markdown, rationale. body_markdown must be publishable "
            "page content using markdown-ish syntax: ## headings, ### subheadings, bullet lists, "
            "numbered steps, and simple tables if useful. If an image/diagram would help, insert a "
            "placeholder line like '[IMAGE PLACEHOLDER: describe the exact image needed]'. "
            "Do not invent facts beyond the transcript. If details are missing, include a short "
            "'Open questions' section instead of guessing."
        )
        payload = {
            "user_request": query,
            "title_hint": title_hint,
            "meeting": {
                "title": meeting.title,
                "summary": meeting.summary,
                "key_topics": meeting.key_topics,
                "decisions": meeting.decisions,
                "action_items": meeting.action_items,
            },
            "transcript_excerpt": transcript_text[:18000],
            "same_space_style_samples": style_payload,
        }
        opts: dict[str, Any] = {"model": self.model, "response_format": {"type": "json_object"}}
        if self.model.startswith(("gpt-5", "o1", "o3", "o4")):
            opts["max_completion_tokens"] = 2200
        else:
            opts["max_tokens"] = 2200
            opts["temperature"] = 0.1
        response = client.chat.completions.create(
            **opts,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        data = json.loads(response.choices[0].message.content or "{}")
        title = normalize_ws(str(data.get("title") or title_hint or "Meeting Notes"))
        body = str(data.get("body_markdown") or "").strip()
        if not body:
            return self._fallback_styled_new_page(session_id, meeting, query)
        rationale = normalize_ws(str(data.get("rationale") or "New page drafted from custom user request and meeting context."))
        return Proposal(
            id=str(uuid.uuid4()),
            change_type="create",
            page_id=None,
            page_title=title[:120],
            section_heading="Overview",
            before_content=None,
            after_content=body,
            timestamp=_utcnow(),
            session_id=session_id,
            rationale=rationale,
            generation_query=query,
            transcript_evidence=[],
            confidence="medium",
            risk="review",
            verifier_note=(
                "Custom new-page proposal. Style was inferred from existing Confluence pages; "
                "review placeholders and open questions before accepting."
            ),
            change_summary=f"Create new page '{title[:80]}'",
            confidence_score=0.72,
            confidence_bin="medium",
        )

    def _fallback_styled_new_page(
        self,
        session_id: str,
        meeting: ExtractedMeeting,
        query: str,
    ) -> Proposal:
        title = self._new_page_title_hint(query, meeting)
        body_lines = [
            f"## {title}",
            "",
            meeting.summary or normalize_ws(query) or "Draft page requested from the meeting.",
        ]
        if meeting.decisions:
            body_lines.extend(["", "## Decisions", *[f"- {item}" for item in meeting.decisions[:8]]])
        if meeting.action_items:
            body_lines.append("")
            body_lines.append("## Action items")
            for item in meeting.action_items[:8]:
                if isinstance(item, dict):
                    body_lines.append(f"- {item.get('description') or item}")
                else:
                    body_lines.append(f"- {item}")
        body_lines.extend(["", "## Open questions", "- Confirm any missing owners, dates, links, or diagrams before publishing."])
        return Proposal(
            id=str(uuid.uuid4()),
            change_type="create",
            page_id=None,
            page_title=title[:120],
            section_heading="Overview",
            before_content=None,
            after_content="\n".join(body_lines).strip(),
            timestamp=_utcnow(),
            session_id=session_id,
            rationale="Fallback new page draft from meeting context.",
            generation_query=query,
            confidence="low",
            risk="review",
            verifier_note="Style sampling or LLM drafting was unavailable; review before accepting.",
            change_summary=f"Create new page '{title[:80]}'",
            confidence_score=0.55,
            confidence_bin="low",
        )

    def _new_page_title_hint(self, query: str, meeting: ExtractedMeeting) -> str:
        cleaned = normalize_ws(re.sub(r"(?i)\b(create|add|make|new|page|confluence|document|doc)\b", " ", query or ""))
        cleaned = re.sub(r"[^A-Za-z0-9 /:_-]", " ", cleaned)
        cleaned = normalize_ws(cleaned)
        if cleaned:
            return cleaned[:90].title()
        if meeting.key_topics:
            return normalize_ws(str(meeting.key_topics[0]))[:90].title()
        return (meeting.title or "Meeting Notes")[:90]

    async def _extract_meeting(
        self,
        transcript: list[dict[str, Any]],
        transcript_text: str,
        *,
        query: str | None = None,
    ) -> ExtractedMeeting:
        try:
            data = await asyncio.to_thread(self._extract_meeting_sync, transcript_text, query or "")
            return self._meeting_from_json(data, transcript)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM meeting extraction failed; using heuristic fallback: %s", exc)
            return self._heuristic_meeting(transcript, transcript_text)

    def _intents_from_action_items(self, meeting: ExtractedMeeting) -> list[ChangeIntent]:
        """Return ChangeIntents for action items not already covered by an existing intent.

        Every action item from a meeting is a concrete fact that could be reflected in
        Confluence documentation. The page retrieval pipeline will naturally filter out
        ones that have no matching page — no keyword filtering needed here.
        """
        existing_subjects = {
            normalize_for_match(i.subject or i.target_hint or i.instruction)
            for i in meeting.change_intents
            if i.subject or i.target_hint or i.instruction
        }
        new_intents: list[ChangeIntent] = []
        for item in meeting.action_items:
            if isinstance(item, dict):
                desc = str(item.get("description") or "").strip()
                owner = str(item.get("owner") or "").strip()
                due = str(item.get("due") or "").strip()
            else:
                desc = str(item).strip()
                owner = ""
                due = ""
            if not desc:
                continue
            desc_norm = normalize_for_match(desc)
            already_covered = any(
                existing and (desc_norm in existing or existing in desc_norm)
                for existing in existing_subjects
            )
            if already_covered:
                continue
            rationale = f"Action item from meeting: {desc}"
            if owner:
                rationale += f" (owner: {owner})"
            if due:
                rationale += f" (due: {due})"
            new_intents.append(
                ChangeIntent(
                    instruction=desc,
                    subject=desc[:120],
                    target_hint=desc[:120],
                    new_value=desc,
                    action="add",
                    rationale=rationale,
                    source="action_item",
                )
            )
        return new_intents

    def _extract_meeting_sync(self, transcript_text: str, query: str) -> dict[str, Any]:
        client = self._get_openai()
        prompt = (
            "You extract structured facts from meeting transcripts that need to be reflected in "
            "Confluence documentation. Identify every concrete fact, decision, metric, status update, "
            "ownership change, date change, completed/reopened task, or agreed action that could be "
            "documented somewhere in Confluence.\n\n"
            "Return JSON only. Rules:\n"
            "- Capture the FINAL agreed state of each fact, not intermediate suggestions.\n"
            "- One change_intent per distinct fact or update.\n"
            "- subject: the specific thing being changed (metric name, project, task, date, person).\n"
            "- target_hint: the most likely Confluence page or section name where this lives.\n"
            "- new_value: the new fact/value if clearly stated in the meeting — leave empty if not explicit.\n"
            "- action: replace (updating existing content), add (new content to record), "
            "complete_task (task was finished), reopen_task (task was re-opened).\n"
            "- evidence: exact transcript lines that support this fact (up to 3).\n"
            "- Leave old_value empty — the pipeline finds the current value from the live page.\n"
            "- Do NOT invent facts. Skip social niceties, vague filler, and process meta-comments.\n\n"
            "JSON shape:\n"
            "{"
            '"title": string, "summary": string, "key_topics": string[], "decisions": string[], '
            '"action_items": [{"description": string, "owner": string|null, "due": string|null}], '
            '"change_intents": [{"instruction": string, "subject": string, "target_hint": string, '
            '"new_value": string, "action": "replace|add|complete_task|reopen_task", '
            '"rationale": string, "evidence": string[]}]'
            "}"
        )
        payload = {
            "optional_user_guidance": query or None,
            "transcript": transcript_text,
        }
        opts: dict[str, Any] = {"model": self.model, "response_format": {"type": "json_object"}}
        if self.model.startswith(("gpt-5", "o1", "o3", "o4")):
            opts["max_completion_tokens"] = 2500
        else:
            opts["max_tokens"] = 2500
            opts["temperature"] = 0.1
        response = client.chat.completions.create(
            **opts,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        return json.loads(raw)

    def _meeting_from_json(self, data: dict[str, Any], transcript: list[dict[str, Any]]) -> ExtractedMeeting:
        participants = sorted(
            {
                str(entry.get("participant") or entry.get("speaker") or "").strip()
                for entry in transcript
                if str(entry.get("participant") or entry.get("speaker") or "").strip()
            }
        )
        intents = []
        for raw in data.get("change_intents") or []:
            if not isinstance(raw, dict):
                continue
            intent = ChangeIntent(
                instruction=normalize_ws(str(raw.get("instruction") or "")),
                subject=normalize_ws(str(raw.get("subject") or "")),
                target_hint=normalize_ws(str(raw.get("target_hint") or "")),
                old_value=normalize_ws(str(raw.get("old_value") or "")),
                new_value=normalize_ws(str(raw.get("new_value") or "")),
                action=normalize_ws(str(raw.get("action") or "replace")).lower() or "replace",
                rationale=normalize_ws(str(raw.get("rationale") or "")),
                evidence=[normalize_ws(str(x)) for x in raw.get("evidence") or [] if normalize_ws(str(x))],
                source=normalize_ws(str(raw.get("source") or "intent_extractor")) or "intent_extractor",
                page_id=normalize_ws(str(raw.get("page_id") or raw.get("pageId") or "")) or None,
                page_title=normalize_ws(str(raw.get("page_title") or raw.get("pageTitle") or "")) or None,
            )
            if intent.instruction or intent.subject or intent.old_value or intent.new_value:
                intents.append(intent)

        return ExtractedMeeting(
            title=normalize_ws(str(data.get("title") or "Meeting Review")),
            summary=normalize_ws(str(data.get("summary") or "")),
            key_topics=[normalize_ws(str(x)) for x in data.get("key_topics") or [] if normalize_ws(str(x))],
            decisions=[normalize_ws(str(x)) for x in data.get("decisions") or [] if normalize_ws(str(x))],
            action_items=[
                item if isinstance(item, dict) else {"description": str(item), "owner": None, "due": None}
                for item in (data.get("action_items") or [])
            ],
            participants=participants,
            change_intents=intents[:60],
        )

    def _heuristic_meeting(self, transcript: list[dict[str, Any]], transcript_text: str) -> ExtractedMeeting:
        participants = sorted(
            {
                str(entry.get("participant") or entry.get("speaker") or "").strip()
                for entry in transcript
                if str(entry.get("participant") or entry.get("speaker") or "").strip()
            }
        )
        intents: list[ChangeIntent] = []
        # Simple fallback for phrases like "change X from A to B" or "move X from A to B".
        pattern = re.compile(
            r"(?P<verb>change|update|move|push|shift|replace)\s+"
            r"(?P<subject>.{3,120}?)\s+"
            r"(?:from|currently\s+is|is)\s+"
            r"(?P<old>.{2,60}?)\s+"
            r"(?:to|with)\s+"
            r"(?P<new>.{2,80}?)(?:[.?!\n]|$)",
            re.IGNORECASE,
        )
        for match in pattern.finditer(transcript_text):
            intents.append(
                ChangeIntent(
                    instruction=normalize_ws(match.group(0)),
                    subject=normalize_ws(match.group("subject")),
                    target_hint=normalize_ws(match.group("subject")),
                    old_value=normalize_ws(match.group("old")),
                    new_value=normalize_ws(match.group("new")),
                    action="replace",
                    evidence=[normalize_ws(match.group(0))],
                    source="heuristic",
                )
            )
        return ExtractedMeeting(
            title="Meeting Review",
            summary="Transcript captured. LLM extraction was unavailable, so only simple explicit changes were extracted.",
            participants=participants,
            change_intents=intents[:10],
        )

    def _merge_intents(
        self,
        primary: list[ChangeIntent],
        secondary: list[ChangeIntent],
    ) -> list[ChangeIntent]:
        merged: list[ChangeIntent] = []
        by_key: dict[str, ChangeIntent] = {}

        def key(intent: ChangeIntent) -> str:
            return "|".join(
                [
                    normalize_for_match(intent.subject or intent.target_hint)[:80],
                    (intent.action or "replace").lower(),
                    normalize_for_match(intent.old_value)[:80],
                    normalize_for_match(intent.new_value)[:80],
                    str(intent.page_id or "").lower(),
                ]
            )

        for intent in [*primary, *secondary]:
            k = key(intent)
            existing = by_key.get(k)
            if existing is None:
                by_key[k] = intent
                merged.append(intent)
                continue
            if not existing.page_id and intent.page_id:
                existing.page_id = intent.page_id
                existing.page_title = intent.page_title
            if len(intent.evidence) > len(existing.evidence):
                existing.evidence = intent.evidence
            if intent.source not in existing.source:
                existing.source = f"{existing.source}+{intent.source}"
        return merged[:40]

    def _add_diagnostic(
        self,
        intent: ChangeIntent | None,
        reason: str,
        detail: str,
        *,
        severity: str = "review",
        page_title: str | None = None,
    ) -> None:
        item = {
            "reason": reason,
            "detail": detail,
            "severity": severity,
            "instruction": intent.instruction if intent else "",
            "subject": intent.subject if intent else "",
            "old_value": intent.old_value if intent else "",
            "new_value": intent.new_value if intent else "",
            "page_id": intent.page_id if intent else None,
            "page_title": page_title or (intent.page_title if intent else None),
            "source": intent.source if intent else "pipeline",
        }
        key = json.dumps(item, sort_keys=True)
        existing = {json.dumps(x, sort_keys=True) for x in self.last_diagnostics}
        if key not in existing:
            self.last_diagnostics.append(item)

    async def _fetch_page_for_retrieval(self, page_id: str, source: str = "search") -> PageCandidate:
        page = await self.client.fetch_page(page_id)
        page.source = source or page.source
        await self._index_page_for_rag(page)
        return page

    async def _index_page_for_rag(self, page: PageCandidate) -> None:
        if not page.page_id:
            return
        try:
            await asyncio.to_thread(self.rag.upsert_page, page)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Vector RAG indexing failed for page %s: %s", page.page_id, exc)


    # ── New RAG retrieval ─────────────────────────────────────────────────────

    async def _retrieve_all_sections(
        self,
        intents: list[ChangeIntent],
    ) -> list[tuple[ChangeIntent, list[VectorSearchHit]]]:
        """Multi-query RAG + rerank for every intent in parallel."""
        results = await asyncio.gather(
            *[self._retrieve_sections_for_intent(intent) for intent in intents],
            return_exceptions=True,
        )
        out: list[tuple[ChangeIntent, list[VectorSearchHit]]] = []
        for intent, result in zip(intents, results):
            if isinstance(result, Exception):
                logger.warning("Section retrieval failed for %r: %s", intent.subject, result)
                self._add_diagnostic(intent, "rag_retrieval_error", str(result), severity="review")
                out.append((intent, []))
            else:
                if not result:
                    self._add_diagnostic(
                        intent,
                        "no_sections_found",
                        "No matching Confluence sections found for this intent.",
                        severity="review",
                    )
                out.append((intent, result))
        return out

    async def _retrieve_sections_for_intent(
        self,
        intent: ChangeIntent,
    ) -> list[VectorSearchHit]:
        queries = self._queries_for_intent_v2(intent)
        if not queries:
            return []
        return await asyncio.to_thread(
            self.rag.search_with_rerank,
            queries,
            top_k_per_query=25,
            top_n=8,
        )

    def _queries_for_intent_v2(self, intent: ChangeIntent) -> list[str]:
        """Up to 4 diverse query variants for multi-query retrieval."""
        queries: list[str] = []
        seen: set[str] = set()

        def _add(text: str) -> None:
            q = normalize_ws(text)[:300]
            if q and len(q) >= 3 and q.lower() not in seen:
                seen.add(q.lower())
                queries.append(q)

        _add(" ".join(p for p in [intent.subject, intent.target_hint] if normalize_ws(p)))
        _add(intent.instruction)
        _add(" ".join(p for p in [intent.new_value, intent.subject] if normalize_ws(p)))
        _add(intent.target_hint)
        return queries[:4]

    # ── Section fetch ─────────────────────────────────────────────────────────

    async def _build_section_cache(
        self,
        intent_sections: list[tuple[ChangeIntent, list[VectorSearchHit]]],
    ) -> tuple[dict[tuple[str, str], str], dict[str, PageCandidate]]:
        """Batch-fetch unique pages, extract the matched sections into a cache."""
        needed: dict[str, set[str]] = {}
        for _intent, hits in intent_sections:
            for hit in hits:
                if hit.page_id:
                    needed.setdefault(hit.page_id, set()).add(hit.heading)

        if not needed:
            return {}, {}

        page_ids = list(needed.keys())
        fetch_results = await asyncio.gather(
            *[self._fetch_page_for_retrieval(pid, "section_cache") for pid in page_ids],
            return_exceptions=True,
        )

        page_cache: dict[str, PageCandidate] = {}
        for pid, result in zip(page_ids, fetch_results):
            if isinstance(result, Exception):
                logger.debug("Section cache fetch failed for %s: %s", pid, result)
            else:
                page_cache[pid] = result

        section_cache: dict[tuple[str, str], str] = {}
        for pid, page in page_cache.items():
            sections = page.sections or extract_sections(page.html)
            by_heading: dict[str, str] = {}
            for section in sections:
                h = normalize_ws(str(section.get("heading") or ""))
                t = normalize_ws(str(section.get("text") or html_to_text(str(section.get("html") or ""))))
                if h and t:
                    by_heading[h.lower()] = t
            for heading in needed.get(pid, set()):
                h_lower = heading.lower()
                if h_lower in by_heading:
                    section_cache[(pid, heading)] = by_heading[h_lower]
                    continue
                for sh, text in by_heading.items():
                    if h_lower in sh or sh in h_lower:
                        section_cache[(pid, heading)] = text
                        break

        return section_cache, page_cache

    # ── Grounded proposal drafting ────────────────────────────────────────────

    def _draft_grounded_proposals_sync(
        self,
        *,
        session_id: str,
        meeting: ExtractedMeeting,
        intent_sections: list[tuple[ChangeIntent, list[VectorSearchHit]]],
        section_cache: dict[tuple[str, str], str],
        page_cache: dict[str, PageCandidate],
        query: str | None,
    ) -> list[dict[str, Any]]:
        import concurrent.futures

        now = _utcnow()
        work_items: list[tuple[ChangeIntent, VectorSearchHit, str]] = []
        for intent, hits in intent_sections:
            for hit in hits:
                if not hit.page_id:
                    continue
                section_text = section_cache.get((hit.page_id, hit.heading)) or hit.text or ""
                if section_text:
                    work_items.append((intent, hit, section_text))

        if not work_items:
            return []

        proposals: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(work_items))) as pool:
            futures = {
                pool.submit(
                    self._draft_one_grounded_sync,
                    session_id, intent, hit, section_text, page_cache, now, query,
                ): (intent, hit)
                for intent, hit, section_text in work_items
            }
            for future, (intent, hit) in futures.items():
                try:
                    proposal = future.result(timeout=40)
                    if proposal:
                        proposals.append(proposal.to_dict())
                except Exception as exc:
                    logger.debug("Grounded draft failed for %r / %r: %s", intent.subject, hit.heading, exc)
        return proposals

    def _draft_one_grounded_sync(
        self,
        session_id: str,
        intent: ChangeIntent,
        hit: VectorSearchHit,
        section_text: str,
        page_cache: dict[str, PageCandidate],
        timestamp: str,
        query: str | None,
    ) -> Proposal | None:
        """LLM draft with transcript evidence + live section text side by side."""
        client = self._get_openai()
        action = (intent.action or "replace").lower()

        task_extra = ""
        if action in {"complete_task", "reopen_task"}:
            desired = "complete" if action == "complete_task" else "incomplete"
            task_extra = (
                f"\nTask action: mark '{intent.subject}' as {desired}. "
                "Set before_content to the current task line and after_content to the updated line. "
                "Set edit_mode to 'replace'."
            )

        prompt = (
            "You are a surgical Confluence editor. Produce the smallest possible text replacement.\n\n"
            "Rules:\n"
            "- before_content: the SHORTEST verbatim substring from the section that contains the "
            "outdated value. If only a single token changed (a number, a name, a date), "
            "before_content is just that token — never the whole sentence.\n"
            "- after_content: the same substring with ONLY the changed value swapped in. "
            "Do NOT copy transcript phrasing into after_content. Do NOT rewrite the sentence. "
            "Only replace what actually changed.\n"
            "- edit_mode: 'replace' when substituting existing text, 'append' only when adding "
            "genuinely new information that has no existing counterpart in the section.\n"
            "- Return edit_mode 'null' if the evidence does not clearly justify a change here.\n"
            "- NEVER rewrite a full sentence when only a single token (number, name, date) changed."
            + task_extra
            + "\n\nReturn JSON only: "
            "{\"edit_mode\": \"replace|append|null\", "
            "\"before_content\": string|null, "
            "\"after_content\": string|null, "
            "\"section_heading\": string, "
            "\"rationale\": string}"
        )
        payload = {
            "meeting_evidence": {
                "instruction": intent.instruction,
                "subject": intent.subject,
                "target_hint": intent.target_hint,
                "new_value": intent.new_value,
                "action": intent.action,
                "evidence": intent.evidence[:5],
            },
            "confluence_section": {
                "page_title": hit.title,
                "section_heading": hit.heading,
                "current_text": section_text[:6000],
            },
        }
        opts: dict[str, Any] = {"model": self.model, "response_format": {"type": "json_object"}}
        if self.model.startswith(("gpt-5", "o1", "o3", "o4")):
            opts["max_completion_tokens"] = 900
        else:
            opts["max_tokens"] = 900
            opts["temperature"] = 0.1
        response = client.chat.completions.create(
            **opts,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        data = json.loads(response.choices[0].message.content or "{}")
        edit_mode = str(data.get("edit_mode") or "").lower().strip()
        if edit_mode in {"null", ""} or not data.get("after_content"):
            return None
        if edit_mode not in {"replace", "append"}:
            edit_mode = "append"

        before = normalize_ws(str(data.get("before_content") or "")) or None
        after = normalize_ws(str(data.get("after_content") or ""))
        if not after or self._looks_like_instruction_text(after):
            return None

        # Verify the replace anchor actually exists in the section.
        if edit_mode == "replace" and before:
            if normalize_for_match(before) not in normalize_for_match(section_text):
                edit_mode = "append"
                before = None

        section_heading = normalize_ws(str(data.get("section_heading") or hit.heading)) or hit.heading
        rationale = normalize_ws(str(data.get("rationale") or intent.rationale or intent.instruction))
        page = page_cache.get(hit.page_id or "")
        page_url = page.url if page else None

        ctx_before, ctx_after = _extract_context_lines(section_text, before) if before else ("", "")

        anchored = edit_mode == "replace" and bool(before)
        confidence: Literal["high", "medium", "low"] = (
            "high" if anchored and intent.evidence
            else "medium" if anchored or hit.score > 0.5
            else "low"
        )
        risk: Literal["safe", "review", "risky"] = "safe" if confidence == "high" else "review"
        conf_score = {"high": 0.9, "medium": 0.72, "low": 0.45}[confidence]

        edit_mode_typed: EditMode = "task_status" if action in {"complete_task", "reopen_task"} else edit_mode  # type: ignore[assignment]

        return Proposal(
            id=str(uuid.uuid4()),
            change_type="edit",
            page_id=hit.page_id,
            page_title=hit.title,
            section_heading=section_heading,
            before_content=before,
            after_content=after,
            context_before=ctx_before or None,
            context_after=ctx_after or None,
            timestamp=timestamp,
            session_id=session_id,
            rationale=rationale,
            generation_query=query,
            transcript_evidence=intent.evidence[:3],
            confidence=confidence,
            risk=risk,
            verifier_note=self._verifier_note_grounded(edit_mode, before, hit.score),
            edit_mode=edit_mode_typed,
            change_summary=self._change_summary(intent, hit.title, "edit", before, after),
            page_url=page_url,
            confidence_score=conf_score,
            confidence_bin=confidence,
        )

    def _verifier_note_grounded(self, edit_mode: str, before: str | None, rerank_score: float) -> str:
        if edit_mode == "replace" and before:
            return "Edit anchored to exact section text; high confidence."
        if rerank_score > 0.5:
            return "Section matched with strong rerank score; review before accepting."
        return "Section matched via semantic similarity; confirm relevance before accepting."

    # ── Coverage audit ────────────────────────────────────────────────────────

    async def _coverage_audit(
        self,
        meeting: ExtractedMeeting,
        proposals: list[dict[str, Any]],
        transcript_text: str,
    ) -> list[dict[str, Any]]:
        def _token_set(text: str) -> set[str]:
            return set(re.findall(r"[a-z0-9]{3,}", text.lower())) - {"the", "and", "for", "this", "that", "with"}

        def _jaccard(a: set[str], b: set[str]) -> float:
            if not a or not b:
                return 0.0
            return len(a & b) / len(a | b)

        covered: set[int] = set()
        for idx, intent in enumerate(meeting.change_intents):
            intent_new = normalize_for_match(intent.new_value)
            intent_subj = normalize_for_match(intent.subject or intent.target_hint)
            intent_tokens = _token_set(
                " ".join(filter(None, [intent.new_value, intent.subject, intent.target_hint, intent.instruction]))
            )
            for proposal in proposals:
                after = normalize_for_match(str(proposal.get("after_content") or ""))
                title = normalize_for_match(str(proposal.get("page_title") or ""))
                section = normalize_for_match(str(proposal.get("section_heading") or ""))
                # Exact string match (original)
                if intent_new and intent_new in after:
                    covered.add(idx)
                    break
                if intent_subj and (
                    (title and (intent_subj in title or title in intent_subj))
                    or (section and (intent_subj in section or section in intent_subj))
                ):
                    covered.add(idx)
                    break
                # Semantic fallback: token Jaccard across combined proposal text
                proposal_tokens = _token_set(
                    " ".join(filter(None, [
                        str(proposal.get("after_content") or ""),
                        str(proposal.get("page_title") or ""),
                        str(proposal.get("section_heading") or ""),
                    ]))
                )
                if _jaccard(intent_tokens, proposal_tokens) >= 0.15:
                    covered.add(idx)
                    break

        for idx, intent in enumerate(meeting.change_intents):
            if idx not in covered and (intent.instruction or intent.subject or intent.new_value):
                self._add_diagnostic(
                    intent,
                    "coverage_audit_unanchored_intent",
                    "Coverage audit could not anchor this extracted intent to a proposal.",
                    severity="review",
                )
        return proposals

    def _looks_like_instruction_text(self, value: str) -> bool:
        text = normalize_ws(value)
        if not text:
            return False
        bracketed = re.findall(r"\[([^\]]{3,240})\]", text)
        if not bracketed:
            return False
        instruction_re = re.compile(
            r"\b("
            r"add|clarify|create|draft|expand|include|insert|mention|rewrite|"
            r"summarize|update|write|minimum|paragraph|section"
            r")\b",
            re.IGNORECASE,
        )
        return any(instruction_re.search(part) for part in bracketed)

    def _create_page_proposal(
        self,
        session_id: str,
        intent: ChangeIntent,
        timestamp: str,
        query: str | None,
    ) -> Proposal:
        title = intent.subject or intent.target_hint or intent.instruction[:80] or "New Confluence Page"
        content = intent.new_value or intent.instruction or title
        return Proposal(
            id=str(uuid.uuid4()),
            change_type="create",
            page_id=None,
            page_title=title,
            section_heading="Overview",
            before_content=None,
            after_content=content,
            timestamp=timestamp,
            session_id=session_id,
            rationale=intent.rationale or "Meeting requested new documentation.",
            generation_query=query,
            transcript_evidence=intent.evidence[:3],
            confidence="medium",
            risk="review",
            verifier_note="Create proposal generated because no existing page was selected for this intent.",
            change_summary=f"Create page '{title}'",
            confidence_score=0.7,
            confidence_bin="medium",
        )

    def _verify_and_dedupe(self, proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Pass 1: exact-key dedup (same page, same change_type, same before/after)
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for proposal in proposals:
            after = normalize_for_match(str(proposal.get("after_content") or ""))
            before = normalize_for_match(str(proposal.get("before_content") or ""))
            key = "|".join(
                [
                    str(proposal.get("page_id") or proposal.get("page_title") or "").lower(),
                    str(proposal.get("change_type") or ""),
                    before[:100],
                    after[:160],
                ]
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(proposal)

        # Pass 2: fuzzy dedup — same page + section, near-identical after_content.
        # Groups proposals by (page_id, section_heading, change_type) and within each
        # group keeps only the highest-confidence representative when token overlap > 85%.
        groups: dict[str, list[dict[str, Any]]] = {}
        for proposal in out:
            gk = "|".join([
                str(proposal.get("page_id") or proposal.get("page_title") or "").lower(),
                str(proposal.get("section_heading") or "").lower(),
                str(proposal.get("change_type") or ""),
            ])
            groups.setdefault(gk, []).append(proposal)

        final: list[dict[str, Any]] = []
        for group in groups.values():
            if len(group) == 1:
                final.append(group[0])
                continue
            kept: list[dict[str, Any]] = []
            for proposal in group:
                after_p = normalize_for_match(str(proposal.get("after_content") or ""))
                absorbed = False
                for existing in kept:
                    after_e = normalize_for_match(str(existing.get("after_content") or ""))
                    if _token_overlap(after_p, after_e) >= 0.85:
                        # Near-duplicate: keep the higher-confidence one
                        if (proposal.get("confidence_score") or 0) > (existing.get("confidence_score") or 0):
                            kept.remove(existing)
                            kept.append(proposal)
                        absorbed = True
                        break
                if not absorbed:
                    kept.append(proposal)
            final.extend(kept)
        return final

    async def _adversarial_verify(
        self,
        meeting: ExtractedMeeting,
        proposals: list[dict[str, Any]],
        transcript_text: str,
    ) -> list[dict[str, Any]]:
        """LLM adversarial verifier for support, wrong-page risk, duplicates, and misses.

        The verifier is advisory. It annotates cards and downgrades risk/confidence,
        but does not silently discard proposals.
        """
        if not proposals and not meeting.change_intents:
            return proposals
        try:
            data = await asyncio.to_thread(
                self._adversarial_verify_sync,
                meeting,
                proposals,
                transcript_text,
            )
            return self._apply_adversarial_verdict(data, proposals)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Adversarial verifier failed; using deterministic checks only: %s", exc)
            return proposals

    def _adversarial_verify_sync(
        self,
        meeting: ExtractedMeeting,
        proposals: list[dict[str, Any]],
        transcript_text: str,
    ) -> dict[str, Any]:
        client = self._get_openai()
        prompt = (
            "You are an adversarial QA verifier for Confluence change proposals. "
            "Check whether each proposal is directly supported by the transcript, "
            "whether it targets the right page, whether it is a duplicate, and whether "
            "any extracted meeting intent is missing from the proposals. "
            "Do not invent new facts. Return JSON only.\n\n"
            "Schema: {"
            "\"proposal_verdicts\": [{\"id\": string, \"supported\": boolean, "
            "\"page_fit\": \"good|uncertain|wrong\", \"duplicate_of\": string|null, "
            "\"confidence\": \"high|medium|low\", \"risk\": \"safe|review|risky\", "
            "\"note\": string}], "
            "\"missed_intents\": [{\"instruction\": string, \"reason\": string}]}"
        )
        payload = {
            "transcript_excerpt": transcript_text[:80000],
            "extracted_intents": [
                {
                    "instruction": i.instruction,
                    "subject": i.subject,
                    "old_value": i.old_value,
                    "new_value": i.new_value,
                    "action": i.action,
                    "page_id": i.page_id,
                    "page_title": i.page_title,
                    "evidence": i.evidence,
                }
                for i in meeting.change_intents[:40]
            ],
            "proposals": [
                {
                    "id": p.get("id"),
                    "change_type": p.get("change_type"),
                    "page_title": p.get("page_title"),
                    "section_heading": p.get("section_heading"),
                    "before_content": p.get("before_content"),
                    "after_content": p.get("after_content"),
                    "rationale": p.get("rationale"),
                    "evidence": p.get("transcript_evidence"),
                }
                for p in proposals[:60]
            ],
        }
        opts: dict[str, Any] = {"model": self.model, "response_format": {"type": "json_object"}}
        if self.model.startswith(("gpt-5", "o1", "o3", "o4")):
            opts["max_completion_tokens"] = 1800
        else:
            opts["max_tokens"] = 1800
            opts["temperature"] = 0.0
        response = client.chat.completions.create(
            **opts,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        return json.loads(response.choices[0].message.content or "{}")

    def _apply_adversarial_verdict(
        self,
        data: dict[str, Any],
        proposals: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        by_id = {str(p.get("id")): p for p in proposals}
        rejected_ids: set[str] = set()
        for verdict in data.get("proposal_verdicts") or []:
            if not isinstance(verdict, dict):
                continue
            proposal = by_id.get(str(verdict.get("id")))
            if not proposal:
                continue

            supported = bool(verdict.get("supported", True))
            page_fit = str(verdict.get("page_fit") or "uncertain").lower()
            duplicate_of = verdict.get("duplicate_of")
            note = normalize_ws(str(verdict.get("note") or ""))

            confidence = str(verdict.get("confidence") or proposal.get("confidence") or "medium").lower()
            risk = str(verdict.get("risk") or proposal.get("risk") or "review").lower()
            if confidence in {"high", "medium", "low"}:
                proposal["confidence"] = confidence
                proposal["confidence_bin"] = confidence
                proposal["confidence_score"] = {"high": 0.9, "medium": 0.7, "low": 0.4}[confidence]
            if risk in {"safe", "review", "risky"}:
                proposal["risk"] = risk

            warnings: list[str] = []
            if not supported:
                warnings.append("[UNSUPPORTED] Transcript support is weak or missing.")
                proposal["confidence"] = "low"
                proposal["confidence_bin"] = "low"
                proposal["confidence_score"] = 0.4
                proposal["risk"] = "risky"
            if page_fit == "wrong":
                warnings.append("[WRONG PAGE] Verifier thinks this page is not the right target.")
                proposal["risk"] = "risky"
            elif page_fit == "uncertain":
                warnings.append("[PAGE FIT UNCERTAIN] Confirm this is the right page.")
                if proposal.get("risk") == "safe":
                    proposal["risk"] = "review"
            if duplicate_of:
                warnings.append(f"[DUPLICATE] May duplicate proposal {duplicate_of}.")
                if proposal.get("risk") == "safe":
                    proposal["risk"] = "review"
            if note:
                warnings.append(note)

            if warnings:
                existing = proposal.get("verifier_note") or ""
                proposal["verifier_note"] = " ".join([existing, *warnings]).strip()

            if not supported or page_fit == "wrong":
                rejected_ids.add(str(proposal.get("id")))
                self._add_diagnostic(
                    None,
                    "verifier_rejected_proposal",
                    note
                    or (
                        "Verifier rejected proposal as unsupported."
                        if not supported
                        else "Verifier rejected proposal because it targets the wrong page."
                    ),
                    severity="risky",
                    page_title=str(proposal.get("page_title") or "") or None,
                )

        missed = data.get("missed_intents") or []
        if missed:
            missed_note = "Adversarial verifier found possible missed intents: " + "; ".join(
                normalize_ws(str((m or {}).get("instruction") or (m or {}).get("reason") or ""))[:140]
                for m in missed[:5]
                if isinstance(m, dict)
            )
            for proposal in proposals:
                existing = proposal.get("verifier_note") or ""
                proposal["verifier_note"] = f"{existing} {missed_note}".strip()
                if proposal.get("risk") == "safe":
                    proposal["risk"] = "review"
        return [proposal for proposal in proposals if str(proposal.get("id")) not in rejected_ids]


    def _verifier_note(self, intent: ChangeIntent, page: PageCandidate, old_found: bool) -> str:
        if old_found:
            return "The old value was found in the live page content; proposal is anchored to an explicit transcript change."
        if intent.old_value:
            return "The page appears relevant, but the exact old value was not found; review before accepting."
        return "The page appears relevant based on the intent subject; review before accepting."

    def _change_summary(
        self,
        intent: ChangeIntent,
        page_title: str,
        change_type: str,
        before: str | None,
        after: str | None,
    ) -> str:
        if change_type == "create":
            return f"Create page '{page_title}'"
        if change_type == "title":
            return f"Rename '{page_title}' to '{after}'"
        if before and after:
            return f"Change '{before}' to '{after}' on '{page_title}'"
        return intent.instruction or f"Update '{page_title}'"

    async def execute(self, proposal: dict[str, Any]) -> dict[str, Any]:
        change_type = proposal.get("change_type")
        if change_type == "create":
            title = str(proposal.get("page_title") or "New Confluence Page")
            html_content = _storage_html(str(proposal.get("after_content") or title))
            result = await self.client.create_page(title, html_content)
            new_page_id = str(result.get("id") or result.get("page_id") or "")
            if new_page_id:
                await self._index_page_for_rag(
                    PageCandidate(
                        page_id=new_page_id,
                        title=title,
                        html=html_content,
                        text=html_to_text(html_content),
                        version=(result.get("version") or {}).get("number") if isinstance(result.get("version"), dict) else None,
                        source="created_page",
                        sections=extract_sections(html_content),
                    )
                )
            return {"success": True, "message": f"Created Confluence page {result.get('id') or title}."}

        page_id = proposal.get("page_id")
        if not page_id:
            return {"success": False, "message": "Proposal has no page_id."}

        page = await self.client.fetch_page(str(page_id))
        if change_type == "title":
            new_title = str(proposal.get("after_content") or "").strip()
            if not new_title:
                return {"success": False, "message": "Title proposal has no new title."}
            await self.client.update_page(str(page_id), page.html, title=new_title, expected_version=page.version)
            await self._index_page_for_rag(self._updated_page_candidate(page, page.html, title=new_title))
            return {"success": True, "message": f"Renamed page to {new_title}."}

        if change_type == "delete":
            return {"success": False, "message": "Delete proposals require manual handling in this new pipeline."}

        before = str(proposal.get("before_content") or "")
        after = str(proposal.get("after_content") or "")
        edit_mode = str(proposal.get("edit_mode") or ("replace" if before else "append"))
        if not after:
            return {"success": False, "message": "Edit proposal has no replacement content."}

        if edit_mode == "task_status":
            task_body = task_body_from_label(after) or task_body_from_label(before)
            desired_status = "complete" if after.strip().lower().startswith("[x]") else "incomplete"
            new_html, replaced = replace_task_status_in_storage(page.html, task_body, desired_status)
            if not replaced:
                return {
                    "success": False,
                    "message": f"Task checkbox was not found on the live page: {task_body[:80]}",
                }
        elif edit_mode == "replace":
            if not before:
                return {"success": False, "message": "Replace proposal has no old text anchor."}
            new_html, replaced = replace_text_in_storage(page.html, before, after)
            if not replaced:
                return {
                    "success": False,
                    "message": f"Old text anchor was not found on the live page: {before[:80]}",
                }
        elif edit_mode == "create_section":
            section = str(proposal.get("section_heading") or "Update")
            new_html = append_new_section(page.html, section, _storage_html(after))
        else:
            addition = _storage_html(after)
            new_html = insert_html_in_section(page.html, proposal.get("section_heading"), addition)

        await self.client.update_page(str(page_id), new_html, expected_version=page.version)
        await self._index_page_for_rag(self._updated_page_candidate(page, new_html))
        return {"success": True, "message": "Applied change to Confluence."}

    def _updated_page_candidate(
        self,
        page: PageCandidate,
        html_content: str,
        *,
        title: str | None = None,
    ) -> PageCandidate:
        next_version = page.version + 1 if page.version is not None else None
        return PageCandidate(
            page_id=page.page_id,
            title=title or page.title,
            space_key=page.space_key,
            url=page.url,
            html=html_content,
            text=html_to_text(html_content),
            version=next_version,
            source="post_apply_reindex",
            score=page.score,
            sections=extract_sections(html_content),
        )

    # ── Post-meeting summary generation (4 parallel LLM calls) ──────────────────

    def _llm_opts(self, max_tokens: int) -> dict[str, Any]:
        opts: dict[str, Any] = {"model": self.model, "response_format": {"type": "json_object"}}
        if self.model.startswith(("gpt-5", "o1", "o3", "o4")):
            opts["max_completion_tokens"] = max_tokens
        else:
            opts["max_tokens"] = max_tokens
            opts["temperature"] = 0.1
        return opts

    async def _cerebras_with_fallback(
        self,
        cerebras: AsyncOpenAI,
        fallback: AsyncOpenAI,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any],
        max_tokens: int,
        label: str,
    ) -> str:
        """One strict Cerebras call; fall back to OpenAI immediately on any failure."""
        try:
            resp = await cerebras.chat.completions.create(
                model=self._CEREBRAS_MODEL,
                temperature=0.1,
                max_tokens=max_tokens,
                response_format=response_format,
                messages=messages,
            )
            content = resp.choices[0].message.content or ""
            if content.strip():
                return content
            logger.warning("[%s] Cerebras returned empty; falling back to OpenAI", label)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] Cerebras failed (%s); falling back to OpenAI", label, exc)

        resp = await fallback.chat.completions.create(
            model=os.getenv("JARVIS_GENERAL_MODEL", "gpt-4o-mini"),
            temperature=0.1,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=messages,
        )
        return resp.choices[0].message.content or ""

    async def _summary_llm(
        self, cerebras: AsyncOpenAI, fallback: AsyncOpenAI, transcript_text: str
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert meeting analyst. Produce an executive summary from the meeting transcript below.\n\n"
                    "title: a crisp 4-8 word title that captures the meeting's core purpose.\n\n"
                    "summary: exactly 2 paragraphs, each 2-3 sentences. "
                    "Paragraph 1 — context and objective: what the meeting was about and why it was called. "
                    "Paragraph 2 — outcomes and next steps: the main decisions reached, agreements made, and immediate actions committed to. "
                    "Write in plain, professional prose. No bullet points. No headers. No fluff. "
                    "Separate the two paragraphs with a single blank line (\\n\\n).\n\n"
                    "key_topics: up to 8 short noun-phrase labels for the main topics discussed."
                ),
            },
            {"role": "user", "content": transcript_text},
        ]
        raw = await self._cerebras_with_fallback(cerebras, fallback, messages, _SUMMARY_SCHEMA, 800, "Summary")
        try:
            return _SummaryResponse.model_validate(json.loads(raw)).model_dump()
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Summary JSON parse failed: %s — raw: %.200s", exc, raw)
            return _SummaryResponse().model_dump()

    async def _decisions_llm(
        self, cerebras: AsyncOpenAI, fallback: AsyncOpenAI, transcript_text: str
    ) -> list[str]:
        messages = [
            {
                "role": "system",
                "content": (
                    "List every concrete decision made in this meeting transcript. "
                    "Each decision is one clear sentence. Include all agreed outcomes, chosen options, and commitments. "
                    "Omit open questions and vague discussion. "
                    'Return JSON: {"decisions": ["<decision 1>", "<decision 2>", ...]}'
                ),
            },
            {"role": "user", "content": transcript_text},
        ]
        raw = await self._cerebras_with_fallback(cerebras, fallback, messages, _DECISIONS_SCHEMA, 400, "Decisions")
        try:
            return [str(d) for d in _DecisionsResponse.model_validate(json.loads(raw)).decisions if d]
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Decisions JSON parse failed: %s — raw: %.200s", exc, raw)
            return []

    async def _action_items_llm(
        self, cerebras: AsyncOpenAI, fallback: AsyncOpenAI, transcript_text: str
    ) -> list[dict[str, Any]]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Extract every action item from this meeting transcript. "
                    "description: what needs to be done. "
                    "owner: person responsible, empty string if unassigned. "
                    "due: deadline if mentioned, empty string if not mentioned. "
                    'Return JSON: {"action_items": [{"description": "...", "owner": "...", "due": "..."}, ...]}'
                ),
            },
            {"role": "user", "content": transcript_text},
        ]
        raw = await self._cerebras_with_fallback(cerebras, fallback, messages, _ACTION_ITEMS_SCHEMA, 500, "Action items")
        try:
            parsed = _ActionItemsResponse.model_validate(json.loads(raw))
            return [
                {
                    "description": item.description,
                    "owner": item.owner or None,
                    "due": item.due or None,
                }
                for item in parsed.action_items
                if item.description
            ]
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Action items JSON parse failed: %s — raw: %.200s", exc, raw)
            return []

    async def _mom_llm(
        self, cerebras: AsyncOpenAI, fallback: AsyncOpenAI, transcript_text: str
    ) -> list[dict[str, Any]]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Generate minutes of meeting from this transcript. "
                    "Each entry covers one distinct topic discussed. "
                    "summary: 1-3 sentences capturing what was said and decided."
                ),
            },
            {"role": "user", "content": transcript_text},
        ]
        raw = await self._cerebras_with_fallback(cerebras, fallback, messages, _MOM_SCHEMA, 700, "MOM")
        try:
            parsed = _MOMResponse.model_validate(json.loads(raw))
            return [
                {"topic": item.topic, "summary": item.summary}
                for item in parsed.mom
                if item.topic
            ]
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("MOM JSON parse failed: %s — raw: %.200s", exc, raw)
            return []

    async def generate_meeting_summary(
        self,
        session_id: str,
        transcript: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Fire 4 parallel Cerebras calls immediately after meeting ends.

        Uses AsyncOpenAI directly (same pattern as the chat interface) so calls
        are native async — no thread pool overhead or sync-client quirks.
        """
        # Strip Jarvis's own spoken replies before summarising. Jarvis answers
        # are informed by RAG-retrieved Confluence content — including them would
        # cause the summary LLM to treat wiki knowledge as meeting discussion.
        transcript_text = format_transcript(_human_only(transcript))
        if not transcript_text:
            return self.summary_response(session_id, None, transcript)

        api_key = os.getenv("CEREBRAS_API_KEY", "")
        if not api_key:
            logger.warning("CEREBRAS_API_KEY not set — falling back to heuristic summary for session %s", session_id)
            return self.summary_response(session_id, None, transcript)

        cerebras_client = AsyncOpenAI(api_key=api_key, base_url=self._CEREBRAS_BASE_URL)
        openai_fallback = AsyncOpenAI()  # uses OPENAI_API_KEY from env

        summary_result, decisions_result, action_items_result, mom_result = await asyncio.gather(
            self._summary_llm(cerebras_client, openai_fallback, transcript_text),
            self._decisions_llm(cerebras_client, openai_fallback, transcript_text),
            self._action_items_llm(cerebras_client, openai_fallback, transcript_text),
            self._mom_llm(cerebras_client, openai_fallback, transcript_text),
            return_exceptions=True,
        )

        summary_data: dict[str, Any] = summary_result if not isinstance(summary_result, BaseException) else {}
        decisions: list[str] = decisions_result if not isinstance(decisions_result, BaseException) else []
        action_items: list[dict] = action_items_result if not isinstance(action_items_result, BaseException) else []
        mom: list[dict] = mom_result if not isinstance(mom_result, BaseException) else []

        if isinstance(summary_result, BaseException):
            logger.error("Summary call failed: %s", summary_result, exc_info=summary_result)
        if isinstance(decisions_result, BaseException):
            logger.error("Decisions call failed: %s", decisions_result, exc_info=decisions_result)
        if isinstance(action_items_result, BaseException):
            logger.error("Action items call failed: %s", action_items_result, exc_info=action_items_result)
        if isinstance(mom_result, BaseException):
            logger.error("MOM call failed: %s", mom_result, exc_info=mom_result)

        participants = sorted({
            str(entry.get("participant") or entry.get("speaker") or "").strip()
            for entry in transcript
            if str(entry.get("participant") or entry.get("speaker") or "").strip()
        })

        today = dt.datetime.now(dt.UTC).strftime("%B %d, %Y")
        return {
            "title": summary_data.get("title") or f"Meeting {session_id[:8]}",
            "session_id": session_id,
            "date": today,
            "summary": summary_data.get("summary") or "",
            "key_topics": summary_data.get("key_topics") or [],
            "action_items": action_items,
            "decisions": decisions,
            "participants": participants,
            "mom": mom,
            "transcript_highlights": transcript_highlights(transcript),
            "stats": {
                "transcript_entries": len(transcript),
                "topic_count": len(summary_data.get("key_topics") or []),
                "decision_count": len(decisions),
                "action_item_count": len(action_items),
            },
        }

    def summary_response(
        self,
        session_id: str,
        meeting: ExtractedMeeting | None,
        transcript: list[dict[str, Any]],
    ) -> dict[str, Any]:
        today = dt.datetime.now(dt.UTC).strftime("%B %d, %Y")
        if meeting is None:
            meeting = self._heuristic_meeting(transcript, format_transcript(transcript))
        return {
            "title": meeting.title or f"Meeting {session_id[:8]}",
            "session_id": session_id,
            "date": today,
            "summary": meeting.summary or "Transcript captured. Generate Confluence changes to analyze documentation updates.",
            "key_topics": meeting.key_topics,
            "action_items": meeting.action_items,
            "decisions": meeting.decisions,
            "participants": meeting.participants,
            "mom": meeting.moments,
            "transcript_highlights": transcript_highlights(transcript),
            "stats": {
                "transcript_entries": len(transcript),
                "topic_count": len(meeting.key_topics),
                "decision_count": len(meeting.decisions),
                "action_item_count": len(meeting.action_items),
            },
        }

    def _get_openai(self) -> OpenAI:
        if self._openai is None:
            self._openai = OpenAI()
        return self._openai

    def _get_cerebras(self) -> OpenAI:
        if self._cerebras is None:
            api_key = os.getenv("CEREBRAS_API_KEY", "")
            self._cerebras = OpenAI(api_key=api_key, base_url=self._CEREBRAS_BASE_URL)
        return self._cerebras

    def _llm_opts_cerebras(self, max_tokens: int, response_format: dict[str, Any] | None = None) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "model": self._CEREBRAS_MODEL,
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }
        if response_format is not None:
            opts["response_format"] = response_format
        return opts

    @property
    def client(self) -> HybridConfluenceClient:
        if self._client is None:
            self._client = HybridConfluenceClient()
        return self._client

    @property
    def rag(self) -> ConfluenceVectorIndex:
        if self._rag is None:
            self._rag = ConfluenceVectorIndex()
        return self._rag


def _utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


def _token_overlap(a: str, b: str) -> float:
    """Jaccard-style token overlap between two normalized strings."""
    if not a or not b:
        return 0.0
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def _storage_html(markdownish: str) -> str:
    lines = [line.rstrip() for line in (markdownish or "").splitlines()]
    html_lines = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            continue
        if stripped.startswith("## "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h2>{html.escape(stripped[3:].strip())}</h2>")
        elif stripped.startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{html.escape(stripped[2:].strip())}</li>")
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<p>{html.escape(stripped)}</p>")
    if in_list:
        html_lines.append("</ul>")
    return "\n".join(html_lines) or "<p></p>"
