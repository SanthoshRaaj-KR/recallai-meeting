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
from typing import Any

from openai import OpenAI

from memory_compaction import add_memory_context

from .confluence import HybridConfluenceClient
from .models import ChangeIntent, ExtractedMeeting, PageCandidate, Proposal
from .rag import ConfluenceVectorIndex
from .text_utils import (
    append_new_section,
    best_section_heading,
    extract_sections,
    extract_tasks,
    format_transcript,
    html_to_text,
    insert_html_in_section,
    normalize_for_match,
    normalize_ws,
    replace_task_status_in_storage,
    replace_text_in_storage,
    task_body_from_label,
    task_status_label,
    transcript_highlights,
)

logger = logging.getLogger(__name__)

EmitFn = Callable[[dict[str, Any]], Awaitable[None]]


class ProposalPipeline:
    """Transcript -> atomic intents -> Confluence page retrieval -> proposal cards.

    This is intentionally self-contained under my-agent. It does not import the
    older confluence_logic package.
    """

    def __init__(self) -> None:
        self.model = os.getenv("MY_AGENT_REVIEW_MODEL", os.getenv("JARVIS_REVIEW_MODEL", "gpt-4o-mini")).strip()
        self.max_candidate_pages = int(os.getenv("MY_AGENT_PIPELINE_MAX_PAGES", "16"))
        self.max_search_terms = int(os.getenv("MY_AGENT_PIPELINE_MAX_SEARCH_TERMS", "24"))
        self.rag_top_k = int(os.getenv("MY_AGENT_PIPELINE_RAG_TOP_K", "8"))
        self._client: HybridConfluenceClient | None = None
        self._rag: ConfluenceVectorIndex | None = None
        self._openai: OpenAI | None = None
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

        await _emit({"type": "stage_start", "stage": "fact_extraction"})
        meeting = await self._extract_meeting(transcript, transcript_text, query=query)
        # Supplement change_intents with action items the LLM may not have converted.
        fallback = self._intents_from_action_items(meeting)
        if fallback:
            meeting.change_intents = self._merge_intents(meeting.change_intents, fallback)[:30]

        await _emit({"type": "stage_start", "stage": "rag_retrieval"})
        rovo_intents = await self._generate_page_grounded_candidates(meeting, transcript_text, query=query)
        if rovo_intents:
            meeting.change_intents = self._merge_intents(meeting.change_intents, rovo_intents)
            await _emit(
                {
                    "type": "stage_progress",
                    "stage": "rag_retrieval",
                    "intent_count": len(meeting.change_intents),
                    "page_grounded_candidates": len(rovo_intents),
                }
            )
        intent_pages = await self._retrieve_pages(meeting.change_intents)

        await _emit({"type": "stage_start", "stage": "drafting"})
        # _draft_proposals contains blocking sync LLM calls (_section_level_draft_sync).
        # Run it in a thread so it doesn't stall the event loop.
        proposals = await asyncio.to_thread(
            self._draft_proposals,
            session_id=session_id,
            meeting=meeting,
            intent_pages=intent_pages,
            query=query,
        )

        await _emit({"type": "stage_start", "stage": "verification"})
        verified = self._verify_and_dedupe(proposals)
        # Run the two independent verifiers in parallel — each makes one LLM call.
        # Both operate on separate copies so neither blocks waiting for the other.
        av_result, rc_result = await asyncio.gather(
            self._adversarial_verify(meeting, [dict(p) for p in verified], transcript_text),
            self._rovo_independent_critic(meeting, [dict(p) for p in verified], transcript_text, query=query),
        )
        # Merge: a proposal survives only if both verifiers kept it; take conservative side.
        _risk_rank = {"safe": 0, "review": 1, "risky": 2}
        _conf_rank = {"high": 2, "medium": 1, "low": 0}
        av_by_id = {str(p.get("id")): p for p in av_result}
        rc_by_id = {str(p.get("id")): p for p in rc_result}
        merged: list[dict[str, Any]] = []
        for p in verified:
            pid = str(p.get("id"))
            av_p = av_by_id.get(pid)
            rc_p = rc_by_id.get(pid)
            if av_p is None or rc_p is None:
                continue  # rejected by at least one verifier
            worst_risk = max(av_p.get("risk", "review"), rc_p.get("risk", "review"), key=lambda r: _risk_rank.get(r, 1))
            worst_conf = min(av_p.get("confidence", "medium"), rc_p.get("confidence", "medium"), key=lambda c: _conf_rank.get(c, 1))
            av_p["risk"] = worst_risk
            av_p["confidence"] = worst_conf
            av_p["confidence_bin"] = worst_conf
            av_p["confidence_score"] = {"high": 0.9, "medium": 0.7, "low": 0.4}[worst_conf]
            rc_note = rc_p.get("verifier_note") or ""
            av_note = av_p.get("verifier_note") or ""
            if rc_note and rc_note not in av_note:
                av_p["verifier_note"] = f"{av_note} {rc_note}".strip()
            merged.append(av_p)
        verified = merged
        verified = await self._coverage_audit(meeting, verified, transcript_text)
        if self.last_diagnostics:
            await _emit(
                {
                    "type": "stage_progress",
                    "stage": "verification",
                    "intent_diagnostics": self.last_diagnostics,
                }
            )
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
        terms = self._candidate_search_terms(meeting, query, query=query)[:6]
        for term in terms:
            try:
                results = await self._hybrid_search_pages(term, limit=limit)
            except Exception:
                continue
            for result in results:
                page_id = str(result.get("page_id") or result.get("id") or "")
                if not page_id or page_id in pages_by_id:
                    continue
                try:
                    page = await self._fetch_page_for_retrieval(page_id, str(result.get("source") or "style_sample"))
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
            "Draft a new Confluence page strictly about the topic specified in user_request. "
            "The user_request defines the EXACT topic and scope — do not document anything from the "
            "meeting that is not directly related to it. The transcript is a source to extract "
            "relevant details from, not a dump to document wholesale.\n\n"
            "Step 1 — Identify the topic: read user_request carefully. That topic is the ONLY subject "
            "of this page. Everything else discussed in the meeting is irrelevant and must be omitted.\n\n"
            "Step 2 — Infer style from same_space_style_samples: heading depth, section order, tone, "
            "bullets vs paragraphs, tables, and how concise the pages are. Write the new page in that style.\n\n"
            "Step 3 — Extract only on-topic facts from the transcript. Ignore off-topic segments entirely.\n\n"
            "Return JSON only with: title, body_markdown, rationale. body_markdown must be publishable "
            "page content using markdown-ish syntax: ## headings, ### subheadings, bullet lists, "
            "numbered steps, and simple tables if useful. If an image/diagram would help, insert a "
            "placeholder line like '[IMAGE PLACEHOLDER: describe the exact image needed]'. "
            "Do not invent facts beyond the transcript/request. If details are missing, include a "
            "short 'Open questions' section instead of guessing."
        )
        payload = {
            "user_request": query,
            "page_topic_scope": (
                f"This page must ONLY cover: {query}. "
                "Ignore all meeting content that is not directly about this topic."
            ),
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
            "You extract structured facts from meeting transcripts that may need to be reflected in "
            "Confluence documentation. The participants will NEVER explicitly say 'update the docs' — "
            "you must proactively identify every concrete fact, decision, metric change, status update, "
            "ownership change, date change, completed/reopened task, or agreed action that could already "
            "be documented somewhere in Confluence and therefore needs updating.\n\n"
            "Return JSON only. Capture the FINAL agreed state of every fact, not intermediate suggestions. "
            "Create one atomic change_intent per distinct fact or update. "
            "Prefer precise old_value/new_value pairs (e.g. old_value='0.91', new_value='0.98'). "
            "If a metric, threshold, status, date, or owner is stated in the meeting, extract it as a "
            "change_intent with the best page/title hint — even with no old_value. "
            "For checklist/task updates use action='complete_task' when the meeting says an item is done "
            "and action='reopen_task' when it is no longer done. For general status fields use action='replace' "
            "with old_value only when stated or inferable. "
            "One fact may appear on MULTIPLE Confluence pages — still emit one intent per fact; the retrieval "
            "pipeline will fan it out to all matching pages. "
            "Do NOT invent facts beyond what was stated. "
            "SKIP ONLY: social niceties (greetings, farewells, thanks), vague filler with no concrete "
            "information, and pure process meta-comments. Everything else is fair game.\n\n"
            "JSON shape:\n"
            "{"
            '"title": string, "summary": string, "key_topics": string[], "decisions": string[], '
            '"action_items": [{"description": string, "owner": string|null, "due": string|null}], '
            '"change_intents": [{"instruction": string, "subject": string, "target_hint": string, '
            '"old_value": string, "new_value": string, "action": "replace|add|remove|rename|create|complete_task|reopen_task", '
            '"rationale": string, "evidence": string[]}]'
            "}\n\n"
            "Example: if the meeting says the Q3 meeting plan moved from 2nd September to 3rd December, "
            "extract old_value='2nd September', new_value='3rd December', action='replace'. "
            "Example: if the meeting says 'we completed the Q2 plan', extract subject='Q2 plan', "
            "target_hint='Q2 plan', new_value='complete', action='complete_task'."
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
            change_intents=intents[:30],
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

    async def _generate_page_grounded_candidates(
        self,
        meeting: ExtractedMeeting,
        transcript_text: str,
        *,
        query: str | None = None,
    ) -> list[ChangeIntent]:
        """Second, independent candidate source using live Confluence page snapshots.

        The first pass asks "what changed in the meeting?" This pass asks
        "given likely relevant pages, what changes are evidenced by both the
        meeting and current page content?" It improves recall without letting
        Rovo/Confluence become the final judge.
        """
        search_terms = self._candidate_search_terms(meeting, transcript_text, query=query)
        if not search_terms:
            return []

        # Search all terms in parallel, then fetch unique pages in parallel.
        search_results = await asyncio.gather(
            *[self._hybrid_search_pages(term, limit=5) for term in search_terms],
            return_exceptions=True,
        )
        page_meta: dict[str, dict] = {}  # page_id → result metadata
        for results in search_results:
            if isinstance(results, Exception):
                continue
            for result in results:
                page_id = str(result.get("page_id") or result.get("id") or "")
                if page_id and page_id not in page_meta:
                    page_meta[page_id] = result
                    if len(page_meta) >= self.max_candidate_pages:
                        break
            if len(page_meta) >= self.max_candidate_pages:
                break

        fetch_results = await asyncio.gather(
            *[self._fetch_page_for_retrieval(pid, str(meta.get("source") or "hybrid_search"))
              for pid, meta in page_meta.items()],
            return_exceptions=True,
        )
        pages_by_id: dict[str, PageCandidate] = {}
        for (pid, meta), page in zip(page_meta.items(), fetch_results):
            if isinstance(page, Exception):
                logger.debug("Page-grounded candidate fetch failed for %s: %s", pid, page)
                continue
            page.score = float(meta.get("retrieval_score") or 0.0)
            pages_by_id[pid] = page

        pages = list(pages_by_id.values())
        if not pages:
            return []

        try:
            data = await asyncio.to_thread(
                self._extract_page_grounded_candidates_sync,
                transcript_text,
                pages,
                query or "",
            )
            return self._candidate_intents_from_json(data, pages)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Page-grounded candidate generation failed: %s", exc)
            return []

    def _candidate_search_terms(
        self,
        meeting: ExtractedMeeting,
        transcript_text: str,
        *,
        query: str | None = None,
    ) -> list[str]:
        values: list[str] = []
        if query:
            values.append(query)
        for intent in meeting.change_intents:
            values.extend([intent.old_value, intent.target_hint, intent.subject, intent.new_value])
        values.extend(meeting.key_topics[:8])
        values.extend(meeting.decisions[:8])
        for item in meeting.action_items[:5]:
            if isinstance(item, dict):
                values.append(str(item.get("description") or ""))
            else:
                values.append(str(item))

        # Fallback broad terms from capitalized phrases / quoted-ish doc names.
        for phrase in re.findall(r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z0-9][A-Za-z0-9]*){1,5}\b", transcript_text):
            values.append(phrase)

        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            term = normalize_ws(str(value))[:300]
            key = term.lower()
            if term and len(term) >= 3 and key not in seen:
                seen.add(key)
                out.append(term)
            if len(out) >= self.max_search_terms:
                break
        return out

    def _extract_page_grounded_candidates_sync(
        self,
        transcript_text: str,
        pages: list[PageCandidate],
        query: str,
    ) -> dict[str, Any]:
        client = self._get_openai()
        page_payload = [
            {
                "page_id": p.page_id,
                "page_title": p.title,
                "content_excerpt": (p.text or html_to_text(p.html))[:2500],
            }
            for p in pages[: self.max_candidate_pages]
        ]
        prompt = (
            "You are an aggressive Confluence change candidate generator. "
            "You receive a meeting transcript and live Confluence page excerpts. "
            "Your job: find every place where a concrete meeting fact (metric, decision, date, owner, "
            "status, threshold, task completion, etc.) matches or relates to content in the page excerpts, "
            "and emit a change_intent for each — even if the meeting participants never said 'update the docs'. "
            "Treat page lines like '[task: incomplete] Q2 plan' as Confluence checkboxes/tasks; "
            "if the transcript says the task is complete, done, shipped, closed, or fully finished, "
            "return action='complete_task' with new_value='complete'. "
            "If the transcript says it is reopened or no longer complete, return action='reopen_task'. "
            "For metrics, thresholds, dates, owners, and statuses: if the meeting states a new value and "
            "a page excerpt shows the same field with any value, emit action='replace'. "
            "If a page excerpt contains a section or field relevant to a meeting fact but no exact old value "
            "is visible, emit action='add'. "
            "The same fact may appear in MULTIPLE page excerpts — emit one change_intent per page where it "
            "belongs. Do not deduplicate across pages. "
            "Do not make vague documentation improvements. Do not invent page IDs. "
            "If an old value appears in a page excerpt, copy it exactly into old_value. "
            "Return JSON only: {\"change_intents\": [{\"instruction\": string, \"subject\": string, "
            "\"target_hint\": string, \"old_value\": string, \"new_value\": string, "
            "\"action\": \"replace|add|remove|rename|create|complete_task|reopen_task\", \"rationale\": string, "
            "\"evidence\": string[], \"page_id\": string|null, \"page_title\": string|null}]}"
        )
        payload = {
            "optional_user_guidance": query or None,
            "transcript": transcript_text,
            "live_confluence_pages": page_payload,
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
        return json.loads(response.choices[0].message.content or "{}")

    def _candidate_intents_from_json(
        self,
        data: dict[str, Any],
        pages: list[PageCandidate],
    ) -> list[ChangeIntent]:
        page_ids = {str(p.page_id) for p in pages if p.page_id}
        page_titles = {p.title.lower(): p for p in pages}
        intents: list[ChangeIntent] = []
        for raw in data.get("change_intents") or []:
            if not isinstance(raw, dict):
                continue
            page_id = normalize_ws(str(raw.get("page_id") or raw.get("pageId") or "")) or None
            page_title = normalize_ws(str(raw.get("page_title") or raw.get("pageTitle") or "")) or None
            if page_id and page_id not in page_ids:
                continue
            if not page_id and page_title and page_title.lower() in page_titles:
                page_id = page_titles[page_title.lower()].page_id
            intent = ChangeIntent(
                instruction=normalize_ws(str(raw.get("instruction") or "")),
                subject=normalize_ws(str(raw.get("subject") or "")),
                target_hint=normalize_ws(str(raw.get("target_hint") or page_title or "")),
                old_value=normalize_ws(str(raw.get("old_value") or "")),
                new_value=normalize_ws(str(raw.get("new_value") or "")),
                action=normalize_ws(str(raw.get("action") or "replace")).lower() or "replace",
                rationale=normalize_ws(str(raw.get("rationale") or "")),
                evidence=[normalize_ws(str(x)) for x in raw.get("evidence") or [] if normalize_ws(str(x))],
                source="page_grounded_candidate",
                page_id=page_id,
                page_title=page_title,
            )
            if intent.instruction or intent.subject or intent.old_value or intent.new_value:
                intents.append(intent)
        return intents[:20]

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

    async def _retrieve_pages(
        self,
        intents: list[ChangeIntent],
    ) -> list[tuple[ChangeIntent, list[PageCandidate]]]:
        """Retrieve candidate pages for ALL intents in parallel."""
        ranked_lists = await asyncio.gather(
            *[self._retrieve_pages_for_intent(intent) for intent in intents],
            return_exceptions=True,
        )
        out: list[tuple[ChangeIntent, list[PageCandidate]]] = []
        for intent, ranked in zip(intents, ranked_lists):
            if isinstance(ranked, Exception):
                logger.warning("Page retrieval failed for intent %r: %s", intent.subject, ranked)
                if intent.instruction or intent.subject or intent.new_value:
                    self._add_diagnostic(intent, "no_matching_page_found", "Page retrieval raised an exception.", severity="review")
                out.append((intent, []))
            else:
                out.append((intent, ranked))
        return out

    async def _retrieve_pages_for_intent(self, intent: ChangeIntent) -> list[PageCandidate]:
        """Retrieve and score candidate pages for a single intent."""
        queries = self._queries_for_intent(intent)
        pages_by_id: dict[str, PageCandidate] = {}
        if intent.page_id:
            try:
                page = await self._fetch_page_for_retrieval(intent.page_id, intent.source or "page_hint")
                page.source = intent.source or "page_hint"
                page.score = self._score_page(intent, page) + 5.0
                pages_by_id[intent.page_id] = page
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not fetch hinted candidate page %s: %s", intent.page_id, exc)
        for query in queries:
            try:
                results = await self._hybrid_search_pages(query, limit=6)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Hybrid Confluence search failed for %r: %s", query, exc)
                continue
            for result in results:
                page_id = result.get("page_id") or result.get("id")
                if not page_id or page_id in pages_by_id:
                    continue
                try:
                    page = await self._fetch_page_for_retrieval(str(page_id), str(result.get("source") or "hybrid_search"))
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Could not fetch candidate page %s: %s", page_id, exc)
                    continue
                page.source = result.get("source") or page.source or "search"
                page.score = self._score_page(intent, page) + float(result.get("retrieval_score") or 0.0)
                pages_by_id[str(page_id)] = page
        ranked = sorted(pages_by_id.values(), key=lambda p: p.score, reverse=True)
        if not ranked and (intent.instruction or intent.subject or intent.new_value):
            self._add_diagnostic(
                intent,
                "no_matching_page_found",
                "No Confluence page matched this extracted meeting intent.",
                severity="review",
            )
        return ranked[: self.max_candidate_pages]

    async def _hybrid_search_pages(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        """Search both live Confluence/Rovo and vector RAG, then merge by page."""
        clean_query = normalize_ws(query)
        if not clean_query:
            return []

        live_results: list[dict[str, Any]] = []
        rag_hits = []
        try:
            live_results = await self.client.search_pages(clean_query, limit=limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Confluence/Rovo search failed for %r: %s", clean_query, exc)
        try:
            rag_hits = await asyncio.to_thread(self.rag.search, clean_query, max(limit, self.rag_top_k))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Vector RAG search failed for %r: %s", clean_query, exc)

        merged: dict[str, dict[str, Any]] = {}
        for idx, result in enumerate(live_results):
            page_id = str(result.get("page_id") or result.get("id") or "")
            title = str(result.get("title") or page_id)
            key = page_id or title.lower()
            if not key:
                continue
            merged[key] = {
                **result,
                "page_id": page_id,
                "title": title,
                "source": result.get("source") or "confluence_search",
                "retrieval_score": float(result.get("retrieval_score") or (1.0 - min(idx, limit) * 0.03)),
            }

        for hit in rag_hits:
            key = hit.page_id or hit.title.lower()
            if not key:
                continue
            existing = merged.get(key)
            if existing:
                if "vector_rag" not in str(existing.get("source") or ""):
                    existing["source"] = f"{existing.get('source') or 'confluence_search'}+vector_rag"
                existing["rag_score"] = hit.score
                existing["rag_heading"] = hit.heading
                existing["rag_excerpt"] = hit.text
                existing["retrieval_score"] = max(float(existing.get("retrieval_score") or 0.0), hit.score + 0.25)
                continue
            merged[key] = {
                "page_id": hit.page_id,
                "title": hit.title,
                "space_key": hit.space_key,
                "version": hit.version,
                "source": "vector_rag",
                "rag_score": hit.score,
                "rag_heading": hit.heading,
                "rag_excerpt": hit.text,
                "retrieval_score": hit.score,
            }

        return sorted(
            merged.values(),
            key=lambda item: float(item.get("retrieval_score") or 0.0),
            reverse=True,
        )[:limit]

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

    def _queries_for_intent(self, intent: ChangeIntent) -> list[str]:
        values = [
            intent.old_value,
            intent.target_hint,
            intent.subject,
            intent.new_value,
            intent.instruction,
        ]
        seen: set[str] = set()
        queries: list[str] = []
        for value in values:
            q = normalize_ws(value)
            key = q.lower()
            if q and key not in seen and len(q) >= 2:
                seen.add(key)
                queries.append(q[:300])
        return queries[:5]

    def _score_page(self, intent: ChangeIntent, page: PageCandidate) -> float:
        text_norm = normalize_for_match(f"{page.title}\n{page.text}\n{page.html}")
        score = 0.0
        if intent.old_value and normalize_for_match(intent.old_value) in text_norm:
            score += 10.0
        for value, weight in ((intent.subject, 3.0), (intent.target_hint, 2.0), (intent.new_value, 1.0)):
            norm = normalize_for_match(value)
            if norm and norm in text_norm:
                score += weight
        if self._desired_task_status(intent):
            for task in extract_tasks(page.html):
                task_norm = normalize_for_match(task.get("body", ""))
                for value, weight in ((intent.subject, 8.0), (intent.target_hint, 6.0), (intent.instruction, 2.0)):
                    norm = normalize_for_match(value)
                    if norm and task_norm and (norm in task_norm or task_norm in norm):
                        score += weight
        return score

    def _desired_task_status(self, intent: ChangeIntent) -> str | None:
        action = (intent.action or "").lower()
        joined = normalize_for_match(
            " ".join([intent.new_value, intent.instruction, intent.rationale])
        )
        if action in {"complete_task", "mark_complete", "mark_done", "done"}:
            return "complete"
        if action in {"reopen_task", "mark_incomplete", "incomplete_task"}:
            return "incomplete"
        if any(phrase in joined for phrase in ("not complete", "not completed", "incomplete", "reopened", "re open")):
            return "incomplete"
        if any(word in joined.split() for word in ("complete", "completed", "done", "finished", "shipped", "closed")):
            return "complete"
        return None

    def _best_task_for_intent(
        self,
        intent: ChangeIntent,
        page: PageCandidate,
    ) -> dict[str, str] | None:
        tasks = extract_tasks(page.html)
        if not tasks:
            return None
        values = [
            intent.subject,
            intent.target_hint,
            intent.old_value,
            intent.new_value,
            intent.instruction,
        ]
        best: tuple[float, dict[str, str] | None] = (0.0, None)
        for task in tasks:
            body_norm = normalize_for_match(task.get("body", ""))
            score = 0.0
            for idx, value in enumerate(values):
                norm = normalize_for_match(value)
                if not norm or not body_norm:
                    continue
                weight = 12.0 - min(idx, 6)
                if norm == body_norm:
                    score += weight * 2
                elif norm in body_norm or body_norm in norm:
                    score += weight
                else:
                    norm_tokens = {t for t in norm.split() if len(t) > 2}
                    body_tokens = {t for t in body_norm.split() if len(t) > 2}
                    overlap = norm_tokens & body_tokens
                    if overlap:
                        score += min(5.0, len(overlap) * 1.5)
            if score > best[0]:
                best = (score, task)
        return best[1] if best[0] >= 4.0 else None

    def _metric_inline_replacement(
        self,
        intent: ChangeIntent,
        page: PageCandidate,
    ) -> tuple[str, str] | None:
        new_number = self._last_numeric_value(intent.new_value)
        if not new_number:
            return None
        metric_names = self._metric_name_candidates(intent)
        if not metric_names:
            return None

        text = page.text or html_to_text(page.html)
        candidates: list[tuple[float, str, str]] = []
        for raw_line in text.splitlines():
            line = normalize_ws(raw_line)
            if not line or len(line) > 400:
                continue
            line_norm = normalize_for_match(line)
            metric = next((name for name in metric_names if normalize_for_match(name) in line_norm), "")
            if not metric:
                continue
            replaced = self._replace_metric_number(line, metric, new_number)
            if not replaced or replaced == line:
                continue
            score = 1.0
            metric_norm = normalize_for_match(metric)
            if line_norm.startswith(metric_norm):
                score += 2.0
            if intent.target_hint and normalize_for_match(intent.target_hint) in normalize_for_match(page.title):
                score += 1.5
            context_tokens = {t for t in normalize_for_match(f"{intent.subject} {intent.target_hint} {intent.instruction}").split() if len(t) > 3}
            line_tokens = {t for t in line_norm.split() if len(t) > 3}
            score += min(2.0, len(context_tokens & line_tokens) * 0.4)
            candidates.append((score, line, replaced))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            self._add_diagnostic(
                intent,
                "ambiguous_metric_inline_match",
                "Multiple metric lines matched with equal confidence; refusing automatic inline replacement.",
                severity="review",
                page_title=page.title,
            )
            return None
        return candidates[0][1], candidates[0][2]

    def _metric_name_candidates(self, intent: ChangeIntent) -> list[str]:
        values = [intent.subject, intent.target_hint]
        match = re.search(
            r"\b([A-Za-z][A-Za-z0-9 _/-]{1,40}?)\s+(?:to|=|is|becomes?|changed?\s+to)\s+\d",
            intent.instruction,
            re.IGNORECASE,
        )
        if match:
            values.append(match.group(1))

        candidates: list[str] = []
        seen: set[str] = set()
        stopwords = {
            "change",
            "changed",
            "metric",
            "score",
            "set",
            "the",
            "to",
            "update",
            "value",
        }
        for value in values:
            cleaned = re.sub(r"\d+(?:\.\d+)?%?", " ", value or "")
            cleaned = re.sub(r"[^A-Za-z0-9 _/-]", " ", cleaned)
            cleaned = normalize_ws(cleaned)
            if not cleaned:
                continue
            words = [word for word in cleaned.split() if word.lower() not in stopwords]
            options = [cleaned, *words]
            for option in options:
                key = option.lower()
                if len(option) >= 2 and key not in seen:
                    seen.add(key)
                    candidates.append(option)
        return candidates[:8]

    def _replace_metric_number(self, line: str, metric: str, new_number: str) -> str | None:
        metric_re = re.escape(metric)
        labelled = re.compile(
            rf"(?i)(\b{metric_re}\b(?:\s+(?:score|rate|metric|value))?\s*(?:is|=|:|-|of)?\s*)(\d+(?:\.\d+)?%?)"
        )
        match = labelled.search(line)
        if match:
            if match.group(2) == new_number:
                return None
            return f"{line[:match.start(2)]}{new_number}{line[match.end(2):]}"

        numbers = list(re.finditer(r"(?<![\w.])\d+(?:\.\d+)?%?(?![\w.])", line))
        if len(numbers) != 1 or numbers[0].group(0) == new_number:
            return None
        match = numbers[0]
        return f"{line[:match.start()]}{new_number}{line[match.end():]}"

    def _last_numeric_value(self, value: str) -> str | None:
        matches = re.findall(r"(?<![\w.])\d+(?:\.\d+)?%?(?![\w.])", value or "")
        return matches[-1] if matches else None

    def _draft_proposals(
        self,
        *,
        session_id: str,
        meeting: ExtractedMeeting,
        intent_pages: list[tuple[ChangeIntent, list[PageCandidate]]],
        query: str | None,
    ) -> list[dict[str, Any]]:
        """Phase 1 (sync): deterministic drafting — skip section-level LLM refinements.
        Phase 2 (sync, parallel-ready): run collected section-level drafts and apply.
        Runs inside asyncio.to_thread so the event loop stays free throughout.
        """
        import concurrent.futures

        proposals: list[Proposal] = []
        # Collect (proposal_index, intent, page, heading, after, mode) for section drafts
        pending_refinements: list[tuple[int, ChangeIntent, PageCandidate, str | None, str, str | None]] = []
        now = _utcnow()

        for intent, pages in intent_pages:
            action = (intent.action or "replace").lower()
            if action == "create" or (not pages and action in {"create", "add"}):
                proposals.append(self._create_page_proposal(session_id, intent, now, query))
                continue

            drafted_for_intent = 0
            for page in pages:
                proposal = self._draft_for_page_no_refine(session_id, intent, page, now, query, pending_refinements, len(proposals))
                if proposal:
                    proposals.append(proposal)
                    drafted_for_intent += 1
            if drafted_for_intent == 0 and pages:
                reason = "old_value_not_found" if intent.old_value else "unsupported_by_page_content"
                detail = (
                    f"Retrieved {len(pages)} candidate page(s), but none contained the required old value."
                    if intent.old_value
                    else f"Retrieved {len(pages)} candidate page(s), but none had enough page/content fit to draft a safe proposal."
                )
                if len(pages) > 1 and not intent.old_value:
                    reason = "ambiguous_multiple_pages"
                    detail = "Multiple pages matched loosely, but no single page was safe enough for a proposal."
                self._add_diagnostic(intent, reason, detail, severity="review")

        # Phase 2: run all section-level LLM drafts in parallel using a thread pool
        if pending_refinements:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(pending_refinements))) as pool:
                futures = {
                    pool.submit(self._section_level_draft_sync, intent, page, heading, after, mode): idx
                    for idx, intent, page, heading, after, mode in pending_refinements
                }
                for future, idx in futures.items():
                    try:
                        refined = future.result(timeout=30)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Section-level draft failed for proposal %d: %s", idx, exc)
                        continue
                    if not refined or idx >= len(proposals):
                        continue
                    p = proposals[idx]
                    p.after_content = refined.get("after_content") or p.after_content
                    p.edit_mode = refined.get("edit_mode") or p.edit_mode  # type: ignore[assignment]
                    p.section_heading = refined.get("section_heading") or p.section_heading

        return [p.to_dict() for p in proposals]

    def _draft_for_page_no_refine(
        self,
        session_id: str,
        intent: ChangeIntent,
        page: PageCandidate,
        timestamp: str,
        query: str | None,
        pending_refinements: list,
        proposal_index: int,
    ) -> Proposal | None:
        """Like _draft_for_page but defers section-level LLM calls to pending_refinements."""
        action = (intent.action or "replace").lower()
        task_status = self._desired_task_status(intent)
        if task_status:
            task_proposal = self._draft_task_status_proposal(session_id, intent, page, timestamp, query, task_status)
            if task_proposal:
                return task_proposal

        text_norm = normalize_for_match(f"{page.title}\n{page.text}\n{page.html}")
        old_norm = normalize_for_match(intent.old_value)
        subject_norm = normalize_for_match(intent.subject)
        target_norm = normalize_for_match(intent.target_hint)

        old_found = bool(old_norm and old_norm in text_norm)
        subject_found = bool((subject_norm and subject_norm in text_norm) or (target_norm and target_norm in text_norm))
        title_change = action == "rename"
        remove = action == "remove"

        if action == "replace" and not old_found and not subject_found:
            return None
        if action == "add" and not subject_found and page.score < 2:
            return None

        change_type = "title" if title_change else "delete" if remove else "edit"
        before = intent.old_value if old_found and not title_change else None
        after = None if remove else (intent.new_value or intent.instruction or intent.subject)
        metric_replacement = None
        if change_type == "edit" and not before and action in {"replace", "update", "add"}:
            metric_replacement = self._metric_inline_replacement(intent, page)
            if metric_replacement:
                before, after = metric_replacement
        edit_mode: str | None = None
        if change_type == "edit":
            edit_mode = "replace" if before else "append"
            if not after:
                return None
            if self._looks_like_instruction_text(after):
                self._add_diagnostic(intent, "instruction_text_after_content",
                    "Draft after_content looked like an editing instruction rather than publishable page content.",
                    severity="review", page_title=page.title)
                return None
        if change_type == "title" and not after:
            return None
        section_heading = best_section_heading(page.html, before or intent.old_value, intent.subject, intent.target_hint, intent.new_value)

        # Defer section-level LLM refinement to Phase 2 instead of calling inline.
        if change_type == "edit" and edit_mode != "replace":
            pending_refinements.append((proposal_index, intent, page, section_heading, after, edit_mode))
            # Proposal is created now with fallback values; Phase 2 will refine it.

        anchored = old_found or metric_replacement is not None
        confidence = "high" if anchored and intent.evidence else "medium" if anchored or subject_found else "low"
        risk = "review" if change_type in {"title", "delete"} else "safe" if confidence == "high" else "review"
        score = 0.9 if confidence == "high" else 0.72 if confidence == "medium" else 0.45

        return Proposal(
            id=str(uuid.uuid4()),
            change_type=change_type,  # type: ignore[arg-type]
            page_id=page.page_id,
            page_title=page.title,
            section_heading=section_heading,
            before_content=before,
            after_content=after,
            timestamp=timestamp,
            session_id=session_id,
            rationale=intent.rationale or intent.instruction,
            generation_query=query,
            transcript_evidence=intent.evidence[:3],
            confidence=confidence,  # type: ignore[arg-type]
            risk=risk,  # type: ignore[arg-type]
            verifier_note=self._verifier_note(intent, page, old_found),
            edit_mode=edit_mode,  # type: ignore[arg-type]
            change_summary=self._change_summary(intent, page.title, change_type, before, after),
            page_url=page.url,
            confidence_score=score,
            confidence_bin=confidence,  # type: ignore[arg-type]
        )

    async def _coverage_audit(
        self,
        meeting: ExtractedMeeting,
        proposals: list[dict[str, Any]],
        transcript_text: str,
    ) -> list[dict[str, Any]]:
        """Annotate proposal set with missed-intent warnings.

        This deliberately does not invent extra proposals. It marks uncovered
        intents so the UI/API result is honest about recall gaps.
        """
        covered: set[int] = set()
        for idx, intent in enumerate(meeting.change_intents):
            intent_old = normalize_for_match(intent.old_value)
            intent_new = normalize_for_match(intent.new_value)
            intent_subj = normalize_for_match(intent.subject or intent.target_hint)
            for proposal in proposals:
                before = normalize_for_match(str(proposal.get("before_content") or ""))
                after = normalize_for_match(str(proposal.get("after_content") or ""))
                title = normalize_for_match(str(proposal.get("page_title") or ""))
                if intent_old and intent_new and intent_old == before and intent_new == after:
                    covered.add(idx)
                    break
                if intent_new and intent_new == after and intent_subj and (intent_subj in title or title in intent_subj):
                    covered.add(idx)
                    break

        missing = [
            intent
            for idx, intent in enumerate(meeting.change_intents)
            if idx not in covered and (intent.instruction or intent.subject or intent.new_value)
        ]
        for intent in missing:
            self._add_diagnostic(
                intent,
                "coverage_audit_unanchored_intent",
                "Coverage audit could not anchor this extracted intent to a proposal.",
                severity="review",
            )
        return proposals

    def _draft_for_page(
        self,
        session_id: str,
        intent: ChangeIntent,
        page: PageCandidate,
        timestamp: str,
        query: str | None,
    ) -> Proposal | None:
        action = (intent.action or "replace").lower()
        task_status = self._desired_task_status(intent)
        if task_status:
            task_proposal = self._draft_task_status_proposal(
                session_id,
                intent,
                page,
                timestamp,
                query,
                task_status,
            )
            if task_proposal:
                return task_proposal

        text_norm = normalize_for_match(f"{page.title}\n{page.text}\n{page.html}")
        old_norm = normalize_for_match(intent.old_value)
        subject_norm = normalize_for_match(intent.subject)
        target_norm = normalize_for_match(intent.target_hint)

        old_found = bool(old_norm and old_norm in text_norm)
        subject_found = bool((subject_norm and subject_norm in text_norm) or (target_norm and target_norm in text_norm))
        title_change = action == "rename"
        remove = action == "remove"

        if action == "replace" and not old_found and not subject_found:
            return None
        if action == "add" and not subject_found and page.score < 2:
            return None

        change_type = "title" if title_change else "delete" if remove else "edit"
        before = intent.old_value if old_found and not title_change else None
        after = None if remove else (intent.new_value or intent.instruction or intent.subject)
        metric_replacement = None
        if change_type == "edit" and not before and action in {"replace", "update", "add"}:
            metric_replacement = self._metric_inline_replacement(intent, page)
            if metric_replacement:
                before, after = metric_replacement
        edit_mode = None
        if change_type == "edit":
            edit_mode = "replace" if before else "append"
            if not after:
                return None
            if self._looks_like_instruction_text(after):
                self._add_diagnostic(
                    intent,
                    "instruction_text_after_content",
                    "Draft after_content looked like an editing instruction rather than publishable page content.",
                    severity="review",
                    page_title=page.title,
                )
                return None
        if change_type == "title" and not after:
            return None
        section_heading = best_section_heading(
            page.html,
            before or intent.old_value,
            intent.subject,
            intent.target_hint,
            intent.new_value,
        )
        if change_type == "edit" and edit_mode != "replace":
            refined = self._section_level_draft(intent, page, section_heading, after, edit_mode)
            if refined:
                after = refined.get("after_content") or after
                edit_mode = refined.get("edit_mode") or edit_mode
                section_heading = refined.get("section_heading") or section_heading

        anchored = old_found or metric_replacement is not None
        confidence = "high" if anchored and intent.evidence else "medium" if anchored or subject_found else "low"
        risk = "review" if change_type in {"title", "delete"} else "safe" if confidence == "high" else "review"
        score = 0.9 if confidence == "high" else 0.72 if confidence == "medium" else 0.45

        return Proposal(
            id=str(uuid.uuid4()),
            change_type=change_type,  # type: ignore[arg-type]
            page_id=page.page_id,
            page_title=page.title,
            section_heading=section_heading,
            before_content=before,
            after_content=after,
            timestamp=timestamp,
            session_id=session_id,
            rationale=intent.rationale or intent.instruction,
            generation_query=query,
            transcript_evidence=intent.evidence[:3],
            confidence=confidence,  # type: ignore[arg-type]
            risk=risk,  # type: ignore[arg-type]
            verifier_note=self._verifier_note(intent, page, old_found),
            edit_mode=edit_mode,  # type: ignore[arg-type]
            change_summary=self._change_summary(intent, page.title, change_type, before, after),
            page_url=page.url,
            confidence_score=score,
            confidence_bin=confidence,  # type: ignore[arg-type]
        )

    def _draft_task_status_proposal(
        self,
        session_id: str,
        intent: ChangeIntent,
        page: PageCandidate,
        timestamp: str,
        query: str | None,
        desired_status: str,
    ) -> Proposal | None:
        task = self._best_task_for_intent(intent, page)
        if not task:
            return None

        current_status = (task.get("status") or "incomplete").lower()
        desired = "complete" if desired_status == "complete" else "incomplete"
        if current_status == desired:
            self._add_diagnostic(
                intent,
                "task_already_in_desired_state",
                f"Task '{task.get('body')}' is already {desired}.",
                severity="info",
                page_title=page.title,
            )
            return None

        body = task.get("body") or intent.subject or intent.target_hint
        before = task_status_label(body, current_status)
        after = task_status_label(body, desired)
        section_heading = best_section_heading(page.html, body, intent.subject, intent.target_hint)
        confidence = "high" if intent.evidence else "medium"
        score = 0.88 if confidence == "high" else 0.74
        verb = "complete" if desired == "complete" else "reopen"

        return Proposal(
            id=str(uuid.uuid4()),
            change_type="edit",
            page_id=page.page_id,
            page_title=page.title,
            section_heading=section_heading,
            before_content=before,
            after_content=after,
            timestamp=timestamp,
            session_id=session_id,
            rationale=intent.rationale or intent.instruction,
            generation_query=query,
            transcript_evidence=intent.evidence[:3],
            confidence=confidence,  # type: ignore[arg-type]
            risk="review",
            verifier_note=(
                "The meeting implies this tracked task changed state, and a matching "
                "Confluence task was found on the live page."
            ),
            edit_mode="task_status",
            change_summary=f"Mark task '{body}' as {verb} on '{page.title}'",
            page_url=page.url,
            confidence_score=score,
            confidence_bin=confidence,  # type: ignore[arg-type]
        )

    def _section_level_draft(
        self,
        intent: ChangeIntent,
        page: PageCandidate,
        current_heading: str | None,
        fallback_after: str | None,
        fallback_mode: str | None,
    ) -> dict[str, str] | None:
        """Constrained LLM drafter for additive/non-exact section edits.

        Exact replacements stay deterministic. This only handles cases where
        we know the page is relevant but need the smallest publishable section
        update.
        """
        try:
            return self._section_level_draft_sync(intent, page, current_heading, fallback_after, fallback_mode)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Section-level drafter failed; using deterministic fallback: %s", exc)
            return None

    def _section_level_draft_sync(
        self,
        intent: ChangeIntent,
        page: PageCandidate,
        current_heading: str | None,
        fallback_after: str | None,
        fallback_mode: str | None,
    ) -> dict[str, str] | None:
        client = self._get_openai()
        sections = page.sections or []
        section_payload = [
            {
                "heading": s.get("heading") or "",
                "text": (s.get("text") or "")[:1200],
            }
            for s in sections[:12]
        ]
        prompt = (
            "You are a precise Confluence section drafter. Given ONE meeting intent "
            "and ONE page's sections, produce the smallest safe edit. "
            "Use only facts directly stated in the intent/evidence. "
            "Prefer append to an existing relevant section. Use create_section only when no existing section fits. "
            "Never rewrite a whole section. Return JSON only: "
            "{\"edit_mode\":\"append|create_section\", \"section_heading\": string|null, "
            "\"after_content\": string, \"reason\": string}"
        )
        payload = {
            "intent": {
                "instruction": intent.instruction,
                "subject": intent.subject,
                "target_hint": intent.target_hint,
                "new_value": intent.new_value,
                "rationale": intent.rationale,
                "evidence": intent.evidence,
            },
            "page": {
                "page_id": page.page_id,
                "title": page.title,
                "current_best_heading": current_heading,
                "sections": section_payload,
            },
            "fallback": {
                "edit_mode": fallback_mode,
                "after_content": fallback_after,
            },
        }
        opts: dict[str, Any] = {"model": self.model, "response_format": {"type": "json_object"}}
        if self.model.startswith(("gpt-5", "o1", "o3", "o4")):
            opts["max_completion_tokens"] = 900
        else:
            opts["max_tokens"] = 900
            opts["temperature"] = 0.0
        response = client.chat.completions.create(
            **opts,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        data = json.loads(response.choices[0].message.content or "{}")
        mode = str(data.get("edit_mode") or fallback_mode or "append").strip().lower()
        if mode not in {"append", "create_section"}:
            mode = "append"
        after = normalize_ws(str(data.get("after_content") or fallback_after or ""))
        if not after:
            return None
        heading = normalize_ws(str(data.get("section_heading") or current_heading or "")) or None
        return {"edit_mode": mode, "section_heading": heading or "", "after_content": after}

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
            "transcript_excerpt": transcript_text[:12000],
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

    async def _rovo_independent_critic(
        self,
        meeting: ExtractedMeeting,
        proposals: list[dict[str, Any]],
        transcript_text: str,
        *,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        """Use Rovo/Confluence retrieval as an independent critic.

        This pass searches/fetches likely pages again and asks a critic model
        whether the final proposal list misses or mis-targets anything. Results
        annotate existing cards and add diagnostics; they do not auto-create or
        auto-apply changes.
        """
        try:
            terms = self._candidate_search_terms(meeting, transcript_text, query=query)[:8]
            search_results = await asyncio.gather(
                *[self._hybrid_search_pages(term, limit=4) for term in terms],
                return_exceptions=True,
            )
            page_meta: dict[str, dict] = {}
            for results in search_results:
                if isinstance(results, Exception):
                    continue
                for result in results:
                    page_id = str(result.get("page_id") or result.get("id") or "")
                    if page_id and page_id not in page_meta:
                        page_meta[page_id] = result
                        if len(page_meta) >= 12:
                            break
                if len(page_meta) >= 12:
                    break
            if not page_meta:
                return proposals
            fetch_results = await asyncio.gather(
                *[self._fetch_page_for_retrieval(pid, str(meta.get("source") or "hybrid_search"))
                  for pid, meta in page_meta.items()],
                return_exceptions=True,
            )
            pages_by_id: dict[str, PageCandidate] = {
                pid: page
                for (pid, _), page in zip(page_meta.items(), fetch_results)
                if not isinstance(page, Exception)
            }
            if not pages_by_id:
                return proposals
            data = await asyncio.to_thread(
                self._rovo_independent_critic_sync,
                meeting,
                proposals,
                transcript_text,
                list(pages_by_id.values()),
            )
            updated = self._apply_adversarial_verdict(data, proposals)
            for missed in data.get("missed_intents") or []:
                if isinstance(missed, dict):
                    self._add_diagnostic(
                        None,
                        "rovo_critic_missed_intent",
                        normalize_ws(str(missed.get("instruction") or missed.get("reason") or "")),
                        severity="review",
                        page_title=normalize_ws(str(missed.get("page_title") or "")) or None,
                    )
            return updated
        except Exception as exc:  # noqa: BLE001
            logger.warning("Rovo independent critic failed: %s", exc)
            return proposals

    def _rovo_independent_critic_sync(
        self,
        meeting: ExtractedMeeting,
        proposals: list[dict[str, Any]],
        transcript_text: str,
        pages: list[PageCandidate],
    ) -> dict[str, Any]:
        client = self._get_openai()
        prompt = (
            "You are an independent Rovo/Confluence critic. You receive transcript, "
            "final proposals, and live Confluence page excerpts found through a separate search. "
            "Find missing changes, wrong target pages, unsupported proposals, and duplicates. "
            "Do not propose nice-to-have doc improvements. Only flag changes grounded in the transcript "
            "and the live page excerpts. Return JSON only using this schema: "
            "{\"proposal_verdicts\":[{\"id\":string,\"supported\":boolean,"
            "\"page_fit\":\"good|uncertain|wrong\",\"duplicate_of\":string|null,"
            "\"confidence\":\"high|medium|low\",\"risk\":\"safe|review|risky\",\"note\":string}],"
            "\"missed_intents\":[{\"instruction\":string,\"reason\":string,\"page_title\":string|null}]}"
        )
        payload = {
            "transcript_excerpt": transcript_text[:14000],
            "extracted_intents": [
                {
                    "instruction": i.instruction,
                    "subject": i.subject,
                    "old_value": i.old_value,
                    "new_value": i.new_value,
                    "action": i.action,
                }
                for i in meeting.change_intents[:40]
            ],
            "final_proposals": [
                {
                    "id": p.get("id"),
                    "page_title": p.get("page_title"),
                    "section_heading": p.get("section_heading"),
                    "before_content": p.get("before_content"),
                    "after_content": p.get("after_content"),
                    "change_type": p.get("change_type"),
                }
                for p in proposals[:60]
            ],
            "independent_live_pages": [
                {
                    "page_id": p.page_id,
                    "page_title": p.title,
                    "content_excerpt": (p.text or html_to_text(p.html))[:2200],
                }
                for p in pages[:12]
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
