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

from .confluence import HybridConfluenceClient
from .models import ChangeIntent, ExtractedMeeting, PageCandidate, Proposal
from .text_utils import (
    append_new_section,
    best_section_heading,
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
        self._client: HybridConfluenceClient | None = None
        self._openai: OpenAI | None = None
        self.last_diagnostics: list[dict[str, Any]] = []

    async def run(
        self,
        *,
        session_id: str,
        transcript: list[dict[str, Any]],
        query: str | None = None,
        emit: EmitFn | None = None,
    ) -> tuple[ExtractedMeeting, list[dict[str, Any]]]:
        self.last_diagnostics = []

        async def _emit(event: dict[str, Any]) -> None:
            if emit:
                await emit(event)

        await _emit({"type": "stage_start", "stage": "transcript_source"})
        transcript_text = format_transcript(transcript)
        if not transcript_text:
            meeting = ExtractedMeeting(
                title="Meeting Review",
                summary="No transcript was captured for this session yet.",
            )
            return meeting, []

        await _emit({"type": "stage_start", "stage": "fact_extraction"})
        meeting = await self._extract_meeting(transcript, transcript_text, query=query)

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
        proposals = self._draft_proposals(
            session_id=session_id,
            meeting=meeting,
            intent_pages=intent_pages,
            query=query,
        )

        await _emit({"type": "stage_start", "stage": "verification"})
        verified = self._verify_and_dedupe(proposals)
        verified = await self._adversarial_verify(meeting, verified, transcript_text)
        verified = await self._rovo_independent_critic(meeting, verified, transcript_text, query=query)
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

    def _extract_meeting_sync(self, transcript_text: str, query: str) -> dict[str, Any]:
        client = self._get_openai()
        prompt = (
            "You extract documentation changes from meeting transcripts. "
            "Return JSON only. Capture the FINAL agreed state, not intermediate suggestions. "
            "Create one atomic change_intent per distinct documentation update. "
            "Prefer precise old_value/new_value pairs when the transcript says a page has an outdated value. "
            "Also extract indirect state changes that imply a documentation update, including items "
            "being completed, shipped, cancelled, deferred, blocked, unblocked, approved, rejected, "
            "renamed, owned by someone else, or moved to a different date/status. "
            "For checklist/task updates use action='complete_task' when the meeting says an item is done "
            "and action='reopen_task' when it is no longer done. For general status fields use action='replace' "
            "with old_value only when stated or inferable from the current wording. "
            "Changes may target multiple pages; keep each intent atomic and include the best page/title hint. "
            "Do not invent facts. If no Confluence/doc change is needed, return change_intents: [].\n\n"
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

        pages_by_id: dict[str, PageCandidate] = {}
        for term in search_terms:
            try:
                results = await self.client.search_pages(term, limit=5)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Page-grounded candidate search failed for %r: %s", term, exc)
                continue
            for result in results:
                page_id = str(result.get("page_id") or result.get("id") or "")
                if not page_id or page_id in pages_by_id:
                    continue
                try:
                    page = await self.client.fetch_page(page_id)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Page-grounded candidate fetch failed for %s: %s", page_id, exc)
                    continue
                pages_by_id[page_id] = page
                if len(pages_by_id) >= self.max_candidate_pages:
                    break
            if len(pages_by_id) >= self.max_candidate_pages:
                break

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
            "You are a strict Confluence change candidate generator. "
            "You receive a meeting transcript and live Confluence page excerpts. "
            "Return candidate changes ONLY when the transcript says something should change "
            "and the page excerpt shows where it belongs or what old value exists. "
            "Treat page lines like '[task: incomplete] Q2 plan' as Confluence checkboxes/tasks; "
            "if the transcript says the task is complete, done, shipped, closed, or fully finished, "
            "return action='complete_task' with new_value='complete'. "
            "If the transcript says it is reopened or no longer complete, return action='reopen_task'. "
            "For non-checkbox status or text updates, infer replace/add changes when the page excerpt "
            "contains the relevant section or current status. "
            "Do not make general documentation improvements. Do not invent page IDs. "
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
        out: list[tuple[ChangeIntent, list[PageCandidate]]] = []
        for intent in intents:
            queries = self._queries_for_intent(intent)
            pages_by_id: dict[str, PageCandidate] = {}
            if intent.page_id:
                try:
                    page = await self.client.fetch_page(intent.page_id)
                    page.source = intent.source or "page_hint"
                    page.score = self._score_page(intent, page) + 5.0
                    pages_by_id[intent.page_id] = page
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Could not fetch hinted candidate page %s: %s", intent.page_id, exc)
            for query in queries:
                try:
                    results = await self.client.search_pages(query, limit=6)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Confluence search failed for %r: %s", query, exc)
                    continue
                for result in results:
                    page_id = result.get("page_id") or result.get("id")
                    if not page_id or page_id in pages_by_id:
                        continue
                    try:
                        page = await self.client.fetch_page(str(page_id))
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Could not fetch candidate page %s: %s", page_id, exc)
                        continue
                    page.source = result.get("source") or page.source or "search"
                    page.score = self._score_page(intent, page)
                    pages_by_id[str(page_id)] = page
            ranked = sorted(pages_by_id.values(), key=lambda p: p.score, reverse=True)
            if not ranked and (intent.instruction or intent.subject or intent.new_value):
                self._add_diagnostic(
                    intent,
                    "no_matching_page_found",
                    "No Confluence page matched this extracted meeting intent.",
                    severity="review",
                )
            out.append((intent, ranked[: self.max_candidate_pages]))
        return out

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

    def _draft_proposals(
        self,
        *,
        session_id: str,
        meeting: ExtractedMeeting,
        intent_pages: list[tuple[ChangeIntent, list[PageCandidate]]],
        query: str | None,
    ) -> list[dict[str, Any]]:
        proposals: list[Proposal] = []
        now = _utcnow()
        for intent, pages in intent_pages:
            action = (intent.action or "replace").lower()
            if action == "create" or (not pages and action in {"create", "add"}):
                proposals.append(self._create_page_proposal(session_id, intent, now, query))
                continue

            drafted_for_intent = 0
            for page in pages:
                proposal = self._draft_for_page(session_id, intent, page, now, query)
                if proposal:
                    proposals.append(proposal)
                    drafted_for_intent += 1
                    # Exact old-value matches are the highest-signal path. One good page
                    # is enough unless retrieval finds another exact duplicate later.
                    if intent.old_value and proposal.before_content:
                        break
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
        return [p.to_dict() for p in proposals]

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
        if missing:
            note = (
                "Coverage audit: no proposal could be anchored for "
                + "; ".join((m.instruction or m.subject)[:120] for m in missing[:5])
            )
            for proposal in proposals:
                existing = proposal.get("verifier_note") or ""
                proposal["verifier_note"] = f"{existing} {note}".strip()
                if proposal.get("risk") == "safe":
                    proposal["risk"] = "review"
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
        edit_mode = None
        if change_type == "edit":
            edit_mode = "replace" if before else "append"
            if not after:
                return None
        if change_type == "title" and not after:
            return None
        section_heading = best_section_heading(
            page.html,
            intent.old_value,
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

        confidence = "high" if old_found and intent.evidence else "medium" if old_found or subject_found else "low"
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
        return out

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
        return proposals

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
            pages_by_id: dict[str, PageCandidate] = {}
            for term in terms:
                try:
                    results = await self.client.search_pages(term, limit=4)
                except Exception:
                    continue
                for result in results:
                    page_id = str(result.get("page_id") or result.get("id") or "")
                    if not page_id or page_id in pages_by_id:
                        continue
                    try:
                        pages_by_id[page_id] = await self.client.fetch_page(page_id)
                    except Exception:
                        continue
                    if len(pages_by_id) >= 12:
                        break
                if len(pages_by_id) >= 12:
                    break
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
        return {"success": True, "message": "Applied change to Confluence."}

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


def _utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


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
