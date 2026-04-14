---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_plan: Not started
status: Milestone complete
stopped_at: "Completed 05-01-PLAN.md: speech rewriter, name-aware fillers, multi-turn referencing"
last_updated: "2026-04-14T05:21:14.798Z"
progress:
  total_phases: 5
  completed_phases: 5
  total_plans: 11
  completed_plans: 11
  percent: 91
---

# Project State

## Current Status

- **Active Milestone:** Milestone 1 — Jarvis Intelligence Enhancement
- **Active Phase:** Phase 01 — Intelligent Question Classification and Conversational Response
- **Current Plan:** Not started
- **Progress:** [█████████░] 91%

## Accumulated Context

### Roadmap Evolution

- Initial roadmap created for Milestone 1: Jarvis Intelligence Enhancement
- Phase 1 added: Intelligent Question Classification and Conversational Response
- Phase 2 added: Meeting transcript access with summarization and opinion generation
- Phase 5 added: Human-likeness improvements: response length rewriter for verbal delivery, name-aware fillers using invoker_participant, micro-ack on wake detection, interruption recovery phrase, multi-turn referencing in prompts, conversational pacing after delivery

### Decisions Made

- **01-01:** gpt-4o-mini with temperature=0 used for LLM fallback classification (deterministic, cost-effective)
- **01-01:** Classifier defaults to 'confluence' on error (safe fallback to existing pipeline)
- **01-01:** MP3 format chosen for audio cache to match Recall API 'kind': 'mp3' requirement
- **01-01:** Lazy audio cache loading to avoid startup cost when not needed
- **01-02:** gpt-4o-mini default for general responder (configurable via JARVIS_GENERAL_MODEL env var)
- **01-02:** max_tokens=150 caps response length for TTS-optimized concise answers
- **01-02:** Classifier runs before state_lock acquisition — lightweight, non-blocking routing
- **01-02:** asyncio.create_task() for general questions — fire-and-forget, non-blocking pipeline
- [Phase 01]: 15-second clarification timeout default (JARVIS_GENERAL_CLARIFICATION_TIMEOUT env var) — short enough to feel conversational, long enough for natural response
- [Phase 01]: Clarification state cleared immediately at handler start — prevents stale state if handler errors mid-way
- [Phase 02]: JARVIS_SUMMARY_MAX_TOKENS defaults to 400, JARVIS_OPINION_MAX_TOKENS to 200 — token caps for meeting summarizer and opinion generator
- [Phase 02]: MEETING_RESPONDER_MODEL reads JARVIS_GENERAL_MODEL env var — shares model config with general responder
- [Phase 02]: D-04: meeting_summary/meeting_opinion routing branches placed after general branch but before Confluence pipeline
- [Phase 03]: _MAX_GENERAL_HISTORY set to 3: stale topic context ages out after 3 Q&A exchanges (TOPIC-01)
- [Phase 03]: async _needs_web_search uses gpt-4o-mini (max_tokens=5, temperature=0.0) — LLM routing replaces regex, generalizes to weather/sports/news (WEBSEARCH-01)
- [Phase 03]: Simple keyword extraction for graph query_context MVP — no LLM call, zero latency; upgrade to entity extraction if entity name mismatch causes misses (GRAPHRAG-01)
- [Phase 03]: graph_context injected into system_prompt (not user message) — meeting facts in model instruction context influence entire general response (CLASSIFY-03)
- [Phase 04]: JARVIS_DEBOUNCE_SECONDS defaults to 1.0s — long enough to catch mid-sentence continuation, short enough to feel responsive
- [Phase 04]: _debounced_dispatch uses asyncio.sleep + cancel-and-restart pattern with accumulated text growing across segments
- [Phase 04]: Filler awaited directly (not via create_task) in _handle_general_question — no concurrent work to overlap, filler IS the gap
- [Phase 04]: force_web_search parameter added to _handle_general_question enabling web_search intent routing
- [Phase 05]: REWRITE-01: _rewrite_for_speech uses gpt-4o-mini (max_tokens=120, temperature=0.5) — condenses >80 char answers to 2-3 spoken sentences with elaboration offer
- [Phase 05]: NAME-01: _get_clean_invoker_name rejects UUIDs/emails/single-chars/all-caps-acronyms — returns first token of invoker_participant
- [Phase 05]: MULTITURN-01: multiturn_reference param appended to system_prompt when conversation history exists in _handle_general_question

### Performance Metrics

| Phase | Plan | Duration | Tasks | Files |
|-------|------|----------|-------|-------|
| 01    | 01   | ~2 min   | 2/2   | 5     |
| 01    | 02   | ~3 min   | 2/2   | 2     |
| Phase 01 P03 | 1m | 1 tasks | 1 files |
| Phase 02 P01 | 4 min | 1 tasks | 2 files |
| Phase 02 P02 | 5 min | 2 tasks | 2 files |
| Quick 260411-le0 | 5 min | 2 tasks | 1 files |
| Quick 260411-mce | 3 min | 2 tasks | 1 files |
| Quick 260411-vfq | 4 min | 2 tasks | 2 files |
| Phase 03 P01 | 3 min | 2 tasks | 5 files |
| Phase 03 P02 | 8 min | 2 tasks | 5 files |
| Phase 04 P01 | 8 min | 2 tasks | 2 files |
| Phase 04 P02 | 5 min | 2 tasks | 2 files |
| Phase 05 P01 | 10 min | 2 tasks | 2 files |

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260411-kvc | Add immediate acknowledgment and brief/detailed clarification flow to meeting summary handler | 2026-04-11 | 32dc63a | [260411-kvc-add-immediate-acknowledgment-and-brief-d](./quick/260411-kvc-add-immediate-acknowledgment-and-brief-d/) |
| 260411-le0 | Smart time-filler before slow ops and no-wake-word re-arm after clarification follow-up questions | 2026-04-11 | 3c19f1c | [260411-le0-smart-time-filler-and-no-wake-word-after](./quick/260411-le0-smart-time-filler-and-no-wake-word-after/) |
| 260411-mce | Serialize TTS output to prevent audio overlap — hold output_lock for estimated playback duration after audio POST | 2026-04-11 | 2d00290 | [260411-mce-serialize-tts-output-to-prevent-audio-ov](./quick/260411-mce-serialize-tts-output-to-prevent-audio-ov/) |
| 260411-mk0 | Only dispatch transcript events when is_final=True to prevent mid-sentence response triggers | 2026-04-11 | e144ef5 | [260411-mk0-only-dispatch-transcript-events-when-is-](./quick/260411-mk0-only-dispatch-transcript-events-when-is-/) |
| 260411-nai | Fix context bleed from Confluence editor history and add selective DuckDuckGo web search | 2026-04-11 | ddaf91b | [260411-nai-fix-context-bleed-in-general-question-ha](./quick/260411-nai-fix-context-bleed-in-general-question-ha/) |
| 260411-vfq | Add cross-handler conversation memory so follow-ups reference prior opinion/summary answers | 2026-04-11 | 6ef2fe3 | [260411-vfq-add-cross-handler-conversation-memory-to](./quick/260411-vfq-add-cross-handler-conversation-memory-to/) |

### Session

- **Last session:** 2026-04-14T05:03:35.656Z
- **Stopped at:** Completed 05-01-PLAN.md: speech rewriter, name-aware fillers, multi-turn referencing
