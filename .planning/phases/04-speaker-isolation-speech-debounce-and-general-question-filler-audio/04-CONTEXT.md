# Phase 4: Speaker Isolation, Speech Debounce, and General Question Filler Audio - Context

**Gathered:** 2026-04-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix three pipeline reliability and UX issues in the Jarvis WebSocket transcript handler:
1. **Speaker isolation** — when multiple mics are active, only the participant who said "Hey Jarvis" should drive the query. Other speakers' transcripts currently bleed into dispatch.
2. **Speech-completion debounce** — final transcript segments dispatch immediately; user may still be thinking. Add a 1-second silence window before processing.
3. **General question filler audio** — `_handle_general_question` has no filler between wake detection and the LLM response arriving. Plays contextual filler to eliminate the awkward silence gap (confluence tasks already do this).

This phase does NOT change the classifier, Graph RAG, conversation history, or TTS pipeline.

</domain>

<decisions>
## Implementation Decisions

### Speaker Isolation
- **D-01:** When `_WAKE_PATTERN` matches a participant's transcript, record `meeting_state["invoker_participant"] = participant` (the speaker's name string from `data_block["participant"]["name"]`)
- **D-02:** During the debounce window, only text segments where `participant == meeting_state["invoker_participant"]` are accepted and accumulated. Other participants' segments are logged but not dispatched.
- **D-03:** Lock resets to `None` after the query is dispatched to `handle_spoken_request` — next "Hey Jarvis" from any participant starts a new lock.
- **D-04:** Add `"invoker_participant": None` to `meeting_state` initial dict.

### Speech-Completion Debounce (1-second window)
- **D-05:** Replace the immediate `asyncio.create_task(handle_spoken_request(query, bot_id))` dispatch with a debounced pattern using a cancellable task.
- **D-06:** Implementation: maintain `meeting_state["_pending_debounce_task"]` (an `asyncio.Task | None`). When an invoker's final segment arrives, cancel any existing debounce task and schedule a new one: `asyncio.create_task(_debounced_dispatch(accumulated_query, bot_id))` where `_debounced_dispatch` does `await asyncio.sleep(1.0)` then calls `handle_spoken_request`.
- **D-07:** If another final segment from the **same invoker** arrives within the 1-second window, cancel the pending task and create a new one with the **accumulated** text (append new segment to prior query text, separated by a space).
- **D-08:** After dispatch fires (sleep completes), clear `meeting_state["invoker_participant"]` and `meeting_state["_pending_debounce_task"]`.
- **D-09:** Add `"_pending_debounce_task": None` to `meeting_state` initial dict.

### General Question Filler Audio
- **D-10:** At the start of `_handle_general_question`, before the `answer_general_question` await, call `_generate_contextual_gap_filler(query)` then speak it via `_speak_guarded(..., allow_stale=True)`.
- **D-11:** `_generate_contextual_gap_filler(query)` already exists at line 645 of `jarvis_agentic.py` — reuse it as-is. It uses gpt-4o-mini to produce a short query-aware acknowledgement (≤10 words).
- **D-12:** The filler is spoken with `allow_stale=True` so it doesn't get suppressed by the generation guard while the main LLM call is in-flight.
- **D-13:** This mirrors how confluence tasks use cached ack audio (lines 791–799) — same UX pattern, just using contextual text filler instead of cached MP3 for general questions.

### Claude's Discretion
- Whether to store the accumulated query text in `meeting_state` or only in the debounce task closure
- Exact log messages for filtered/dropped transcripts from non-invoker participants
- Whether bare wake invocations ("Hey Jarvis" with no query) also trigger the debounce window or bypass it immediately

</decisions>

<specifics>
## Specific Ideas

- The 1-second debounce constant should be configurable via env var: `JARVIS_DEBOUNCE_SECONDS` defaulting to `1.0`
- Non-invoker transcripts during an active lock should log at DEBUG level: "Ignoring transcript from {participant} — active invoker is {invoker_participant}"
- The debounce cancellation+restart is the same pattern as many autocomplete debounce implementations — cancel old task, create new one

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Core Pipeline
- `confluence_logic/jarvis_agentic.py` — WebSocket handler (lines ~1355–1414), `process_transcript_event()` (line 1310), `_handle_general_question()` (line ~987), `_speak_guarded()` (line 615), `_generate_contextual_gap_filler()` (line 645), `meeting_state` dict (line 125)
- `confluence_logic/audio_cache.py` — `get_random_ack_audio()` — existing cached ack pattern for reference (NOT used for general filler — contextual filler is used instead)

### Prior Phase Context
- `.planning/phases/01-intelligent-question-classification-and-conversational-response/01-CONTEXT.md` — WAV/ack caching decisions from Phase 1
- `.planning/phases/03-classifier-and-context-intelligence/03-CONTEXT.md` — graph_rag wiring decisions (transcript append hook at line ~1401)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_generate_contextual_gap_filler(query)` (line 645): already exists, returns a short LLM-generated ack phrase for the query — use this for D-10/D-11
- `_speak_guarded(text, bot_id, generation, allow_stale=True)` (line 615): use with `allow_stale=True` for filler so it isn't generation-guarded
- `meeting_state["last_user_speech_at"]` (line 142): already tracks speech timing; debounce window is separate from this (different purpose)

### Established Patterns
- Fire-and-forget with `asyncio.create_task()` — used throughout; debounce task follows same pattern
- `meeting_state` dict is the central mutable state store — add `invoker_participant` and `_pending_debounce_task` here
- All blocking I/O wrapped in `asyncio.to_thread()` — `_generate_contextual_gap_filler` already follows this

### Integration Points
- WebSocket handler `while True` loop (lines ~1359–1410): speaker filter and debounce logic slots in here, replacing the immediate `asyncio.create_task(handle_spoken_request(...))` call
- `_handle_general_question` (line ~987): add filler call at the top, before `answer_general_question` await
- `meeting_state` initial dict (line 125): add `"invoker_participant": None` and `"_pending_debounce_task": None`

</code_context>

<deferred>
## Deferred Ideas

- Persistent speaker identity across sessions (voice fingerprinting) — out of scope
- Debounce window per-intent (shorter for confluence commands, longer for questions) — backlog
- Visual indicator to the invoker that Jarvis is listening (not feasible without UI access) — out of scope

</deferred>

---

*Phase: 04-speaker-isolation-speech-debounce-and-general-question-filler-audio*
*Context gathered: 2026-04-13*
