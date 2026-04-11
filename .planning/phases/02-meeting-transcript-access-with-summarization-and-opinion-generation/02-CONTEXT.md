# Phase 2: Meeting Transcript Access with Summarization and Opinion Generation - Context

**Gathered:** 2026-04-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Give Jarvis awareness of the full meeting conversation so it can:
1. Summarize everything spoken since the bot joined when asked
2. Form and deliver confident, first-person opinions grounded in what was discussed

This phase covers routing, transcript access, and the two new response handlers.
Real-time transcription, UI, or persistent storage of summaries are out of scope.

</domain>

<decisions>
## Implementation Decisions

### Classifier Intents
- **D-01:** Add two new classifier intents: `meeting_summary` and `meeting_opinion`, alongside existing `confluence` and `general` intents.
- **D-02:** `meeting_summary` catches phrases like "summarize the meeting", "what was said", "catch me up", "what did I miss".
- **D-03:** `meeting_opinion` catches phrases like "what do you think", "what's your take", "how should we proceed", "which option is better".
- **D-04:** Both new intents are routed in `handle_spoken_request()` before the existing general question fallback.

### Transcript Scope
- **D-05:** Both summarization and opinion handlers use the **full** `meeting_state["transcript_log"]` — everything since the bot joined.
- **D-06:** Transcript is passed to the LLM as a formatted string of `"[Participant]: [text]"` lines, in chronological order.
- **D-07:** If `transcript_log` is empty, Jarvis responds with a short TTS-friendly fallback: "I haven't heard anything in the meeting yet."

### Opinion Personality
- **D-08:** Opinions are delivered in **confident first-person**: "I think you should go with X because..."
- **D-09:** Jarvis briefly acknowledges the meeting source at the start: "Based on what I heard..." or "From the discussion..." — one short grounding phrase, then the opinion.
- **D-10:** Opinions are grounded in what was actually discussed — Jarvis references specific points or trade-offs it heard, not generic knowledge alone.

### Response Length (env-configurable)
- **D-11:** Summary token cap: `JARVIS_SUMMARY_MAX_TOKENS` (default: 400).
- **D-12:** Opinion token cap: `JARVIS_OPINION_MAX_TOKENS` (default: 200).
- **D-13:** Both handlers follow the Phase 1 convention of TTS-optimized responses: no markdown, no bullet points, spoken in natural sentences.
- **D-14:** Use same model as general responder: `JARVIS_GENERAL_MODEL` (default: gpt-4o-mini).

### Claude's Discretion
- Exact LLM system prompt wording for each handler
- How to truncate the transcript if it exceeds LLM context limits (simple truncation from the start is fine)
- Whether to extract participant names for the summary prompt header

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Existing pipeline (Phase 1 artifacts to build on)
- `confluence_logic/classifier.py` — Current classifier structure; new intents extend `classify_intent()` return values
- `confluence_logic/jarvis_agentic.py` — `handle_spoken_request()` routing logic, `meeting_state["transcript_log"]` structure, `_handle_general_question()` pattern to follow for new handlers
- `confluence_logic/general_responder.py` — Reference implementation for LLM response handlers (model, token cap, system prompt pattern, TTS conventions)

No external specs — requirements are fully captured in decisions above.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `meeting_state["transcript_log"]`: Already populated by the WebSocket handler with `{participant, text, timestamp}` dicts — no new data capture needed.
- `general_responder.py` → `answer_general_question()`: Template for new `summarize_meeting()` and `generate_opinion()` functions (same model, env-configurable token cap, TTS-safe system prompt pattern).
- `classifier.py` → `classify_intent()`: Returns a string intent. Extending to return `"meeting_summary"` or `"meeting_opinion"` requires adding to the LLM prompt's category list and fast-path heuristics.
- `audio_cache.py` → `get_random_ack_audio()`: Already used for confluence tasks. Can also play an ack before summary generation (since summaries take longer).

### Established Patterns
- Async fire-and-forget via `asyncio.create_task()` — use same pattern for new handlers.
- `JARVIS_*` env vars for runtime configuration — follow same naming convention.
- Fallback to live TTS when audio cache is empty — apply same fallback to ack before summary.

### Integration Points
- `handle_spoken_request()` in `jarvis_agentic.py`: Add two new routing branches after the classifier gate (currently routes `general` and `confluence`).
- `generate_wav_assets.py`: No changes needed — ack audio is already shared.

</code_context>

<specifics>
## Specific Ideas

- Example trigger from user: "Hey Jarvis, what do you think is the best way to proceed?" after a team discussion about OpenAI model selection — Jarvis should reference the choices discussed and give a direct recommendation.
- Jarvis should sound like a thoughtful colleague, not a meeting recorder. For opinions, it picks a side.
- Grounding phrase examples: "Based on what I heard...", "From the discussion so far...", "Given what the team discussed..."

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 02-meeting-transcript-access-with-summarization-and-opinion-generation*
*Context gathered: 2026-04-11*
