# Phase 7: Fully Agentic Meeting Pipeline Redesign - Context

**Gathered:** 2026-04-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace the existing end-of-meeting batch summarization pipeline (SummarizerAgent → Pinecone → RetrieverAgent → OrchestratorAgent) with a real-time rolling pipeline that summarizes meeting content in configurable batches during the meeting, writes structured .md files per meeting, routes past-meeting queries through a metadata-driven History Manager Agent, and answers user questions via voice using either live meeting context or fetched historical content.

This phase does NOT add new query types, new Slack commands, or changes to the Recall.ai integration. It redesigns the data flow between transcript input and answer output.

</domain>

<decisions>
## Implementation Decisions

### Summarizer Agent — Batch Flush Trigger

- **D-01:** Flush the sentence buffer when EITHER of two conditions is met: (1) N sentences accumulated (configurable, default 10), OR (2) 2 minutes have elapsed since the last flush — whichever comes first. The time ceiling is also configurable.
- **D-02:** After a "hey jarvis" mid-meeting interrupt triggers an immediate flush, reset the sentence buffer to zero and start a new batch fresh. The count does not carry over from the interrupted batch.

### Meeting Writer Agent — .md File Format

- **D-03:** Each flushed batch is written as a new timestamped section appended to the meeting's .md file. Section header format: `## HH:MM AM/PM — Batch N` (e.g., `## 10:32 AM — Batch 3`).
- **D-04:** Speaker attribution appears at the section level, not inline. Each section ends with `**Speakers:** Alice, Bob` listing speakers active in that batch.
- **D-05:** The .md file begins with a full meeting header containing: title (derived from meeting metadata or channel name), date/time, Slack channel, and a participants list. This header is written when the meeting starts (first batch flush or meeting open event).
- **D-06:** New batch sections are always appended to the end of the .md file. Existing content is never rewritten or regenerated. This makes the file safe to read concurrently while being written.

### History Manager Agent — Past Meeting Selection

- **D-07:** The History Manager receives the user's question + the full JSON meeting index (metadata: meeting title, date/time, channel, high-level overview). An LLM call selects the best matching meeting. This handles natural language references like "last Thursday's standup" or "the Q1 planning session".
- **D-08:** If the LLM identifies multiple equally-relevant meetings (ambiguous match), the History Manager presents a disambiguation list to the user (meeting title, channel, date) and waits for selection before loading the .md file and calling the Answering Agent. This reuses the existing Phase 5 disambiguation UX pattern.

### Pinecone RAG — Coexistence Strategy

- **D-09:** Keep both retrieval paths. The History Manager uses .md files as the primary retrieval path. Pinecone is consulted as a semantic fallback when .md-based matching is unclear (e.g., no strong keyword or date match in the index). This preserves semantic search depth without requiring a full rewrite of the retrieval stack.
- **D-10:** New meeting summaries continue to be written to BOTH Pinecone (at end of meeting, existing ingestion pipeline) AND the new .md files (via Meeting Writer Agent during meeting). Belt-and-suspenders — single source of truth for voice queries is .md files, but Pinecone vectors remain available for future semantic use.

### Claude's Discretion

- File naming convention for meeting .md files (e.g., `meetings/{channel_id}/{date}_{meeting_id}.md`)
- JSON index structure and schema (fields: meeting_id, title, date, channel, overview, md_path, participants)
- How the Meeting Writer Agent is invoked (as a separate agent class or as a utility function called by the Summarizer)
- Exact LLM prompt for History Manager's meeting selection step

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Existing Agent Implementations
- `agents/summarizer.py` — Current SummarizerAgent (processes end-of-meeting transcript → MeetingRecord). New Summarizer Agent replaces its trigger model (rolling batches vs. end-of-meeting), but the structured output pattern (Pydantic output_type) should be preserved.
- `agents/orchestrator.py` — OrchestratorAgent with existing disambiguation flow (lines ~200+). History Manager's disambiguation should reuse or extend this pattern.
- `agents/answer_agent.py` — AnswerAgent that Phase 7's Answering Agent will call with .md content as context.

### Storage Layer
- `storage/models.py` — MeetingRecord Pydantic model. New JSON index schema should be compatible or extend this.
- `storage/metadata_store.py` — MetadataStore for JSON read/write. New Meeting Writer Agent should follow this async I/O pattern.
- `storage/pinecone_client.py` — Existing Pinecone upsert/query. Phase 7 keeps this active for end-of-meeting ingestion.

### Core Application
- `jarvis.py` — WebSocket event handler (wake-word detection, transcript accumulation, handle_query). The Summarizer Agent's sentence buffer lives here or is wired from here. The "hey jarvis" interrupt flush must integrate with the existing wake-word detection path.

### Planning Artifacts
- `.planning/STATE.md` §Decisions — All Phase 1-6 decisions that constrain this phase (asyncio.Lock pattern, gpt-4o-mini model, agents-as-tools pattern, disambiguation trigger logic).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `agents/summarizer.py` `SummarizerAgent`: Structured LLM output via `Agent(output_type=SummaryOutput)` — reuse this pattern for the new rolling Summarizer Agent
- `agents/orchestrator.py` disambiguation flow: Already implements "present list, wait for selection" — reuse for History Manager's ambiguous-match path
- `agents/answer_agent.py` `AnswerAgent`: Receives context string + question, returns `AnswerOutput`. Will be called by the new pipeline with .md file content as context.
- `storage/metadata_store.py` `MetadataStore`: Async JSON file I/O with `asyncio.Lock`. New JSON index writes should follow the same locking pattern.
- `jarvis.py` `_split_sentences()` + `speak_chunked()`: Sentence-level utilities added in Phase 6. The Summarizer's sentence counting can reuse `_split_sentences()`.
- `jarvis.py` wake-word handler: Already detects "hey jarvis" and spawns `handle_query`. The interrupt flush must hook into this exact detection path.

### Established Patterns
- `asyncio.Lock` for all shared state (Phase 1 mandate) — sentence buffer and .md write operations must be lock-protected
- `gpt-4o-mini` for all LLM calls (Phase 6 mandate) — History Manager selection LLM call + Answering Agent both use this model
- `Agent(output_type=PydanticModel)` for structured agent output — use for new Summarizer batch output
- `asyncio.create_task()` for fire-and-forget from WebSocket handler — Meeting Writer invocation should follow this

### Integration Points
- `jarvis.py` WebSocket transcript handler: New sentence buffer lives here; append to buffer on each `transcript.data` event; check count + time ceiling; flush on threshold or interrupt
- `jarvis.py` `handle_query()`: History Manager is invoked from here when query is classified as `memory_query`; current meeting context comes from the .md file rather than raw `transcript_log`
- `meetings/` directory: Already exists with 2 JSON files from prior meetings — new .md files go alongside these

</code_context>

<specifics>
## Specific Ideas

- The "rolling summarizer" feel is key — the .md file should be readable live during the meeting, not just as a post-processing artifact
- The Meeting Writer Agent's output should look like a professional transcription service document — clean sections, not raw LLM output
- The JSON index is the fast lookup path; the .md file is the full-content path. History Manager reads index to pick the meeting, then reads .md for full content to pass to Answering Agent.
- The interrupt flush ("hey jarvis" mid-batch) should be immediate and blocking — the Answering Agent should receive the just-flushed batch content as part of its context for that query

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 07-fully-agentic-meeting-pipeline-redesign*
*Context gathered: 2026-04-05*
