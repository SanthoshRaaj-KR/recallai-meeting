# Phase 7: Fully Agentic Meeting Pipeline Redesign - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-05
**Phase:** 07-fully-agentic-meeting-pipeline-redesign
**Areas discussed:** Batch flush trigger, Meeting .md file format, History Manager selection, Pinecone RAG strategy

---

## Batch Flush Trigger

| Option | Description | Selected |
|--------|-------------|----------|
| Sentence count only | Flush every N sentences. Simple and predictable. | |
| Sentence count + time ceiling | Flush at N sentences OR every T minutes, whichever first. | ✓ |
| You decide | Claude picks strategy. | |

**User's choice:** Sentence count + time ceiling
**Notes:** Default N=10 sentences, default T=2 minutes. Both should be configurable.

---

## After Interrupt Flush

| Option | Description | Selected |
|--------|-------------|----------|
| Start fresh | Reset buffer to zero after flush. | ✓ |
| Continue count | Resume count from where it was. | |
| You decide | Claude decides. | |

**User's choice:** Start fresh
**Notes:** After "hey jarvis" triggers an immediate flush, the sentence counter resets to zero.

---

## Meeting .md File Format — Batch Presentation

| Option | Description | Selected |
|--------|-------------|----------|
| Timestamped sections | Each batch = new section with timestamp header. | ✓ |
| Continuous narrative | Batches appended as flowing paragraphs. | |
| You decide | Claude picks most professional format. | |

**User's choice:** Timestamped sections
**Notes:** Format: `## 10:32 AM — Batch 3`

---

## Meeting .md File Format — Speaker Attribution

| Option | Description | Selected |
|--------|-------------|----------|
| Section-level list | `**Speakers:** Alice, Bob` at end of each section. | ✓ |
| Inline per sentence | Speaker names embedded in prose. | |
| Both | Section header + inline attribution. | |

**User's choice:** Section-level list
**Notes:** Clean, doesn't clutter prose.

---

## Meeting .md File Format — Header

| Option | Description | Selected |
|--------|-------------|----------|
| Full meeting header | Title, date/time, channel, participants list at top. | ✓ |
| Minimal header | Just meeting ID and start time. | |
| You decide | Claude designs sensible header. | |

**User's choice:** Full meeting header

---

## Meeting .md File — Write Mode

| Option | Description | Selected |
|--------|-------------|----------|
| Append new section | New batches appended, existing content never touched. | ✓ |
| Regenerate entire file | Rewrite full .md from all batches each time. | |
| You decide | Claude picks safer approach. | |

**User's choice:** Append new section

---

## History Manager — Selection Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| LLM reads index + decides | Pass JSON index + question to LLM; LLM picks best match. | ✓ |
| Keyword + date rule matching | String similarity + dateparser, no LLM call for selection. | |
| You decide | Claude picks balance of accuracy and cost. | |

**User's choice:** LLM reads index + decides
**Notes:** Handles natural language date references like "last Thursday's standup".

---

## History Manager — Multiple Match Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Pick single best match | LLM or rule picks most relevant one. | |
| Load top 2-3 meetings | Pass multiple meeting contexts to Answering Agent. | |
| Ask user to disambiguate | Present list, wait for user choice. | ✓ |

**User's choice:** Ask user to disambiguate
**Notes:** Reuses existing Phase 5 disambiguation UX pattern from OrchestratorAgent.

---

## Pinecone — Retrieval Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Fully replace with .md retrieval | Remove/disable Pinecone queries. | |
| Leave Pinecone but stop querying | Keep code, don't query in new pipeline. | |
| Keep both paths (hybrid) | .md files primary, Pinecone semantic fallback. | ✓ |

**User's choice:** Keep both paths (hybrid)
**Notes:** History Manager uses .md as primary, Pinecone as fallback for unclear matches.

---

## Pinecone — New Meeting Write Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Only .md files going forward | No new Pinecone upserts. | |
| Write to both | Continue Pinecone upserts + new .md files. | ✓ |
| You decide | Claude picks based on architecture fit. | |

**User's choice:** Write to both
**Notes:** Belt-and-suspenders. .md files are primary for voice queries; Pinecone remains for future semantic use.

---

## Claude's Discretion

- File naming convention for meeting .md files
- JSON meeting index schema
- Whether Meeting Writer Agent is a separate agent class or utility called by Summarizer
- Exact LLM prompt design for History Manager's selection step

## Deferred Ideas

None.
