# Phase 2: Meeting Transcript Access - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-11
**Phase:** 02-meeting-transcript-access-with-summarization-and-opinion-generation
**Areas discussed:** Trigger phrases, Transcript scope, Opinion personality, Response length & format

---

## Trigger Phrases

| Option | Description | Selected |
|--------|-------------|----------|
| New classifier intent | Add `meeting_summary` and `meeting_opinion` as classifier outputs | ✓ |
| Keyword detection in general handler | Detect phrases inside `_handle_general_question()`, inject context | |
| Always inject transcript | Every general question gets transcript context | |

**User's choice:** New classifier intent
**Notes:** Two distinct intents — one for summarization, one for opinions. Keeps routing clean.

---

## Transcript Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Full meeting | Everything since bot joined, truncated to fit LLM context | ✓ |
| Last N minutes | Configurable window via env var | |
| Smart split | Full for summary, recent for opinion | |

**User's choice:** Full meeting (everything since bot joined)
**Notes:** `transcript_log` already captures the full history. Both handlers get the full transcript.

---

## Opinion Personality

| Option | Description | Selected |
|--------|-------------|----------|
| Confident first-person | "I think you should go with X because..." — genuine point of view | ✓ |
| Pros/cons facilitator | Lays out trade-offs, gently leans toward one | |
| Pure neutral | Summarizes perspectives without adding its own stance | |

**User's choice:** Confident first-person
**Notes:** Jarvis should sound like a colleague with a genuine point of view, not a neutral recorder.

### Follow-up: Meeting source acknowledgement

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, briefly acknowledge | "Based on what I heard..." prefix | ✓ |
| No, just give the opinion | Confident without flagging the source | |

**User's choice:** Yes, briefly acknowledge
**Notes:** Short grounding phrase before the opinion. Transparent about drawing from meeting context.

---

## Response Length & Format

| Option | Description | Selected |
|--------|-------------|----------|
| Summary: 400, Opinion: 200 | Different caps by mode | |
| Both: 300 (uniform) | Same cap for both | |
| Configurable via env vars | JARVIS_SUMMARY_MAX_TOKENS / JARVIS_OPINION_MAX_TOKENS, defaults 400/200 | ✓ |

**User's choice:** Configurable via env vars
**Notes:** Runtime control without code changes. Defaults: summary=400, opinion=200.

---

## Claude's Discretion

- Exact LLM system prompt wording for each handler
- Transcript truncation strategy when exceeding context limits
- Whether to extract and label participant names in the summary prompt

## Deferred Ideas

None surfaced during discussion.
