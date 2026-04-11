# Roadmap

## Milestone 1: Jarvis Intelligence Enhancement

> Make Jarvis a smarter, more conversational AI assistant — able to classify intent, answer general questions, cache responses efficiently, and follow up with users when clarification is needed.

### Phase 1: Intelligent Question Classification and Conversational Response

**Goal:** Differentiate between confluence-edit intents and general questions; answer general questions with real AI responses; cache hardcoded WAV replies locally; support conversational follow-up without requiring wake word.

**Depends on:** —

**Requirements:** CLASSIFY-01, WAV-01, WAV-02, GENERAL-01, CLARIFY-01

**Plans:** 3/3 plans complete

Plans:
- [x] 01-01-PLAN.md — Intent classifier module + WAV asset generation script + audio cache
- [x] 01-02-PLAN.md — General question responder + pipeline integration (classifier routing, cached acks)
- [x] 01-03-PLAN.md — Clarification follow-up listening state (no wake word on follow-up)

### Phase 2: Meeting transcript access with summarization and opinion generation

**Goal:** Give Jarvis awareness of the full meeting conversation — summarize everything spoken and deliver first-person opinions grounded in what was discussed.
**Requirements:** TRANSCRIPT-01, SUMMARY-01, OPINION-01, CLASSIFY-02, ROUTE-01
**Depends on:** Phase 1
**Plans:** 2/2 plans complete

Plans:
- [x] 02-01-PLAN.md — New meeting_responder.py module with summarize_meeting() and generate_opinion() handlers
- [x] 02-02-PLAN.md — Extend classifier with meeting_summary/meeting_opinion intents + route in handle_spoken_request()

---
