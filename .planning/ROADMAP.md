# Roadmap

## Milestone 1: Jarvis Intelligence Enhancement

> Make Jarvis a smarter, more conversational AI assistant — able to classify intent, answer general questions, cache responses efficiently, and follow up with users when clarification is needed.

### Phase 1: Intelligent Question Classification and Conversational Response

**Goal:** Differentiate between confluence-edit intents and general questions; answer general questions with real AI responses; cache hardcoded WAV replies locally; support conversational follow-up without requiring wake word.

**Depends on:** —

**Requirements:** CLASSIFY-01, WAV-01, WAV-02, GENERAL-01, CLARIFY-01

**Plans:** 3 plans

Plans:
- [x] 01-01-PLAN.md — Intent classifier module + WAV asset generation script + audio cache
- [x] 01-02-PLAN.md — General question responder + pipeline integration (classifier routing, cached acks)
- [ ] 01-03-PLAN.md — Clarification follow-up listening state (no wake word on follow-up)

---
