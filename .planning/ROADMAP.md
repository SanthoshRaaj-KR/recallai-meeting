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

### Phase 3: Classifier and Context Intelligence — Topic Tracking, Web Search, and Graph RAG

**Goal:** Fix classifier accuracy for meeting-context questions; track topic shifts so Jarvis doesn't answer stale topics; expand web search to cover weather and real-time queries; implement Graph RAG on meeting transcript for entity-aware context retrieval.

**Depends on:** Phase 2

**Requirements:** CLASSIFY-03, TOPIC-01, WEBSEARCH-01, GRAPHRAG-01

**Plans:** 2/2 plans complete

Plans:
- [x] 03-01-PLAN.md — Sliding window cap (3 exchanges), LLM web search router, dependency install
- [x] 03-02-PLAN.md — Graph RAG module (graph_rag.py) + pipeline wiring (ingest hook, query injection)

### Phase 4: Speaker Isolation, Speech Debounce, and General Question Filler Audio

**Goal:** Filter multi-speaker transcript overlap so only the Jarvis-invoker's speech enters the pipeline; add a 1-second speech-completion debounce before processing; play filler WAV audio during general question LLM response generation to eliminate the awkward silence gap.

**Depends on:** Phase 3

**Requirements:** SPEAKER-01, DEBOUNCE-01, FILLER-02

**Plans:** 2/2 plans complete

Plans:
- [x] 04-01-PLAN.md — Speaker isolation (invoker lock) + speech debounce (cancellable 1s window) in websocket_endpoint
- [x] 04-02-PLAN.md — Contextual gap filler audio before LLM answer in _handle_general_question

### Phase 5: Human-likeness improvements: response length rewriter for verbal delivery, name-aware fillers using invoker_participant, micro-ack on wake detection, interruption recovery phrase, multi-turn referencing in prompts, conversational pacing after delivery

**Goal:** Make Jarvis sound and behave like a human colleague by condensing LLM answers for verbal delivery, addressing participants by name, emitting immediate micro-acknowledgments on wake detection, recovering gracefully from interruptions, referencing prior conversation naturally, and holding conversational pauses after speaking.

**Depends on:** Phase 4

**Requirements:** REWRITE-01, NAME-01, MULTITURN-01, MICROACK-01, INTERRUPT-01, PACING-01

**Plans:** 1/2 plans executed

Plans:
- [x] 05-01-PLAN.md — Response length rewriter + name-aware fillers + multi-turn referencing (LLM prompt changes)
- [ ] 05-02-PLAN.md — Micro-ack on wake detection + interruption recovery + post-speech pacing hold (timing/audio)

### Phase 6: Pipeline intelligence and safety improvements

**Goal:** Harden the Jarvis pipeline with meeting-aware Confluence edits, delete safety gates, action items extraction, speaker queries, confidence signaling, fuzzy wake word matching, upgraded web search via Tavily, and garbled query recovery.

**Requirements:** MEETCTX-01, DELGATE-01, TAVILY-01, GARBLED-01, WAKEALIAS-01, CONFIDENCE-01, ACTIONITEMS-01, SPEAKERQ-01

**Depends on:** Phase 5

**Plans:** 3/4 plans executed

Plans:
- [x] 06-01-PLAN.md — Meeting context for Confluence edits + delete confirmation safety gate
- [x] 06-02-PLAN.md — Tavily web search upgrade + garbled query recovery
- [x] 06-03-PLAN.md — Fuzzy wake word aliases + confidence signaling
- [ ] 06-04-PLAN.md — Action items extraction + participant speaker queries

---
