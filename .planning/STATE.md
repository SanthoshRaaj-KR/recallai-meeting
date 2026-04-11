# Project State

## Current Status

- **Active Milestone:** Milestone 1 — Jarvis Intelligence Enhancement
- **Active Phase:** Phase 01 — Intelligent Question Classification and Conversational Response
- **Current Plan:** 01-03 (next to execute)
- **Progress:** 2/3 plans complete in Phase 01

## Accumulated Context

### Roadmap Evolution

- Initial roadmap created for Milestone 1: Jarvis Intelligence Enhancement
- Phase 1 added: Intelligent Question Classification and Conversational Response

### Decisions Made

- **01-01:** gpt-4o-mini with temperature=0 used for LLM fallback classification (deterministic, cost-effective)
- **01-01:** Classifier defaults to 'confluence' on error (safe fallback to existing pipeline)
- **01-01:** MP3 format chosen for audio cache to match Recall API 'kind': 'mp3' requirement
- **01-01:** Lazy audio cache loading to avoid startup cost when not needed
- **01-02:** gpt-4o-mini default for general responder (configurable via JARVIS_GENERAL_MODEL env var)
- **01-02:** max_tokens=150 caps response length for TTS-optimized concise answers
- **01-02:** Classifier runs before state_lock acquisition — lightweight, non-blocking routing
- **01-02:** asyncio.create_task() for general questions — fire-and-forget, non-blocking pipeline

### Performance Metrics

| Phase | Plan | Duration | Tasks | Files |
|-------|------|----------|-------|-------|
| 01    | 01   | ~2 min   | 2/2   | 5     |
| 01    | 02   | ~3 min   | 2/2   | 2     |

### Session

- **Last session:** 2026-04-11
- **Stopped at:** Completed 01-02-PLAN.md (General responder + classifier routing + cached ack playback integrated into main pipeline)
