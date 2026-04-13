---
phase: 04-speaker-isolation-speech-debounce-and-general-question-filler-audio
verified: 2026-04-13T21:00:00Z
status: passed
score: 10/10 must-haves verified
re_verification: false
---

# Phase 4: Speaker Isolation, Speech Debounce, and General Question Filler Audio — Verification Report

**Phase Goal:** Filter multi-speaker transcript overlap so only the Jarvis-invoker's speech enters the pipeline; add a 1-second speech-completion debounce before processing; play contextual filler audio during general question LLM response generation to eliminate the awkward silence gap.

**Verified:** 2026-04-13T21:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | When a participant triggers the wake word, only their subsequent transcript segments are accepted until dispatch | VERIFIED | `meeting_state["invoker_participant"] = participant` set at line 1440; filter at lines 1431-1435 drops non-invoker segments |
| 2 | Non-invoker transcripts during an active debounce window are silently dropped (logged at DEBUG) | VERIFIED | Lines 1431-1435: `if invoker and participant != invoker: logger.debug("Ignoring transcript from %s — active invoker is %s", ...)` |
| 3 | A final invoker segment does not dispatch immediately — it schedules `_debounced_dispatch` with 1-second sleep | VERIFIED | Line 1448: `asyncio.create_task(_debounced_dispatch(accumulated, bot_id))`; `_debounced_dispatch` at line 654 does `await asyncio.sleep(JARVIS_DEBOUNCE_SECONDS)` |
| 4 | A second final segment from the same invoker within the window cancels the prior task and extends with accumulated text | VERIFIED | Lines 1445-1448: `if pending and not pending.done(): pending.cancel()` then new `create_task(_debounced_dispatch(accumulated, bot_id))` with grown `accumulated` |
| 5 | After dispatch, `invoker_participant` and `_pending_debounce_task` are both reset to None | VERIFIED | Lines 655-657 in `_debounced_dispatch`: both fields set to None, `_accumulated_query` cleared to `""` |
| 6 | `JARVIS_DEBOUNCE_SECONDS` env var controls the sleep duration, defaulting to 1.0 | VERIFIED | Line 65: `JARVIS_DEBOUNCE_SECONDS = float(os.getenv("JARVIS_DEBOUNCE_SECONDS", "1.0"))` |
| 7 | A contextual filler phrase is spoken before the LLM answer is generated in `_handle_general_question` | VERIFIED | Lines 1021-1023: `filler = await _generate_contextual_gap_filler(query)` then `await _speak_guarded(filler, ...)` immediately before `answer = await answer_general_question(...)` at line 1024 |
| 8 | The filler is spoken with `allow_stale=True` so it is not suppressed by the generation guard | VERIFIED | Line 1023: `await _speak_guarded(filler, bot_id, generation, allow_stale=True)` |
| 9 | The filler call reuses `_generate_contextual_gap_filler(query)` — no new function | VERIFIED | Line 1022 calls the existing function at line 661; no new filler function introduced |
| 10 | The existing answer flow (`_speak_guarded(answer, ..., allow_stale=True)`) is unchanged after the filler | VERIFIED | Line 1028: `await _speak_guarded(answer, bot_id, generation, allow_stale=True)` is intact and unchanged |

**Score:** 10/10 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `confluence_logic/jarvis_agentic.py` — JARVIS_DEBOUNCE_SECONDS | Env var constant, default 1.0 | VERIFIED | Line 65: `JARVIS_DEBOUNCE_SECONDS = float(os.getenv("JARVIS_DEBOUNCE_SECONDS", "1.0"))` |
| `confluence_logic/jarvis_agentic.py` — meeting_state new fields | `invoker_participant`, `_pending_debounce_task`, `_accumulated_query` | VERIFIED | Lines 147-149: all three fields present with correct defaults (None, None, "") |
| `confluence_logic/jarvis_agentic.py` — `_debounced_dispatch` | Cancellable async coroutine | VERIFIED | Lines 649-658: substantive implementation — sleeps, clears state, awaits `handle_spoken_request` |
| `confluence_logic/jarvis_agentic.py` — speaker isolation filter in websocket_endpoint | Participant check with DEBUG log and continue | VERIFIED | Lines 1430-1453: full isolation + debounce dispatch block present |
| `confluence_logic/jarvis_agentic.py` — filler in `_handle_general_question` | `_generate_contextual_gap_filler(query)` call before `answer_general_question` | VERIFIED | Lines 1021-1024: filler generated and spoken, then `answer_general_question` called |
| `confluence_logic/tests/test_jarvis_agentic.py` — SPEAKER-01/DEBOUNCE-01 tests | 5 new tests + `_reset_meeting_state` extension | VERIFIED | Lines 27-29: reset extended; 5 tests at lines 411-470 all present and substantive |
| `confluence_logic/tests/test_jarvis_agentic.py` — FILLER-02 test | `test_handle_general_question_speaks_filler_before_answer` | VERIFIED | Lines 473-502: test present, uses call_order tracking for ordering assertion |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `websocket_endpoint::participant check` | `meeting_state[invoker_participant]` | `if invoker and participant != invoker: logger.debug + continue` | WIRED | Lines 1431-1435 — pattern exactly as planned |
| `websocket_endpoint::query branch` | `_debounced_dispatch` | `asyncio.create_task(_debounced_dispatch(accumulated, bot_id))` | WIRED | Line 1448 — `create_task` with `_debounced_dispatch`; old direct `handle_spoken_request` call is gone (grep returns empty) |
| `_debounced_dispatch` | `handle_spoken_request` | `await asyncio.sleep(JARVIS_DEBOUNCE_SECONDS)` then `await handle_spoken_request` | WIRED | Lines 654, 658 — sleep then direct await; `handle_spoken_request` is only called from here |
| `_handle_general_question::filler` | `_generate_contextual_gap_filler` | `filler = await _generate_contextual_gap_filler(query)` | WIRED | Line 1022 — direct await inside `_handle_general_question` |
| `_handle_general_question::filler_speak` | `_speak_guarded` | `await _speak_guarded(filler, bot_id, generation, allow_stale=True)` | WIRED | Line 1023 — correct signature with `allow_stale=True` |

---

### Data-Flow Trace (Level 4)

Not applicable. Phase artifacts are pipeline control-flow components (speaker filter, debounce coordinator, filler invocation) that delegate to existing LLM and TTS functions. They do not independently render dynamic data — `_generate_contextual_gap_filler` and `answer_general_question` are existing, previously verified data producers.

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Module imports without error | `python -c "import confluence_logic.jarvis_agentic as ja; print('OK')"` | OK | PASS |
| All 5 SPEAKER-01/DEBOUNCE-01 tests pass | `pytest -k "test_speaker_isolation or test_debounce" -v` | 5 passed | PASS |
| FILLER-02 test passes | `pytest -k "test_handle_general_question_speaks_filler_before_answer" -v` | 1 passed | PASS |
| Old direct dispatch removed | `grep "asyncio.create_task(handle_spoken_request"` | no matches | PASS |
| `_debounced_dispatch` only caller of `handle_spoken_request` | `grep "await handle_spoken_request"` | 1 match at line 658 | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| SPEAKER-01 | 04-01-PLAN.md | Only the invoking speaker's transcript segments are processed during an active wake-word session | SATISFIED | `invoker_participant` lock set on wake detection (line 1440); non-invoker segments filtered at lines 1431-1435; lock cleared after dispatch (line 655) |
| DEBOUNCE-01 | 04-01-PLAN.md | Transcript dispatch is delayed by a cancellable 1-second window; mid-sentence continuation extends the window | SATISFIED | `_debounced_dispatch` (lines 649-658) implements sleep + state clear + dispatch; cancel-and-restart at lines 1445-1448 |
| FILLER-02 | 04-02-PLAN.md | Contextual acknowledgment audio is spoken before the LLM answer in `_handle_general_question` | SATISFIED | Lines 1021-1023 insert filler call before `answer_general_question`; FILLER-02 test passes confirming ordering |
| D-10 | 04-02-PLAN.md (implementation decision, not a formal requirement ID) | Filler generation placed at top of `_handle_general_question` before LLM await | SATISFIED | D-10 is satisfied as part of FILLER-02 implementation at lines 1021-1023 |

No REQUIREMENTS.md file exists in `.planning/` — requirements are defined inline in the ROADMAP.md and CONTEXT.md. No orphaned requirements detected. All three formal requirement IDs (SPEAKER-01, DEBOUNCE-01, FILLER-02) are accounted for.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | — | — | No anti-patterns found |

No TODO/FIXME markers, placeholder returns, empty handlers, or hardcoded empty data detected in the phase-modified files for the new code paths.

---

### Human Verification Required

#### 1. Multi-speaker mic bleed prevention (live meeting)

**Test:** Join a meeting with two active microphones. Have participant A say "Hey Jarvis, what time is it?" while participant B is also speaking. Observe whether Jarvis processes only participant A's follow-up segments.

**Expected:** Only participant A's transcript drives the query; participant B's simultaneous speech is silently dropped and logged at DEBUG.

**Why human:** Requires a live Recall.ai bot session with two concurrent speakers — cannot simulate the WebSocket data flow with unit tests alone.

#### 2. Debounce window extension (mid-sentence interruption)

**Test:** In a live session, trigger "Hey Jarvis, what is the..." then after ~0.5 seconds say "...meeting agenda?" Observe whether a single dispatch fires after the second segment, not two.

**Expected:** The first pending task is cancelled, the second task fires ~1 second after the final segment with the accumulated text "what is the meeting agenda".

**Why human:** Requires real async timing across a live WebSocket connection; unit test mocks asyncio.sleep(0.0) and cannot validate real-world cancellation timing.

#### 3. Filler audio audibility (UX)

**Test:** Ask Jarvis a general question (not Confluence-specific). Measure the time from wake detection to first audio output versus the time to LLM response delivery.

**Expected:** An acknowledgment phrase (e.g., "Let me check that for you.") is audible within ~1 second of the query, well before the full LLM answer arrives.

**Why human:** TTS output quality and perceived latency cannot be measured programmatically from the test suite.

---

### Gaps Summary

No gaps. All 10 observable truths verified. All artifacts exist, are substantive (not stubs), and are fully wired. Key links confirmed end-to-end. Old direct `handle_spoken_request` dispatch path removed. All 6 new tests pass. Commits 7d38032, 3977bbf, and 07c9896 confirmed present in git log.

---

_Verified: 2026-04-13T21:00:00Z_
_Verifier: Claude (gsd-verifier)_
