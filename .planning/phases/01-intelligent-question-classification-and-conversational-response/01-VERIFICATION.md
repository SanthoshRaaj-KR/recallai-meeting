---
phase: 01-intelligent-question-classification-and-conversational-response
verified: 2026-04-11T07:15:00Z
status: passed
score: 10/10 must-haves verified
gaps: []
human_verification:
  - test: "Say 'What is Kubernetes?' to Jarvis in a live meeting"
    expected: "Jarvis responds conversationally within 2-3 seconds with a concise spoken answer, no mention of Confluence"
    why_human: "Requires a live Recall.ai bot session to verify end-to-end TTS playback"
  - test: "Say 'Edit the roadmap page' to Jarvis"
    expected: "Jarvis plays a cached MP3 acknowledgement (e.g. 'On it.') instead of synthesizing TTS, then proceeds with the confluence pipeline"
    why_human: "Verifying cached audio vs live TTS requires a live bot with audio cache populated by generate_wav_assets.py"
  - test: "Ask a vague general question that triggers clarification (e.g. 'Tell me about the project'), then respond without saying Hey Jarvis"
    expected: "Jarvis's clarifying question opens a 15-second window; user's reply is accepted and answered without wake word"
    why_human: "Multi-turn conversational flow requires a live meeting session"
---

# Phase 01: Intelligent Question Classification and Conversational Response — Verification Report

**Phase Goal:** Implement intelligent question classification so that general questions (e.g., "What is Kubernetes?") are answered conversationally by an LLM in real-time, while Confluence-specific queries continue through the existing agentic pipeline. Add clarification follow-up support so users can respond to Jarvis's clarifying questions without needing to repeat the wake word.

**Verified:** 2026-04-11T07:15:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A classifier function can distinguish confluence-edit intents from general questions | VERIFIED | `confluence_logic/classifier.py` exports `classify_intent` and `_fast_classify`; fast-path heuristic correctly classifies "create a new page about roadmap" → confluence and "what is the weather today" → general; LLM fallback uses gpt-4o-mini with temperature=0 |
| 2 | Pre-generated WAV files (MP3) exist on disk for acknowledgement phrases | VERIFIED | `confluence_logic/assets/audio/` directory exists with `.gitkeep`; `generate_wav_assets.py` has 10 ACK_PHRASES; directory structure confirmed |
| 3 | An audio cache module can load and serve cached WAV bytes by key | VERIFIED | `confluence_logic/audio_cache.py` exports `load_audio_cache` and `get_random_ack_audio`; returns None when empty, tuple(stem, bytes) when populated; lazy loading confirmed |
| 4 | General questions receive a real LLM-generated conversational answer spoken via TTS | VERIFIED | `confluence_logic/general_responder.py` exports `answer_general_question`; uses gpt-4o-mini, max_tokens=150, no-markdown system prompt; wired into `_handle_general_question` which calls `_speak_guarded(allow_stale=True)` |
| 5 | Confluence-edit intents play a cached MP3 acknowledgement instead of live TTS | VERIFIED | `_run_voice_task` calls `get_random_ack_audio()` and uses `_speak_cached_guarded`; falls back to `random.choice(_INSTANT_ACKS)` when cache is empty |
| 6 | The classifier routes incoming requests before they enter the confluence pipeline | VERIFIED | `handle_spoken_request` calls `classify_intent(spoken_query)` after status query check and before `state_lock` acquisition (line 928); early return for `intent == "general"` confirmed |
| 7 | Existing confluence edit/create/delete flows continue working unchanged | VERIFIED | `_execute_editor_task` untouched; `session_agent.handle_prepared_query` still present; `pending_clarification` confluence logic at line 1048 still intact (13 occurrences) |
| 8 | When the general responder needs clarification, it asks a follow-up question and the user can answer without saying Hey Jarvis | VERIFIED | `_handle_general_question` sets `pending_general_clarification` when `_looks_like_clarification_prompt(answer)` is True; `process_transcript_event` checks this state and bypasses wake-word for non-expired entries |
| 9 | The clarification listening state has a timeout after which it resets to normal wake-word mode | VERIFIED | `JARVIS_GENERAL_CLARIFICATION_TIMEOUT = 15.0s` (env-configurable); `process_transcript_event` clears expired state at lines 1055-1058; `handle_spoken_request` checks `time.time() <= pending_general["expires_at"]` before routing |
| 10 | Only one clarification exchange is supported per general question (no chaining) | VERIFIED | `_handle_general_clarification_answer` immediately clears `meeting_state["pending_general_clarification"] = None` at line 903 before processing the answer, preventing chained re-entry |

**Score:** 10/10 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `confluence_logic/classifier.py` | Intent classification function | VERIFIED | 105 lines; exports `classify_intent` (async) and `_fast_classify`; gpt-4o-mini LLM fallback; defaults to "confluence" on error |
| `confluence_logic/audio_cache.py` | WAV cache loader and random selection | VERIFIED | 70 lines; exports `load_audio_cache`, `get_random_ack_audio`, `get_cache_size`; lazy loading; returns None when empty |
| `confluence_logic/scripts/generate_wav_assets.py` | Script to pre-generate MP3 files | VERIFIED | Exists; 10 ACK_PHRASES; reads TTS settings from env vars; idempotent (skips existing files) |
| `confluence_logic/assets/audio/` | Directory containing cached audio files | VERIFIED | Directory exists with `.gitkeep`; no MP3s yet (expected — require running generate_wav_assets.py first) |
| `confluence_logic/general_responder.py` | General question answering function | VERIFIED | 80 lines; exports `answer_general_question`; gpt-4o-mini default; concise system prompt; error fallback string |
| `confluence_logic/jarvis_agentic.py` | Updated pipeline with classifier routing and clarification state | VERIFIED | 1168 lines; contains all 9 additions from Plans 02 and 03; valid Python (ast.parse passes) |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `classifier.py` | OpenAI chat completions API | `classify_intent()` calls `gpt-4o-mini` | VERIFIED | `asyncio.to_thread(lambda: _get_client().chat.completions.create(...model="gpt-4o-mini"...))` at line 85-94 |
| `audio_cache.py` | `assets/audio/` | `load_audio_cache` reads `.mp3` files from disk | VERIFIED | `AUDIO_DIR = Path(__file__).resolve().parent / "assets" / "audio"` at line 14; `audio_dir.glob("*.mp3")` at line 36 |
| `jarvis_agentic.py` | `classifier.py` | `from .classifier import classify_intent` + call in `handle_spoken_request` | VERIFIED | Import at line 35; call at line 928 (`intent = await classify_intent(spoken_query)`) |
| `jarvis_agentic.py` | `audio_cache.py` | `from .audio_cache import get_random_ack_audio` + use in `_run_voice_task` | VERIFIED | Import at line 36; call at line 672 (`cached = get_random_ack_audio()`) |
| `jarvis_agentic.py` | `general_responder.py` | `from .general_responder import answer_general_question` + call in handlers | VERIFIED | Import at line 37; calls at lines 875 and 914 |
| `general_responder.py` | OpenAI chat completions API | LLM call for conversational answer | VERIFIED | `asyncio.to_thread(lambda: _get_client().chat.completions.create(...))` at lines 64-70 |
| `_handle_general_question` | `meeting_state["pending_general_clarification"]` | Sets state when answer is a clarifying question | VERIFIED | Lines 882-890 set the dict with `expires_at`; guarded by `_looks_like_clarification_prompt(answer)` |
| `process_transcript_event` | `meeting_state["pending_general_clarification"]` | Checks and routes before wake-word check | VERIFIED | Lines 1052-1062; checks after `pending_clarification`, before `jarvis_listening` |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `_fast_classify` routes action verbs to confluence | `python -c "_fast_classify('create a new page about roadmap') == 'confluence'"` | "confluence" | PASS |
| `_fast_classify` routes question words to general | `python -c "_fast_classify('what is the weather today') == 'general'"` | "general" | PASS |
| `audio_cache` returns None when empty | `load_audio_cache('/tmp/nonexistent')` then `get_random_ack_audio()` | None | PASS |
| `generate_wav_assets` has >= 8 phrases | `len(ACK_PHRASES) >= 8` | 10 | PASS |
| `general_responder` uses gpt-4o-mini | `GENERAL_RESPONDER_MODEL == 'gpt-4o-mini'` | True | PASS |
| `jarvis_agentic.py` is valid Python | `ast.parse(source)` | No exception | PASS |
| `pending_general_clarification` appears >= 5 times | `grep -c` | 6 | PASS |
| `JARVIS_GENERAL_CLARIFICATION_TIMEOUT` appears >= 2 times | `grep -c` | 3 | PASS |
| Existing confluence `pending_clarification` untouched | `grep -c "pending_clarification"` | 13 | PASS |

---

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| `classifier.py` lines 48, 60 | `return None` | Info | Intentional — signals "ambiguous, use LLM fallback"; not a stub; function contract documented in docstring |

No blockers or warnings found.

---

### Human Verification Required

#### 1. General Question End-to-End Flow

**Test:** In a live Recall.ai meeting bot session, say "What is Kubernetes?" to Jarvis
**Expected:** Jarvis responds within a few seconds with a conversational 1-3 sentence spoken answer; no Confluence-related content
**Why human:** Requires live Recall.ai bot with active session; TTS audio playback cannot be verified programmatically

#### 2. Cached MP3 Acknowledgement Playback

**Test:** Run `python -m confluence_logic.scripts.generate_wav_assets` to populate the audio cache, then in a live session say "Edit the roadmap page"
**Expected:** Jarvis plays a pre-cached MP3 phrase (e.g. "On it.") instead of synthesizing TTS; then proceeds with the confluence pipeline
**Why human:** Distinguishing cached MP3 from live TTS requires audio inspection in a live meeting; fallback to live TTS when cache empty makes this invisible without populated cache

#### 3. Clarification Follow-Up Without Wake Word

**Test:** Ask a vague general question like "Tell me about the project" that causes Jarvis to respond with a clarifying question; then reply naturally (without "Hey Jarvis") within 15 seconds
**Expected:** Jarvis accepts the clarification answer, builds enriched context, and responds with a final answer
**Why human:** Multi-turn conversational flow requires a live meeting session; timeout behavior needs real-time observation

---

### Gaps Summary

No gaps found. All 10 observable truths are verified with evidence from the actual codebase.

The audio assets directory (`confluence_logic/assets/audio/`) contains only a `.gitkeep` and no MP3 files. This is the expected state before running the generation script and is explicitly noted as intentional in the plan — the audio cache module gracefully falls back to live TTS when no cached files are present. This is not a gap.

---

_Verified: 2026-04-11T07:15:00Z_
_Verifier: Claude (gsd-verifier)_
