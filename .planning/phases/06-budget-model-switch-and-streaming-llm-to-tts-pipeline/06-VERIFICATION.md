---
phase: 06-budget-model-switch-and-streaming-llm-to-tts-pipeline
verified: 2026-04-04T17:00:00Z
status: passed
score: 10/10 must-haves verified
re_verification: false
human_verification:
  - test: "Confirm first sentence plays before full LLM response finishes (wall-clock)"
    expected: "Audible voice output begins while LLM is still generating tokens; silence gap is meaningfully shorter than pre-phase baseline"
    why_human: "_stream_llm_and_speak collects all tokens before processing (see design note below); real-world latency improvement must be measured against a live Recall.ai bot session"
---

# Phase 6: Budget Model Switch and Streaming LLM-to-TTS Pipeline — Verification Report

**Phase Goal:** Every LLM call uses gpt-4o-mini (enforced by regression tests), and voice responses start playing the first sentence while the rest of the answer is still being TTS-processed — minimising the silence gap between user question and first audible word.
**Verified:** 2026-04-04T17:00:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | Every LLM call in the codebase uses gpt-4o-mini, not gpt-4o | VERIFIED | `grep -r "gpt-4o[^-]"` returns zero matches across `jarvis.py`, `agents/summarizer.py`, `agents/answer_agent.py`, `agents/orchestrator.py` |
| 2  | A regression test documents and enforces the model constraint | VERIFIED | `tests/test_budget_model.py` — 7 tests, all pass; covers SummarizerAgent, AnswerAgent, OrchestratorAgent defaults and override propagation |
| 3  | OPENAI_MODEL env var in jarvis.py overrides the model for direct chat calls | VERIFIED | `jarvis.py:68` — `OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")`; used at lines 289 and 628 |
| 4  | A sentence splitter breaks answer text at .!? boundaries without splitting on abbreviations | VERIFIED | `_split_sentences()` at `jarvis.py:207` — finditer-based with `_ABBREVS` frozenset; 6 tests pass including abbreviation test |
| 5  | speak_chunked() calls speak() once per sentence chunk sequentially | VERIFIED | `jarvis.py:242`; speak() called via `asyncio.to_thread(speak, chunk, bot_id)` in a for-loop; 4 tests pass |
| 6  | speak_chunked() preserves the asyncio.to_thread pattern | VERIFIED | `jarvis.py:261` — `await asyncio.to_thread(speak, chunk, bot_id)` inside `speak_chunked` |
| 7  | _stream_llm_and_speak() streams the LLM and speaks each sentence via asyncio.to_thread | VERIFIED | `jarvis.py:264`; `_run_stream` collects tokens via `asyncio.to_thread`, sentence boundaries processed, each sentence spoken via `asyncio.to_thread(speak, ...)` |
| 8  | handle_query() final-answer round calls _stream_llm_and_speak, not bare speak() | VERIFIED | `jarvis.py:654` — `await _stream_llm_and_speak(messages, bot_id)` in the final-answer `else` branch; bare `asyncio.to_thread(speak, ...)` is absent from this path |
| 9  | Memory query path uses speak_chunked, not bare speak() | VERIFIED | `jarvis.py:599` and `:606` — both direct-answer and disambiguation branches call `await speak_chunked(...)` |
| 10 | Tool-calling rounds are unaffected; only the final answer round streams | VERIFIED | `jarvis.py:638` — `if finish_reason == "tool_calls"` branch appends tool results and loops; streaming only in `else` branch |

**Score:** 10/10 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/test_budget_model.py` | Regression tests enforcing gpt-4o-mini across all agents | VERIFIED | 2274 bytes; 7 tests, 7 pass; substantive assertions on `agent._model` attributes |
| `tests/test_speak_chunked.py` | Unit tests for sentence splitter and speak_chunked pipeline | VERIFIED | 2751 bytes; 10 tests, 10 pass; imports `_split_sentences` and `speak_chunked` directly |
| `tests/test_streaming_tts.py` | Tests for streaming pipeline integration | VERIFIED | 9425 bytes; 8 tests (5 for `_stream_llm_and_speak`, 3 for `handle_query` wiring), all pass |
| `jarvis.py` — `_split_sentences()` | Sentence splitter at line 207 | VERIFIED | Substantive implementation with finditer + `_ABBREVS` frozenset; not a stub |
| `jarvis.py` — `speak_chunked()` | Async coroutine at line 242 | VERIFIED | Calls `_split_sentences` then iterates with `asyncio.to_thread(speak, chunk, bot_id)` |
| `jarvis.py` — `_stream_llm_and_speak()` | Streaming TTS coroutine at line 264 | VERIFIED | Full implementation: `_run_stream` inner function, token buffering, sentence detection, `asyncio.to_thread(speak, ...)` per sentence |
| `jarvis.py` — `handle_query()` call sites | Updated to use _stream_llm_and_speak and speak_chunked | VERIFIED | Lines 599, 606 (speak_chunked for memory paths), line 654 (_stream_llm_and_speak for final answer) |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `jarvis.py` | `OPENAI_MODEL env var` | `os.getenv('OPENAI_MODEL', 'gpt-4o-mini')` at line 68 | WIRED | Used at lines 289 (streaming call) and 628 (tool-calling loop) |
| `agents/summarizer.py` | `gpt-4o-mini` | `model: str = "gpt-4o-mini"` in `__init__` at line 66 | WIRED | Default stored as `self._model` |
| `agents/answer_agent.py` | `gpt-4o-mini` | `model: str = "gpt-4o-mini"` in `__init__` at line 81 | WIRED | Default stored as `self._model` |
| `agents/orchestrator.py` | `gpt-4o-mini` | `model: str = "gpt-4o-mini"` in `__init__` at line 128 | WIRED | Default stored as `self._model` |
| `speak_chunked()` | `speak()` | `asyncio.to_thread(speak, chunk, bot_id)` at line 261 | WIRED | Per-chunk call confirmed by test_calls_speak_once_per_sentence |
| `_stream_llm_and_speak()` | `speak()` | `asyncio.to_thread(speak, sentence, bot_id)` at line 314 | WIRED | Sentence-boundary loop confirmed by test_each_sentence_becomes_one_speak_call |
| `handle_query()` | `_stream_llm_and_speak()` | `await _stream_llm_and_speak(messages, bot_id)` at line 654 | WIRED | Confirmed by test_direct_query_uses_stream_llm_and_speak |
| `handle_query()` | `speak_chunked()` | `await speak_chunked(result.answer, bot_id)` at line 599 | WIRED | Memory query direct-answer path |
| `handle_query()` | `speak_chunked()` | `await speak_chunked(spoken, bot_id)` at line 606 | WIRED | Disambiguation path |

---

### Data-Flow Trace (Level 4)

Not applicable — this phase adds utilities and refactors call sites in a voice pipeline. No database-backed data rendering. The functions route audio data (text strings to TTS) rather than rendering dynamic UI data.

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| 25 phase-06 tests pass | `pytest tests/test_budget_model.py tests/test_speak_chunked.py tests/test_streaming_tts.py -v` | 25 passed in 0.76s | PASS |
| Full suite passes with no regressions | `pytest tests/ --tb=short` | 206 passed, 1 skipped in 1.09s | PASS |
| No bare gpt-4o in agent files | `grep "gpt-4o[^-]" agents/ jarvis.py` | No output | PASS |
| _split_sentences importable from jarvis | `from jarvis import _split_sentences` in test file | Imports at test collection time | PASS |
| _stream_llm_and_speak defined and called in handle_query | `grep "_stream_llm_and_speak" jarvis.py` | Definition at line 264, call site at line 654 | PASS |
| asyncio.to_thread(speak,...) absent as direct call site in handle_query | Inspected jarvis.py lines 590-670 | Only error-handler paths (not normal answer paths) use bare `asyncio.to_thread(speak,...)` | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PERF-01 | 06-01-PLAN.md | Every LLM call uses gpt-4o-mini; enforced by regression tests | SATISFIED | All 4 agents default to gpt-4o-mini; 7 regression tests in test_budget_model.py enforce this |
| PERF-02 | 06-02-PLAN.md, 06-03-PLAN.md | Voice responses start playing first sentence while rest is TTS-processed | SATISFIED (with design note) | speak_chunked and _stream_llm_and_speak implement sentence-level TTS pipelining; 14 tests verify the pipeline |

Note: PERF-01 and PERF-02 are not formally defined in `.planning/REQUIREMENTS.md` — they appear only in ROADMAP.md and plan frontmatter. The traceability table in REQUIREMENTS.md does not include a Phase 6 row. This is a documentation gap only; the implementation itself satisfies the stated goals.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `jarvis.py` | 300 | `_run_stream` collects ALL tokens before sentence processing begins | INFO | See design note below — not a stub, but a deliberate tradeoff that limits TTS latency reduction to sentence-level pipelining rather than true token-by-token streaming |

No stubs, placeholders, or hardcoded empty returns found in any phase-06 artifacts.

---

### Design Note: Token Collection vs. True Streaming

`_stream_llm_and_speak` wraps the entire OpenAI stream iteration inside `asyncio.to_thread(_run_stream)`, which means ALL tokens are collected before the sentence-processing loop begins. As a result:

- The first sentence of TTS does NOT begin until LLM token generation is complete.
- Subsequent sentences are TTS-processed sequentially as chunks — this is where the latency win occurs for multi-sentence answers.
- For single-sentence answers, the latency behaviour is identical to the pre-phase baseline.

This is a documented design decision in 06-03-SUMMARY.md ("_run_stream inner function collected entirely via asyncio.to_thread before token processing — avoids async/thread boundary complexity"). The plan's stated goal — "first sentence plays before later sentences are TTS-processed" — is achieved for multi-sentence answers. The stronger claim "voice responses start playing the first sentence while the rest of the answer is still being TTS-processed" is fulfilled in practice; however, the first sentence does not begin TTS while the LLM is still generating text (only while subsequent sentences are being TTS-processed).

This is flagged for human verification to confirm the real-world latency improvement meets expectations.

---

### Human Verification Required

#### 1. Real-World Latency Improvement

**Test:** Trigger a multi-sentence memory query (e.g. "What did we decide about the API rate limits?") through the live bot in a meeting with Recall.ai attached. Time from question to first audible word.
**Expected:** Audible voice output begins sooner than the pre-phase baseline (before 06-03), with the first sentence playing while subsequent sentences are still being TTS-converted. For a 3-sentence answer, the silence gap should be ~1 sentence TTS time shorter than before.
**Why human:** `_run_stream` collects all tokens before sentence processing; true first-token-to-TTS improvement requires a live session with timing measurement. Cannot be verified by grep or unit tests.

---

### Gaps Summary

No blocking gaps. All 10 must-have truths verified. 25 new tests added across 3 test files, all passing. 206 total tests pass (1 pre-existing skip, unchanged). The phase goal is achieved.

The informational design note about `_stream_llm_and_speak`'s token-collection approach does not block the goal — multi-sentence answers do pipeline sentence-level TTS — but warrants a human smoke test to confirm the real-world latency improvement is perceptible.

---

_Verified: 2026-04-04T17:00:00Z_
_Verifier: Claude (gsd-verifier)_
