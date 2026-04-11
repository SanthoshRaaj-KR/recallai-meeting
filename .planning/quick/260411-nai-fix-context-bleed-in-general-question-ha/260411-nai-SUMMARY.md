---
task: 260411-nai
title: Fix Context Bleed in General Question Handler + Selective Web Search
date: 2026-04-11
commits:
  - 393485b
  - ddaf91b
files_modified:
  - confluence_logic/jarvis_agentic.py
  - confluence_logic/general_responder.py
tests: 12 passed
---

# Quick Task 260411-nai: Fix Context Bleed + Selective Web Search

**One-liner:** Isolated general Q&A conversation history from Confluence editor history and added selective DuckDuckGo instant-answer injection for freshness-sensitive questions.

## Fix 1 — Context bleed: separate general conversation history

**Root cause:** `_handle_general_question` called `session_agent.get_recent_history_text()` which returns the Confluence `EditorAgent`'s history — tuples like `("fix my confluence page", "Page updated successfully")`. This caused the general responder LLM to answer about the previous Confluence topic.

**Changes in `confluence_logic/jarvis_agentic.py`:**

1. Added `"general_history": []` to the `meeting_state` dict initializer.
2. Added `_MAX_GENERAL_HISTORY = 4` constant.
3. Added `_remember_general_exchange(question, answer)` — appends to `meeting_state["general_history"]`, trims to last 4 turns.
4. Added `_format_general_history()` — renders history as `User: ...\nAssistant: ...` lines, returns `"[none]"` when empty.
5. In `_handle_general_question`: replaced `session_agent.get_recent_history_text()` with `_format_general_history()`. Added `_remember_general_exchange(query, answer)` after `_speak_guarded`.
6. In `_handle_general_clarification_answer`: added `_remember_general_exchange(answer_text, final_answer)` after `_speak_guarded`.

**Commit:** `393485b`

---

## Fix 2 — Selective web search in general_responder.py

**Changes in `confluence_logic/general_responder.py`:**

1. Added `import re` and `import requests as _requests`.
2. Added `_FRESHNESS_PATTERNS` regex matching keywords like `latest`, `current`, `price`, `version`, `which models`, etc.
3. Added `_needs_web_search(question)` — returns True if the question matches the freshness pattern.
4. Added `_quick_web_search(query)` — hits DuckDuckGo Instant Answer API (`api.duckduckgo.com`), prefers `AbstractText` then `Answer`, caps at 600 chars, 3s timeout, swallows all exceptions as non-fatal.
5. Updated `answer_general_question`: when `_needs_web_search` returns True, runs `_quick_web_search` in a thread, injects result into the user message as `[Live web search result — use this for accuracy]`.
6. Updated system prompt to remove Confluence-specific framing (the classifier already routes correctly; the system prompt no longer needs to remind the LLM).

**Commit:** `ddaf91b`

---

## Verification

- Both files parse cleanly: `ast.parse()` OK
- All 12 existing tests pass (`python -m pytest tests/ -q --tb=short`)

## Deviations

None — plan executed exactly as written.
