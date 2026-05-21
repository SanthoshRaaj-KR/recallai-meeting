---
phase: 04-pipeline-quality-fixes
reviewed: 2026-05-15T00:00:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - confluence_logic/agents/fact_extraction_agent.py
  - confluence_logic/agents/drafter_agent.py
findings:
  critical: 2
  warning: 4
  info: 3
  total: 9
status: issues_found
---

# Phase 04: Code Review Report

**Reviewed:** 2026-05-15T00:00:00Z
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found

## Summary

Both files implement the fact-extraction and drafting stages of the post-meeting Confluence pipeline. The code is generally well-structured, with good logging, graceful degradation, and clear separation of the "old" `_run_drafter` path from the newer `_run_intent_drafter` path. However, two logic bugs can silently corrupt pipeline output (one causes ChangeIntents with non-empty actions to be merged under the wrong deduplication key; the other can force a valid `edit`/`title`/`delete` proposal to be coerced to a no-op skip), and two security/reliability gaps (invalid model name ships in defaults; `_fact_agent` is constructed at import time) deserve attention before production.

---

## Critical Issues

### CR-01: `not any(key)` guard in `_merge_facts` skips only all-empty keys — silently collapses all intents with empty `subject` into one

**File:** `confluence_logic/agents/fact_extraction_agent.py:279`

**Issue:**
The deduplication loop in `_merge_facts` is meant to skip intents that have no usable key signal (empty `subject` AND empty `action`). The guard `if not any(key)` achieves this only when BOTH tuple elements are falsy. When `subject=""` and `action="replace"` (the default), `not any(("", "replace"))` evaluates to `False`, so the intent is not skipped. All such intents compete for the single deduplication slot `("", "replace")`, and only the last one from all chunks survives — every other blank-subject change intent from earlier chunks is silently discarded regardless of its `instruction` or `new_value`. Because the LLM default `action` is `"replace"`, this affects every chunk where the model omitted the `subject` field.

The intent was almost certainly `if not all(key)` — skip the intent if either part of the key is missing.

```python
# CURRENT (wrong): only skips when BOTH subject AND action are empty
if not any(key):
    continue

# FIX: skip when EITHER subject OR action is empty (key is unusable)
if not all(key):
    continue
```

### CR-02: `_normalize_intent_draft` coerces `"delete"` and `"title"` to `"create"` for zero-RAG pages

**File:** `confluence_logic/agents/drafter_agent.py:563-564`

**Issue:**
The zero-RAG guard reads:
```python
if not page_id and change_type not in {"create"}:
    change_type = "create"
```
This means that when `page_id is None`, any `change_type` other than `"create"` is silently coerced to `"create"` — including `"delete"`, `"title"`, and `"edit"`. The comment says this path is for "new pages" but the LLM may legitimately return `change_type="edit"` for a zero-RAG page when `applies=True` (e.g. the qualifier passed a known page without a `page_id` in the enriched dict due to a graph gap). In that scenario the change is silently converted from an `edit` to a `create`, producing a new-page proposal instead of an edit, which is a wrong outcome and can result in duplicate Confluence pages being created.

The fix is to only coerce to `"create"` when the action is genuinely creative — leave `"edit"` alone since it is valid for a page that the system doesn't have an ID for:

```python
# Only force create for non-edit actions when page_id is absent
if not page_id and change_type not in {"create", "edit"}:
    logger.warning(
        "DrafterAgent: coercing change_type %r to 'create' for zero-RAG page (page_id=None)",
        change_type,
    )
    change_type = "create"
```

---

## Warnings

### WR-01: `JARVIS_AGENT_MODEL` default is `"gpt-5-mini"` — an invalid OpenAI model name

**File:** `confluence_logic/agents/fact_extraction_agent.py:20`
**File:** `confluence_logic/agents/drafter_agent.py:16`

**Issue:**
Both files read `JARVIS_AGENT_MODEL` with a default of `"gpt-5-mini"`, and the CLAUDE.md table explicitly notes this is an **invalid model name**. If `JARVIS_AGENT_MODEL` is not set in the environment, all `Runner.run()` calls will fail with an OpenAI "model not found" error at runtime. The `_extract_chunk` and both drafter functions catch `Exception` broadly and return `None`/`ExtractedFacts()`, so failures are silent in production. The correct model names are `gpt-4o-mini` (available now) or a valid GPT-5 variant per the project's model ceiling.

```python
# Fix: use a model name that actually exists as the fallback
JARVIS_AGENT_MODEL = os.getenv("JARVIS_AGENT_MODEL", "gpt-4o-mini").strip()
```

### WR-02: `_fact_agent` Agent is constructed at module import time — causes side-effects and test-coupling

**File:** `confluence_logic/agents/fact_extraction_agent.py:178-183`

**Issue:**
```python
_fact_agent = Agent(
    name="FactExtractionAgent",
    model=JARVIS_AGENT_MODEL,
    instructions=FACT_EXTRACTION_PROMPT,
    output_type=AgentOutputSchema(ExtractedFacts, strict_json_schema=False),
)
```
This runs `Agent(...)` unconditionally the moment the module is imported. If the `agents` SDK performs any I/O, validation, or network calls in `__init__`, this will fail or block at import time. More practically, `JARVIS_AGENT_MODEL` is captured at import, so changing the env variable after import has no effect. Tests that import this module must be careful to set `JARVIS_AGENT_MODEL` before the first import — a well-known Python test trap. The `_run_intent_drafter` and `_run_drafter` functions in `drafter_agent.py` correctly construct agents lazily per-call; `fact_extraction_agent.py` should do the same.

```python
# Fix: construct lazily inside _extract_chunk
async def _extract_chunk(text: str) -> ExtractedFacts:
    agent = Agent(
        name="FactExtractionAgent",
        model=JARVIS_AGENT_MODEL,
        instructions=FACT_EXTRACTION_PROMPT,
        output_type=AgentOutputSchema(ExtractedFacts, strict_json_schema=False),
    )
    result = await Runner.run(agent, text)
    ...
```

### WR-03: `JARVIS_FACT_CHUNK_OVERLAP >= JARVIS_FACT_CHUNK_CHARS` causes infinite loop in chunking

**File:** `confluence_logic/agents/fact_extraction_agent.py:336-343`

**Issue:**
The chunk loop advances `start` by `JARVIS_FACT_CHUNK_CHARS - JARVIS_FACT_CHUNK_OVERLAP` characters per iteration. If an operator sets `JARVIS_FACT_CHUNK_OVERLAP` to a value >= `JARVIS_FACT_CHUNK_CHARS` (e.g. misconfigured to `55000` or higher when chunk size is `55000`), the advance is zero or negative, and the `while start < len(text)` loop never terminates. There is no validation of this invariant.

```python
# Fix: add a guard after reading the env vars
if JARVIS_FACT_CHUNK_OVERLAP >= JARVIS_FACT_CHUNK_CHARS:
    raise ValueError(
        f"JARVIS_FACT_CHUNK_OVERLAP ({JARVIS_FACT_CHUNK_OVERLAP}) must be less than "
        f"JARVIS_FACT_CHUNK_CHARS ({JARVIS_FACT_CHUNK_CHARS})"
    )
```

### WR-04: `_build_meeting_context` `tail_budget` overlaps with `head_budget` on short transcripts, and `summary_json` fallback only fires when `facts` provides no data — silently skips summary when facts is non-None but empty

**File:** `confluence_logic/agents/drafter_agent.py:89-99`

**Issue:**
The `summary_json` branch at line 89 is guarded by `if summary_json and not brief_lines`. This means if `facts` is a non-None `ExtractedFacts` object with all empty lists, `brief_lines` will still be `[]`, and `summary_json` WILL be used as the fallback — which is the correct behavior. However, if `facts` contains even a single non-empty field (e.g. one decision), `brief_lines` is populated and `summary_json` is entirely ignored, even when it contains complementary information (e.g. key topics or minutes-of-meeting not present in `ExtractedFacts`). The meeting brief sent to the drafter will be incomplete if `facts` is partially populated.

Additionally, the `head + tail` excerpt (lines 111-113) can in theory emit up to `budget + len("\n[... middle omitted ...]\n")` characters, making the output larger than `DRAFTER_TRANSCRIPT_BUDGET`. This is minor but violates the stated budget.

```python
# Fix: merge both sources instead of OR-ing them
if summary_json:
    if summary_json.get("decisions") and not decisions:
        brief_lines.append("DECISIONS:\n" + ...)
    if summary_json.get("key_topics"):
        brief_lines.append("KEY TOPICS: " + ...)
    # ... etc — use summary_json as supplementary, not exclusive fallback
```

---

## Info

### IN-01: `from confluence_logic.db.vector_store import PineconeStore` placed mid-module with `# noqa: E402`

**File:** `confluence_logic/agents/fact_extraction_agent.py:371`

**Issue:**
The import is placed after all the function definitions and after the module-level `_fact_agent` singleton construction. The `noqa: E402` suppresses the linter warning. While functionally valid, mid-module imports are easy to miss and break the "all imports at top" convention followed by every other module in this codebase. Move it to the top of the file alongside the other imports (after the stdlib and third-party imports).

### IN-02: `_run_drafter` (`_run_drafter` / original drafter) constructs a new `Agent` object on every call

**File:** `confluence_logic/agents/drafter_agent.py:219-223`

**Issue:**
```python
agent = Agent(
    name=f"DrafterAgent-{page_id or 'new'}",
    model=DRAFTER_MODEL,
    instructions=DRAFTER_SYSTEM_PROMPT,
)
```
A new `Agent` instance is created for every page drafted. Given that system prompt and model are fixed, this allocates unnecessary objects. In `_run_intent_drafter` (line 502-506) the same pattern is used. If the `agents` SDK caches anything by identity, repeated construction defeats that. Consider a module-level or function-scoped cached instance (similar to `_fact_agent` for the fact extractor) — or at minimum document why per-call construction is intentional (e.g. if the `name` suffix is load-bearing for SDK tracing).

### IN-03: `_find_relevant_transcript_window` silently truncates `subject` to 40 chars before searching

**File:** `confluence_logic/agents/drafter_agent.py:29`

**Issue:**
```python
needle = subject.strip().lower()[:40]
```
If `intent.subject` is longer than 40 characters (e.g. `"payments service on-call rotation owner"`), the search needle is truncated, which may fail to find a match that would be found with the full string. There is no warning or log when truncation occurs. Either document this as intentional (to avoid searching for too-long needles) or raise the cap to match realistic subject lengths (e.g. 80-100 chars).

---

_Reviewed: 2026-05-15T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
