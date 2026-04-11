---
type: execute
wave: 1
depends_on: []
files_modified:
  - confluence_logic/meeting_responder.py
  - confluence_logic/jarvis_agentic.py
autonomous: true
must_haves:
  truths:
    - "When summary type is ambiguous, Jarvis immediately says 'Sure! Just give me a sec.' then asks brief vs detailed"
    - "When summary type is already specified in query, Jarvis immediately says 'Sure! Just give me a sec.' and skips the clarifying question"
    - "summarize_meeting() with detail_level='brief' returns a shorter bullet-style response capped at JARVIS_SUMMARY_BRIEF_MAX_TOKENS (default 150)"
    - "summarize_meeting() with detail_level='detailed' returns a full narrative response capped at JARVIS_SUMMARY_MAX_TOKENS (default 400)"
    - "The clarification response ('brief' or 'detailed') bypasses the wake word and is routed to the summary handler"
  artifacts:
    - path: "confluence_logic/meeting_responder.py"
      provides: "summarize_meeting with detail_level parameter and dual prompt/token logic"
    - path: "confluence_logic/jarvis_agentic.py"
      provides: "_handle_meeting_summary with ack, optional clarification, and listen state"
  key_links:
    - from: "jarvis_agentic.py _handle_meeting_summary"
      to: "meeting_state['pending_summary_clarification']"
      via: "set after speaking clarifying question"
    - from: "process_transcript_event / handle_spoken_request"
      to: "_handle_summary_clarification_answer"
      via: "pending_summary_clarification state check"
---

<objective>
Add immediate spoken acknowledgment ("Sure! Just give me a sec.") to the meeting summary handler, and introduce a brief/detailed clarification flow when the user does not specify summary type. When type is already specified in the original query the acknowledgment fires and generation begins directly.

Purpose: Makes the bot feel immediately responsive (no silent pause during LLM generation) and lets users choose summary verbosity.
Output: Updated meeting_responder.py with detail_level support; updated jarvis_agentic.py with ack + clarification flow.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
</execution_context>

<context>
@.planning/STATE.md

<!-- Key interfaces for the executor — no codebase exploration needed. -->
<interfaces>
From confluence_logic/meeting_responder.py:
```python
JARVIS_SUMMARY_MAX_TOKENS = int(os.getenv("JARVIS_SUMMARY_MAX_TOKENS", "400"))
# Add: JARVIS_SUMMARY_BRIEF_MAX_TOKENS = int(os.getenv("JARVIS_SUMMARY_BRIEF_MAX_TOKENS", "150"))

async def summarize_meeting(transcript_log: List[Dict[str, Any]]) -> str:
# Change signature to:
async def summarize_meeting(transcript_log: List[Dict[str, Any]], detail_level: str = "detailed") -> str:
```

From confluence_logic/jarvis_agentic.py:
```python
# TTS mechanism (wrap in asyncio.to_thread for async contexts):
speak(text: str, bot_id: str) -> bool          # blocking, synthesizes TTS and POSTs to Recall
async def _speak_guarded(text, bot_id, generation, allow_stale=False) -> bool  # use this

# State dict (global):
meeting_state = {
    "pending_general_clarification": None,  # pattern to follow for summary clarification
    "output_generation": int,
    "transcript_log": list,
    ...
}

# Current handler (replace body):
async def _handle_meeting_summary(bot_id: str) -> None: ...

# Current routing in handle_spoken_request (lines ~958-961):
if intent == "meeting_summary":
    asyncio.create_task(_handle_meeting_summary(bot_id))
    return

# Change to pass query:
if intent == "meeting_summary":
    asyncio.create_task(_handle_meeting_summary(spoken_query, bot_id))
    return

# Routing for clarification answers in handle_spoken_request:
# (follow the pending_general_clarification pattern — check pending_summary_clarification first)

# process_transcript_event bypasses wake word for pending_general_clarification;
# add the same bypass for pending_summary_clarification.
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add detail_level parameter to summarize_meeting()</name>
  <files>confluence_logic/meeting_responder.py</files>
  <action>
1. Add a new module-level constant after JARVIS_SUMMARY_MAX_TOKENS:
   ```python
   JARVIS_SUMMARY_BRIEF_MAX_TOKENS = int(os.getenv("JARVIS_SUMMARY_BRIEF_MAX_TOKENS", "150"))
   ```

2. Change the signature of summarize_meeting():
   ```python
   async def summarize_meeting(transcript_log: List[Dict[str, Any]], detail_level: str = "detailed") -> str:
   ```

3. Inside summarize_meeting(), branch on detail_level BEFORE building the system_prompt:
   - If detail_level == "brief":
     - system_prompt: "You are Jarvis, an AI assistant attending a live meeting. The user wants a brief summary. Give a short, punchy bullet-point style summary of the 3-5 most important points discussed. No more than 5 bullet points. Speak naturally — say 'Here are the main points:' then list them conversationally."
     - max_tokens: JARVIS_SUMMARY_BRIEF_MAX_TOKENS
   - Else (detailed, default):
     - Keep existing system_prompt and max_tokens (JARVIS_SUMMARY_MAX_TOKENS) unchanged.

4. The user_prompt and all other logic remain identical. No other changes.
  </action>
  <verify>python3 -c "import ast, sys; ast.parse(open('confluence_logic/meeting_responder.py').read()); print('syntax ok')"</verify>
  <done>summarize_meeting() accepts detail_level kwarg; brief uses JARVIS_SUMMARY_BRIEF_MAX_TOKENS and a bullet-style prompt; detailed is unchanged from current behavior.</done>
</task>

<task type="auto">
  <name>Task 2: Add ack + clarification flow to _handle_meeting_summary() in jarvis_agentic.py</name>
  <files>confluence_logic/jarvis_agentic.py</files>
  <action>
**Step A — Add pending_summary_clarification to meeting_state init (line ~118):**
Add a new key alongside pending_general_clarification:
```python
"pending_summary_clarification": None,  # {"bot_id": str, "expires_at": float}
```

**Step B — Add the clarification answer handler (new function, place after _handle_general_clarification_answer ~line 919):**
```python
JARVIS_SUMMARY_CLARIFICATION_TIMEOUT = float(os.getenv("JARVIS_SUMMARY_CLARIFICATION_TIMEOUT", "15.0"))

async def _handle_summary_clarification_answer(answer_text: str, pending: dict) -> None:
    """Handle user's brief/detailed response and generate the appropriate summary."""
    bot_id = pending["bot_id"]
    generation = meeting_state["output_generation"]
    meeting_state["pending_summary_clarification"] = None

    normalized = answer_text.strip().lower()
    if any(w in normalized for w in ("brief", "short", "quick", "concise")):
        detail_level = "brief"
    else:
        detail_level = "detailed"

    try:
        transcript_log = list(meeting_state["transcript_log"])
        answer = await summarize_meeting(transcript_log, detail_level=detail_level)
        await _speak_guarded(answer, bot_id, generation, allow_stale=True)
    except Exception as e:
        logger.error("Summary clarification resolution failed: %s", e)
```

**Step C — Replace _handle_meeting_summary() body (starting ~line 922):**
Change signature to accept `query`:
```python
async def _handle_meeting_summary(query: str, bot_id: str) -> None:
    """Speak acknowledgment, optionally ask brief/detailed, then generate summary."""
    generation = meeting_state["output_generation"]
    normalized_query = query.strip().lower()

    # Detect detail level from original query
    if any(w in normalized_query for w in ("brief", "short", "quick", "concise")):
        detail_level = "brief"
        specified = True
    elif any(w in normalized_query for w in ("detailed", "full", "long", "complete", "thorough")):
        detail_level = "detailed"
        specified = True
    else:
        detail_level = None
        specified = False

    # Immediately acknowledge (fire before any generation)
    await _speak_guarded("Sure! Just give me a sec.", bot_id, generation, allow_stale=True)

    if specified:
        # Type already known — generate directly
        try:
            transcript_log = list(meeting_state["transcript_log"])
            answer = await summarize_meeting(transcript_log, detail_level=detail_level)
            await _speak_guarded(answer, bot_id, generation, allow_stale=True)
        except Exception as e:
            logger.error("Meeting summary handling failed: %s", e)
    else:
        # Ask clarifying question and enter listen state
        await _speak_guarded("Do you want a detailed or a brief summary?", bot_id, generation, allow_stale=True)
        meeting_state["pending_summary_clarification"] = {
            "bot_id": bot_id,
            "expires_at": time.time() + JARVIS_SUMMARY_CLARIFICATION_TIMEOUT,
        }
        logger.info("Summary clarification mode activated (timeout: %.0fs)", JARVIS_SUMMARY_CLARIFICATION_TIMEOUT)
```

**Step D — Update handle_spoken_request() routing (~line 958) to pass query:**
Change:
```python
asyncio.create_task(_handle_meeting_summary(bot_id))
```
To:
```python
asyncio.create_task(_handle_meeting_summary(spoken_query, bot_id))
```

**Step E — Route clarification answers in handle_spoken_request():**
At the top of handle_spoken_request(), BEFORE the intent classification block, add a check for pending_summary_clarification (mirror the pending_general check around line 968):
```python
pending_summary = meeting_state.get("pending_summary_clarification")
if pending_summary:
    logger.info("Routing to summary clarification handler: %s", spoken_query[:60])
    asyncio.create_task(_handle_summary_clarification_answer(spoken_query, pending_summary))
    return
```

**Step F — Bypass wake word in process_transcript_event() for pending_summary_clarification:**
In process_transcript_event() (around line 1086, after the pending_general_clarification block), add:
```python
# Check for pending summary clarification (no wake word needed)
pending_summary = meeting_state.get("pending_summary_clarification")
if pending_summary:
    if time.time() > pending_summary["expires_at"]:
        logger.info("Summary clarification timeout expired, clearing state.")
        meeting_state["pending_summary_clarification"] = None
    else:
        meeting_state["jarvis_listening"] = False
        return sentence
```
  </action>
  <verify>python3 -c "import ast, sys; ast.parse(open('confluence_logic/jarvis_agentic.py').read()); print('syntax ok')"</verify>
  <done>
  - _handle_meeting_summary(query, bot_id) detects brief/detailed keywords in query
  - Immediate "Sure! Just give me a sec." fires via _speak_guarded before any generation
  - If type unspecified: asks "Do you want a detailed or a brief summary?" and sets pending_summary_clarification
  - If type specified: calls summarize_meeting(transcript_log, detail_level=...) directly
  - process_transcript_event bypasses wake word when pending_summary_clarification is active
  - handle_spoken_request routes to _handle_summary_clarification_answer before intent classification
  </done>
</task>

</tasks>

<verification>
```bash
# Syntax check both files
python3 -c "import ast; ast.parse(open('confluence_logic/meeting_responder.py').read()); print('meeting_responder.py ok')"
python3 -c "import ast; ast.parse(open('confluence_logic/jarvis_agentic.py').read()); print('jarvis_agentic.py ok')"

# Confirm new state key present
grep -n "pending_summary_clarification" confluence_logic/jarvis_agentic.py

# Confirm detail_level parameter present
grep -n "detail_level" confluence_logic/meeting_responder.py confluence_logic/jarvis_agentic.py

# Confirm ack string
grep -n "Just give me a sec" confluence_logic/jarvis_agentic.py

# Confirm JARVIS_SUMMARY_BRIEF_MAX_TOKENS
grep -n "JARVIS_SUMMARY_BRIEF_MAX_TOKENS" confluence_logic/meeting_responder.py
```
</verification>

<success_criteria>
- "Sure! Just give me a sec." is spoken immediately for all meeting summary requests (both specified and unspecified type)
- Queries containing "brief"/"short"/"quick"/"concise" go directly to summarize_meeting(detail_level="brief")
- Queries containing "detailed"/"full"/"long"/"complete"/"thorough" go directly to summarize_meeting(detail_level="detailed")
- Ambiguous queries trigger "Do you want a detailed or a brief summary?" and the user's spoken answer is captured without a wake word (within timeout)
- brief summaries use JARVIS_SUMMARY_BRIEF_MAX_TOKENS (default 150) with a bullet-point prompt
- detailed summaries use JARVIS_SUMMARY_MAX_TOKENS (default 400) with the existing narrative prompt
- _handle_meeting_opinion is unchanged
</success_criteria>

<output>
No SUMMARY.md required for quick plans.
</output>
