---
phase: quick
plan: 260411-vfq
type: execute
wave: 1
depends_on: []
files_modified:
  - confluence_logic/classifier.py
  - confluence_logic/jarvis_agentic.py
autonomous: true
requirements: []
must_haves:
  truths:
    - "'what do you feel' fast-paths to meeting_opinion without hitting the LLM classifier"
    - "After Jarvis answers an opinion or summary question, the exchange is stored in meeting_state"
    - "Follow-up queries like 'explain that simpler' reuse the prior exchange as context in _handle_general_question"
  artifacts:
    - path: "confluence_logic/classifier.py"
      provides: "Extended _OPINION_TRIGGERS with 'feel' and 'thoughts'"
      contains: "feel"
    - path: "confluence_logic/jarvis_agentic.py"
      provides: "last_jarvis_response state, _is_followup helper, cross-handler memory wiring"
      contains: "last_jarvis_response"
  key_links:
    - from: "confluence_logic/jarvis_agentic.py::_handle_meeting_opinion"
      to: "meeting_state['last_jarvis_response']"
      via: "write after generate_opinion returns"
      pattern: "last_jarvis_response.*intent.*meeting_opinion"
    - from: "confluence_logic/jarvis_agentic.py::_handle_general_question"
      to: "meeting_state['last_jarvis_response']"
      via: "_is_followup check before LLM call"
      pattern: "_is_followup"
---

<objective>
Add cross-handler conversation memory so Jarvis can handle follow-up questions that reference a prior meeting opinion or summary answer (e.g., "explain that simpler", "elaborate on that").

Purpose: Currently, if a user asks "what do you think about X?" (opinion handler) then follows up with "explain that simpler" (general handler), the general handler has no context about the prior exchange. This plan wires a shared `last_jarvis_response` memory slot across handlers.

Output: Modified `classifier.py` and `jarvis_agentic.py` with cross-handler memory.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@confluence_logic/classifier.py
@confluence_logic/jarvis_agentic.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add opinion triggers and last_jarvis_response state</name>
  <files>confluence_logic/classifier.py, confluence_logic/jarvis_agentic.py</files>
  <action>
1. In `confluence_logic/classifier.py`, line 50-52: Add `"feel"` and `"thoughts"` to the `_OPINION_TRIGGERS` frozenset. The current set is:
   ```python
   _OPINION_TRIGGERS = frozenset({
       "think", "opinion", "recommend", "recommendation", "suggestion", "suggest", "prefer", "choose",
   })
   ```
   Add the two new words to the same set.

2. In `confluence_logic/jarvis_agentic.py`, in the `meeting_state` dict (lines 125-144): Add a new key `"last_jarvis_response": None` after the `"general_history"` key. When populated, this will hold a dict with schema `{"intent": str, "query": str, "answer": str}`.

3. In `confluence_logic/jarvis_agentic.py`, add a module-level helper function `_is_followup(query: str) -> bool` near the other helper functions (around line 155-177). It should:
   - Lowercase the query
   - Return True if the query contains any of these words/phrases: "it", "that", "simpler", "explain", "elaborate", "again", "more", "rephrase", "clarify"
   - Use word boundary matching (re.search with `\b`) to avoid false positives on words like "iterate" containing "it". Specifically for "it" and "that", use `r"\bit\b"` and `r"\bthat\b"`. For the longer words ("simpler", "explain", "elaborate", "again", "more", "rephrase", "clarify"), simple `in` check on the lowered query is fine since they are unambiguous.
  </action>
  <verify>
    <automated>cd /Users/akshathr/Clones/recallai-meeting && python -c "
from confluence_logic.classifier import _OPINION_TRIGGERS
assert 'feel' in _OPINION_TRIGGERS, 'feel missing'
assert 'thoughts' in _OPINION_TRIGGERS, 'thoughts missing'
print('PASS: opinion triggers')

from confluence_logic.jarvis_agentic import meeting_state, _is_followup
assert 'last_jarvis_response' in meeting_state, 'last_jarvis_response missing from state'
assert meeting_state['last_jarvis_response'] is None, 'should init to None'
assert _is_followup('explain that simpler') == True
assert _is_followup('what is iterate') == False, 'should not match iterate'
assert _is_followup('tell me more') == True
assert _is_followup('what is the weather') == False
print('PASS: state and followup helper')
"</automated>
  </verify>
  <done>_OPINION_TRIGGERS contains "feel" and "thoughts"; meeting_state has "last_jarvis_response" key initialized to None; _is_followup correctly identifies follow-up queries with word-boundary safety for short words like "it".</done>
</task>

<task type="auto">
  <name>Task 2: Wire cross-handler memory — write after opinion/summary, read in general</name>
  <files>confluence_logic/jarvis_agentic.py</files>
  <action>
1. In `_handle_meeting_opinion` (line 1117): After the `answer = await opinion_task` line (line 1129) and before speaking it, store the exchange:
   ```python
   meeting_state["last_jarvis_response"] = {
       "intent": "meeting_opinion",
       "query": query,
       "answer": answer,
   }
   ```
   Place this right after `answer = await opinion_task` and before `await _speak_guarded(answer, ...)`.

2. In `_handle_meeting_summary` (line 1075): After `answer = await summary_task` (line 1102), store the exchange similarly:
   ```python
   meeting_state["last_jarvis_response"] = {
       "intent": "meeting_summary",
       "query": query,
       "answer": answer,
   }
   ```
   Place after line 1102, before `await _speak_guarded(answer, ...)` on line 1103. Only do this in the `if specified:` branch where an answer is actually generated (not in the clarification branch).

3. In `_handle_general_question` (line 971): At the start of the function, after `generation = meeting_state["output_generation"]` (line 974), add logic to detect and use cross-handler context:
   ```python
   # Cross-handler follow-up: prepend prior meeting response as context
   prior = meeting_state.get("last_jarvis_response")
   if prior and _is_followup(query):
       cross_context = f"User: {prior['query']}\nAssistant: {prior['answer']}"
       conversation_history = cross_context + "\n" + conversation_history if conversation_history and conversation_history.strip() != "[none]" else cross_context
   ```
   This must happen AFTER `conversation_history = _format_general_history()` (line 976) so we can prepend to it. So the order is:
   - line 976: `conversation_history = _format_general_history()`
   - NEW: check prior + _is_followup, prepend if match
   - line 977: `answer = await answer_general_question(query, conversation_history)`

   Do NOT clear `last_jarvis_response` after use — it should persist until the next opinion/summary overwrites it, allowing multiple follow-ups.
  </action>
  <verify>
    <automated>cd /Users/akshathr/Clones/recallai-meeting && python -c "
import ast, inspect
from confluence_logic.jarvis_agentic import _handle_meeting_opinion, _handle_general_question, _handle_meeting_summary

# Check that last_jarvis_response is referenced in all three handlers
src_opinion = inspect.getsource(_handle_meeting_opinion)
assert 'last_jarvis_response' in src_opinion, 'opinion handler missing last_jarvis_response write'

src_summary = inspect.getsource(_handle_meeting_summary)
assert 'last_jarvis_response' in src_summary, 'summary handler missing last_jarvis_response write'

src_general = inspect.getsource(_handle_general_question)
assert '_is_followup' in src_general, 'general handler missing _is_followup check'
assert 'last_jarvis_response' in src_general, 'general handler missing last_jarvis_response read'
print('PASS: cross-handler memory wired')
"</automated>
  </verify>
  <done>_handle_meeting_opinion and _handle_meeting_summary write the exchange to meeting_state["last_jarvis_response"] after generating the answer. _handle_general_question checks _is_followup and prepends the prior exchange to conversation_history before calling the LLM.</done>
</task>

</tasks>

<verification>
1. Automated verify commands for both tasks pass
2. "feel" and "thoughts" in _OPINION_TRIGGERS
3. meeting_state["last_jarvis_response"] initialized to None
4. _is_followup uses word-boundary matching for "it"/"that"
5. All three handlers reference last_jarvis_response correctly
</verification>

<success_criteria>
- Queries like "what do you feel about..." fast-path to meeting_opinion via classifier
- After an opinion or summary answer, the exchange is persisted in meeting_state
- Follow-up queries containing referential words prepend the prior exchange as LLM context
- No regression in existing general_history conversation tracking
</success_criteria>

<output>
After completion, create `.planning/quick/260411-vfq-add-cross-handler-conversation-memory-to/260411-vfq-SUMMARY.md`
</output>
