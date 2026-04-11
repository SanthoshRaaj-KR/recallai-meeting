---
id: 260411-mce
type: quick
file: confluence_logic/jarvis_agentic.py
autonomous: true
---

<objective>
Serialize TTS output to prevent audio overlap by holding the output lock for the estimated playback duration after each audio POST returns.

Root cause: `_speak_guarded` and `_speak_cached_guarded` release `_output_lock` the instant the HTTP POST response arrives, but audio is still playing on the client. A concurrent handler acquires the lock immediately and sends its own clip, causing both to play simultaneously.

Fix: After a successful audio POST, keep the lock held while sleeping for the estimated playback duration. Allow early exit if a new output generation starts (user interruption).
</objective>

<tasks>

<task type="auto">
  <name>Task 1: Add `_estimate_speech_duration` and update `_speak_guarded`</name>
  <files>confluence_logic/jarvis_agentic.py</files>
  <action>
Add the helper function immediately before `_speak_guarded` (around line 553):

```python
def _estimate_speech_duration(text: str) -> float:
    """Estimate playback duration in seconds from text word count (2.5 words/sec, min 0.5s)."""
    words = len((text or "").split())
    return max(0.5, words / 2.5)
```

Then update `_speak_guarded` (currently lines 553-565) to hold the lock for the estimated duration after the audio POST succeeds. The full updated function:

```python
async def _speak_guarded(text: str, bot_id: str, generation: int, allow_stale: bool = False) -> bool:
    output_lock = _get_output_lock()
    async with output_lock:
        if not allow_stale and generation != meeting_state["output_generation"]:
            return False
        while True:
            remaining = meeting_state["last_user_speech_at"] + JARVIS_SPEECH_HOLD_SECONDS - time.time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(remaining, 0.2))
            if not allow_stale and generation != meeting_state["output_generation"]:
                return False
        ok = await asyncio.to_thread(speak, text, bot_id)
        if ok:
            duration = _estimate_speech_duration(text)
            elapsed = 0.0
            while elapsed < duration:
                await asyncio.sleep(0.1)
                elapsed += 0.1
                if generation != meeting_state["output_generation"]:
                    break
        return ok
```

Note: the `generation != meeting_state["output_generation"]` early-exit inside the sleep loop intentionally allows a newer wake-word/interruption to cancel the wait so the new response is not delayed.
  </action>
  <verify>python3 -c "import ast, sys; ast.parse(open('confluence_logic/jarvis_agentic.py').read()); print('syntax ok')"</verify>
  <done>`_estimate_speech_duration` exists in the file, `_speak_guarded` captures the return value of `asyncio.to_thread(speak, ...)` into `ok`, and sleeps for the estimated duration inside the lock before returning `ok`.</done>
</task>

<task type="auto">
  <name>Task 2: Add `_estimate_cached_duration` and update `_speak_cached_guarded`</name>
  <files>confluence_logic/jarvis_agentic.py</files>
  <action>
Add the helper function immediately before `_speak_cached_guarded` (around line 451):

```python
def _estimate_cached_duration(audio_bytes: bytes) -> float:
    """Estimate MP3 playback duration from byte size (assumes ~32 kbps = 4000 bytes/sec, min 0.3s)."""
    return max(0.3, len(audio_bytes) / 4000)
```

Then update `_speak_cached_guarded` (currently lines 451-464) to hold the lock for the estimated duration after the audio POST succeeds. The full updated function:

```python
async def _speak_cached_guarded(audio_bytes: bytes, bot_id: str, generation: int) -> bool:
    """Like _speak_guarded but sends pre-cached audio bytes instead of synthesizing."""
    output_lock = _get_output_lock()
    async with output_lock:
        if generation != meeting_state["output_generation"]:
            return False
        while True:
            remaining = meeting_state["last_user_speech_at"] + JARVIS_SPEECH_HOLD_SECONDS - time.time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(remaining, 0.2))
            if generation != meeting_state["output_generation"]:
                return False
        ok = await asyncio.to_thread(speak_cached_audio, audio_bytes, bot_id)
        if ok:
            duration = _estimate_cached_duration(audio_bytes)
            elapsed = 0.0
            while elapsed < duration:
                await asyncio.sleep(0.1)
                elapsed += 0.1
                if generation != meeting_state["output_generation"]:
                    break
        return ok
```
  </action>
  <verify>python3 -c "import ast, sys; ast.parse(open('confluence_logic/jarvis_agentic.py').read()); print('syntax ok')"</verify>
  <done>`_estimate_cached_duration` exists in the file, `_speak_cached_guarded` captures the return value of `asyncio.to_thread(speak_cached_audio, ...)` into `ok`, and sleeps for the estimated byte-based duration inside the lock before returning `ok`.</done>
</task>

</tasks>

<success_criteria>
- Both helper functions (`_estimate_speech_duration`, `_estimate_cached_duration`) exist in `jarvis_agentic.py`
- Both guarded speak functions hold the output lock for their estimated playback duration after a successful POST
- Both include the early-exit on generation change so user interruptions are not unnecessarily delayed
- File passes Python AST parse check (no syntax errors)
- No other files are modified
</success_criteria>

<output>
After completion, create `.planning/quick/260411-mce-serialize-tts-output-to-prevent-audio-ov/260411-mce-SUMMARY.md`
</output>
