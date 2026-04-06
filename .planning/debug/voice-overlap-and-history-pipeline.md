---
status: fixing
trigger: "Investigate two bugs: voice-async-overlap and history-pipeline-logic"
created: 2026-04-05T00:00:00Z
updated: 2026-04-05T00:01:00Z
---

## Current Focus

hypothesis: CONFIRMED — all bugs identified and fixes applied.
test: Syntax check passed. Code inspection confirms all speak paths now go through _speak_lock; header race now guarded by _header_lock.
expecting: No audio overlap; no double-header write.
next_action: human verification

## Symptoms

expected: Only one voice plays at a time; new speech waits for current speech to finish. Memory queries efficiently route, fall back gracefully, return clean answers.
actual: Multiple speak()/speak_chunked() calls can fire concurrently with no serialization. Logic/correctness gaps in the flush pipeline.
errors: No crash — audible overlap at runtime / logic concern
reproduction: Two async paths trigger speak() simultaneously — e.g. _flush_sentence_buffer fires while handle_query calls speak_chunked(); or rapid wake-word triggers.
started: Phase 6 added speak_chunked() and _stream_llm_and_speak(); Phase 7 added _flush_sentence_buffer().

## Eliminated

(none yet)

## Evidence

- timestamp: 2026-04-05T00:00:00Z
  checked: jarvis.py speak(), speak_chunked(), _stream_llm_and_speak() — all call sites
  found: |
    All three speak functions are plain async functions with zero mutual exclusion.
    Call sites in websocket_endpoint:
      Line 887: asyncio.create_task(asyncio.to_thread(speak, "Yes?", bot_id))  — fire-and-forget
      Line 881: asyncio.create_task(handle_query(query, bot_id))               — fire-and-forget
      Line 870: asyncio.create_task(_flush_sentence_buffer(bot_id))            — fire-and-forget
    handle_query() internally calls speak_chunked() (lines 644, 650, 700) and
    _stream_llm_and_speak() (line 695). speak_chunked() loops calling speak() sequentially.
    _stream_llm_and_speak() calls speak() inside a token loop.
    No asyncio.Lock, asyncio.Queue, or any other guard exists anywhere across these paths.
  implication: >
    Any two concurrent asyncio.create_task() calls that reach a speak path will
    interleave freely. Concrete race scenarios:
    (A) Line 870 _flush_sentence_buffer task runs and... but wait — _flush_sentence_buffer
        does NOT call speak(). It calls rolling_summarizer + meeting_writer. No audio overlap
        from that path specifically. The audio overlap is between:
    (B) Line 887 speak("Yes?") task runs concurrently with line 881 handle_query() task
        when wake word is detected with empty query — both are fire-and-forget tasks started
        in the same websocket loop iteration (lines 886-887 after force flush).
        Actually no — line 887 only fires when query is "" (bare wake word). Line 881
        fires when query is non-empty. They are mutually exclusive branches.
    (C) Two rapid wake-word triggers: first handle_query task is still running (speaking a
        multi-sentence answer via _stream_llm_and_speak / speak_chunked) when a second
        wake-word fires and spawns a new handle_query task. Both tasks are calling
        speak() / asyncio.to_thread(speak, ...) concurrently — CONFIRMED OVERLAP.
    (D) Line 887 speaks "Yes?" while a prior handle_query task is still in its speak loop —
        confirmed overlap because "Yes?" fires as a new task without checking if voice is busy.

- timestamp: 2026-04-05T00:00:00Z
  checked: _flush_sentence_buffer race between asyncio.create_task (line 870) and force=True (lines 880, 886)
  found: |
    Line 870: asyncio.create_task(_flush_sentence_buffer(bot_id))   -- fires on every sentence
    Line 880: await _flush_sentence_buffer(bot_id, force=True)      -- awaited on full-query wake word
    Line 886: await _flush_sentence_buffer(bot_id, force=True)      -- awaited on bare wake word
    The force=True calls are awaited, so they complete before handle_query is spawned.
    HOWEVER: the create_task on line 870 is NOT cancelled before the force flush on 880/886.
    The task created on line 870 may be scheduled (but not yet running) when the force flush
    on line 880 runs. After the force flush drains and clears the buffer, the line-870 task
    will wake up, acquire the lock, find the buffer empty, and return early (the 'not should_flush
    or not _sentence_buffer' guard on line 763 handles this). So the double-flush is harmless in
    practice — the second flush of an already-drained buffer is a no-op.
    VERDICT: Not a true race condition. The guard clause prevents double-flush corruption.

- timestamp: 2026-04-05T00:00:00Z
  checked: _meeting_header_written flag in _flush_sentence_buffer
  found: |
    _meeting_header_written is read (line 789) and written (line 791) OUTSIDE the
    _sentence_buffer_lock. The lock is released after draining the buffer (line 772 —
    "Outside lock: LLM call + file I/O"). Two concurrent flush tasks (e.g. line 870 task
    racing with a force flush that was already awaited) could both read
    _meeting_header_written == False, both call write_meeting_header(), and write the
    header twice, corrupting the .md file.
    In practice: the guard clause (evidence above) means only one flush runs at a time
    per buffer contents — so the race window is very small. But it IS a latent bug:
    if write_meeting_header() is slow and two flushes could interleave (e.g. disconnect
    flush on line 915 racing with a line-870 task that was created just before disconnect),
    the header write race is real. The MeetingWriterAgent._lock protects the I/O, but
    write_meeting_header opens in "w" mode — a second write would truncate and re-write
    the header, losing the first batch that was appended between the two header writes.
  implication: >
    Double-header write would truncate the .md at the header, losing all previously
    appended batch content. Fix: guard the _meeting_header_written flag and the
    write_meeting_header() call under its own dedicated lock (separate from the buffer
    lock), OR move the header-written check inside the buffer lock block.

- timestamp: 2026-04-05T00:00:00Z
  checked: handle_query() routing for memory queries — HistoryManagerAgent vs OrchestratorAgent
  found: |
    Lines 620-651: if _is_memory_query(query): tries HistoryManagerAgent first.
    history_manager is ALWAYS non-None (created unconditionally at line 144-148 regardless
    of whether Pinecone is configured). So the fallback to OrchestratorAgent (lines 635-640)
    can only be reached if HistoryManagerAgent.run() raises an exception. This is correct
    behavior but the comment on line 619 ("prefer HistoryManagerAgent... over OrchestratorAgent")
    makes it sound like an intentional priority — it is, but the condition on line 624
    `if history_manager is not None` is always True.
    MINOR: The OrchestratorAgent fallback at lines 635-640 is dead code unless
    HistoryManagerAgent throws. Not a bug — just misleading guard.

- timestamp: 2026-04-05T00:00:00Z
  checked: HistoryManagerAgent.run() — missing meetings/ directory handling
  found: |
    MeetingWriterAgent.read_index() (line 158-164) checks `if not self._index_path.exists(): return []`.
    The index path is ./meetings/meeting_index.json. If the meetings/ directory doesn't exist,
    Path.exists() returns False — it does not throw FileNotFoundError for missing parents.
    So read_index() correctly returns [] when the directory has never been created.
    HistoryManagerAgent.run() handles the empty list case (lines 217-224) with a graceful
    "No meeting history found" response.
    VERDICT: No bug — the missing-directory case is handled correctly.

- timestamp: 2026-04-05T00:00:00Z
  checked: N+1 / repeated read_index() I/O
  found: |
    read_index() is called once per handle_query invocation (line 214 in history_manager.py).
    There is no caching layer — every voice query re-reads the JSON file from disk.
    For a typical meeting with O(10-100) meetings in the index, this is negligible.
    The MeetingWriterAgent._lock is an asyncio.Lock, so concurrent read_index() calls
    will serialize — no corruption risk. Not a performance problem at current scale.
    VERDICT: No immediate bug, but a cache would be a future optimization.

- timestamp: 2026-04-05T00:00:00Z
  checked: _flush_sentence_buffer — header/batch/upsert order
  found: |
    Order in code (lines 789-805):
      1. if not _meeting_header_written: write_meeting_header() — correct, header before first batch
      2. append_batch() — correct
      3. upsert_index() — correct, index reflects latest batch speakers/overview
    Order is correct. The header is written before the first batch append. upsert_index
    is called after every batch so the index overview stays current.
    VERDICT: Ordering is correct.

- timestamp: 2026-04-05T00:00:00Z
  checked: read_index() acquires MeetingWriterAgent._lock; upsert_index() also acquires it
  found: |
    If a flush task is mid-way through upsert_index() (holding _lock), and handle_query()
    triggers HistoryManagerAgent which calls read_index() (also needing _lock), the read
    will correctly wait. No deadlock possible — lock is not re-entrant, and no code holds
    the lock while calling anything that also acquires the lock.
    VERDICT: Lock usage is correct and safe.

- timestamp: 2026-04-05T00:00:00Z
  checked: speak() uses hardcoded "temp_jarvis.mp3" filename (line 208)
  found: |
    speak() writes to "temp_jarvis.mp3" unconditionally. When two concurrent speak() calls
    run via asyncio.to_thread(), both threads will race on the same temp file:
    - Thread A writes TTS to temp_jarvis.mp3
    - Thread B writes a different TTS to temp_jarvis.mp3 (overwriting A's audio)
    - Thread A reads temp_jarvis.mp3 — reads B's audio and sends it
    - Thread B reads temp_jarvis.mp3 — reads its own audio (may be already deleted by A's finally)
    This is a file-level data race on top of the audio overlap. Even if the Recall.ai API
    queues audio correctly, the wrong audio content could be sent.
  implication: >
    Critical: concurrent speak() calls corrupt each other's temp files.
    Fix requires BOTH serialization (speak lock) AND unique temp filenames per call.

## Resolution

root_cause: |
  BUG 1 — Voice Overlap (two distinct sub-bugs):
  (1a) No serialization lock guards the speak path. Any two concurrent asyncio tasks that
       call speak(), speak_chunked(), or _stream_llm_and_speak() will interleave audio calls.
       Primary scenarios: two rapid wake-word triggers each spawning handle_query tasks;
       "Yes?" acknowledgement speaking while a prior handle_query is still running.
  (1b) speak() uses a hardcoded temp filename "temp_jarvis.mp3". Concurrent calls from
       asyncio.to_thread() race on the same file — one thread's TTS overwrites another's
       before the first thread reads it.

  BUG 2 — History Pipeline Logic:
  (2a) _meeting_header_written is read and written outside the buffer lock. A disconnect
       flush (line 915) racing with a line-870 task could both see _meeting_header_written==False
       and both call write_meeting_header() (opening the file in "w" mode), truncating the .md
       and losing appended batch content.
  (2b) Minor: `if history_manager is not None` guard (line 624) is always True — history_manager
       is unconditionally instantiated. The OrchestratorAgent fallback comment is misleading
       but the logic is functionally correct.
  No other history pipeline bugs found. Directory handling, flush ordering, lock correctness,
  and fallback chain are all correct.

fix: |
  Bug 1a — Added module-level _speak_lock = asyncio.Lock(). All speak paths now acquire it:
    - speak_chunked() wraps its entire sentence loop with `async with _speak_lock`
    - _stream_llm_and_speak() collects all sentences first (outside lock), then speaks
      them all under `async with _speak_lock`
    - bare wake-word "Yes?" ack is now an async helper _ack_yes() that acquires _speak_lock
    - handle_query() error/fallback paths replaced bare asyncio.to_thread(speak,...) with
      speak_chunked() which already acquires the lock

  Bug 1b — speak() now uses `f"temp_jarvis_{threading.get_ident()}.mp3"` as the temp
  filename, making it unique per OS thread. Since asyncio.to_thread() runs each call in
  a separate thread pool worker, concurrent calls cannot overwrite each other's file.

  Bug 2a — Added module-level _header_lock = asyncio.Lock(). The _meeting_header_written
  check and write_meeting_header() call in _flush_sentence_buffer() are now wrapped with
  `async with _header_lock`, preventing two concurrent flush tasks from both writing the
  header and truncating the .md file.

verification: syntax check passed (python -m ast). Logic confirmed by inspection.
files_changed:
  - jarvis.py
