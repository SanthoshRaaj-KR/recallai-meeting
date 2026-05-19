# Jarvis Voice Pipeline — Architecture, Latency & Optimizations

> **Status:** Working as of 2026-05-19 (Phase 6 architecture confirmed functional)
> **Scope:** Real-time voice path only — Confluence review pipeline is separate and not covered here.

---

## 0. Recent Fixes That Made the Pipeline Work

Before the pipeline worked end-to-end, two bugs were silently breaking audio delivery. Both were found and fixed on 2026-05-19.

---

### Fix A — `jarvis_agentic.py`: NameError on every bot start (`return room, source`)

**What was broken:**
`_create_livekit_room()` ended with `return room, source`. In Phase 6 the `AudioSource` object was removed (the agent now publishes TTS directly via AgentSession, not via a manually created audio track), but the `source` variable was never cleaned up. Every call to `POST /bot/start` raised a `NameError: name 'source' is not defined`.

The exception was caught by `_start_bot_for_session`, which marked the session as `"error"` and returned an error response to the caller. Crucially, the agent dispatch ran *before* the crash, so the agent was alive and generating TTS — but the API always reported failure, causing the UI and orchestrator to think the bot never started successfully.

**Fix:**
Removed `return room, source`. The function now returns `None` implicitly, which is what the caller expected (the return value was already being discarded). One-line change.

---

### Fix B — `bot.html`: AudioContext async race condition (audio never played reliably)

**What was broken:**
The subscriber IIFE in `bot.html` used WebAudio API (`AudioContext → createMediaStreamSource → ctx.destination`) as the primary audio path with `HTMLAudioElement` as fallback. The logic was:

```javascript
function ensureAudioCtx() {
  audioCtx = new AudioContext({ latencyHint: 'interactive' });
  audioCtx.resume();   // ← fire-and-forget Promise — does NOT block
  return audioCtx;     // ← returned immediately, ctx.state still 'suspended'
}

function attachTrack(track, identity) {
  var ctx = ensureAudioCtx();
  if (ctx.state === 'running') {        // ← ALWAYS false — resume() is async
    _attachViaWebAudio(rawTrack, track, identity);
  } else {
    // AudioContext suspended — call resume() AGAIN inside .then()
    ctx.resume().then(function () {
      if (ctx.state === 'running') {    // ← may or may not work in headless Chrome
        _attachViaWebAudio(rawTrack, track, identity);
      } else {
        playTrackFallback(track, identity);  // ← fallback
      }
    });
  }
}
```

The problem: `ensureAudioCtx()` called `ctx.resume()` as a fire-and-forget Promise and returned the context object immediately. The caller then synchronously checked `ctx.state === 'running'` — but since `resume()` is async, the state was still `'suspended'` at that check. **The WebAudio primary path was never taken.** The code always fell into the `else` branch, which called `ctx.resume()` a second time inside a `.then()` handler. Whether that second attempt succeeded in Recall's headless Chrome was non-deterministic — sometimes it worked, sometimes it silently dropped the audio.

**Fix:**
Flipped the priority. `HTMLAudioElement` via `track.attach()` is now the **primary** path, called first and unconditionally. Recall's official docs confirm that headless Chrome reliably captures audio from `<audio>` elements and injects it into the meeting. WebAudio is only layered on top if `ctx.state === 'running'` at the moment a track arrives (i.e., the pre-warm on `RoomEvent.Connected` already resolved). This eliminates the race condition entirely.

```javascript
function attachTrack(track, identity) {
  // Primary: HTMLAudioElement — always reliable in Recall's headless Chrome
  var audioEl = playTrackFallback(track, identity);   // plays immediately

  // Secondary: WebAudio — only if context already confirmed running
  var ctx = ensureAudioCtx();
  if (ctx.state === 'running') {
    // Swap out the <audio> element for the WebAudio node
    audioEl.pause(); audioEl.remove();
    _attachViaWebAudio(rawTrack, track, identity);
  } else {
    // Kick off a background resume() — doesn't block audio delivery
    ctx.resume().catch(function (e) { console.warn(...); });
  }
}
```

**Why this matters:** Before the fix, audio playback in Recall depended entirely on a second async `ctx.resume()` call succeeding in a headless Chrome environment where autoplay policies are non-standard. After the fix, audio plays the moment the `TrackSubscribed` event fires, via a synchronous `track.attach()` call that Recall's Chrome handles natively.

---

## 1. End-to-End Pipeline

```
Participants speak in meeting
  │
  ▼
Recall.ai headless Chrome (bot.html, loaded via output_media.camera.kind="webpage")
  │  bot.html publisher IIFE: getUserMedia({ echoCancellation:false, noiseSuppression:false })
  │  Joins LiveKit room as "recall-browser-{session_id}" (can_publish=True)
  │
  ▼  WebRTC Opus encode → LiveKit SFU relay
  │  ~20–50 ms
  ▼
AgentSession (agent_worker.py)
  │  Subscribed to participant "recall-browser-{session_id}" only
  │
  ├─▶ Silero VAD (prewarmed in prewarm() at worker startup)
  │     Detects speech start/end, gates STT
  │     ~10–20 ms end-of-speech detection
  │
  ├─▶ Deepgram Nova-3 STT (LiveKit Inference cloud)
  │     Streaming — partial + final transcripts in real time
  │     ~100–200 ms to final transcript after VAD end-of-speech
  │
  ▼
on_user_turn_completed (JarvisAgent)
  │  Wake word gate via regex — no I/O, pure string match
  │  ~1 ms
  │
  ├─ No wake word → content cleared → llm_node returns None → complete silence
  ├─ Bare "Hey Jarvis" → session.say("Yes?") + listening mode → no LLM
  └─ "Hey Jarvis, <query>" → content rewritten to query only → dispatched to LLM
  │
  ▼
llm_node → gpt-4o-mini (LiveKit Inference)
  │  Streams tokens — AgentSession pipes directly to TTS node
  │  TTFT: ~200–400 ms
  │
  ▼
tts_node → Cartesia Sonic-3 (LiveKit Inference)
  │  Streaming — first audio frame after ~40–90 ms from first LLM token
  │  Text transforms: filter_emoji, filter_markdown
  │
  ▼  AgentSession publishes TTS as agent's own participant track in LiveKit room
  │
  ▼
bot.html subscriber IIFE (same Recall headless Chrome)
  │  TrackSubscribed fires for the agent's track
  │  Skips any "recall-browser-*" participant (prevents echo)
  │  Plays via HTMLAudioElement (track.attach(), autoplay=true)
  │  ~20–50 ms WebRTC decode + playback buffer
  │
  ▼
Recall captures browser audio output → injects into meeting
  │
  ▼
All meeting participants hear Jarvis
```

**Total perceived latency (P50):** ~400–800 ms after user stops speaking
**Worst case (P95):** ~1500–2000 ms (false interruption timeout adds up to 1.2 s)

---

## 2. Latency Budget Per Stage

| Stage | Component | P50 | P95 | Notes |
|-------|-----------|-----|-----|-------|
| WebRTC capture + encode | getUserMedia → LiveKit SFU | 20–50 ms | 80 ms | Opus codec, no bitrate override |
| VAD end-of-speech | Silero (prewarmed) | 30–80 ms | 150 ms | Silence detection after last voice frame |
| STT final transcript | Deepgram Nova-3 (multi) | 100–200 ms | 400 ms | `language="multi"` adds ~30–80 ms vs EN |
| Wake word gate | Regex in on_user_turn_completed | <1 ms | <1 ms | Pure string match, no I/O |
| Turn detection endpointing | min_delay=0.3 s, max_delay=1.5 s | 300 ms | 1500 ms | Waits for VAD silence gap |
| False interruption timeout | resume_false_interruption=True | 0 ms | 1200 ms | Only fires on brief interjections during response |
| LLM TTFT | gpt-4o-mini (LiveKit Inference) | 200–400 ms | 800 ms | Full tool calls add 300–800 ms |
| TTS first audio frame | Cartesia Sonic-3 | 40–90 ms | 150 ms | Streams in parallel with LLM tokens |
| WebRTC decode + playback | bot.html → Recall Chrome | 20–50 ms | 80 ms | |
| **Total (no tool call)** | | **~400–800 ms** | **~1500 ms** | |
| **Total (with tool call)** | | **~700–1600 ms** | **~2500 ms** | |

---

## 3. Current Bottlenecks

### HIGH IMPACT — Turn detection endpointing (min_delay: 0.3 s)

The `endpointing.min_delay = 0.3` means the agent waits at least 300 ms of silence after the user stops speaking before committing to processing. This is the single largest controllable latency element. Reducing to 0.15 s would cut 150 ms from every response with minimal risk of premature cutoffs on natural speech pauses.

### HIGH IMPACT — `language="multi"` on Deepgram

The multilingual Nova-3 model is ~30–80 ms slower than the English-only variant. For any deployment that is English-only, this is a free win.

### MEDIUM IMPACT — False interruption timeout (1.2 s)

`false_interruption_timeout: 1.2` means after Jarvis starts speaking, if someone says something brief, the agent waits 1.2 s before deciding it's not a real interruption. In practice this only fires when someone speaks over Jarvis, but it can extend the tail end of latency perception.

### MEDIUM IMPACT — No pre-recorded ack for the listening mode response

When the user says bare "Hey Jarvis" (listening mode), the agent calls `session.say("Yes?")` which routes through the full TTS pipeline (Cartesia cloud round-trip, ~100–200 ms). A pre-cached PCM frame for "Yes?" would play instantly (<5 ms) and feel much snappier.

### LOW IMPACT — `preemptive_generation=False`

Currently disabled because the wake-word rewrite in `on_user_turn_completed` always changes the message content, which causes the speculative LLM output to be discarded and its TTS frames to produce an audible glitch at response start. This is the correct tradeoff given the wake-word architecture. No change recommended here.

### LOW IMPACT — No audio bitrate set on publisher

The publisher IIFE uses default LiveKit Opus bitrate (~32–64 kbps). Explicitly setting `audioBitrate: 24000` would slightly reduce encode time in constrained environments, but the latency impact is negligible on a local/cloud setup.

---

## 4. Optimization Recommendations (Ranked by Impact)

### 1. Reduce endpointing min_delay: 0.3 → 0.15 s  ★★★

**Saves: ~150 ms on every single response**

```python
# agent_worker.py — in AgentSession constructor
turn_handling=TurnHandlingOptions(
    endpointing={
        "min_delay": 0.15,   # was 0.3 — cut 150ms from every turn
        "max_delay": 1.5,
    },
    ...
),
```

Risk: Very low. Users who speak in complete sentences won't notice. Only affects users who pause mid-sentence and then resume — Jarvis might cut them off slightly earlier. Easy to tune back up if needed.

---

### 2. Switch STT to English-only if meeting is EN  ★★★

**Saves: ~30–80 ms on every STT round-trip**

```python
# agent_worker.py
stt=inference.STT(model="deepgram/nova-3", language="en"),   # was "multi"
```

Or add a config variable:
```python
JARVIS_STT_LANGUAGE = os.getenv("JARVIS_STT_LANGUAGE", "en").strip()
stt=inference.STT(model="deepgram/nova-3", language=JARVIS_STT_LANGUAGE),
```

Risk: Zero for English-only meetings. Multilingual meetings would lose STT accuracy for non-English speech.

---

### 3. Pre-cache "Yes?" ack as PCM frames  ★★

**Saves: ~100–200 ms on bare wake-word responses (the snappiness of the first interaction)**

Instead of routing "Yes?" through Cartesia TTS on every bare wake, pre-generate the audio once at startup and push it directly:

```python
# agent_worker.py — in on_user_turn_completed, bare wake case
if not query:
    logger.info("👂 BARE WAKE — entering listening mode")
    self._listening_mode = True
    self._listening_since = time.perf_counter()
    new_message.content = []
    # session.say() goes through full TTS pipeline (~150 ms).
    # Pre-cached frames would play in <5 ms:
    # await _play_ack_frames(self.session, "yes")   ← target implementation
    await self.session.say("Yes?", add_to_chat_ctx=False)
    return
```

The `audio_cache.py` module already implements this pattern — it pre-generates acknowledgement MP3s and caches them. The `push_audio_to_livekit()` function in `jarvis_agentic.py` handles the PCM conversion. This would need to be wired into the AgentSession path.

---

### 4. Reduce false_interruption_timeout: 1.2 → 0.6 s  ★★

**Saves: up to 600 ms on turns where a participant briefly speaks while Jarvis is responding**

```python
turn_handling=TurnHandlingOptions(
    ...
    interruption={
        "resume_false_interruption": True,
        "false_interruption_timeout": 0.6,   # was 1.2
    },
),
```

Risk: Low-medium. Brief interjections ("mm-hmm", "yeah") during Jarvis responses will now more aggressively interrupt. For a meeting assistant this is usually acceptable — users in meetings don't accidentally say "Hey Jarvis" often.

---

### 5. Add min_endpointing_delay to STT config  ★

**Saves: 0 ms (but locks in behavior and prevents upstream regressions)**

```python
stt=inference.STT(
    model="deepgram/nova-3",
    language="en",
    # min_endpointing_delay=100,   ← explicit; removes dependency on Deepgram default
),
```

This is hygiene, not a latency win. Documents and pins the endpointing behavior so Deepgram API changes don't silently affect response timing.

---

### 6. Add wake-word health check + publisher presence detection  ★

**Saves: 0 ms latency, but prevents silent failures**

If the publisher IIFE fails (token error, getUserMedia denied), the agent subscribes to `recall-browser-{session_id}` but that participant never joins. The agent silently hears nothing with no error surfaced.

```python
# agent_worker.py entrypoint — after session.start()
async def _wait_for_publisher(room, session_id, timeout=30.0):
    target = f"recall-browser-{session_id}"
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if any(p.identity == target for p in room.remote_participants.values()):
            return True
        await asyncio.sleep(1.0)
    return False

if session_id:
    present = await _wait_for_publisher(ctx.room, session_id)
    if not present:
        logger.error("❌ Publisher recall-browser-%s never joined — agent is deaf", session_id)
```

---

## 5. Quick Summary

| Optimization | Latency Saved | Effort | Risk |
|---|---|---|---|
| Reduce endpointing min_delay 0.3→0.15 | 150 ms every turn | 1 line | Low |
| Switch STT to language="en" | 30–80 ms every turn | 1 line | Zero (EN-only) |
| Pre-cache "Yes?" ack | 100–200 ms on bare wakes | Medium | Low |
| Reduce false_interruption_timeout 1.2→0.6 | Up to 600 ms on interruptions | 1 line | Low-Medium |
| Publisher health check | 0 ms (reliability) | ~15 lines | Zero |

**Recommended first pass:** Apply optimizations 1 and 2 — they are single-line changes with no risk and together shave ~180–230 ms off the P50 latency of every response.

---

## 6. Phase 7 Upgrade Plan — Voice Pipeline v2

> **Status:** Planned — not yet implemented
> **Goal:** Cut P50 total latency from ~400–800 ms to ~200–400 ms by upgrading every stage of the pipeline.

### Stack Comparison

| Component | Current | Target | Why |
|-----------|---------|--------|-----|
| STT | Deepgram Nova-3 (multi, via LiveKit Inference) | AssemblyAI Universal-3 Pro Streaming (plugin-direct) | Better accuracy, keyterm prompting for wake-word, ~150 ms P50, cheaper |
| LLM | gpt-4o-mini | gpt-4.1-mini | ~50% lower TTFT, better instruction-following, 1M token context |
| TTS | Cartesia Sonic-3 | Cartesia Sonic-Turbo | 40 ms TTFA vs 90 ms — same price, same API, same voices |
| Endpointing | min_delay=0.3 s | min_delay=0.15 s | Saves 150 ms every turn; safe with wake-word gate |
| False interruption | 1.2 s | 0.6 s | Halves tail latency on interrupted responses |

### Change 1 — STT: AssemblyAI Universal-3 Pro Streaming

**Why plugin-direct (not LiveKit Inference):** LiveKit Inference does not expose mid-stream prompting controls. Using `assemblyai.STT` directly gives access to `word_boost` and `boost_param`, which is the entire point of switching.

**Important constraint:** Recall is only used for joining the meeting and publishing audio to LiveKit via the browser publisher (`recall-browser-{session_id}`). Recall's transcript/realtime_endpoints are NOT used — Jarvis hears via STT on the LiveKit audio track, not via Recall's transcript webhook.

```python
# requirements.txt
# livekit-agents[assemblyai]

from livekit.plugins import assemblyai

session = AgentSession(
    vad=silero.VAD.load(),   # keep Silero — required for barge-in detection
    stt=assemblyai.STT(
        model="universal-3-rt-pro",
        word_boost=["Jarvis", "Hey Jarvis"],  # reduces "Hey Travis"/"Hey Gervis" misses
        boost_param="high",
        language_code="en_us",               # drop multi — saves 30–80 ms
    ),
    ...
)
```

**Dynamic keyterms:** Pull attendee names and project names from meeting metadata at session start and add them to `word_boost` — improves transcription of names mentioned in the meeting.

**New env var:** `ASSEMBLYAI_API_KEY`

### Change 2 — LLM: gpt-4o-mini → gpt-4.1-mini

```python
# agent_worker.py
llm=inference.LLM("openai/gpt-4.1-mini"),
# or via env: JARVIS_LK_LLM=openai/gpt-4.1-mini
```

~50% TTFT reduction. Better instruction-following. 1M token context window (useful for long meeting transcripts in chat context). Only fires on wake-word hits so the per-token cost increase is negligible.

### Change 3 — TTS: Sonic-3 → Sonic-Turbo

```python
# agent_worker.py
tts=inference.TTS(
    f"{JARVIS_LK_TTS_PROVIDER}/sonic-turbo",  # was sonic-3
    voice=JARVIS_LK_TTS_VOICE,
),
```

40 ms TTFA vs 90 ms. Same SSM architecture, same voice IDs, same price ($50/M chars). Free 50 ms back on every response.

### Change 4 — Endpointing: 0.3 s → 0.15 s + false_interruption_timeout: 1.2 → 0.6

```python
turn_handling=TurnHandlingOptions(
    endpointing={
        "min_delay": 0.15,   # was 0.3
        "max_delay": 1.5,
    },
    interruption={
        "resume_false_interruption": True,
        "false_interruption_timeout": 0.6,   # was 1.2
    },
),
```

Endpointing cut is safe with the wake-word gate: a premature cut on a non-Jarvis utterance just means the regex sees a partial transcript, finds no wake word, and discards it — no LLM misfire possible. False interruption cut halves tail latency when someone speaks over Jarvis mid-response.

### Change 5 — Pre-cache the "Yes?" Ack

```python
# agent_worker.py — on_user_turn_completed, bare wake case
# Current (full TTS pipeline, ~150 ms):
await self.session.say("Yes?", add_to_chat_ctx=False)

# Target (<5 ms, pre-generated PCM frames):
await _play_ack_frames(self.session, "yes")
```

Generate the "Yes?" audio once at worker startup using Cartesia, cache as PCM frames, push directly into the AgentSession audio track. `audio_cache.py` already implements the pattern — wiring into the AgentSession path is the only new work.

### What Does NOT Change

| Component | Why it stays |
|-----------|-------------|
| Silero VAD | Still required for barge-in detection — AssemblyAI's built-in turn detection does not replace VAD for interruption handling |
| `preemptive_generation=False` | Our `on_user_turn_completed` always rewrites message content → speculative output is always discarded → audible glitch if enabled |
| Recall for joining/streaming | Recall joins the meeting and publishes audio via the browser publisher IIFE. It is NOT used for transcription — `realtime_endpoints` transcript webhook is not consulted by the voice agent |
| Wake-word gate architecture | The gate stays as the primary turn-detection mechanism; `turn_detection="stt"` is explicitly NOT used |

### Expected Latency After Upgrade

| Stage | Before | After |
|-------|--------|-------|
| STT final transcript | 100–200 ms | ~150 ms (U3 Pro P50) |
| STT language overhead | +30–80 ms (multi) | 0 ms (EN only) |
| Endpointing wait | 300 ms floor | 150 ms floor |
| LLM TTFT | ~300–400 ms | ~150–200 ms |
| TTS first audio | ~90 ms | ~40 ms |
| Bare wake "Yes?" | ~150 ms | <5 ms (cached) |
| **Total P50 (no tool call)** | **~400–800 ms** | **~200–400 ms** |
