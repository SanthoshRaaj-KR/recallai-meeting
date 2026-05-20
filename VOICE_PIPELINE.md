# Jarvis Voice Pipeline — Complete Architecture & Technical Reference

> **Status:** Phase 7 implemented (2026-05-20). All five latency optimisations live.
> **Scope:** Real-time voice path only. Confluence review pipeline (post-meeting) is separate.

---

## Table of Contents

1. [What Jarvis Does in a Meeting](#1-what-jarvis-does-in-a-meeting)
2. [Full End-to-End Pipeline](#2-full-end-to-end-pipeline)
3. [How LiveKit Powers the Voice Pipeline](#3-how-livekit-powers-the-voice-pipeline)
4. [Audio Pipeline Deep Dive](#4-audio-pipeline-deep-dive)
5. [Latency Budget (Phase 7 State)](#5-latency-budget-phase-7-state)
6. [Tool System — What the Agent Can Do](#6-tool-system--what-the-agent-can-do)
7. [Key Files](#7-key-files)
8. [Environment Variables](#8-environment-variables)
9. [Phase History & Fixes Applied](#9-phase-history--fixes-applied)
10. [Phase 7 Upgrade Spec (Reference)](#10-phase-7-upgrade-spec-reference)

---

## 1. What Jarvis Does in a Meeting

Jarvis is a wake-word-activated AI assistant that joins meetings as a bot and listens to all participants. When someone says **"Hey Jarvis, &lt;question&gt;"**, the agent:

1. Detects the wake word and query in the transcript
2. Calls the appropriate tool (Confluence search, meeting summary, general Q&A, etc.)
3. Synthesises a spoken response and plays it back into the meeting

No wake word → total silence. Bare "Hey Jarvis" (no query) → brief "Yes?" acknowledgement. All processing is in real time; there is no buffering or post-meeting batch step in the voice path.

---

## 2. Full End-to-End Pipeline

```
Participants speak in the meeting
  │
  ▼
Recall.ai headless Chrome joins the meeting as a bot
  │  bot.html publisher IIFE runs:
  │    navigator.mediaDevices.getUserMedia({ audio: true, ... })
  │    ↓ captures the browser's mixed meeting audio
  │  LiveKit JS SDK pubRoom.localParticipant.publishTrack(audioTrack)
  │    identity = "recall-browser-{session_id}"   ← must match pub_token exactly
  │
  ▼  WebRTC Opus encode → LiveKit SFU → relay to agent process
  │  ~20–50 ms encode+network
  ▼
AgentSession (agent_worker.py, livekit-agents 1.5.9)
  │  Subscribes ONLY to the "recall-browser-{session_id}" participant track
  │
  ├─▶ Silero VAD (pre-warmed at worker start via prewarm())
  │     Runs on every decoded audio frame
  │     Fires on_voice_start / on_voice_end events to gate STT
  │     ~10–30 ms end-of-speech detection latency
  │
  ├─▶ AssemblyAI Universal-3 Pro Streaming STT  (Phase 7)
  │     Plugin-direct: livekit-plugins-assemblyai 1.5.9
  │     model="u3-rt-pro", keyterms_prompt=["Jarvis","Hey Jarvis"]
  │     language_detection=False  → removes multilingual overhead
  │     Streams partial + final transcripts in real time
  │     ~150 ms P50 to final transcript after VAD end-of-speech
  │
  ▼
on_user_turn_completed(turn_ctx, new_message)   ← JarvisAgent method
  │  Pure regex wake-word gate — no I/O, no await
  │  ~1 ms
  │
  ├── No wake word     → new_message.content = [] → llm_node returns None → silence
  ├── Bare wake        → new_message.content = [] → _play_ack_frames() → silence
  └── Wake + query     → new_message.content = [query_only] → llm_node dispatches
  │
  ▼
llm_node(chat_ctx, tools, model_settings)
  │  If content is empty: returns None (agent.default.llm_node is skipped entirely)
  │  If content is present: delegates to Agent.default.llm_node
  │    → inference.LLM("openai/gpt-5.4-nano")  via LiveKit Inference
  │    → streams tokens; tools are dispatched as needed (Confluence, meeting, general)
  │    TTFT: ~150–250 ms
  │
  ▼
tts_node(text_stream, model_settings)
  │  Wraps Agent.default.tts_node
  │  inference.TTS("cartesia/sonic-turbo", voice=JARVIS_LK_TTS_VOICE)
  │  Text transforms: filter_emoji, filter_markdown (removes stars, dashes, etc.)
  │  First audio frame: ~40 ms after first LLM token
  │
  ▼  AgentSession publishes TTS audio as the agent's own LiveKit participant track
  │
  ▼
bot.html subscriber IIFE  (same Recall headless Chrome, different LiveKit room token)
  │  TrackSubscribed fires when the agent's track arrives
  │  Filters out "recall-browser-*" tracks (prevents echo)
  │  Primary path: track.attach() → <audio> element → autoplay=true
  │  ~20–50 ms WebRTC decode + playback buffer
  │
  ▼
Recall captures the browser's audio output → injects into the meeting
  │
  ▼
All meeting participants hear Jarvis respond
```

---

## 3. How LiveKit Powers the Voice Pipeline

LiveKit is the real-time media infrastructure that makes the whole pipeline work. Here is what it does and why it is the right choice.

### 3.1 What LiveKit Is

[LiveKit](https://livekit.io) is an open-source WebRTC SFU (Selective Forwarding Unit) plus a cloud service that provides:

- A **media relay** — routes audio/video between participants without mixing or decoding at the server
- An **Agents SDK** (`livekit-agents`) — a Python framework for building AI voice agents that connect to LiveKit rooms
- A **plugin ecosystem** — drop-in STT, LLM, and TTS plugins (`livekit-plugins-*`) with a unified interface

### 3.2 LiveKit Rooms and Participants

Every Jarvis meeting session uses **two LiveKit rooms** (one token pair):

| Token | Room | Identity | Purpose |
|-------|------|----------|---------|
| `pub_token` | `{session_id}` | `recall-browser-{session_id}` | Publisher: Recall Chrome sends meeting audio INTO LiveKit |
| `token` (subscriber) | `{session_id}` | `recall-listener-{session_id}` | Subscriber: second bot identity for teardown parity |
| Agent track | `{session_id}` | auto-assigned by AgentSession | Publisher: Jarvis TTS audio goes OUT via LiveKit |

The Recall headless Chrome loads `bot.html` (served from `GET /bot-page`). The publisher IIFE inside bot.html uses the **LiveKit JavaScript SDK** to:
1. Capture the browser's mixed meeting audio via `getUserMedia`
2. Join the LiveKit room as `recall-browser-{session_id}`
3. Publish the audio track

The AgentSession in `agent_worker.py` subscribes specifically to that participant's track — and only that participant's track.

### 3.3 AgentSession — The Core Abstraction

`AgentSession` (from `livekit.agents`) is the orchestrator that wires STT → LLM → TTS into a single pipeline:

```python
session = AgentSession(
    stt=assemblyai.STT(...),        # transcribes incoming audio
    llm=inference.LLM("openai/gpt-5.4-nano"),   # generates responses
    tts=inference.TTS("cartesia/sonic-turbo"),   # synthesises speech
    vad=silero.VAD.load(),          # detects voice activity (speech/silence)
    turn_handling=TurnHandlingOptions(...),      # endpointing + interruption
    preemptive_generation=False,    # disabled — wake-word rewrite always fires
    tts_text_transforms=["filter_emoji", "filter_markdown"],
)
```

AgentSession manages the full lifecycle:
- Subscribing to the target participant's audio track
- Routing audio frames through VAD → STT
- Calling `on_user_turn_completed` when a transcript is ready
- Calling `llm_node` to generate a response stream
- Calling `tts_node` to convert text to audio frames
- Publishing the audio frames as the agent's own participant track

This replaces what would otherwise require hundreds of lines of WebRTC, audio format conversion, and pipeline management code.

### 3.4 LiveKit Inference — Cloud Plugin Gateway

`inference.LLM` and `inference.TTS` route through **LiveKit's Inference service** — a cloud gateway that proxies to OpenAI, Cartesia, and other providers. This means:

- API keys are managed centrally (LiveKit handles them, not the agent process)
- The agent code uses `"openai/gpt-5.4-nano"` / `"cartesia/sonic-turbo"` as opaque model identifiers
- Latency is comparable to calling OpenAI/Cartesia directly (the gateway is low-overhead)

For STT, AssemblyAI is used **plugin-direct** (`livekit-plugins-assemblyai`) rather than through LiveKit Inference, because the Inference gateway does not expose AssemblyAI's `keyterms_prompt` parameter, which is essential for reliable wake-word recognition in noisy meeting audio.

### 3.5 Silero VAD — Pre-warmed Voice Activity Detection

The Silero Voice Activity Detector runs on every decoded audio frame and:
- **Starts** a turn when it detects speech energy above the threshold
- **Ends** a turn when it detects a silence gap ≥ `endpointing.min_delay` (0.15 s after Phase 7)

Without VAD, the STT would send audio continuously, wasting credits and producing partial transcripts from background noise. VAD is pre-warmed at worker startup (`prewarm()`) so the first turn has no warm-up latency.

**Why Silero stays even with AssemblyAI STT:** AssemblyAI has its own built-in turn detection, but it cannot replace Silero for **barge-in detection** — the ability to interrupt Jarvis while it is speaking. Silero's VAD events are what the AgentSession uses to decide "the user is interrupting me" vs "this is background noise".

### 3.6 The PCM Ack Path (Phase 7 D-05)

For bare wake-word responses ("Yes?"), the agent bypasses the full TTS pipeline:

```
"Hey Jarvis" detected
  ↓
_play_ack_frames(session, "yes")
  ↓
get_wake_ack_audio()  →  (key, mp3_bytes) from in-memory audio_cache
  ↓
_mp3_bytes_to_frames(mp3_bytes)
  →  AudioStreamDecoder(sample_rate=48000, num_channels=1, format="mp3")
  →  async generator of rtc.AudioFrame objects
  ↓
session.say(audio=_mp3_bytes_to_frames(mp3_bytes), add_to_chat_ctx=False)
  →  AgentSession feeds frames directly into the agent's publish track
  →  <5 ms perceived latency (vs ~150 ms Cartesia round-trip)
```

The `yes.mp3` file is loaded into memory at `prewarm()` time via `load_audio_cache()`. If the cache is empty (file missing), the code falls back to live TTS with a logged warning.

---

## 4. Audio Pipeline Deep Dive

### 4.1 Inbound Audio Path (Meeting → Jarvis)

```
Meeting participant's microphone
  → meeting platform (Teams, Zoom, Meet) audio mixing
  → Recall.ai bot's Chrome browser audio output (mixed)
  → bot.html: navigator.mediaDevices.getUserMedia({ audio: true, echoCancellation: false })
  → LiveKit JS SDK: room.localParticipant.publishTrack(audioTrack)
  → Opus encode → WebRTC → LiveKit SFU
  → AgentSession: subscribes to "recall-browser-{session_id}" participant
  → Opus decode → PCM audio frames (48 kHz, mono)
  → Silero VAD: frame-level energy check
  → [if voice detected] AssemblyAI STT: streams PCM to AssemblyAI cloud
  → AssemblyAI returns final transcript
  → on_user_turn_completed fires with transcript text
```

**Note on echoCancellation=false:** The getUserMedia call deliberately disables echo cancellation and noise suppression. Recall's Chrome is already isolated (headless, no speakers); applying browser echo cancellation would degrade the audio quality of the mixed meeting audio.

### 4.2 Outbound Audio Path (Jarvis → Meeting)

```
llm_node returns text stream
  → tts_node calls Agent.default.tts_node
  → inference.TTS("cartesia/sonic-turbo") streams audio frames
  → AgentSession publishes frames as the agent participant's audio track in LiveKit
  → WebRTC Opus encode → LiveKit SFU
  → bot.html subscriber IIFE: TrackSubscribed event fires
  → track.attach() → <audio> element (primary, always works in headless Chrome)
  → Recall captures Chrome's audio output (the <audio> element playback)
  → Recall injects captured audio into the meeting
  → All participants hear Jarvis
```

**Primary path is HTMLAudioElement, not WebAudio:** Earlier versions used the WebAudio API (`AudioContext → createMediaStreamSource`). This caused a race condition — `AudioContext.resume()` is async, but the code checked `ctx.state === 'running'` synchronously, so the WebAudio path was never taken. `track.attach()` (which creates an `<audio>` element) is synchronous and works reliably in Recall's headless Chrome. The WebAudio path was demoted to a secondary/optional enhancement.

### 4.3 Wake Word Gate

The wake-word gate in `on_user_turn_completed` is a pure regex match — no I/O, no network, no LLM call:

```python
_WAKE_PATTERN = re.compile(
    r"(?:hey|yo|ok|hi|okay)[,\s]+(?:jarvis|jarvas|jervis|jarvus|jarves|jarvi|jarv)[,.\s!?]*\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)
```

Three cases:
- **No match** → `new_message.content = []` → `llm_node` sees empty content → returns `None` → agent is silent
- **Match with empty capture** → bare wake word → play pre-cached PCM ack → enter listening mode (next utterance is treated as the query, even without a wake word)
- **Match with non-empty capture** → query present → `new_message.content = [query]` → LLM dispatched with query only (wake word stripped)

The listening mode has an 8-second timeout: if no query arrives within 8 s of "Hey Jarvis", listening mode resets silently.

---

## 5. Latency Budget (Phase 7 State)

| Stage | Component | P50 | P95 | Notes |
|-------|-----------|-----|-----|-------|
| WebRTC capture + encode | getUserMedia → LiveKit SFU | 20–50 ms | 80 ms | Opus at default bitrate |
| VAD end-of-speech | Silero (pre-warmed) | 10–30 ms | 80 ms | |
| STT final transcript | AssemblyAI U3-RT-Pro | ~150 ms | 350 ms | `language_detection=False` removes multilingual overhead |
| Wake word gate | Regex | <1 ms | <1 ms | Pure string match |
| Turn endpointing | `min_delay=0.15 s` | 150 ms | 1500 ms | Waits for VAD silence gap |
| False interruption timeout | `0.6 s` (Phase 7) | 0 ms | 600 ms | Only fires on brief interjections |
| LLM TTFT | gpt-5.4-nano (LiveKit Inference) | 150–250 ms | 500 ms | Tool calls add 300–800 ms |
| TTS first audio frame | Cartesia Sonic-Turbo | ~40 ms | 120 ms | Streams in parallel with LLM tokens |
| WebRTC decode + playback | bot.html → Recall Chrome | 20–50 ms | 80 ms | |
| **Bare wake "Yes?" ack** | Pre-cached PCM | **<5 ms** | **<5 ms** | Phase 7 D-05 |
| **Total (no tool call)** | | **~200–400 ms** | **~1000 ms** | |
| **Total (with tool call)** | | **~500–1200 ms** | **~2000 ms** | |

P50 improved from **~400–800 ms** (pre-Phase 7) to **~200–400 ms** (Phase 7).

---

## 6. Tool System — What the Agent Can Do

The agent has access to 18 tools defined in `agent_bridge.py` and passed to `Agent(tools=JARVIS_TOOLS)`.

### General / Utility

| Tool | Description |
|------|-------------|
| `get_current_datetime` | Returns current date and time. Jarvis always calls this for time questions instead of guessing. |

### Meeting Intelligence

| Tool | Description |
|------|-------------|
| `summarize_meeting_tool` | Summarise what has been said so far. Accepts `detail_level="brief"` or `"full"`. |
| `generate_opinion_tool` | Give Jarvis's opinion or recommendation on the current discussion topic. |
| `extract_action_items_tool` | Extract commitments, action items, and next steps from the transcript. |
| `summarize_speaker_tool` | Summarise what a specific participant has said. |
| `answer_general_question_tool` | Answer any factual or general question; optionally forces a Tavily web search for live data. |

All meeting tools read the live `transcript_log` for the session via `get_transcript_log_for_session()`.

### Confluence — Basic

| Tool | Description |
|------|-------------|
| `search_confluence_pages` | Simple keyword search via the Confluence REST API. |
| `list_confluence_pages` | List recent pages in the workspace. |
| `fetch_confluence_page` | Fetch a page's HTML and headings; optionally isolate a section. |
| `create_confluence_page` | Create a new page with plain-text body. |
| `edit_confluence_section` | Edit or append to a named section in a page. |
| `delete_confluence_section` | Delete content within a named section. |
| `delete_confluence_page` | Permanently delete an entire page. |

### Confluence — Advanced (Graph RAG + Pinecone + Version Retry)

| Tool | Description |
|------|-------------|
| `search_workspace_knowledge_tool` | Parallel search: Pinecone vector search + live Confluence REST, merged and ranked by semantic similarity. More powerful than `search_confluence_pages` for fuzzy/semantic lookups. |
| `fetch_live_page_tool` | Fetch a page including its current **version number**. Required before any commit operation — use the returned version as `expected_version`. |
| `commit_document_edit_tool` | Edit a page section with automatic version-conflict retry (up to 3 attempts, 0.5×2^n backoff). Supports `append=True` to add content without overwriting. |
| `commit_delete_tool` | Delete a page section with automatic version-conflict retry. |
| `update_page_title_tool` | Rename a Confluence page with automatic version-conflict retry. |

**Typical commit workflow:**
```
1. search_workspace_knowledge_tool("sprint planning") → find page_id
2. fetch_live_page_tool(page_id, heading="Goals") → get version + section content
3. commit_document_edit_tool(page_id, version, "Goals", new_content) → apply
```

---

## 7. Key Files

| File | Purpose |
|------|---------|
| `confluence_logic/agent_worker.py` | AgentSession setup, wake-word gate, LLM/TTS nodes, LiveKit worker entry point |
| `confluence_logic/agent_bridge.py` | All 18 `@function_tool` definitions; `JARVIS_TOOLS` export |
| `confluence_logic/audio_cache.py` | In-memory pre-generated TTS MP3 cache (`yes.mp3`, filler clips) |
| `confluence_logic/jarvis_agentic.py` | FastAPI app; bot lifecycle, pub/sub token minting, LiveKit room management |
| `confluence_logic/review/api.py` | Bot start/stop/status endpoints; Supabase meeting persistence |
| `static/bot.html` | Served to Recall Chrome via `/bot-page`; publisher + subscriber IIFEs |
| `confluence_logic/connectors/confluence.py` | Confluence REST API client |
| `confluence_logic/db/vector_store.py` | Pinecone client |
| `confluence_logic/meeting_responder.py` | LLM-based meeting summarisation, opinions, action items |
| `confluence_logic/general_responder.py` | General Q&A with optional Tavily web search |
| `Confluence/requirements.txt` | All Python dependencies (livekit-agents + plugins pinned at 1.5.9) |

---

## 8. Environment Variables

| Variable | Default | Used By | Notes |
|----------|---------|---------|-------|
| `LIVEKIT_URL` | — | `agent_worker.py`, `jarvis_agentic.py` | `wss://...livekit.cloud` format |
| `LIVEKIT_API_KEY` | — | token minting in `jarvis_agentic.py` | |
| `LIVEKIT_API_SECRET` | — | token minting in `jarvis_agentic.py` | |
| `ASSEMBLYAI_API_KEY` | — | `livekit-plugins-assemblyai` | Separate from legacy `ASSEMBLY_API` (Recall BYOB) |
| `OPENAI_API_KEY` | — | LLM + TTS via LiveKit Inference | |
| `RECALL_API_KEY` | — | `jarvis_agentic.py` | Recall.ai bot management |
| `JARVIS_LK_LLM` | `openai/gpt-5.4-nano` | `agent_worker.py` | Override the LLM model string |
| `JARVIS_LK_TTS_PROVIDER` | `cartesia` | `agent_worker.py` | TTS provider prefix |
| `JARVIS_LK_TTS_VOICE` | `9626c31c-bec5-4cca-baa8-f8ba9e84c8bc` | `agent_worker.py` | Cartesia voice UUID |
| `JARVIS_AGENT_WORKER_NAME` | `jarvis-agent` | `agent_worker.py` | Worker name for LiveKit dispatch |
| `JARVIS_WAKE_ALIASES` | `""` | `agent_worker.py` | Pipe-separated extra wake word aliases |

---

## 9. Phase History & Fixes Applied

### Fix A (2026-05-19) — `jarvis_agentic.py` NameError on bot start

`_create_livekit_room()` returned `room, source` but `source` was removed in Phase 6 when the manually-created AudioSource was replaced by AgentSession's built-in TTS publish. Every `POST /bot/start` raised `NameError: name 'source' is not defined`. Fixed by removing `return room, source`.

### Fix B (2026-05-19) — `bot.html` AudioContext race condition

The subscriber IIFE used `AudioContext.resume()` fire-and-forget then synchronously checked `ctx.state === 'running'`. Since `resume()` is async, the state was always `'suspended'` at the check, so the WebAudio path was never taken. Headless Chrome audio playback was non-deterministic. Fixed by making `track.attach() → <audio>` the unconditional primary path.

### Phase 6 (2026-05-18) — Browser publisher architecture

Replaced the Phase 4 relay WebSocket + AudioResampler Python path with a browser-native publisher: Recall's Chrome loads `bot.html`, runs `getUserMedia`, and publishes the mixed audio track directly to LiveKit. Eliminated relay server complexity, reduced latency, removed `recall-relay-{id}` participant identity.

### Phase 7 (2026-05-20) — Voice pipeline v2 (all five optimisations live)

| Change | Before | After |
|--------|--------|-------|
| STT | Deepgram Nova-3 multi (LiveKit Inference) | AssemblyAI U3-RT-Pro plugin-direct |
| LLM | openai/gpt-4.1-mini | openai/gpt-5.4-nano |
| TTS | Cartesia Sonic-3 | Cartesia Sonic-Turbo |
| Endpointing min_delay | 0.3 s | 0.15 s |
| False interruption timeout | 1.2 s | 0.6 s |
| Bare wake ack | session.say("Yes?") via Cartesia ~150 ms | Pre-cached PCM frames <5 ms |

---

## 10. Phase 7 Upgrade Spec (Reference)

### D-01: AssemblyAI Universal-3 Pro Streaming

Plugin-direct (NOT via LiveKit Inference) so `keyterms_prompt` is accessible:

```python
from livekit.plugins import assemblyai
stt=assemblyai.STT(
    model="u3-rt-pro",
    keyterms_prompt=["Jarvis", "Hey Jarvis"],
    language_detection=False,
)
```

Requirement: `ASSEMBLYAI_API_KEY` in `.env`. Dependency: `livekit-plugins-assemblyai==1.5.9` in `requirements.txt`.

### D-02: LLM — gpt-5.4-nano

```python
JARVIS_LK_LLM = os.getenv("JARVIS_LK_LLM", "openai/gpt-5.4-nano")
llm=inference.LLM(JARVIS_LK_LLM)
```

Override via `JARVIS_LK_LLM` env var if a different model is needed.

### D-03: TTS — Cartesia Sonic-Turbo

```python
tts=inference.TTS(f"{JARVIS_LK_TTS_PROVIDER}/sonic-turbo", voice=JARVIS_LK_TTS_VOICE)
```

Voice UUID unchanged — Sonic-Turbo uses the same voice IDs as Sonic-3. ~40 ms TTFA vs ~90 ms.

### D-04: Endpointing tightening

```python
turn_handling=TurnHandlingOptions(
    endpointing={"min_delay": 0.15, "max_delay": 1.5},
    interruption={"resume_false_interruption": True, "false_interruption_timeout": 0.6},
)
```

Safe with wake-word gate: premature cuts on non-Jarvis speech are discarded by the regex.

### D-05: Pre-cached PCM ack

```python
# In on_user_turn_completed bare-wake case:
await _play_ack_frames(self.session, "yes")

# _play_ack_frames implementation:
async def _play_ack_frames(session, key="yes"):
    cached = get_wake_ack_audio()          # returns (key, mp3_bytes) or None
    if cached is None:
        await session.say("Yes?", add_to_chat_ctx=False)  # fallback
        return
    _, mp3_bytes = cached
    await session.say("", audio=_mp3_bytes_to_frames(mp3_bytes), add_to_chat_ctx=False)
```

`audio_cache.py` pre-generates the MP3 at startup. `AudioStreamDecoder` decodes MP3 to 48 kHz mono PCM frames on the fly. The `session.say(audio=...)` API feeds frames directly into the agent's publish track, bypassing Cartesia entirely.
