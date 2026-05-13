import asyncio
import argparse
import logging
import os
import time
from collections import deque
from typing import Iterable

from dotenv import load_dotenv

from .agents.editor_agent import EditorAgent
from . import jarvis_agentic as meeting


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_SIM_BOT_ID = "local-terminal-bot"
_DEFAULT_SPEAKER = "Local User"


def _reset_meeting_simulator_state() -> None:
    meeting.meeting_state["bot_id"] = _SIM_BOT_ID
    meeting.meeting_state["meeting_url"] = "local-terminal"
    meeting.meeting_state["transcript_log"] = []
    meeting.meeting_state["is_active"] = True
    meeting.meeting_state["session_status"] = "local_terminal"
    meeting.meeting_state["started_at"] = time.time()
    meeting.meeting_state["ended_at"] = None
    meeting.meeting_state["end_reason"] = None
    meeting.meeting_state["jarvis_listening"] = False
    meeting.meeting_state["jarvis_listening_at"] = 0.0
    meeting.meeting_state["current_task"] = None
    meeting.meeting_state["pending_requests"] = deque()
    meeting.meeting_state["pending_clarification"] = None
    meeting.meeting_state["pending_general_clarification"] = None
    meeting.meeting_state["pending_summary_clarification"] = None
    meeting.meeting_state["current_task_id"] = None
    meeting.meeting_state["current_phase"] = "idle"
    meeting.meeting_state["current_request"] = None
    meeting.meeting_state["output_generation"] = 0
    meeting.meeting_state["mutation_started"] = False
    meeting.meeting_state["cancel_requested"] = False
    meeting.meeting_state["last_user_speech_at"] = 0.0
    meeting.meeting_state["parallel_runners"] = []
    meeting.meeting_state["general_history"] = []
    meeting.meeting_state["last_jarvis_response"] = None
    meeting.meeting_state["gap_filler_generation"] = None
    meeting.meeting_state["wake_query_ack_pending"] = False
    meeting.meeting_state["invoker_participant"] = None
    meeting.meeting_state["_pending_debounce_task"] = None
    meeting.meeting_state["_accumulated_query"] = ""
    meeting.bind_bot_to_session(_SIM_BOT_ID, meeting.meeting_state.get("session_id") or "default")


def _print_transcript(entries: Iterable[dict]) -> None:
    entries = list(entries)
    if not entries:
        print("Transcript is empty.\n")
        return

    for idx, entry in enumerate(entries, start=1):
        participant = entry.get("participant", "Unknown")
        text = entry.get("text", "")
        print(f"{idx:03d}. {participant}: {text}")
    print()


def _install_terminal_speech_hooks() -> None:
    meeting.JARVIS_POST_SPEECH_PAUSE_SECONDS = 0.0
    meeting.JARVIS_SPEECH_HOLD_SECONDS = 0.0

    async def terminal_speak_guarded(
        text: str,
        bot_id: str,
        generation: int,
        allow_stale: bool = False,
        _preloaded_all=None,
    ) -> bool:
        if not allow_stale and generation != meeting.meeting_state["output_generation"]:
            return False
        text = (text or "").strip()
        if not text:
            return False
        print(f"Jarvis> {text}\n")
        meeting._record_jarvis_transcript(text)
        return True

    async def terminal_speak_cached_guarded(audio_bytes: bytes, bot_id: str, generation: int) -> bool:
        return True

    async def terminal_speak_gap_filler(query: str, bot_id: str, generation: int) -> None:
        async with meeting._get_state_lock():
            if meeting.meeting_state.get("gap_filler_generation") == generation:
                return
            meeting.meeting_state["gap_filler_generation"] = generation
            play_wake_ack = bool(meeting.meeting_state.get("wake_query_ack_pending"))
            meeting.meeting_state["wake_query_ack_pending"] = False

        if play_wake_ack:
            await terminal_speak_guarded(meeting.JARVIS_WAKE_ACK, bot_id, generation, allow_stale=True)

        print("Jarvis> Working on it...\n")

    async def terminal_speak_streaming(sentence_gen, gap_filler_task: asyncio.Task, bot_id: str, generation: int):
        full_sentences = []
        try:
            async for sentence in sentence_gen:
                if generation != meeting.meeting_state["output_generation"]:
                    return None
                sentence = (sentence or "").strip()
                if sentence:
                    full_sentences.append(sentence)
        finally:
            await gap_filler_task

        answer = " ".join(full_sentences).strip()
        if not answer:
            return None
        if generation != meeting.meeting_state["output_generation"]:
            return None
        print(f"Jarvis> {answer}\n")
        meeting._record_jarvis_transcript(answer)
        return answer

    meeting._speak_guarded = terminal_speak_guarded
    meeting._speak_cached_guarded = terminal_speak_cached_guarded
    meeting._speak_gap_filler = terminal_speak_gap_filler
    meeting._speak_streaming = terminal_speak_streaming


async def _wait_for_spawned_tasks(before: set[asyncio.Task], timeout: float = 120.0) -> None:
    current = asyncio.current_task()
    deadline = time.monotonic() + timeout

    while True:
        pending = {
            task
            for task in asyncio.all_tasks()
            if task is not current and task not in before and not task.done()
        }
        if not pending:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print("Jarvis> Still working in the background; returning to the prompt.\n")
            return
        done, _ = await asyncio.wait(pending, timeout=min(remaining, 0.5))
        for task in done:
            try:
                task.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                print(f"Jarvis> Background task error: {exc}\n")


async def _feed_terminal_transcript(participant: str, sentence: str) -> None:
    now = time.time()
    meeting.meeting_state["last_user_speech_at"] = now
    entry = meeting._append_transcript_log_entry(participant=participant, text=sentence, timestamp=now)
    if entry:
        logging.info("Transcript %s: %s", entry["participant"], entry["text"])

    invoker = meeting.meeting_state.get("invoker_participant")
    if invoker and participant != invoker:
        logging.debug("Ignoring command routing from %s; active invoker is %s", participant, invoker)
        return

    query = meeting.process_transcript_event(sentence, now)
    if query:
        meeting.meeting_state["invoker_participant"] = participant
        accumulated = meeting.meeting_state.get("_accumulated_query", "")
        accumulated = (accumulated + " " + query).strip() if accumulated else query
        meeting.meeting_state["_accumulated_query"] = accumulated

        from_wake_invocation = meeting.meeting_state.pop("_query_from_wake_invocation", False)
        from_listening = meeting.meeting_state.pop("_query_from_listening", False)
        if from_wake_invocation and not from_listening:
            meeting.meeting_state["wake_query_ack_pending"] = True

        before = asyncio.all_tasks()
        await meeting._debounced_dispatch(accumulated, _SIM_BOT_ID)
        await _wait_for_spawned_tasks(before)
    elif meeting.meeting_state["jarvis_listening"] and meeting.is_bare_wake_invocation(sentence):
        meeting.meeting_state["invoker_participant"] = participant
        before = asyncio.all_tasks()
        task = asyncio.create_task(meeting._handle_bare_wake(_SIM_BOT_ID))
        await task
        await _wait_for_spawned_tasks(before)


async def _run_editor_repl() -> None:
    load_dotenv()
    model = os.getenv("JARVIS_AGENT_MODEL", "gpt-5-mini")
    agent = EditorAgent(model=model)
    logging.info("Jarvis local REPL using model: %s", model)

    print("Jarvis local REPL")
    print("Type a request and press Enter. Type 'exit' to quit. Type '/reset' to clear memory.\n")

    while True:
        try:
            query = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Exiting.")
            return
        if query.lower() == "/reset":
            agent.clear_memory()
            print("Jarvis> Memory cleared.\n")
            continue

        try:
            answer = await agent.handle_query(query)
            print(f"Jarvis> {answer}\n")
        except Exception as exc:
            print(f"Jarvis> Error: {exc}\n")


async def _run_meeting_repl() -> None:
    load_dotenv()
    _reset_meeting_simulator_state()
    _install_terminal_speech_hooks()

    speaker = _DEFAULT_SPEAKER
    print("Jarvis local meeting simulator")
    print("Type meeting speech and press Enter. Only wake-word lines dispatch Jarvis.")
    print("Commands: /speaker NAME, /transcript, /state, /reset, /help, exit\n")

    while True:
        try:
            line = input(f"{speaker}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if not line:
            continue

        lowered = line.lower()
        if lowered in {"exit", "quit"}:
            print("Exiting.")
            return
        if lowered == "/help":
            print("Type normal meeting speech to add transcript.")
            print("Use the wake word to ask Jarvis, e.g. 'hey jarvis summarize the meeting'.")
            print("Use '/speaker Priya' to change the simulated speaker.\n")
            continue
        if lowered == "/reset":
            _reset_meeting_simulator_state()
            speaker = _DEFAULT_SPEAKER
            print("Jarvis> Meeting simulator reset.\n")
            continue
        if lowered == "/transcript":
            _print_transcript(meeting.meeting_state["transcript_log"])
            continue
        if lowered == "/state":
            print(f"Transcript entries: {len(meeting.meeting_state['transcript_log'])}")
            print(f"Listening: {meeting.meeting_state['jarvis_listening']}")
            print(f"Current phase: {meeting.meeting_state['current_phase']}")
            print(f"Current request: {meeting.meeting_state['current_request']}")
            print(f"Pending clarification: {meeting.meeting_state.get('pending_clarification')}")
            print(f"Pending summary clarification: {meeting.meeting_state.get('pending_summary_clarification')}")
            print(f"Last response: {meeting.meeting_state.get('last_jarvis_response')}\n")
            continue
        if lowered.startswith("/speaker "):
            next_speaker = line[len("/speaker "):].strip()
            if not next_speaker:
                print("Usage: /speaker NAME\n")
                continue
            speaker = next_speaker
            print(f"Speaker set to {speaker}.\n")
            continue

        try:
            await _feed_terminal_transcript(speaker, line)
        except Exception as exc:
            print(f"Jarvis> Error: {exc}\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Jarvis local terminal REPL")
    parser.add_argument(
        "--editor",
        action="store_true",
        help="Run the original editor-agent REPL instead of the Recall-free meeting simulator.",
    )
    args = parser.parse_args()

    if args.editor:
        await _run_editor_repl()
    else:
        await _run_meeting_repl()


if __name__ == "__main__":
    asyncio.run(main())
