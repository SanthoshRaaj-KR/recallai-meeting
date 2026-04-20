"""
End-to-end pipeline evaluation for the Jarvis meeting assistant.

Injects a realistic synthetic meeting transcript and runs a battery of
questions through the full pipeline (classifier → handler → LLM → output).
Captures spoken text instead of calling TTS. Reports timing, filler words,
answers, and a scored quality summary.

Usage:
    python -m tests.e2e_pipeline_eval
"""
import asyncio
import sys
import time
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import AsyncMock, patch

# Make sure the package root is on the path when run directly.
sys.path.insert(0, ".")

from confluence_logic import jarvis_agentic as ja
from confluence_logic.classifier import classify_intent


# ---------------------------------------------------------------------------
# Synthetic meeting transcript — realistic STT-style speech including
# informal banter, Indian-company specific names, tech jargon, and
# deliberate STT artifacts to stress the normalisation pipeline.
# ---------------------------------------------------------------------------

TRANSCRIPT = [
    {"participant": "Rahul Sharma", "text": "Okay guys, I think everyone's joined now. Should we get started?", "timestamp": 0},
    {"participant": "Priya Nair", "text": "Yeah I'm here, sorry I was on mute for a sec.", "timestamp": 5},
    {"participant": "Tanvi Mehta", "text": "Same here, good to go.", "timestamp": 8},
    {"participant": "Rahul Sharma", "text": "Alright. So the main thing today is our Q2 roadmap review and the payment integration status. Kiran, do you want to kick off with the razor pay integration?", "timestamp": 12},
    {"participant": "Kiran Patel", "text": "Sure. So we finished the razor pay checkout flow last sprint. The a p i is working fine in staging, we've done about 200 test transactions and it looks solid. The one issue is the webhook handling — we're getting some duplicate events because the pipe line doesn't deduplicate properly.", "timestamp": 20},
    {"participant": "Rahul Sharma", "text": "Okay so the webhook deduplication is the main blocker?", "timestamp": 45},
    {"participant": "Kiran Patel", "text": "Right. We need to store the event ID in the database and check before processing. It's maybe two days of work.", "timestamp": 50},
    {"participant": "Priya Nair", "text": "Actually I already have a ticket for that in Jira. Kiran I'll assign it to you today.", "timestamp": 58},
    {"participant": "Kiran Patel", "text": "Perfect, thanks Priya.", "timestamp": 62},
    {"participant": "Tanvi Mehta", "text": "What about the UPI flow? Last I heard Anjali was working on it.", "timestamp": 65},
    {"participant": "Rahul Sharma", "text": "Yeah Anjali you want to give an update?", "timestamp": 70},
    {"participant": "Anjali Singh", "text": "So the UPI intent flow is done but we're blocked on the merchant ID. Finance needs to approve the live credentials before we can test in production. I've been following up with Vikram but he's been OOO.", "timestamp": 74},
    {"participant": "Priya Nair", "text": "That's a problem. Our Q2 deadline is end of June. If Vikram is out that pushes us back at least a week.", "timestamp": 95},
    {"participant": "Rahul Sharma", "text": "Alright we need to escalate this. I'll ping Meera in finance directly. Anjali can you send me a note with exactly what we need so I can include it?", "timestamp": 105},
    {"participant": "Anjali Singh", "text": "Yes of course, I'll send it after this call.", "timestamp": 115},
    {"participant": "Tanvi Mehta", "text": "While we're on payments — Kiran have you looked at the performance side? Like the calf ka queue for async payment events?", "timestamp": 120},
    {"participant": "Kiran Patel", "text": "Yeah we're using calf ka for the event streaming. Consumer lag was high last week but we scaled the consumer group and it's back to normal. P99 latency is under 200 milliseconds.", "timestamp": 128},
    {"participant": "Rahul Sharma", "text": "Good. What about the recommendation engine? Tanvi that's yours right?", "timestamp": 145},
    {"participant": "Tanvi Mehta", "text": "Yes. So we ran the b m twenty five experiments last week against the vector search approach. B M 25 is faster — about 40 milliseconds versus 120 — but recall quality is worse, especially for long-tail queries. My recommendation is to use hybrid: b m twenty five for the first pass, then re-rank with embeddings.", "timestamp": 150},
    {"participant": "Rahul Sharma", "text": "And cost-wise?", "timestamp": 180},
    {"participant": "Tanvi Mehta", "text": "The re-ranking adds some overhead but it's negligible at our scale. Maybe 10 to 15 rupees per million queries extra.", "timestamp": 184},
    {"participant": "Priya Nair", "text": "That seems very reasonable. I'd go with the hybrid approach.", "timestamp": 195},
    {"participant": "Tanvi Mehta", "text": "Agreed. I'll write up the implementation plan this week.", "timestamp": 200},
    {"participant": "Rahul Sharma", "text": "One more thing before we close — the infrastructure move to Kubernetes. Miheer you were supposed to have a status update right?", "timestamp": 205},
    {"participant": "Mihir Joshi", "text": "Yeah so, we've containerized about 60 percent of services. The main bottleneck is the legacy authentication service — it has hard-coded paths and some nasty startup logic. I think we need another three weeks to migrate it cleanly.", "timestamp": 212},
    {"participant": "Priya Nair", "text": "Three weeks puts us into July. Can we parallelize at all?", "timestamp": 235},
    {"participant": "Mihir Joshi", "text": "I can bring in Ravi to help but he's currently on the cat a gory management feature. Either we delay that or we split his time.", "timestamp": 240},
    {"participant": "Rahul Sharma", "text": "Let's keep Ravi on category management, that's higher priority. Miheer three weeks is fine, just make sure staging is stable before we cut prod.", "timestamp": 252},
    {"participant": "Mihir Joshi", "text": "Will do.", "timestamp": 262},
    {"participant": "Priya Nair", "text": "Okay should we do a quick decision log? We decided: one, Kiran to fix webhook deduplication, two, Rahul to escalate UPI credentials with finance, three, Tanvi to write hybrid search implementation plan, four, Miheer to continue Kubernetes migration with three week timeline.", "timestamp": 265},
    {"participant": "Rahul Sharma", "text": "Great summary. Anything else?", "timestamp": 290},
    {"participant": "Anjali Singh", "text": "Nothing from me.", "timestamp": 295},
    {"participant": "Kiran Patel", "text": "All good.", "timestamp": 297},
    {"participant": "Tanvi Mehta", "text": "One thing — can we move next week's sync to Thursday? I have a conflict on Wednesday.", "timestamp": 300},
    {"participant": "Rahul Sharma", "text": "Thursday works for me. Let's do Thursday at 11.", "timestamp": 307},
    {"participant": "Priya Nair", "text": "Perfect. I'll update the invite.", "timestamp": 312},
    {"participant": "Rahul Sharma", "text": "Great, thanks everyone. Good meeting!", "timestamp": 315},
]


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    question: str
    expected_intent: str
    expectation: str  # what a correct answer should contain / address

@dataclass
class Result:
    question: str
    expected_intent: str
    actual_intent: str
    filler_text: str
    answer_text: str
    elapsed_ms: float
    expectation: str
    score: int = 0        # 0–3 filled in during scoring
    notes: str = ""


TEST_CASES: List[TestCase] = [
    # --- Meeting-context questions ---
    TestCase(
        question="Can you summarize the meeting so far?",
        expected_intent="meeting_summary",
        expectation="Should mention Razorpay, UPI, Kubernetes, recommendation engine / BM25",
    ),
    TestCase(
        question="What are the action items from this meeting?",
        expected_intent="action_items",
        expectation="Should list: Kiran webhook fix, Rahul escalate UPI, Tanvi hybrid search plan, Mihir K8s migration",
    ),
    TestCase(
        question="What did Anjali say?",
        expected_intent="speaker_query",
        expectation="Should summarize Anjali's UPI update and merchant ID blocker",
    ),
    TestCase(
        question="What did Tanvee say?",  # deliberate misspelling to test fuzzy matching
        expected_intent="speaker_query",
        expectation="Should still match Tanvi and summarize her BM25/hybrid recommendation",
    ),
    TestCase(
        question="What did Miheer con tribute to the meeting?",  # split verb STT artifact
        expected_intent="speaker_query",
        expectation="Should match Mihir and summarize Kubernetes migration update",
    ),
    TestCase(
        question="What do you think about the hybrid search approach?",
        expected_intent="meeting_opinion",
        expectation="Should give opinion grounded in BM25 vs vector tradeoff from meeting",
    ),
    TestCase(
        question="How should we handle the UPI credentials situation?",
        expected_intent="meeting_opinion",
        expectation="Should reference Vikram OOO, escalation to Meera, Anjali's note",
    ),
    # --- General knowledge questions (web search / LLM knowledge) ---
    TestCase(
        question="What is BM25 and how does it work?",
        expected_intent="general",
        expectation="Should explain BM25 as a ranking function / TF-IDF variant, no hallucination",
    ),
    TestCase(
        question="What is the Razorpay API rate limit?",
        expected_intent="general",
        expectation="Should attempt web search or clearly state it doesn't know exact limits, no wrong numbers",
    ),
    TestCase(
        question="What's the current IPL score?",
        expected_intent="general",
        expectation="Should trigger web search; should say it can't confirm live score if no API key, no made-up score",
    ),
    TestCase(
        question="What is Kafka used for?",
        expected_intent="general",
        expectation="Should explain event streaming / message queue correctly",
    ),
    TestCase(
        question="Give me a brief summary",
        expected_intent="meeting_summary",
        expectation="Brief summary — 3 to 5 points from the meeting",
    ),
]


# ---------------------------------------------------------------------------
# Harness helpers
# ---------------------------------------------------------------------------

def _reset_state():
    ja.meeting_state["bot_id"] = "test-bot"
    ja.meeting_state["transcript_log"] = list(TRANSCRIPT)
    ja.meeting_state["is_active"] = True
    ja.meeting_state["jarvis_listening"] = False
    ja.meeting_state["current_task"] = None
    ja.meeting_state["pending_requests"].clear()
    ja.meeting_state["pending_clarification"] = None
    ja.meeting_state["pending_general_clarification"] = None
    ja.meeting_state["pending_summary_clarification"] = None
    ja.meeting_state["current_task_id"] = None
    ja.meeting_state["current_phase"] = "idle"
    ja.meeting_state["current_request"] = None
    ja.meeting_state["output_generation"] = 0
    ja.meeting_state["mutation_started"] = False
    ja.meeting_state["cancel_requested"] = False
    ja.meeting_state["last_user_speech_at"] = 0.0
    ja.meeting_state["general_history"] = []
    ja.meeting_state["last_jarvis_response"] = None
    ja.meeting_state["invoker_participant"] = None
    ja.meeting_state["_pending_debounce_task"] = None
    ja.meeting_state["_accumulated_query"] = ""


async def _run_case(tc: TestCase) -> Result:
    """Run one test case, capturing all speech output and timing."""
    _reset_state()

    captured: List[str] = []
    filler_captured: List[str] = []
    is_first_speak = [True]  # track filler vs answer

    async def mock_speak_guarded(text: str, bot_id: str, generation: int, allow_stale: bool = False) -> bool:
        if text:
            if is_first_speak[0]:
                filler_captured.append(text)
                is_first_speak[0] = False
            else:
                captured.append(text)
        return True

    async def mock_speak_filler(bot_id: str, generation: int) -> None:
        pass

    async def mock_speak_cached_guarded(audio_bytes: bytes, bot_id: str, generation: int) -> bool:
        return True

    async def mock_sleep(secs: float) -> None:
        pass  # skip post-speech pause so tests run fast

    # Determine which handler to call based on intent
    actual_intent = await classify_intent(tc.question)

    start = time.perf_counter()

    with (
        patch.object(ja, "_speak_guarded", side_effect=mock_speak_guarded),
        patch.object(ja, "_speak_filler", side_effect=mock_speak_filler),
        patch.object(ja, "_speak_cached_guarded", side_effect=mock_speak_cached_guarded),
        patch("asyncio.sleep", side_effect=mock_sleep),
    ):
        try:
            if actual_intent == "meeting_summary":
                await ja._handle_meeting_summary(tc.question, "test-bot")
            elif actual_intent == "meeting_opinion":
                await ja._handle_meeting_opinion(tc.question, "test-bot")
            elif actual_intent == "action_items":
                await ja._handle_action_items(tc.question, "test-bot")
            elif actual_intent == "speaker_query":
                await ja._handle_speaker_query(tc.question, "test-bot")
            elif actual_intent == "general":
                await ja._handle_general_question(tc.question, "test-bot")
            else:
                captured.append(f"[Unhandled intent: {actual_intent}]")
        except Exception as exc:
            captured.append(f"[Handler error: {exc}]")

    elapsed_ms = (time.perf_counter() - start) * 1000

    return Result(
        question=tc.question,
        expected_intent=tc.expected_intent,
        actual_intent=actual_intent,
        filler_text=" | ".join(filler_captured) if filler_captured else "(none)",
        answer_text=" | ".join(captured) if captured else "(no output captured)",
        elapsed_ms=elapsed_ms,
        expectation=tc.expectation,
    )


# ---------------------------------------------------------------------------
# Scoring (deterministic heuristic — not LLM scored for speed)
# ---------------------------------------------------------------------------

def _score_result(r: Result) -> tuple[int, str]:
    """
    Score 0–3:
      3 = Intent correct + answer clearly addresses expectation keywords
      2 = Intent correct + answer present but surface-level or missing context
      1 = Intent wrong OR answer very short / unhelpful
      0 = No answer or hard failure
    """
    notes = []

    if not r.answer_text or r.answer_text.startswith("(no output"):
        return 0, "No answer produced"

    if "[Handler error" in r.answer_text or "[Unhandled intent" in r.answer_text:
        return 0, r.answer_text

    intent_ok = r.actual_intent == r.expected_intent
    if not intent_ok:
        notes.append(f"Intent mismatch: expected '{r.expected_intent}', got '{r.actual_intent}'")

    answer_lower = r.answer_text.lower()

    # Check for fallback/failure phrases
    failure_phrases = [
        "sorry, i couldn't", "i had trouble", "i wasn't able",
        "couldn't generate", "i don't see anyone", "hasn't said anything",
    ]
    if any(p in answer_lower for p in failure_phrases):
        notes.append("Answer appears to be an error fallback")
        return 1 if intent_ok else 0, "; ".join(notes)

    # Keyword coverage from expectation
    # Extract key terms (capitalised words, quoted phrases) from the expectation
    expectation_keywords = [
        w.lower() for w in r.expectation.split()
        if len(w) > 4 and w[0].isupper() and w.lower() not in ("should", "their", "these", "which", "still", "match")
    ]
    matches = sum(1 for kw in expectation_keywords if kw in answer_lower)
    coverage = matches / max(len(expectation_keywords), 1)

    if intent_ok and coverage >= 0.4:
        score = 3
        if coverage < 0.6:
            notes.append(f"Partial keyword coverage ({coverage:.0%})")
            score = 2
    elif intent_ok:
        score = 2
        notes.append(f"Low keyword coverage ({coverage:.0%})")
    else:
        score = 1
        notes.append(f"Keyword coverage: {coverage:.0%}")

    return score, "; ".join(notes) if notes else "OK"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _render_report(results: List[Result]) -> str:
    WIDTH = 100
    lines = []
    lines.append("=" * WIDTH)
    lines.append("  JARVIS PIPELINE END-TO-END EVALUATION REPORT")
    lines.append("=" * WIDTH)
    lines.append(f"  Transcript: {len(TRANSCRIPT)} entries, {len(set(e['participant'] for e in TRANSCRIPT))} speakers")
    lines.append(f"  Test cases: {len(results)}")
    lines.append("")

    total_score = 0
    max_score = len(results) * 3

    for i, r in enumerate(results, 1):
        intent_badge = "✓" if r.actual_intent == r.expected_intent else "✗"
        score_bar = "●" * r.score + "○" * (3 - r.score)
        lines.append(f"┌─ [{i:02d}] {r.question}")
        lines.append(f"│   Intent:   {intent_badge} expected={r.expected_intent}  got={r.actual_intent}")
        lines.append(f"│   Latency:  {r.elapsed_ms:.0f}ms")
        lines.append(f"│   Filler:   {textwrap.shorten(r.filler_text, 90)}")
        lines.append(f"│   Answer:   {textwrap.shorten(r.answer_text, 90)}")
        lines.append(f"│   Score:    [{score_bar}] {r.score}/3  — {r.notes}")
        lines.append(f"│   Expects:  {r.expectation}")
        lines.append("└" + "─" * (WIDTH - 1))
        lines.append("")
        total_score += r.score

    pct = total_score / max_score * 100
    lines.append("=" * WIDTH)
    lines.append(f"  TOTAL SCORE: {total_score}/{max_score}  ({pct:.0f}%)")

    if pct >= 80:
        grade = "PASS ✓"
    elif pct >= 60:
        grade = "MARGINAL ⚠"
    else:
        grade = "FAIL ✗"
    lines.append(f"  GRADE: {grade}")
    lines.append("=" * WIDTH)

    # --- Timing stats ---
    latencies = [r.elapsed_ms for r in results]
    lines.append("")
    lines.append("  LATENCY BREAKDOWN")
    lines.append(f"  {'Question':<55} {'Intent':<18} {'ms':>6}")
    lines.append("  " + "-" * 80)
    for r in results:
        lines.append(f"  {r.question[:54]:<55} {r.actual_intent:<18} {r.elapsed_ms:>6.0f}")
    lines.append(f"  {'AVERAGE':<55} {'':18} {sum(latencies)/len(latencies):>6.0f}")
    lines.append(f"  {'MAX':<55} {'':18} {max(latencies):>6.0f}")
    lines.append("")

    # --- Improvement suggestions ---
    lines.append("=" * WIDTH)
    lines.append("  IMPROVEMENT SUGGESTIONS")
    lines.append("=" * WIDTH)
    suggestions = _generate_suggestions(results)
    for s in suggestions:
        lines.append(f"  • {s}")
    lines.append("")

    return "\n".join(lines)


def _generate_suggestions(results: List[Result]) -> List[str]:
    suggestions = []

    intent_mismatches = [r for r in results if r.actual_intent != r.expected_intent]
    if intent_mismatches:
        suggestions.append(
            f"Intent classification errors on {len(intent_mismatches)} cases: "
            + ", ".join(f'"{r.question[:40]}"→{r.actual_intent}(expected {r.expected_intent})' for r in intent_mismatches)
        )

    speaker_failures = [
        r for r in results
        if r.expected_intent == "speaker_query" and (
            "don't see anyone" in r.answer_text.lower() or "hasn't said" in r.answer_text.lower() or r.score <= 1
        )
    ]
    if speaker_failures:
        suggestions.append(
            f"Speaker matching failed for {len(speaker_failures)} case(s) — "
            "fuzzy threshold may need tuning or STT normalization isn't applied to speaker names"
        )

    slow_cases = [r for r in results if r.elapsed_ms > 8000]
    if slow_cases:
        suggestions.append(
            f"{len(slow_cases)} case(s) exceeded 8s — consider streaming or parallel filler+answer generation"
        )

    filler_failures = [r for r in results if r.filler_text == "(none)" and r.actual_intent != "confluence"]
    if filler_failures:
        suggestions.append(
            f"{len(filler_failures)} case(s) produced no filler speech — gap filler generator may be failing silently"
        )

    low_scorers = [r for r in results if r.score <= 1]
    if low_scorers:
        suggestions.append(
            f"{len(low_scorers)} case(s) scored ≤1/3 — "
            "review LLM prompts for meeting context injection and transcript truncation"
        )

    avg_latency = sum(r.elapsed_ms for r in results) / len(results)
    if avg_latency > 5000:
        suggestions.append(
            f"Average latency {avg_latency:.0f}ms is high — "
            "consider caching entity extraction results and running filler generation in parallel with LLM call"
        )

    if not suggestions:
        suggestions.append("No major issues detected — pipeline is performing well on all dimensions.")

    return suggestions


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    print(f"\nRunning {len(TEST_CASES)} test cases against synthetic meeting transcript...\n")
    results = []
    for i, tc in enumerate(TEST_CASES, 1):
        print(f"  [{i:02d}/{len(TEST_CASES)}] {tc.question[:70]}", end="", flush=True)
        try:
            r = await _run_case(tc)
            r.score, r.notes = _score_result(r)
            results.append(r)
            print(f"  → {r.actual_intent}  {r.elapsed_ms:.0f}ms  [{r.score}/3]")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append(Result(
                question=tc.question,
                expected_intent=tc.expected_intent,
                actual_intent="error",
                filler_text="",
                answer_text=f"[Exception: {exc}]",
                elapsed_ms=0,
                expectation=tc.expectation,
                score=0,
                notes=str(exc),
            ))

    print()
    print(_render_report(results))


if __name__ == "__main__":
    asyncio.run(main())
