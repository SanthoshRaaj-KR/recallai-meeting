"""
AI Interview System - Recall.ai + OpenAI Agents SDK
====================================================

This system combines:
- Recall.ai for video meeting integration (joins Zoom/Teams/Meet)
- OpenAI Agents SDK for intelligent decision-making
- Dynamic orchestration with specialist agents
- Real-time transcription via webhooks
- TTS audio output to meeting
"""

import json
import os
import sys
import time
import queue
import base64
import logging
import asyncio
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from threading import Thread
from pathlib import Path

import requests
import uvicorn
from dotenv import load_dotenv
from openai import OpenAI
from gtts import gTTS
from fastapi import FastAPI, Request
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from agents import Agent, Runner, RunContextWrapper, function_tool


# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()
client = OpenAI()

# Recall.ai Configuration
RECALL_API_KEY = os.getenv("RECALL_API_KEY")
RECALL_REGION = os.getenv("RECALL_REGION", "ap-northeast-1") 
RECALL_BASE_URL = f"https://{RECALL_REGION}.recall.ai/api/v1"
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # Your public webhook URL

# Interview Configuration
MAX_QUESTIONS = 2
MAX_WARNINGS = 3
MAX_PROBES_PER_QUESTION = 2
SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", "3.5"))
MAX_WAIT_FOR_ANSWER = 60

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app for webhooks
app = FastAPI()

# Global state (thread-safe queue for transcripts)
transcript_queue = queue.Queue()
interview_state = {
    "bot_id": None,
    "candidate_name": "",
    "topic": "",
    "bot_name": "Interview Bot",
    "is_active": False
}


# ============================================================================
# CONTEXT & STATE
# ============================================================================

@dataclass
class InterviewState:
    """Shared state for all agents"""
    name: str
    topic: str
    bot_id: str
    conversation_log: list = field(default_factory=list)
    asked_questions: list = field(default_factory=list)
    warnings: int = 0
    questions_count: int = 0
    current_question: Optional[str] = None
    current_answer: Optional[str] = None
    probe_count: int = 0
    probe_history: list = field(default_factory=list)


# ============================================================================
# PYDANTIC MODELS (Structured Outputs)
# ============================================================================

class QuestionOutput(BaseModel):
    """Generated question"""
    acknowledgement: str = Field(description="Brief acknowledgement or empty")
    question: str = Field(description="The interview question")


class EvaluationOutput(BaseModel):
    """Answer evaluation"""
    score: int = Field(ge=0, le=10)
    feedback: str = Field(description="Constructive feedback")


class ProbeOutput(BaseModel):
    """Probe follow-up"""
    probe: str = Field(description="Follow-up question")


class DuplicateOutput(BaseModel):
    """Duplicate check"""
    is_duplicate: bool
    reason: str


class AbuseOutput(BaseModel):
    """Abuse detection"""
    is_abusive: bool
    reason: str
    confidence: float = Field(ge=0, le=1)


class OrchestratorDecision(BaseModel):
    """Master orchestrator decision"""
    action: str = Field(
        description="ASK_QUESTION, PROBE_ANSWER, ACCEPT_ANSWER, HANDLE_WARNING, TERMINATE_INTERVIEW"
    )
    reason: str = Field(description="Reasoning for this action")
    confidence: float = Field(ge=0, le=1)


# ============================================================================
# RECALL.AI INTEGRATION
# ============================================================================

def create_recall_bot(meeting_url: str, bot_name: str = "Interview Bot") -> Optional[dict]:
    
    """
    Create a Recall.ai bot that joins the meeting with real-time transcription
    """
    if not RECALL_API_KEY:
        logger.error("RECALL_API_KEY not set")
        return None
    
    if not WEBHOOK_URL:
        logger.error("WEBHOOK_URL not set")
        return None

    # 🔥 THE FIX: Define the headers variable here!
    headers = {
        "Authorization": f"Token {RECALL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 🔥 NEW: Convert your HTTPS Webhook URL to a WSS WebSocket URL
    ws_url = WEBHOOK_URL.replace("https://", "wss://") + "/recall-audio-stream"
    
    payload = {
        "meeting_url": meeting_url,
        "bot_name": bot_name,
        "recording_config": {
            "transcript": {
                "provider": {
                    "recallai_streaming": {
                        "mode": "prioritize_low_latency",
                        "language_code": "en"
                    }
                }
            },
            "realtime_endpoints": [
                {
                    "type": "websocket", # 🔥 CHANGED from webhook
                    "url": ws_url,       # 🔥 Use the WSS URL
                    "events": [
                        "transcript.partial_data",     # 🔥 Use partial for speed
                        "participant_events.speech_on" # For barge-in detection
                    ]
                }
            ]
        }
    }
    
    try:
        response = requests.post(
            f"{RECALL_BASE_URL}/bot/",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 201:
            bot_data = response.json()
            logger.info(f"✅ Bot created: {bot_data['id']}")
            
            status_changes = bot_data.get("status_changes", [])
            if status_changes:
                logger.info(f"   Status: {status_changes[-1].get('code', 'unknown')}")
            
            return bot_data
        else:
            logger.error(f"❌ Failed to create bot: {response.status_code}")
            logger.error(response.text)
            return None
    
    except Exception as e:
        logger.error(f"❌ Exception creating bot: {e}")
        return None


def speak_to_meeting(text: str, bot_id: str) -> bool:
    """
    Convert text to speech and send to meeting via Recall.ai
    """
    temp_file = "temp_tts.mp3"
    
    try:
        # Generate TTS
        tts = gTTS(text, lang="en")
        tts.save(temp_file)
        
        # Read and encode
        with open(temp_file, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        
        # Send to Recall.ai
        response = requests.post(
            f"{RECALL_BASE_URL}/bot/{bot_id}/output_audio/",
            headers={
                "Authorization": f"Token {RECALL_API_KEY}",
                "Content-Type": "application/json"
            },
            json={"kind": "mp3", "b64_data": audio_b64},
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info(f"🔊 Spoke: {text[:60]}...")
            return True
        else:
            logger.error(f"❌ Failed to send audio: {response.status_code}")
            return False
    
    except Exception as e:
        logger.error(f"❌ Error in speak_to_meeting: {e}")
        return False
    
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def leave_meeting(bot_id: str) -> bool:
    """
    Command the bot to leave the meeting
    """
    try:
        response = requests.post(
            f"{RECALL_BASE_URL}/bot/{bot_id}/leave_call",
            headers={"Authorization": f"Token {RECALL_API_KEY}"},
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info("✅ Bot leaving meeting")
            return True
        else:
            logger.error(f"❌ Failed to leave: {response.status_code}")
            return False
    
    except Exception as e:
        logger.error(f"❌ Error leaving meeting: {e}")
        return False


def wait_for_answer(max_wait: int = MAX_WAIT_FOR_ANSWER) -> str:
    """
    Collect transcripts with silence detection
    """
    collected = []
    last_time = None
    start = time.time()
    
    while True:
        now = time.time()
        
        # Timeout
        if now - start > max_wait:
            logger.warning("⏱️ Answer timeout")
            break
        
        try:
            text = transcript_queue.get(timeout=0.5)
            collected.append(text)
            last_time = time.time()
        
        except queue.Empty:
            if collected and last_time:
                silence = now - last_time
                
                # Silence threshold reached
                if silence > SILENCE_THRESHOLD:
                    # Double-check for late chunks
                    try:
                        text = transcript_queue.get(timeout=0.8)
                        collected.append(text)
                        last_time = time.time()
                        continue
                    except queue.Empty:
                        logger.info(f"✅ Answer complete (silence: {silence:.1f}s)")
                        break
    
    answer = " ".join(collected).strip()
    return answer


# ============================================================================
# WEBHOOK HANDLERS
# ============================================================================

@app.websocket("/recall-audio-stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("🔌 WebSocket Connected!")
    
    while True:
        try:
            payload = await websocket.receive_json()
            if payload.get("event") == "transcript.partial_data":
                data_block = payload.get("data", {}).get("data", {})
                words = data_block.get("words", [])
                sentence = " ".join(w.get("text", "") for w in words)

                if sentence.strip():
                    # 🔥 RESET THE TIMER: Update last spoken time
                    interview_state["last_speech_time"] = time.time()
                    interview_state["is_speaking"] = True
                    
                    # Put raw text into a queue for the aggregator
                    transcript_queue.put(sentence)

        except Exception as e:
            break


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "bot_active": interview_state["is_active"],
        "bot_id": interview_state["bot_id"]
    }


# ============================================================================
# SPECIALIST AGENTS
# ============================================================================

question_agent = Agent(
    name="Question Generator",
    model="gpt-4o-mini",
    instructions="""Generate a unique, focused technical interview question.

Rules:
- One clear, concise question
- Cover diverse aspects: algorithms, concepts, applications, trade-offs
- DO NOT repeat variations of already-asked questions
- Consider candidate's level based on previous answers""",
    output_type=QuestionOutput,
)

evaluator_agent = Agent(
    name="Answer Evaluator",
    model="gpt-4o-mini",
    instructions="""Evaluate candidate's answer professionally.

Score 0-10:
- 9-10: Exceptional depth and accuracy
- 7-8: Strong understanding, well explained
- 5-6: Adequate, mostly correct
- 3-4: Weak or vague
- 0-2: Incorrect or irrelevant

Give constructive feedback.""",
    output_type=EvaluationOutput,
)

probe_agent = Agent(
    name="Probe Generator",
    model="gpt-4o-mini",
    instructions="""Generate helpful follow-up to draw out better answer.

Guidelines:
- Stay on same topic, don't introduce new concepts
- Be encouraging and supportive
- Keep it short (1-2 sentences)
- Address what's missing or unclear""",
    output_type=ProbeOutput,
)

duplicate_agent = Agent(
    name="Duplicate Checker",
    model="gpt-4o-mini",
    instructions="""Check if new question duplicates previous questions.

Duplicates: Same core concept, just rephrased
Not duplicates: Different topics or different aspects

Be strict - we want diverse questions.""",
    output_type=DuplicateOutput,
)

abuse_agent = Agent(
    name="Abuse Detector",
    model="gpt-4o-mini",
    instructions="""Detect if answer is intentionally off-topic or malicious.

Flag as abusive:
- Gibberish or random text
- Jokes/memes instead of answers
- Explicit refusal
- Completely unrelated content

Do NOT flag:
- Honest "I don't know"
- Incorrect but earnest attempts
- Vague or incomplete answers

Benefit of the doubt to candidate.""",
    output_type=AbuseOutput,
)


# ============================================================================
# TOOLS FOR AGENTS
# ============================================================================

@function_tool
def speak_text(ctx: RunContextWrapper[InterviewState], text: str) -> str:
    """
    Speak text to the meeting participants.
    
    Args:
        text: The text to speak
    
    Returns:
        Success confirmation
    """
    bot_id = ctx.context.bot_id
    success = speak_to_meeting(text, bot_id)
    
    if success:
        return f"Successfully spoke: {text}"
    else:
        return f"Failed to speak: {text}"


@function_tool
def get_candidate_answer(ctx: RunContextWrapper[InterviewState]) -> str:
    """
    Wait for and collect the candidate's answer from the meeting.
    Uses silence detection to determine when answer is complete.
    
    Returns:
        The candidate's answer
    """
    logger.info("👂 Listening for candidate answer...")
    answer = wait_for_answer()
    
    if not answer:
        return "[No answer received - silence or timeout]"
    
    logger.info(f"🧠 Candidate: {answer}")
    ctx.context.current_answer = answer
    
    return answer


@function_tool
def get_interview_status(ctx: RunContextWrapper[InterviewState]) -> str:
    """
    Get current interview status and history.
    """
    state = ctx.context
    
    status = f"""INTERVIEW STATUS
================
Candidate: {state.name}
Topic: {state.topic}
Progress: {state.questions_count}/{MAX_QUESTIONS} questions
Warnings: {state.warnings}/{MAX_WARNINGS}
Current probe count: {state.probe_count}/{MAX_PROBES_PER_QUESTION}

ASKED QUESTIONS:
{chr(10).join(f'{i+1}. {q}' for i, q in enumerate(state.asked_questions)) or '[None yet]'}

RECENT SCORES:
{chr(10).join(f'Q{i+1}: {e["score"]}/10' for i, e in enumerate(state.conversation_log[-3:])) or '[No evaluations yet]'}

CURRENT CONTEXT:
Question: {state.current_question or '[Generating next question]'}
Answer: {state.current_answer or '[Waiting for answer]'}
Probe history: {len(state.probe_history)} exchanges
"""
    return status


# ============================================================================
# MASTER ORCHESTRATOR AGENT
# ============================================================================

master_orchestrator = Agent(
    name="Interview Orchestrator",
    model="gpt-4o-mini",
    instructions="""You orchestrate the technical interview dynamically in a video meeting.

DECISION ACTIONS:
1. ASK_QUESTION - Generate and ask next question
2. PROBE_ANSWER - Ask follow-up on current answer (if vague and probes remain)
3. ACCEPT_ANSWER - Evaluate current answer and move to next question
4. HANDLE_WARNING - Issue warning for off-topic answer
5. TERMINATE_INTERVIEW - End interview (reached limits or max warnings)

DECISION LOGIC (you decide dynamically):
- If no current question OR just accepted previous → ASK_QUESTION
- If answer is vague but candidate trying AND probes < max → PROBE_ANSWER
- If answer is specific enough OR probes exhausted → ACCEPT_ANSWER
- If answer is abuse/off-topic → HANDLE_WARNING
- If reached MAX_QUESTIONS or MAX_WARNINGS → TERMINATE_INTERVIEW

PROBING STRATEGY (smart, not hardcoded):
- Vague but effort shown: Probe with encouragement
- Repeated "I don't know": Accept and move on (don't torture)
- Building up understanding: Probe to help them
- Stuck/repeating same vague answer: Accept with low score
- Detailed and complete: Accept immediately

You're in a real video call - be professional and supportive.""",
    output_type=OrchestratorDecision,
    tools=[
        speak_text,
        get_candidate_answer,
        get_interview_status
    ],
)


# ============================================================================
# MAIN INTERVIEW ORCHESTRATION
# ============================================================================

async def run_interview_async(state: InterviewState):
    """
    Main interview loop orchestrated by master agent
    """
    bot_id = state.bot_id
    
    # Wait for bot to join
    logger.info("⏳ Waiting for bot to join the call...")
    time.sleep(10)
    
    # Greeting
    greeting = f"Hello! I'm conducting a technical interview on {state.topic}. Let's begin."
    speak_to_meeting(greeting, bot_id)
    time.sleep(4)
    
    # Main interview loop
    loop_iteration = 0
    max_iterations = MAX_QUESTIONS * (MAX_PROBES_PER_QUESTION + 3) * 2  # Safety limit
    
    while state.questions_count < MAX_QUESTIONS and state.warnings < MAX_WARNINGS:
        loop_iteration += 1
        
        # Safety check for infinite loops
        if loop_iteration > max_iterations:
            logger.error(f"❌ Loop safety limit reached ({loop_iteration} iterations)")
            logger.error(f"   This suggests the orchestrator is not making progress")
            speak_to_meeting("I apologize, but I'm experiencing technical difficulties. Ending interview.", bot_id)
            break
        
        try:
            # Build context for orchestrator
            status_info = f"""
=== CURRENT INTERVIEW STATE ===
Candidate: {state.name}
Topic: {state.topic}
Questions completed: {state.questions_count}/{MAX_QUESTIONS}
Warnings: {state.warnings}/{MAX_WARNINGS}

=== CURRENT QUESTION CONTEXT ===
Current question: {state.current_question or "[No active question - need to ASK_QUESTION]"}
Current answer: {state.current_answer or "[No answer yet]"}
Probes used on this question: {state.probe_count}/{MAX_PROBES_PER_QUESTION}

=== CONVERSATION HISTORY FOR CURRENT QUESTION ===
{json.dumps(state.probe_history, indent=2) if state.probe_history else "[No exchanges yet for current question]"}

=== FULL INTERVIEW HISTORY ===
{json.dumps(state.conversation_log, indent=2) if state.conversation_log else "[No completed questions yet]"}

=== WHAT TO DECIDE ===
Based on the above state, choose your next action:
- If no current question: ASK_QUESTION
- If have answer but need more detail and probes remain: PROBE_ANSWER
- If answer is good enough or probes exhausted: ACCEPT_ANSWER
- If answer is off-topic: HANDLE_WARNING
- If reached limits: TERMINATE_INTERVIEW

Make your decision now."""
            
            # Get orchestrator decision
            logger.info("\n" + "="*70)
            logger.info("🤖 Consulting Master Orchestrator...")
            logger.info(f"   Current state: Q{state.questions_count + 1}/{MAX_QUESTIONS}, "
                       f"Warnings: {state.warnings}/{MAX_WARNINGS}, "
                       f"Probes: {state.probe_count}/{MAX_PROBES_PER_QUESTION}")
            logger.info("="*70)
            result = await Runner.run(master_orchestrator, status_info, context=state)
            decision: OrchestratorDecision = result.final_output_as(OrchestratorDecision)
            
            logger.info(f"\n{'='*70}")
            logger.info(f"🎯 ORCHESTRATOR DECISION: {decision.action}")
            logger.info(f"📝 Reasoning: {decision.reason}")
            logger.info(f"💯 Confidence: {decision.confidence:.0%}")
            logger.info(f"{'='*70}\n")
            
            # Validate decision
            valid_actions = [
                "ASK_QUESTION", "PROBE_ANSWER", "ACCEPT_ANSWER",
                "HANDLE_WARNING", "TERMINATE_INTERVIEW"
            ]
            if decision.action not in valid_actions:
                logger.error(f"❌ Invalid action: {decision.action}, forcing TERMINATE")
                decision.action = "TERMINATE_INTERVIEW"
            
            # Execute decision
            if decision.action == "TERMINATE_INTERVIEW":
                speak_to_meeting("Thank you for your time.", bot_id)
                break
            
            elif decision.action == "HANDLE_WARNING":
                state.warnings += 1
                msg = f"That seems off-topic. Warning {state.warnings} of {MAX_WARNINGS}."
                speak_to_meeting(msg, bot_id)

                # 🔥 CRITICAL FIX
                state.current_answer = None
                state.probe_history = []
                
                if state.warnings >= MAX_WARNINGS:
                    speak_to_meeting("Interview terminated.", bot_id)
                    generate_report(state, terminated=True)
                    return

                time.sleep(2)
                continue

            
            elif decision.action == "ASK_QUESTION":
                # Reset for new question
                state.current_question = None
                state.current_answer = None
                state.probe_count = 0
                state.probe_history = []
                
                # Generate question
                q_context = f"""Topic: {state.topic}
Question {state.questions_count + 1}/{MAX_QUESTIONS}

Already asked:
{json.dumps(state.asked_questions, indent=2)}

Generate a UNIQUE question on a different aspect."""
                
                q_result = await Runner.run(question_agent, q_context, context=state)
                q_output: QuestionOutput = q_result.final_output_as(QuestionOutput)
                
                # Check for duplicates
                dup_context = f"""Already asked: {json.dumps(state.asked_questions)}
New: {q_output.question}"""
                
                dup_result = await Runner.run(duplicate_agent, dup_context, context=state)
                dup_output: DuplicateOutput = dup_result.final_output_as(DuplicateOutput)
                
                if dup_output.is_duplicate:
                    logger.warning("⚠️ Duplicate detected, retrying...")
                    continue
                
                # Good question!
                state.current_question = q_output.question
                state.asked_questions.append(q_output.question)
                
                # Speak to candidate
                if q_output.acknowledgement and state.questions_count > 0:
                    speak_to_meeting(q_output.acknowledgement, bot_id)
                    time.sleep(2)
                
                speak_to_meeting(q_output.question, bot_id)
                logger.info(f"\n❓ Q{state.questions_count + 1}: {q_output.question}")
                time.sleep(3)
                
                # Get answer
                answer = wait_for_answer()
                
                if not answer:
                    speak_to_meeting("I didn't hear a response. Could you please answer?", bot_id)
                    time.sleep(2)
                    continue
                
                logger.info(f"🧠 Answer: {answer}")
                state.current_answer = answer
                
                # Check for abuse
                abuse_context = f"""Question: {state.current_question}
Answer: {answer}"""
                
                abuse_result = await Runner.run(abuse_agent, abuse_context, context=state)
                abuse_output: AbuseOutput = abuse_result.final_output_as(AbuseOutput)
                
                if abuse_output.is_abusive and abuse_output.confidence > 0.7:
                    state.warnings += 1
                    msg = f"That seems off-topic. Warning {state.warnings} of {MAX_WARNINGS}."
                    speak_to_meeting(msg, bot_id)
                    logger.warning(f"⚠️ {msg}")
                    continue
                
                # Store first answer in probe history
                state.probe_history.append({"answer": answer, "probe": None})
                
                # Now loop back to let orchestrator decide: PROBE or ACCEPT
                logger.info("🔄 Answer collected, consulting orchestrator for next action...")
                continue
            
            elif decision.action == "PROBE_ANSWER":
                if state.probe_count >= MAX_PROBES_PER_QUESTION:
                    logger.info("⏭️ Max probes reached, forcing acceptance on next iteration")
                    continue
                
                # Generate probe
                probe_context = f"""Question: {state.current_question}

Conversation so far:
{json.dumps(state.probe_history, indent=2)}

Generate probe {state.probe_count + 1}/{MAX_PROBES_PER_QUESTION}"""
                
                probe_result = await Runner.run(probe_agent, probe_context, context=state)
                probe_output: ProbeOutput = probe_result.final_output_as(ProbeOutput)
                
                speak_to_meeting(probe_output.probe, bot_id)
                logger.info(f"🤔 Probe {state.probe_count + 1}: {probe_output.probe}")
                
                state.probe_count += 1
                state.probe_history[-1]["probe"] = probe_output.probe
                
                time.sleep(2)
                
                # Get response to probe
                answer = wait_for_answer()
                
                if not answer:
                    speak_to_meeting("Could you please answer?", bot_id)
                    time.sleep(2)
                    continue
                
                logger.info(f"🧠 Probe response: {answer}")
                state.current_answer = answer
                state.probe_history.append({"answer": answer, "probe": None})
                
                # Loop back to consult orchestrator again
                logger.info("🔄 Probe response collected, consulting orchestrator...")
                continue
            
            elif decision.action == "ACCEPT_ANSWER":
                # Evaluate answer
                eval_context = f"""Question: {state.current_question}

Full conversation:
{json.dumps(state.probe_history, indent=2)}

Previous context:
{json.dumps(state.conversation_log[-2:], indent=2) if state.conversation_log else '[First question]'}"""
                
                eval_result = await Runner.run(evaluator_agent, eval_context, context=state)
                evaluation: EvaluationOutput = eval_result.final_output_as(EvaluationOutput)
                
                # Log the Q&A
                state.conversation_log.append({
                    "question": state.current_question,
                    "answer": state.probe_history[0]["answer"] if state.probe_history else state.current_answer,
                    "score": evaluation.score,
                    "feedback": evaluation.feedback,
                    "probes_used": state.probe_count,
                    "probe_history": state.probe_history.copy()
                })
                
                logger.info(f"✅ Accepted - Score: {evaluation.score}/10")
                logger.info(f"   {evaluation.feedback}")
                
                state.questions_count += 1
                
                # Reset for next question
                state.current_question = None
                state.current_answer = None
                state.probe_count = 0
                state.probe_history = []
                
                # Give brief pause before next question
                time.sleep(2)
                
                # Loop back to ask next question
                logger.info(f"🔄 Moving to question {state.questions_count + 1}/{MAX_QUESTIONS}...")
                continue
            
            else:
                # Safety net - should never reach here
                logger.error(f"❌ Unexpected action state: {decision.action}")
                logger.error(f"   This should not happen - check orchestrator logic")
                logger.error(f"   Forcing continue to retry...")
                continue
        
        except Exception as e:
            logger.error(f"❌ Error in interview loop: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Interview complete
    logger.info("\n" + "="*60)
    logger.info("INTERVIEW COMPLETE")
    logger.info("="*60)
    
    generate_report(state, terminated=False)
    
    # Final message
    total_score = sum(e["score"] for e in state.conversation_log) if state.conversation_log else 0
    max_possible = len(state.conversation_log) * 10 if state.conversation_log else 0
    
    closing = f"Thank you for participating. Your final score is {total_score} out of {max_possible}. Goodbye!"
    speak_to_meeting(closing, bot_id)
    time.sleep(5)


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(state: InterviewState, terminated: bool = False):
    """Generate final interview report"""
    log = state.conversation_log
    
    if not log:
        total_score = avg_score = 0
    else:
        total_score = sum(x["score"] for x in log)
        avg_score = total_score / len(log)
    
    max_possible = len(log) * 10 if log else 0
    status = "❌ TERMINATED" if terminated else "✅ COMPLETED"
    
    report = f"""# Interview Report

**Status:** {status}
**Candidate:** {state.name}
**Topic:** {state.topic}
**Date:** {datetime.utcnow().isoformat()}

## Summary

- **Total Score:** {total_score}/{max_possible}
- **Average:** {avg_score:.1f}/10
- **Questions:** {len(log)}/{MAX_QUESTIONS}
- **Warnings:** {state.warnings}/{MAX_WARNINGS}

---

"""
    
    for i, entry in enumerate(log, 1):
        probes = entry.get('probes_used', 0)
        report += f"""## Question {i}

**Q:** {entry['question']}

"""
        
        if probes > 0 and 'probe_history' in entry:
            report += "**Conversation:**\n\n"
            for j, ex in enumerate(entry['probe_history'], 1):
                if j == 1:
                    report += f"*Initial:* {ex['answer']}\n\n"
                else:
                    report += f"*Probe {j-1}:* {ex.get('probe', 'N/A')}\n\n"
                    report += f"*Response:* {ex['answer']}\n\n"
        else:
            report += f"**A:** {entry['answer']}\n\n"
        
        report += f"**Score:** {entry['score']}/10{f' (after {probes} probes)' if probes else ''}\n"
        report += f"**Feedback:** {entry['feedback']}\n\n---\n\n"
    
    filename = f"interview_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
    with open(filename, "w") as f:
        f.write(report)
    
    logger.info(f"\n📄 Report: {filename}")


# ============================================================================
# WEBHOOK SERVER
# ============================================================================

def start_webhook_server():
    """Start FastAPI webhook server in background thread"""
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main_async():
    """Main async entry point"""
    print("=" * 70)
    print("🤖 AI Interview System - Recall.ai + OpenAI Agents SDK")
    print("   Intelligent Meeting Bot with Dynamic Orchestration")
    print("=" * 70)
    
    # Check configuration
    if not RECALL_API_KEY:
        print("❌ RECALL_API_KEY not found in .env file")
        sys.exit(1)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in .env file")
        sys.exit(1)
    
    if not WEBHOOK_URL:
        print("❌ WEBHOOK_URL not found in .env file")
        print("   Example: WEBHOOK_URL=https://your-domain.com")
        sys.exit(1)
    
    # Start webhook server
    webhook_thread = Thread(target=start_webhook_server, daemon=True)
    webhook_thread.start()
    print("✅ Webhook server started on port 8000")
    time.sleep(2)
    
    # Get interview details
    candidate_name = input("\nCandidate name: ")
    topic = input("Interview topic: ")
    meeting_url = input("Meeting URL (Zoom/Meet/Teams): ")
    
    bot_name = f"Interview Bot - {candidate_name}"
    
    # Create bot
    print(f"\n🚀 Creating bot for meeting...")
    bot_data = create_recall_bot(meeting_url, bot_name)
    
    if not bot_data:
        print("❌ Failed to create bot")
        sys.exit(1)
    
    bot_id = bot_data['id']
    
    # Save bot ID
    with open("bot_id.txt", "w") as f:
        f.write(bot_id)
    
    print(f"✅ Bot ID: {bot_id}")
    print(f"   Saved to bot_id.txt")
    print(f"   The bot will join the meeting shortly...")
    print(f"   ⚠️  Make sure to admit the bot if in waiting room!")
    
    # Create interview state
    state = InterviewState(
        name=candidate_name,
        topic=topic,
        bot_id=bot_id
    )
    
    interview_state["bot_id"] = bot_id
    interview_state["candidate_name"] = candidate_name
    interview_state["topic"] = topic
    interview_state["bot_name"] = bot_name
    interview_state["is_active"] = True
    
    # Run interview
    try:
        await run_interview_async(state)
    except KeyboardInterrupt:
        print("\n\n⚠️ Interview interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during interview: {e}")
        import traceback
        traceback.print_exc()
    finally:
        interview_state["is_active"] = False
        leave_meeting(bot_id)


def main():
    """Synchronous main wrapper"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()