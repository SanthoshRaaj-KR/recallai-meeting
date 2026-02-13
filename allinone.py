import json
import os
import sys
import time
import requests
import base64
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from gtts import gTTS
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
import uvicorn
from threading import Thread
import queue
import logging

# ---------------- CONFIG ---------------- #

load_dotenv()
client = OpenAI()

# 🔥 NEW LOGIC: Ensure we have the Domain for WebSocket
# Expecting WEBHOOK_URL to be like "https://xyz.ngrok-free.app"
NGROK_URL = os.getenv("WEBHOOK_URL") 
WS_URL = NGROK_URL.replace("https://", "wss://") + "/recall-audio-stream"

# Recall.ai Configuration
RECALL_API_KEY = os.getenv("RECALL_API_KEY")
RECALL_REGION = os.getenv("RECALL_REGION", "ap-northeast-1")
RECALL_BASE_URL = f"https://{RECALL_REGION}.recall.ai/api/v1"

# Bot Configuration
MAX_QUESTIONS = 2
MAX_WARNINGS = 3
MAX_PROBES_PER_QUESTION = 2

# 🔥 NEW LOGIC: Magic Phrase
DONE_PHRASE = "done speaking"

app = FastAPI()

logging.basicConfig(level=logging.INFO)

# Global state
# We use a simple queue to pass the FINAL answer from WS to the Main Thread
answer_queue = queue.Queue()

interview_state = {
    "bot_id": None,
    "candidate_name": "",
    "topic": "",
    "bot_name": "Interview Bot"
}


# ---------------- RECALL.AI BOT SPEECH ---------------- #

def speak_to_meeting(text, bot_id):
    """Generates audio and sends it to the bot."""
    print(f"🗣️ Bot Speaking: {text}")
    tts = gTTS(text, lang="en")
    tts.save("temp.mp3")

    try:
        with open("temp.mp3", "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        requests.post(
            f"{RECALL_BASE_URL}/bot/{bot_id}/output_audio/",
            headers={
                "Authorization": f"Token {RECALL_API_KEY}",
                "Content-Type": "application/json"
            },
            json={"kind": "mp3", "b64_data": audio_b64}
        )
    finally:
        if os.path.exists("temp.mp3"):
            os.remove("temp.mp3")

# 🔥 NEW LOGIC: Barge-In Function
def stop_bot_speech(bot_id):
    """Instantly stops the bot from talking (Barge-In)"""
    # Note: Currently Recall doesn't have a direct 'stop_audio' endpoint exposed publicly 
    # in all regions, but sending an empty audio packet or a specific command 
    # is the standard workaround. For now, we will just print the log.
    # If your region supports 'delete_output_audio', you would call it here.
    print(f"🛑 INTERRUPTION DETECTED! (Simulated Stop Command to Bot {bot_id})")


# ---------------- RECALL.AI BOT MANAGEMENT ---------------- #

def create_recall_bot(meeting_url, bot_name="Technical Interview Bot"):
    """Create a Recall.ai bot with WebSocket configuration"""
    headers = {
        "Authorization": f"Token {RECALL_API_KEY}",
        "Content-Type": "application/json"
    }
    
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
            # 🔥 NEW LOGIC: WebSocket Configuration
            "realtime_endpoints": [
                {
                    "type": "websocket",
                    "url": WS_URL,
                    "events": [
                        "transcript.partial_data",    # To detect "Done speaking" ASAP
                        "participant_events.speech_on" # To detect Interruptions
                    ]
                }
            ]
        }
    }

    response = requests.post(f"{RECALL_BASE_URL}/bot/", headers=headers, json=payload)
    
    if response.status_code == 201:
        bot_data = response.json()
        print(f"✅ Bot created: {bot_data['id']}")
        return bot_data
    else:
        print(f"❌ Failed to create bot: {response.status_code} - {response.text}")
        return None

def leave_meeting(bot_id):
    requests.post(f"{RECALL_BASE_URL}/bot/{bot_id}/leave_call", headers={"Authorization": f"Token {RECALL_API_KEY}"})
    print(f"✅ Bot leaving meeting")


# ---------------- WEBHOOK/WEBSOCKET SERVER ---------------- #

# 🔥 NEW LOGIC: WebSocket Handler
@app.websocket("/recall-audio-stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔌 WebSocket Connected!")
    
    # Buffer to hold current sentence being spoken
    current_buffer = ""

    try:
        while True:
            data = await websocket.receive_json()
            event = data.get("event")

            # 1. HANDLE BARGE-IN (Interruption)
            if event == "participant_events.speech_on":
                bot_id = interview_state["bot_id"]
                if bot_id:
                    stop_bot_speech(bot_id)

            # 2. HANDLE TEXT STREAM
            if event == "transcript.partial_data":
                # Extract words
                words = data["data"]["data"]["words"]
                sentence = " ".join(w["text"] for w in words).lower()
                
                # Check for Magic Phrase
                if DONE_PHRASE in sentence:
                    print(f"⚡ KEYWORD DETECTED: '{DONE_PHRASE}'")
                    
                    # Clean the phrase out
                    clean_answer = sentence.replace(DONE_PHRASE, "").strip()
                    
                    # Send to main thread
                    answer_queue.put(clean_answer)
                    
    except WebSocketDisconnect:
        print("❌ WebSocket Disconnected")
    except Exception as e:
        print(f"❌ WebSocket Error: {e}")


@app.get("/health")
async def health():
    return {"status": "healthy"}


# ---------------- AI AGENT FUNCTIONS ---------------- #

def call_agent_responses_api(instructions, message_content, max_retries=3):
    """Call OpenAI API with retry logic"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": message_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            return json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"⚠️ API error: {e}")
                return None
    return None


# ---------------- AGENT INSTRUCTIONS ---------------- #

DUPLICATE_CHECKER_INSTRUCTIONS = """You check if a question is semantically duplicate to already-asked questions.

Return JSON:
{
  "is_duplicate": true/false,
  "reason": "brief explanation"
}"""

INTERVIEWER_INSTRUCTIONS = """You are a senior technical interviewer.

CRITICAL RULES:
- Ask ONLY ONE concise, focused question
- DO NOT repeat variations of already-asked questions
- Cover broad range: algorithms, concepts, applications, differences, use cases

Return JSON:
{
  "acknowledgement": "brief acknowledgement or empty string",
  "question": "your UNIQUE question on different aspect"
}"""

EVALUATOR_INSTRUCTIONS = """You are an expert technical interviewer evaluating answers.

Evaluate considering:
1. Technical accuracy
2. Depth of understanding
3. Clarity of explanation
4. Consistency with previous answers

Return JSON:
{
  "score": 0-10,
  "feedback": "2-3 sentence constructive feedback"
}"""

ABUSE_GUARD_INSTRUCTIONS = """Determine if the answer is intentionally irrelevant or malicious.

RED FLAGS:
- Gibberish or random characters
- Jokes or memes as answers
- Explicit refusal to engage
- Completely off-topic content

DO NOT FLAG:
- Honest "I don't know"
- Partial knowledge attempts
- Confusion about the question
- Incorrect but earnest attempts

Return JSON:
{
  "abusive": true/false,
  "reason": "brief explanation if abusive"
}"""

PROBE_DECISION_INSTRUCTIONS = """You analyze probe conversations to make smart decisions.

ANALYSIS GUIDELINES:
1. HONEST UNCERTAINTY (repeated "I don't know") → ACCEPT with score 4-5
2. VAGUE BUT TRYING (circling topic) → PROBE if attempts remain
3. BUILDING UP (progressive answers) → PROBE to help
4. STUCK/REPEATING (same vague answer) → ACCEPT with low score
5. GOOD ENOUGH (specific details) → ACCEPT with score 6-8

Return JSON:
{
  "decision": "ACCEPT" or "PROBE",
  "reason": "brief explanation",
  "score": 0-10 (only if ACCEPT),
  "feedback": "interviewer feedback (only if ACCEPT)"
}"""

PROBE_GENERATOR_INSTRUCTIONS = """You generate intelligent follow-up probes.

Generate follow-up that:
- Addresses what's missing/unclear
- Stays on SAME question topic
- Does NOT introduce new concepts
- Sounds natural and supportive
- Is SHORT (1-2 sentences max)

Return JSON:
{
  "probe": "your natural-sounding follow-up"
}"""


# ---------------- AGENT FUNCTIONS ---------------- #

def is_duplicate_question(new_question, asked_questions):
    if not asked_questions:
        return False
    
    asked_list = "\n".join([f"- {q}" for q in asked_questions])
    message = f"""Already asked questions:
{asked_list}

New question:
{new_question}

Is the new question essentially asking the same thing?"""
    
    result = call_agent_responses_api(DUPLICATE_CHECKER_INSTRUCTIONS, message)
    return result.get("is_duplicate", False) if result else False


def interviewer(topic, conversation_log, q_number, skip_acknowledgement=False):
    successful_context = ""
    for entry in conversation_log:
        if entry.get('score', 0) >= 5:
            successful_context += f"\nQ: {entry['question']}\nA: {entry['answer']}\n"
    
    all_questions = ""
    for entry in conversation_log:
        all_questions += f"- {entry['question']}\n"
    
    ack_instruction = "DO NOT include acknowledgement" if skip_acknowledgement else "Include brief acknowledgement"
    
    message = f"""Interview topic: {topic}
{ack_instruction}
Question {q_number}/{MAX_QUESTIONS}

ALREADY ASKED QUESTIONS (DO NOT REPEAT):
{all_questions if all_questions else "[None]"}

Successfully answered questions:
{successful_context if successful_context else "[None - ask fundamentals]"}

Ask about a COMPLETELY DIFFERENT aspect of {topic}."""
    
    result = call_agent_responses_api(INTERVIEWER_INSTRUCTIONS, message)
    return result or {
        "acknowledgement": "",
        "question": f"Can you explain a key concept in {topic}?"
    }


def evaluator(question, answer, full_context):
    context_str = ""
    for entry in full_context:
        if entry.get('score', 0) >= 5:
            context_str += f"\nQ: {entry['question']}\nA: {entry['answer']}\nScore: {entry['score']}/10\n"
    
    message = f"""Previous answers:
{context_str if context_str else "[First answer]"}

Current Question: {question}
Current Answer: {answer}

Evaluate this answer."""
    
    result = call_agent_responses_api(EVALUATOR_INSTRUCTIONS, message)
    return result or {"score": 5, "feedback": "Unable to evaluate."}


def abuse_guard(question, answer):
    message = f"""Question: {question}
Answer: {answer}

Is this answer intentionally irrelevant or malicious?"""
    
    result = call_agent_responses_api(ABUSE_GUARD_INSTRUCTIONS, message)
    return result.get("abusive", False) if result else False


def probe_decision_agent(question, probe_history):
    conversation = f"Original Question: {question}\n\n"
    for i, exchange in enumerate(probe_history, 1):
        if i == 1:
            conversation += f"Initial Answer: {exchange['answer']}\n"
        else:
            conversation += f"Probe {i-1}: {exchange['probe']}\n"
            conversation += f"Response {i}: {exchange['answer']}\n"
    
    message = conversation + "\n\nWhat should we do next?"
    
    result = call_agent_responses_api(PROBE_DECISION_INSTRUCTIONS, message)
    return result or {
        "decision": "ACCEPT",
        "reason": "Unable to analyze",
        "score": 5,
        "feedback": "Answer recorded."
    }


def probe_generator(question, probe_history, probe_number):
    conversation = f"Original Question: {question}\n\n"
    for i, exchange in enumerate(probe_history, 1):
        if i == 1:
            conversation += f"Initial Answer: {exchange['answer']}\n"
        else:
            conversation += f"Probe {i-1}: {exchange['probe']}\n"
            conversation += f"Response {i}: {exchange['answer']}\n"
    
    intensity = "gentle and encouraging" if probe_number == 1 else "more direct"
    message = f"""{conversation}

Generate a {intensity} follow-up.
Probe attempt: {probe_number}/{MAX_PROBES_PER_QUESTION}"""
    
    result = call_agent_responses_api(PROBE_GENERATOR_INSTRUCTIONS, message)
    return result.get("probe", "Could you elaborate?") if result else "Could you elaborate?"


# ---------------- SILENCE DETECTION ---------------- #

def wait_for_answer(max_wait=60):
    collected = []
    last_time = time.time()
    start = time.time()

    while True:
        if time.time() - start > max_wait:
            break

        try:
            text = transcript_queue.get(timeout=0.5)
            collected.append(text)
            last_time = time.time()
        except queue.Empty:
            if collected and time.time() - last_time > SILENCE_THRESHOLD:
                break

    return " ".join(collected)




# ---------------- INTERVIEW LOOP ---------------- #

def run_interview(bot_id, name, topic):
    """Run the automated technical interview"""
    log = []
    q_count = 0
    warnings = 0
    
    # Wait for bot to join
    print("⏳ Waiting for bot to join the call...")
    time.sleep(10)  

    
    print("✅ Bot is in the call!")
    
    # Greeting
    time.sleep(2)
    greeting = f"Hello! I'm conducting a technical interview on {topic}. Let's begin."
    speak_to_meeting(greeting, bot_id)
    time.sleep(4)
    
    while q_count < MAX_QUESTIONS:
        probe_count = 0
        probe_history = []
        skip_ack = False
        
        asked_questions = [entry['question'] for entry in log]
        
        # Generate unique question
        max_attempts = 3
        for attempt in range(max_attempts):
            q = interviewer(topic, log, q_count + 1, skip_acknowledgement=skip_ack)
            ack = q.get("acknowledgement", "")
            question = q.get("question", "")
            
            if not question:
                continue
            
            if is_duplicate_question(question, asked_questions):
                print(f"⚠️ Duplicate detected (attempt {attempt + 1}), regenerating...")
                continue
            
            break
        else:
            q_count += 1
            continue
        
        # Acknowledge previous answer
        if q_count > 0 and ack and not skip_ack:
            speak_to_meeting(ack, bot_id)
            time.sleep(2)
        
        # Ask question
        speak_to_meeting(question, bot_id)
        print(f"\n❓ Question {q_count + 1}/{MAX_QUESTIONS}: {question}")
        time.sleep(3)
        
        # Answer loop with intelligent probing
        while True:
            answer = wait_for_answer()
            
            if not answer:
                speak_to_meeting("I didn't hear a response. Could you please answer?", bot_id)
                time.sleep(2)
                continue
            
            print(f"🧠 Candidate answer: {answer}")
            
            # Check for abuse
            if abuse_guard(question, answer):
                warnings += 1
                warn_msg = f"That response seems off-topic. Warning {warnings} of {MAX_WARNINGS}."
                speak_to_meeting(warn_msg, bot_id)
                print(f"⚠️ {warn_msg}")
                
                if warnings >= MAX_WARNINGS:
                    speak_to_meeting("Interview terminated due to repeated off-topic responses.", bot_id)
                    generate_report(name, topic, log, warnings, terminated=True)
                    leave_meeting(bot_id)
                    return
                
                time.sleep(2)
                continue
            
            # Add to probe history
            probe_history.append({"answer": answer, "probe": None})
            
            # Intelligent decision
            decision = probe_decision_agent(question, probe_history)
            
            print(f"🤖 Agent decision: {decision['decision']}")
            print(f"   Reason: {decision['reason']}")
            
            if decision['decision'] == 'ACCEPT':
                score = decision.get('score', 5)
                feedback = decision.get('feedback', 'Answer recorded.')
                
                log.append({
                    "question": question,
                    "answer": answer,
                    "score": score,
                    "feedback": feedback,
                    "probes_used": probe_count,
                    "probe_history": probe_history
                })
                
                q_count += 1
                print(f"✅ Answer accepted (Score: {score}/10)")
                break
            
            elif decision['decision'] == 'PROBE':
                if probe_count >= MAX_PROBES_PER_QUESTION:
                    print(f"⏭️ Probe limit reached, accepting answer")
                    
                    score = decision.get('score', 3)
                    feedback = decision.get('feedback', f"Incomplete after {probe_count} probes.")
                    
                    log.append({
                        "question": question,
                        "answer": answer,
                        "score": score,
                        "feedback": feedback,
                        "probes_used": probe_count,
                        "probe_history": probe_history
                    })
                    
                    q_count += 1
                    skip_ack = True
                    break
                
                probe_count += 1
                probe_text = probe_generator(question, probe_history, probe_count)
                
                probe_history[-1]['probe'] = probe_text
                
                print(f"🤔 Probe {probe_count}/{MAX_PROBES_PER_QUESTION}: {probe_text}")
                speak_to_meeting(probe_text, bot_id)
                time.sleep(2)
                continue
    
    # Interview complete
    generate_report(name, topic, log, warnings, terminated=False)
    
    total_score = sum(x["score"] for x in log) if log else 0
    max_possible = len(log) * 10 if log else 0
    
    closing = f"Thank you for participating. Your final score is {total_score} out of {max_possible}. Goodbye!"
    speak_to_meeting(closing, bot_id)
    time.sleep(5)
    
    leave_meeting(bot_id)


# ---------------- REPORT GENERATION ---------------- #

def generate_report(name, topic, log, warnings, terminated=False):
    """Generate interview report"""
    total_score = sum(x["score"] for x in log) if log else 0
    avg_score = total_score / len(log) if log else 0
    max_possible = len(log) * 10 if log else 0
    status = "❌ TERMINATED" if terminated else "✅ COMPLETED"
    
    md = f"""# Interview Report

**Status:** {status}  
**Candidate:** {name}  
**Topic:** {topic}  
**Date:** {datetime.utcnow().isoformat()}  

## Summary

- **Total Score:** {total_score}/{max_possible}
- **Average Score:** {avg_score:.1f}/10
- **Questions Answered:** {len(log)}/{MAX_QUESTIONS}
- **Warnings Issued:** {warnings}/{MAX_WARNINGS}

---

"""

    for i, x in enumerate(log, 1):
        probes_used = x.get('probes_used', 0)
        probes_info = f" (after {probes_used} probes)" if probes_used > 0 else ""
        
        md += f"""## Question {i}

**Question:** {x['question']}

"""
        
        if probes_used > 0 and 'probe_history' in x:
            md += "**Conversation:**\n\n"
            for j, exchange in enumerate(x['probe_history'], 1):
                if j == 1:
                    md += f"*Initial Answer:* {exchange['answer']}\n\n"
                else:
                    md += f"*Probe {j-1}:* {exchange.get('probe', 'N/A')}\n\n"
                    md += f"*Response {j}:* {exchange['answer']}\n\n"
        else:
            md += f"**Answer:**  \n{x['answer']}\n\n"
        
        md += f"""**Score:** {x['score']}/10{probes_info}  
**Feedback:** {x['feedback']}

---

"""

    filename = f"interview_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
    with open(filename, "w") as f:
        f.write(md)
    
    print(f"\n📄 Report generated: {filename}")


# ---------------- MAIN ---------------- #

def start_webhook_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)


def main():
    print("=" * 60)
    print("🤖 Recall.ai Automated Technical Interview Bot")
    print("   (With Voice Output)")
    print("=" * 60)
    
    # Check configuration
    if not RECALL_API_KEY:
        print("❌ RECALL_API_KEY not found in .env file")
        sys.exit(1)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in .env file")
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
    
    interview_state["candidate_name"] = candidate_name
    interview_state["topic"] = topic
    interview_state["bot_name"] = f"Interview Bot - {candidate_name}"
    
    # Create bot
    print(f"\n🚀 Creating bot for meeting: {meeting_url}")
    bot_data = create_recall_bot(meeting_url, interview_state["bot_name"])
    
    if not bot_data:
        print("❌ Failed to create bot")
        sys.exit(1)
    
    bot_id = bot_data['id']
    interview_state["bot_id"] = bot_id
    
    # Save bot ID to file (like in your example)
    with open("bot_id.txt", "w") as f:
        f.write(bot_id)
    
    print(f"✅ Bot ID: {bot_id}")
    print(f"   Saved to bot_id.txt")
    print(f"   The bot will join the meeting shortly...")
    print(f"   Make sure to admit the bot if in waiting room!")
    
    # Run the interview
    try:
        run_interview(bot_id, candidate_name, topic)
    except KeyboardInterrupt:
        print("\n\n⚠️ Interview interrupted by user")
        leave_meeting(bot_id)
    except Exception as e:
        print(f"\n\n❌ Error during interview: {e}")
        import traceback
        traceback.print_exc()
        leave_meeting(bot_id)


if __name__ == "__main__":
    main()