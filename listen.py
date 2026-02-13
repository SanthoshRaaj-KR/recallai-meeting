from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import asyncio
import os
import random
import base64
import requests
from dotenv import load_dotenv
from gtts import gTTS
from pydantic import BaseModel, Field
from agents import Agent, Runner

# ================================
# ENV
# ================================

load_dotenv()
RECALL_API_KEY = os.getenv("RECALL_API_KEY")

# ================================
# INTERVIEW CONTROL
# ================================

MAX_QUESTIONS = 5
COMPLETION_THRESHOLD = 0.90  # 90% confidence to move on
CHECK_INTERVAL = 2.0  # Check every 2 seconds

current_buffer = ""  # Latest transcript chunk
accumulated_answer = ""  # Full answer accumulated over time
answer_finalized = False
questions_asked = 0
checking_task = None  # Background task for continuous checking

# load bot id
try:
    with open("bot_id.txt", "r") as f:
        BOT_ID = f.read().strip()
except FileNotFoundError:
    BOT_ID = None

# ================================
# QUESTIONS
# ================================

ML_QUESTIONS = [
    "Explain what overfitting is in machine learning.",
    "What is the difference between bias and variance?",
    "How does gradient descent work?",
    "What is the purpose of a validation set?",
    "Explain the difference between supervised and unsupervised learning.",
]

CURRENT_QUESTION = ""

# ================================
# FASTAPI
# ================================

app = FastAPI()

# ================================
# AGENT OUTPUT
# ================================

class SentenceScoreOutput(BaseModel):
    is_complete_sentence: bool = Field(description="Whether this is a grammatically complete sentence")
    confidence_score: float = Field(ge=0, le=1, description="Confidence that the sentence is complete and coherent")
    reasoning: str = Field(description="Brief explanation of the score")

# ================================
# AGENT
# ================================

sentence_judge = Agent(
    name="Sentence Completion Judge",
    model="gpt-4o-mini",
    instructions="""
You are a judge that determines if the accumulated answer is a complete, coherent response.

Evaluate the ENTIRE accumulated answer:
- Is it grammatically complete?
- Does it express a complete thought?
- Does it properly answer the question?
- Are there trailing words that suggest incompleteness?

Return:
- is_complete_sentence: true/false
- confidence_score: 0.0 to 1.0 (how confident you are)
- reasoning: brief explanation

Be strict - only mark as complete if it's truly a finished answer.
""",
    output_type=SentenceScoreOutput,
)

# ================================
# HELPERS
# ================================

async def evaluate_answer(accumulated_text: str):
    """Evaluate if the accumulated answer is complete."""
    if not accumulated_text.strip():
        return None
    
    prompt = f"""
Question: {CURRENT_QUESTION}

Accumulated Answer: {accumulated_text}

Is this answer complete?
"""
    
    result = await Runner.run(sentence_judge, prompt)
    return result.final_output_as(SentenceScoreOutput)


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def speak_question():
    global questions_asked, CURRENT_QUESTION

    if not BOT_ID:
        print("❌ No BOT_ID available.")
        return

    if questions_asked >= MAX_QUESTIONS:
        print("🏁 Interview finished.")
        return

    CURRENT_QUESTION = random.choice(ML_QUESTIONS)
    print(f"\n{'='*60}")
    print(f"🤖 QUESTION #{questions_asked + 1}: {CURRENT_QUESTION}")
    print(f"{'='*60}\n")

    tts = gTTS(CURRENT_QUESTION, lang="en")
    tts.save("temp_question.mp3")

    with open("temp_question.mp3", "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    requests.post(
        f"https://ap-northeast-1.recall.ai/api/v1/bot/{BOT_ID}/output_audio/",
        headers={
            "Authorization": f"Token {RECALL_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"kind": "mp3", "b64_data": audio_b64},
    )

    questions_asked += 1


# ================================
# 🔥 CONTINUOUS CHECKING LOOP 🔥
# ================================

async def continuous_answer_checker():
    """
    Background task that:
    1. Appends buffer to accumulated answer
    2. Checks if accumulated answer is complete
    3. Moves to next question if 90%+ confidence
    """
    global current_buffer, accumulated_answer, answer_finalized
    
    print("🔄 Starting continuous answer checker...")
    
    last_checked_buffer = ""
    
    while True:
        try:
            await asyncio.sleep(CHECK_INTERVAL)
            
            # Skip if answer already finalized
            if answer_finalized:
                continue
            
            # 🔥 STEP 1: Append new buffer content to accumulated answer
            if current_buffer and current_buffer != last_checked_buffer:
                # Only append if it's actually new content
                if current_buffer not in accumulated_answer:
                    accumulated_answer = current_buffer  # Replace with latest (it already contains everything)
                
                last_checked_buffer = current_buffer
                
                word_count = count_words(accumulated_answer)
                print(f"\n📝 [ACCUMULATION] Updated answer ({word_count} words)")
                print(f"   Current: {accumulated_answer[:100]}...")
            
            # Skip if accumulated answer is too short
            if count_words(accumulated_answer) < 5:
                continue
            
            # 🔥 STEP 2: Evaluate the ACCUMULATED answer
            print(f"\n🔍 [AUTO-CHECK] Evaluating accumulated answer...")
            
            result = await evaluate_answer(accumulated_answer)
            
            if not result:
                continue
            
            # Display results
            print(f"┌─ ANSWER ANALYSIS ───────────────────────────")
            print(f"│ Question: {CURRENT_QUESTION[:50]}...")
            print(f"│ Answer ({count_words(accumulated_answer)} words): {accumulated_answer[:60]}...")
            print(f"│")
            print(f"│ ✓ Complete:     {result.is_complete_sentence}")
            print(f"│ 📊 Confidence:  {result.confidence_score:.2%}")
            print(f"│ 💭 {result.reasoning}")
            print(f"└─────────────────────────────────────────────")
            
            # 🔥 STEP 3: If confidence >= 90%, move to next question
            if result.confidence_score >= COMPLETION_THRESHOLD:
                print(f"\n✅ ANSWER COMPLETE! ({result.confidence_score:.2%} confidence)")
                print(f"\n📝 FINAL ANSWER:")
                print(f"   {accumulated_answer}")
                print(f"\n🎯 Moving to next question...\n")
                
                answer_finalized = True
                
                await asyncio.sleep(2)
                
                # 🔥 STEP 4: Reset everything for next question
                current_buffer = ""
                accumulated_answer = ""
                answer_finalized = False
                last_checked_buffer = ""
                
                speak_question()
            else:
                print(f"⏳ Waiting for completion... ({result.confidence_score:.2%} < {COMPLETION_THRESHOLD:.0%})")
                print(f"   Keep speaking to build your answer...\n")
                
        except Exception as e:
            print(f"❌ Error in continuous checker: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(1)


# ================================
# WEBSOCKET
# ================================

@app.websocket("/recall-audio-stream")
async def websocket_endpoint(websocket: WebSocket):
    global current_buffer, accumulated_answer, answer_finalized, checking_task

    print("🔌 WebSocket connected!")
    await websocket.accept()

    # 🔥 Start continuous checking task
    if checking_task is None:
        checking_task = asyncio.create_task(continuous_answer_checker())
        print("✅ Continuous checker started!")

    # 🔥 Ask first question once bot connects
    if questions_asked == 0:
        await asyncio.sleep(2)
        speak_question()

    try:
        while True:
            data = await websocket.receive_json()
            event = data.get("event")

            if event != "transcript.partial_data":
                continue

            data_block = data.get("data", {}).get("data", {})
            participant = data_block.get("participant", {}).get("name", "Unknown")
            is_final = data_block.get("is_final", False)
            words = data_block.get("words", [])
            sentence = " ".join(w.get("text", "") for w in words).strip()

            if not sentence:
                continue

            # Skip bot's own speech
            if "bot" in participant.lower() or "generalai" in participant.lower():
                continue

            # 🔥 Update buffer with latest transcript
            # This buffer gets continuously updated by the transcript stream
            current_buffer = sentence
            
            # Show live updates
            if not is_final:
                print(f"\r💬 LIVE: {sentence[:100]}...", end="", flush=True)
            else:
                print(f"\n✓ FINAL: {sentence}")

    except WebSocketDisconnect:
        print("\n❌ WebSocket Disconnected")
        if checking_task:
            checking_task.cancel()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        if checking_task:
            checking_task.cancel()


# ================================
# MAIN
# ================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)