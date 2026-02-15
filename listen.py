import json
import os
import sys
import asyncio
import base64
import requests
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from dotenv import load_dotenv
from openai import AsyncOpenAI
from gtts import gTTS
from pydantic import BaseModel, Field
from agents import Agent, Runner

# ================= CONFIGURATION =================
load_dotenv()
client = AsyncOpenAI()
RECALL_API_KEY = os.getenv("RECALL_API_KEY")

app = FastAPI()

# ================= STATE =================
state = {
    "accumulated_answer": "",       
    "questions_asked": 0,
    "current_question": "Introduction",
    "bot_id": None,
    "chunk_count": 0  # 🔥 Track how many times they spoke for this question
}

ML_QUESTIONS = [
    "Explain what overfitting is in machine learning.",
    "What is the difference between bias and variance?",
    "How does gradient descent work?",
    "What is the purpose of a validation set?",
    "Explain the difference between supervised and unsupervised learning.",
]

# ================= HELPERS =================
def load_bot_id():
    try:
        with open("bot_id.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def speak_next_question():
    state["bot_id"] = load_bot_id()
    if not state["bot_id"]:
        print("❌ Error: No BOT_ID found. Run main.py first.")
        return

    if state["questions_asked"] >= len(ML_QUESTIONS):
        print("🏁 Interview finished.")
        return

    question = ML_QUESTIONS[state["questions_asked"]]
    state["current_question"] = question
    state["questions_asked"] += 1

    print(f"\n{'='*60}")
    print(f"🤖 ASKING Q#{state['questions_asked']}: {question}")
    print(f"{'='*60}\n")

    try:
        tts = gTTS(question, lang="en")
        tts.save("temp_q.mp3")
        with open("temp_q.mp3", "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        requests.post(
            f"https://ap-northeast-1.recall.ai/api/v1/bot/{state['bot_id']}/output_audio/",
            headers={"Authorization": f"Token {RECALL_API_KEY}", "Content-Type": "application/json"},
            json={"kind": "mp3", "b64_data": audio_b64},
            timeout=5
        )
    except Exception as e:
        print(f"❌ Failed to send audio: {e}")

# ================= SHARPER AI JUDGE =================
class SentenceScoreOutput(BaseModel):
    is_complete: bool = Field(description="True if the user is done speaking.")
    confidence: float = Field(ge=0, le=1, description="Confidence (0-1).")
    reasoning: str = Field(description="Reasoning.")

sentence_judge = Agent(
    name="Turn-Taking Judge",
    model="gpt-4o-mini",
   instructions="""
You are a turn-taking judge. Your ONLY job is to decide whether the speaker has finished their thought.

FOCUS ON INTENT COMPLETION, NOT GRAMMAR.

CORE RULES:

1. Clear stop signals = COMPLETE (True, confidence 1.0):
   If the user says things like:
   - "I don't know"
   - "I'm not sure"
   - "Pass"
   - "Skip"
   - "That's it"
   - "Next question"
   - expresses frustration or clearly gives up
   → Mark as COMPLETE immediately.

2. Thought finished = COMPLETE:
   If the user's idea appears complete in meaning (even if grammar is messy, broken, or informal), mark COMPLETE.
   Ignore grammar mistakes, filler words, and imperfect phrasing.

3. Thought still forming = INCOMPLETE:
   Mark INCOMPLETE if the user is clearly mid-thought, such as:
   - trailing off ("and...", "so...", "because...")
   - obviously continuing an explanation
   - sentence feels cut off or unfinished in meaning

4. Be decisive:
   If the user has likely finished their idea, prefer COMPLETE.
   Only return INCOMPLETE when continuation is very likely.

OUTPUT:
- is_complete: True or False
- confidence: 0.0 to 1.0
- reasoning: brief explanation focused on thought completion
""",
    output_type=SentenceScoreOutput,
)

async def check_if_user_is_done(text: str):
    if not text.strip(): return None
    result = await Runner.run(sentence_judge, f"User said: {text}")
    return result.final_output_as(SentenceScoreOutput)

# ================= WEBSOCKET HANDLER =================
@app.websocket("/recall-audio-stream")
async def websocket_endpoint(websocket: WebSocket):
    print("🔌 WebSocket Connected!")
    await websocket.accept()

    # Ask first question
    if state["questions_asked"] == 0:
        await asyncio.sleep(2)
        speak_next_question()

    try:
        while True:
            data = await websocket.receive_json()
            event = data.get("event")

            # 🔥 FINALIZED CHUNKS ONLY
            if event == "transcript.data":
                data_block = data["data"]["data"]
                participant = data_block["participant"]["name"]
                
                # Ignore Bot
                if "genrealai" in str(participant).lower() or "bot" in str(participant).lower():
                    continue

                words = data_block.get("words", [])
                sentence = " ".join(w["text"] for w in words).strip()
                
                if sentence:
                    print(f"\n📦 [NEW CHUNK]: \"{sentence}\"")
                    
                    state["accumulated_answer"] += " " + sentence
                    state["chunk_count"] += 1  # Increment counter
                    full_answer = state["accumulated_answer"].strip()
                    
                    print(f"📄 [FULL ANSWER]: \"{full_answer}\"")

                    print(f"🔍 Judging...", end="\r")
                    result = await check_if_user_is_done(full_answer)
                    
                    if result:
                        status = "✅ COMPLETE" if result.is_complete else "⏳ WAITING"
                        print(f"{status} (Conf: {result.confidence:.0%}) | Reason: {result.reasoning}")

                        # 🔥 THE STALEMATE BREAKER
                        # If they have sent 3+ chunks, we lower the bar to avoid loops.
                        force_move = (state["chunk_count"] >= 3 and result.confidence > 0.6)

                        if (result.is_complete and result.confidence > 0.85) or force_move:
                            
                            if force_move:
                                print(f"⚠️ FORCE MOVING (3+ chunks received)")

                            print(f"\n🚀 FINAL ANSWER LOCKED: {full_answer}")
                            print("-" * 50)
                            
                            # Reset
                            state["accumulated_answer"] = ""
                            state["chunk_count"] = 0
                            
                            await asyncio.sleep(1)
                            speak_next_question()
                        else:
                            print(f"   (Waiting for more...)")

    except WebSocketDisconnect:
        print("\n❌ WebSocket Disconnected")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)