import asyncio
import numpy as np
import sounddevice as sd
import queue
import threading
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from agents import Agent, Runner
from dotenv import load_dotenv
from openai import AsyncOpenAI
import os
load_dotenv()

client = AsyncOpenAI()
print("Key present:", bool(os.getenv("OPENAI_API_KEY")))
# ================================
# CONFIGURATION
# ================================

BUFFER_WORD_LIMIT = 12  # Max words before force-append
SILENCE_THRESHOLD = 7.0  # Seconds of silence before checking
COMPLETION_THRESHOLD = 0.90  # 90% confidence to move on
SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHUNK_DURATION = 1.0  # Process audio every 1 second

# ================================
# STATE VARIABLES
# ================================

current_buffer = ""  # Current transcript chunk
accumulated_answer = ""  # Full answer built over time
last_speech_time = None  # Track when last speech detected
is_listening = True
audio_queue = queue.Queue()

# ================================
# QUESTIONS
# ================================

QUESTIONS = [
    "Explain what overfitting is in machine learning.",
    "What is the difference between bias and variance?",
    "How does gradient descent work?",
    "What is the purpose of a validation set?",
    "Explain the difference between supervised and unsupervised learning.",
]

current_question_index = 0

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
    model="gpt-4o",
    instructions="""
You are a judge that determines if the text is a grammatically complete sentence.

Focus ONLY on:
- Is it grammatically complete?
- Does it express a complete thought?
- Are there trailing words that suggest incompleteness?

DO NOT judge:
- Whether the answer is correct
- Technical accuracy
- Quality of content

Return:
- is_complete_sentence: true/false
- confidence_score: 0.0 to 1.0 (how confident you are about completeness)
- reasoning: brief explanation

""",
    output_type=SentenceScoreOutput,
)


# ================================
# HELPERS
# ================================

def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


async def evaluate_sentence(text: str):
    """Evaluate if the accumulated answer is a complete sentence."""
    if not text.strip():
        return None
    
    result = await Runner.run(sentence_judge, text)
    return result.final_output_as(SentenceScoreOutput)


def ask_question():
    """Display current question."""
    global current_question_index
    
    if current_question_index >= len(QUESTIONS):
        print("\n🏁 All questions completed!")
        return None
    
    question = QUESTIONS[current_question_index]
    print(f"\n{'='*60}")
    print(f"🤖 QUESTION #{current_question_index + 1}: {question}")
    print(f"{'='*60}")
    print("🎤 Start speaking your answer...\n")
    return question


def get_silence_duration():
    """Get seconds since last speech."""
    global last_speech_time
    if last_speech_time is None:
        return 0
    return (datetime.now() - last_speech_time).total_seconds()


# ================================
# AUDIO PROCESSING
# ================================

def audio_callback(indata, frames, time_info, status):
    """Callback for audio stream - captures audio chunks."""
    if status:
        print(f"⚠️  Audio status: {status}")
    audio_queue.put(indata.copy())


async def transcribe_audio_chunk_remote(audio_data):
    try:
        audio_float = audio_data.flatten().astype(np.float32)

        if np.max(np.abs(audio_float)) > 0:
            audio_float = audio_float / np.max(np.abs(audio_float))

        # save temp wav
        import tempfile, soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_float, SAMPLE_RATE)

            with open(f.name, "rb") as audio_file:
                transcript = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en",
                )

        return transcript.text.strip()

    except Exception as e:
        print(f"❌ Remote transcription error: {e}")
        return ""

async def process_transcript(text: str):
    """Process transcribed text - append to buffer and check conditions."""
    global current_buffer, accumulated_answer, last_speech_time, current_question_index
    
    if not text:
        return
    
    # Update last speech time
    last_speech_time = datetime.now()
    
    # Append to buffer
    if current_buffer:
        current_buffer = current_buffer + " " + text
    else:
        current_buffer = text
    
    buffer_word_count = count_words(current_buffer)
    
    print(f"\n💬 Heard: {text}")
    print(f"📝 Buffer ({buffer_word_count} words): {current_buffer[:100]}...")
    
    # 🔥 CONDITION 1: Buffer exceeds word limit
    if buffer_word_count >= BUFFER_WORD_LIMIT:
        print(f"\n🔄 Buffer limit reached ({buffer_word_count} ≥ {BUFFER_WORD_LIMIT})!")
        await append_buffer_and_check()


async def append_buffer_and_check():
    """Append buffer to answer and check for completion."""
    global current_buffer, accumulated_answer, current_question_index
    
    if not current_buffer.strip():
        return
    
    # Append buffer to accumulated answer
    if accumulated_answer:
        accumulated_answer = accumulated_answer + " " + current_buffer
    else:
        accumulated_answer = current_buffer
    
    # Clear buffer
    current_buffer = ""
    
    # Show accumulated answer
    answer_word_count = count_words(accumulated_answer)
    print(f"\n📄 Accumulated Answer ({answer_word_count} words):")
    print(f"   {accumulated_answer}")
    
    # 🔥 Evaluate the accumulated answer
    print(f"\n🔍 Checking if sentence is complete...")
    
    result = await evaluate_sentence(accumulated_answer)
    
    if result:
        print(f"\n┌─ SENTENCE COMPLETION ANALYSIS ──────────────")
        print(f"│ ✓ Complete:     {result.is_complete_sentence}")
        print(f"│ 📊 Confidence:  {result.confidence_score:.2%}")
        print(f"│ 💭 {result.reasoning[:70]}...")
        print(f"└─────────────────────────────────────────────\n")
        
        # 🔥 If sentence complete with 90%+ confidence, move to next question
        if result.confidence_score >= COMPLETION_THRESHOLD:
            print(f"✅ SENTENCE COMPLETE! ({result.confidence_score:.2%} confidence)")
            print(f"\n📝 FINAL ANSWER: {accumulated_answer}\n")
            
            # Reset for next question
            current_buffer = ""
            accumulated_answer = ""
            current_question_index += 1
            
            # Ask next question
            if ask_question() is None:
                global is_listening
                is_listening = False
        else:
            print(f"⏳ Not complete yet ({result.confidence_score:.2%} < {COMPLETION_THRESHOLD:.0%})")
            print(f"   Keep speaking to complete your sentence...\n")


# ================================
# SILENCE MONITOR
# ================================

async def silence_monitor():
    """Monitor for silence and trigger checks."""
    global current_buffer, last_speech_time
    
    print("🔇 Silence monitor started...")
    
    while is_listening:
        await asyncio.sleep(1)  # Check every second
        
        silence_duration = get_silence_duration()
        
        # 🔥 CONDITION 2: Silence detected for threshold duration
        if silence_duration >= SILENCE_THRESHOLD and current_buffer.strip():
            print(f"\n🔇 Silence detected ({silence_duration:.1f}s ≥ {SILENCE_THRESHOLD}s)")
            print(f"   Appending buffer to answer...\n")
            
            await append_buffer_and_check()
            
            # Reset last speech time to avoid repeated triggers
            last_speech_time = None


# ================================
# AUDIO LISTENER
# ================================

async def audio_listener():
    """Listen to audio and transcribe in real-time."""
    global last_speech_time, is_listening
    
    print("🎤 Audio listener started...")
    print(f"📊 Sample rate: {SAMPLE_RATE}Hz")
    print(f"⏱️  Chunk duration: {CHUNK_DURATION}s\n")
    
    # Start audio stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback,
        blocksize=int(SAMPLE_RATE * CHUNK_DURATION)
    )
    
    stream.start()
    
    # Initialize last speech time
    last_speech_time = datetime.now()
    
    accumulated_audio = []
    
    try:
        while is_listening:
            # Get audio chunk
            if not audio_queue.empty():
                audio_chunk = audio_queue.get()
                accumulated_audio.append(audio_chunk)
                
                # Process every second of audio
                if len(accumulated_audio) >= 1:
                    # Combine chunks
                    combined_audio = np.concatenate(accumulated_audio)
                    
                    # Check if there's actual sound (not just silence)
                    audio_energy = np.sqrt(np.mean(combined_audio**2))
                    
                    if audio_energy > 0.01:  # Threshold for detecting speech
                        # Transcribe
                        text = await transcribe_audio_chunk_remote(combined_audio)
                        
                        if text:
                            await process_transcript(text)
                    
                    # Clear accumulated audio
                    accumulated_audio = []
            
            await asyncio.sleep(0.1)  # Small delay to prevent busy loop
            
    finally:
        stream.stop()
        stream.close()


# ================================
# MAIN
# ================================

async def main():
    """Main async loop."""
    print("\n" + "="*60)
    print("🎯 LIVE WHISPER INTERVIEW SYSTEM")
    print("="*60)
    print(f"\n⚙️  Configuration:")
    print(f"   - Buffer word limit: {BUFFER_WORD_LIMIT} words")
    print(f"   - Silence threshold: {SILENCE_THRESHOLD} seconds")
    print(f"   - Completion threshold: {COMPLETION_THRESHOLD:.0%}")
    print(f"\n📖 How it works:")
    print(f"   1. Speak your answer (Whisper transcribes in real-time)")
    print(f"   2. Transcript appends to buffer")
    print(f"   3. Buffer ≥ {BUFFER_WORD_LIMIT} words OR {SILENCE_THRESHOLD}s silence → append to answer")
    print(f"   4. Check if answer is complete sentence")
    print(f"   5. If {COMPLETION_THRESHOLD:.0%}+ confident → next question")
    print(f"\n🎤 Make sure your microphone is connected!")
    print(f"   Press Ctrl+C to stop\n")
    
    input("Press Enter to start...")
    
    # Ask first question
    ask_question()
    
    # Start both tasks
    await asyncio.gather(
        audio_listener(),
        silence_monitor()
    )
    
    print("\n👋 Interview completed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")