from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import base64
import json
import pyaudio
import queue
import threading

app = FastAPI()

# --- CONFIGURATION ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

# Jitter Buffer Settings
BUFFER_SIZE = 10 

# Audio Queue (The Waiting Room)
audio_queue = queue.Queue()

def play_audio():
    """Separate thread that plays audio smoothly from the queue"""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)
    
    print("🎧 Player Thread Started...")
    
    # 1. PRE-BUFFERING: Wait until we have enough packets to start
    # This ensures we don't start playing and immediately run out.
    while True:
        if audio_queue.qsize() >= BUFFER_SIZE:
            print("🚀 Buffer full! Starting playback...")
            break
            
    # 2. CONTINUOUS PLAYBACK
    while True:
        try:
            data = audio_queue.get() # Get next chunk
            stream.write(data)       # Play it
        except Exception as e:
            print(f"Error playing: {e}")
            break

# Start the player in the background
threading.Thread(target=play_audio, daemon=True).start()

@app.websocket("/recall-audio-stream")
async def audio_stream(websocket: WebSocket):
    print("🔌 Waiting for Bot...")
    await websocket.accept()
    print("✅ Connected!")

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)

            if payload.get("event") == "audio_mixed_raw.data":
                b64_audio = payload["data"]["data"]["buffer"]
                audio_bytes = base64.b64decode(b64_audio)
                
                # INSTEAD OF PLAYING DIRECTLY, PUT IN QUEUE
                audio_queue.put(audio_bytes)

    except WebSocketDisconnect:
        print("❌ Bot Disconnected.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)