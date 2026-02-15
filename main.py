import requests
import os
import base64
from dotenv import load_dotenv
from gtts import gTTS

load_dotenv()
API_KEY = os.getenv("RECALL_API_KEY")

NGROK_DOMAIN = "b88a-136-233-9-107.ngrok-free.app" 

WEBSOCKET_URL = f"wss://{NGROK_DOMAIN}/recall-audio-stream"

print(f"🔌 Connecting to: {WEBSOCKET_URL}")

MEETING_URL = "https://meet.google.com/cqs-whty-xnv" 

print(f"🚀 Spawning bot...")

# 1. Spawn the Bot
payload = {
    "meeting_url": MEETING_URL,
    "bot_name": "GenrealAI",
    "recording_config": {
        "audio_mixed_raw": {},
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
                "type": "websocket",
                "url": WEBSOCKET_URL,
                "events": [
                    "audio_mixed_raw.data",    # The Raw Audio
                    "transcript.data"  # The Real-time Text
                ]
            }
        ]
    }
}

response = requests.post(
    "https://ap-northeast-1.recall.ai/api/v1/bot/",
    headers={"Authorization": f"Token {API_KEY}", "Content-Type": "application/json"},
    json=payload
)

if response.status_code in [200, 201]:
    bot_id = response.json().get('id')
    print(f"✅ SUCCESS! Bot Spawned. ID: {bot_id}")

    # 👇 THIS IS THE CRITICAL FIX 👇
    # Save the Bot ID to a file so speak.py can find it
    with open("bot_id.txt", "w") as f:
        f.write(bot_id)
    print("📁 Saved ID to 'bot_id.txt'. You can now run 'speak.py'!")
    # 👆 ------------------------ 👆

    # 2. Generate Intro Audio
    print("🎤 Generating intro audio...")
    tts = gTTS("Hello, I am ready.", lang='en')
    tts.save("intro.mp3")
    
    with open("intro.mp3", "rb") as audio_file:
        audio_b64 = base64.b64encode(audio_file.read()).decode('utf-8')

    # 3. Send Intro Audio
    requests.post(
        f"https://ap-northeast-1.recall.ai/api/v1/bot/{bot_id}/output_audio/",
        headers={"Authorization": f"Token {API_KEY}", "Content-Type": "application/json"},
        json={"kind": "mp3", "b64_data": audio_b64}
    )
    print("✅ Intro sent.")

else:
    print(f"\n❌ FAILED (Status {response.status_code})")
    print(f"Error Message: {response.text}")