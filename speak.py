import requests
import os
import base64
from dotenv import load_dotenv
from gtts import gTTS

load_dotenv()
API_KEY = os.getenv("RECALL_API_KEY")

# Read the Bot ID automatically from the file we saved in main.py
try:
    with open("bot_id.txt", "r") as f:
        BOT_ID = f.read().strip()
    print(f"🤖 Connected to Bot ID: {BOT_ID}")
except FileNotFoundError:
    print("❌ Error: Could not find 'bot_id.txt'. Run main.py first!")
    exit()

print("-------------------------------------------------")
print("⌨️  Type a message and press Enter to make the bot speak.")
print("❌  Type 'exit' to quit.")
print("-------------------------------------------------")

while True:
    user_text = input("You > ")
    
    if user_text.lower() in ["exit", "quit"]:
        break
        
    if not user_text.strip():
        continue

    print("   ... generating audio ...")
    
    # 1. Convert Text to Audio (MP3)
    try:
        tts = gTTS(user_text, lang='en')
        tts.save("temp_speech.mp3")
        
        # 2. Convert MP3 to Base64
        with open("temp_speech.mp3", "rb") as audio_file:
            audio_b64 = base64.b64encode(audio_file.read()).decode('utf-8')

        # 3. Send to Recall API
        response = requests.post(
            f"https://ap-northeast-1.recall.ai/api/v1/bot/{BOT_ID}/output_audio/",
            headers={
                "Authorization": f"Token {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "kind": "mp3",
                "b64_data": audio_b64
            }
        )

        if response.status_code == 200:
            print("   ✅ Sent!")
        else:
            print(f"   ❌ Failed: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("Goodbye!")