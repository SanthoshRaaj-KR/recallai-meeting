from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import base64
import json

app = FastAPI()

@app.websocket("/recall-audio-stream")
async def audio_stream(websocket: WebSocket):
    print("🔌 Waiting for Recall.ai connection...")
    await websocket.accept()
    print("✅ Connected! Receiving Audio Stream...")
    
    # Open a file to save the raw audio (so you can prove it works)
    file_path = "meeting_audio.pcm"
    with open(file_path, "wb") as f:
        try:
            while True:
                # 1. Receive JSON packet from Recall
                data = await websocket.receive_text()
                payload = json.loads(data)
                event_type = payload.get("event")

                # 2. Handle Audio Data
                if event_type == "audio_mixed_raw.data":
                    # Docs: payload -> data -> data -> buffer
                    b64_audio = payload["data"]["data"]["buffer"]
                    
                    # Decode Base64 -> Raw Bytes
                    audio_bytes = base64.b64decode(b64_audio)
                    
                    # Save to file
                    f.write(audio_bytes)
                    print(f"🔊 Received {len(audio_bytes)} bytes | Saved to {file_path}")

                # 3. Handle Transcript Data (Optional)
                elif event_type == "transcript.partial_data":
                    words = payload["data"]["data"]["words"]
                    if words:
                        text = " ".join([w["text"] for w in words])
                        print(f"💬 Transcript: {text}")

        except WebSocketDisconnect:
            print("❌ Recall.ai disconnected.")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Standard HTTP port 8000, but supports WS
    uvicorn.run(app, host="0.0.0.0", port=8000)