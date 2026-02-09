from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

@app.post("/recall-webhook")
async def receive_transcript(request: Request):
    try:
        # 1. Get the JSON payload
        payload = await request.json()
        event = payload.get("event")

        # 2. Filter for Transcript Data
        if event == "transcript.data":
            # DIGGING INTO THE JSON PATH FROM YOUR LOGS:
            # payload -> data -> data -> words
            # payload -> data -> data -> participant
            
            main_data = payload.get("data", {}).get("data", {})
            
            # Extract Speaker Name
            participant = main_data.get("participant", {}).get("name", "Unknown Speaker")
            
            # Extract Words and join them into a sentence
            words_list = main_data.get("words", [])
            sentence = " ".join([w.get("text", "") for w in words_list])

            # 3. PRINT THE CLEAN OUTPUT
            # Only print if there is actual text to show
            if sentence.strip():
                print(f"💬 {participant}: {sentence}")

    except Exception as e:
        print(f"❌ Error processing data: {e}")

    return {"status": "ok"}

if __name__ == "__main__":
    print("🚀 Server running... Waiting for speech...")
    uvicorn.run(app, host="0.0.0.0", port=8000)