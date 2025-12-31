import torch
import torchaudio
import numpy as np
import base64
import asyncio
import re
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from chatterbox.tts_turbo import ChatterboxTurboTTS

app = FastAPI()

print("Loading Chatterbox Turbo...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxTurboTTS.from_pretrained(device=DEVICE)

REFERENCE_AUDIO_PATH = "/home/cloud/STT-Livekit-RTC/test_audio_2.wav" 
print(f"Model loaded on {DEVICE}")


def split_text(text):

    chunks = re.split(r'(?<=[.!?]) +', text)
    return [c.strip() for c in chunks if c.strip()]

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            
            data = await websocket.receive_text()
            print(f"Received text: {data[:]}...")

            sentences = split_text(data)

            for i, sentence in enumerate(sentences):

                print(f"Generating chunk {i+1}/{len(sentences)}: {sentence[:]}...")
               
                wav_tensor = await asyncio.to_thread(
                    model.generate, 
                    sentence, 
                    audio_prompt_path=REFERENCE_AUDIO_PATH
                )

                audio_data = wav_tensor.cpu().numpy().squeeze()

                audio_bytes = audio_data.astype(np.float32).tobytes()
     
                encoded_audio = base64.b64encode(audio_bytes).decode('utf-8')
                
                await websocket.send_json({
                    "audio": encoded_audio,
                    "sample_rate": model.sr,
                    "chunk_id": i,
                    "is_last": (i == len(sentences) - 1)
                })

                await asyncio.sleep(0.01)

            print("Finished streaming response.")

    except Exception as e:
        print(f"Connection closed or error: {e}")


@app.get("/")
async def get():
    with open("index.html", "r") as f:
        return HTMLResponse(f.read())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)