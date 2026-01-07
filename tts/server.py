import torch
import numpy as np
import base64
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from kokoro import KPipeline

app = FastAPI()

def fixed_infer(model, ps, pack, speed=1):
    if callable(speed):
        speed = speed(len(ps))
    return model(ps, pack, speed, return_output=True)

KPipeline.infer = staticmethod(fixed_infer)

print("Loading Kokoro...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device=DEVICE)

print("Loading Voice Tensor...")
voice_tensor = torch.load('british_voice_style.pt', weights_only=True)
if voice_tensor.ndim == 3:
    voice_tensor = voice_tensor[-1]
if voice_tensor.ndim == 1:
    voice_tensor = voice_tensor.unsqueeze(0)

print(f"Model loaded on {DEVICE}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received text: {data[:]}...")

            stream = pipeline(
                data, 
                voice=voice_tensor, 
                speed=1.0, 
                split_pattern=r'(?<=[.!?]) +'
            )

            chunk_id = 0
            while True:
                # Run inference in a separate thread to keep WS responsive
                result = await asyncio.to_thread(next, stream, None)
                
                if result is None:
                    break
                
                _, _, audio = result
                
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()
                
                audio_bytes = audio.astype(np.float32).tobytes()
                encoded_audio = base64.b64encode(audio_bytes).decode('utf-8')
                
                await websocket.send_json({
                    "audio": encoded_audio,
                    "sample_rate": 24000,
                    "chunk_id": chunk_id,
                    "is_last": False 
                })
                
                chunk_id += 1

            # Send a marker indicating the stream for this text is finished
            await websocket.send_json({
                "audio": "",
                "sample_rate": 24000,
                "chunk_id": chunk_id,
                "is_last": True
            })

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
