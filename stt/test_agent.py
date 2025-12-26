import asyncio
import logging
import os
import numpy as np
import whisper
import noisereduce as nr
import aiohttp
from scipy import signal
from livekit import rtc, api
from livekit.plugins import silero
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pro-local-agent")

ROOM_NAME = "my-room"

# 1. Load Whisper Model
print("Loading Whisper model...")
model = whisper.load_model("base")
print("✅ Model loaded.")

async def main():
    agent_source = rtc.AudioSource(48000, 1)
    agent_track = rtc.LocalAudioTrack.create_audio_track("agent_output", agent_source)

    async with aiohttp.ClientSession() as http_session:
        room = rtc.Room()

        @room.on("track_subscribed")
        def on_track_subscribed(track, publication, participant):
            if track.kind == rtc.TrackKind.KIND_AUDIO and participant.identity != "python-agent":
                logger.info(f"Detected audio from {participant.identity}")
                asyncio.create_task(process_track(track, room, agent_source))

        token = api.AccessToken(
            os.getenv("LIVEKIT_API_KEY"),
            os.getenv("LIVEKIT_API_SECRET")
        ).with_identity("python-agent").with_name("Python Agent").with_grants(
            api.VideoGrants(room_join=True, room=ROOM_NAME)
        ).to_jwt()

        logger.info(f"Connecting to {ROOM_NAME}...")
        try:
            await room.connect(os.getenv("LIVEKIT_URL"), token)
            logger.info("✅ Connected. Speak into your mic!")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return

        await room.local_participant.publish_track(agent_track)
        await asyncio.Event().wait()

async def process_track(track, room, audio_source):
    # Initialize Silero VAD with slightly faster settings
    vad = silero.VAD.load(
        min_silence_duration=0.5, # Stop quickly after speech ends
        min_speech_duration=0.1   # Trigger quickly on short words
    )
    vad_stream = vad.stream()
    
    audio_stream = rtc.AudioStream(track)
    audio_buffer = [] 
    main_loop = asyncio.get_event_loop()
    MAX_BUFFER_FRAMES = 500 

    logger.info("Pipeline Started (Debug Mode)")

    async for event in audio_stream:
        # --- DEBUG: Print Audio Volume ---
        # This proves if Python is actually hearing you
        data_int16 = np.frombuffer(event.frame.data, dtype=np.int16)
        vol = np.sqrt(np.mean(data_int16.astype(np.float32)**2)) / 32768.0
        
        if vol > 0.005: # Only print if there is sound
            print(f"🎤 Audio Level: {vol:.4f}", end="\r")

        # 1. VAD Check
        vad_results = vad_stream.push_frame(event.frame)
        audio_buffer.append(data_int16)

        # 2. Process Results
        for res in (vad_results or []):
            if res.type == silero.VADEventType.START_OF_SPEECH:
                print("\n🗣️  Started speaking...")
                if len(audio_buffer) > 10:
                    audio_buffer = audio_buffer[-10:] # Keep 200ms pre-roll
            
            elif res.type == silero.VADEventType.END_OF_SPEECH:
                print("✅ Finished speaking. Transcribing...")
                
                if audio_buffer:
                    full_audio_48k = np.concatenate(audio_buffer)
                    audio_buffer = [] 
                    
                    await main_loop.run_in_executor(
                        None, 
                        lambda: process_audio_chunk(full_audio_48k, room, main_loop)
                    )

        # 3. Buffer Safety
        if len(audio_buffer) > MAX_BUFFER_FRAMES:
            audio_buffer = audio_buffer[-100:]

def process_audio_chunk(audio_data_48k, room, loop):
    try:
        # A. Resample 48k -> 16k
        samples_16k = int(len(audio_data_48k) * 16000 / 48000)
        audio_16k = signal.resample(audio_data_48k, samples_16k)
        audio_float32 = audio_16k.astype(np.float32) / 32768.0

        # B. Noise Reduction
        reduced_audio = nr.reduce_noise(y=audio_float32, sr=16000, stationary=True, prop_decrease=0.75)

        # C. Whisper
        result = model.transcribe(reduced_audio, fp16=False)
        text = result['text'].strip()
        
        if text:
            print(f"📝 Transcribed: {text}")
            asyncio.run_coroutine_threadsafe(
                room.local_participant.publish_data(text, reliable=True),
                loop
            )
            
    except Exception as e:
        print(f"Processing Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass