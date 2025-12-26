import asyncio
import logging
import os
import numpy as np
import whisper
from livekit import rtc, api
from livekit.plugins import noise_cancellation
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("local-whisper")

ROOM_NAME = "my-room"

# Load model once at startup
print("Loading Whisper model...")
model = whisper.load_model("base")
print("Model loaded.")

async def main():
    agent_source = rtc.AudioSource(48000, 1)
    agent_track = rtc.LocalAudioTrack.create_audio_track("denoised_output", agent_source)

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
        logger.info("Connected.")
    except Exception as e:
        logger.error(f"Failed to connect: {e}")
        return

    await room.local_participant.publish_track(agent_track)
    await asyncio.Event().wait()

async def process_track(track, room, audio_source):
    clean_stream = rtc.AudioStream(track, noise_cancellation=noise_cancellation.BVC())
    
    audio_buffer = [] 
    MIN_VOLUME = 0.01
    SILENCE_DURATION = 1.0
    
    last_speech_time = asyncio.get_event_loop().time()
    is_speaking = False

    logger.info("Pipeline Started")

    try:
        async for event in clean_stream:
            # --- AUDIO PLAYBACK DISABLED ---
            # await audio_source.capture_frame(event.frame)

            frame = event.frame
            
            # Convert to numpy and calculate volume safely
            data_int16 = np.frombuffer(frame.data, dtype=np.int16)
            data_float = data_int16.astype(np.float32)
            volume = np.sqrt(np.mean(data_float**2)) / 32768.0
            
            current_time = asyncio.get_event_loop().time()

            # Logic: If loud enough, add to buffer. If silent for X seconds, process buffer.
            if volume > MIN_VOLUME:
                if not is_speaking:
                    is_speaking = True
                    print("Speaking...", end="\r")
                last_speech_time = current_time
                audio_buffer.append(data_int16)
            else:
                if is_speaking:
                    audio_buffer.append(data_int16)
                    if current_time - last_speech_time > SILENCE_DURATION:
                        is_speaking = False
                        
                        if audio_buffer:
                            # 1. Merge buffer
                            full_audio = np.concatenate(audio_buffer)
                            audio_buffer = []
                            
                            # 2. Convert to float32 (Whisper requirement)
                            audio_float32 = full_audio.astype(np.float32) / 32768.0

                            # 3. Resample 48k -> 16k (Whisper requirement)
                            if frame.sample_rate == 48000:
                                audio_float32 = audio_float32[::3]

                            # 4. Run Whisper in executor (Blocking call moved to background)
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(None, lambda: model.transcribe(audio_float32, fp16=False))
                            
                            text = result['text'].strip()
                            if text:
                                logger.info(f"Transcribed: {text}")
                                await room.local_participant.publish_data(text, reliable=True)
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass