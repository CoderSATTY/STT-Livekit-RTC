import asyncio
import logging
import os
import aiohttp
import numpy as np
import io
import wave
from livekit import rtc, api
from livekit.plugins import noise_cancellation
from groq import AsyncGroq  # <--- OFFICIAL GROQ CLIENT
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("groq-direct-agent")

ROOM_NAME = "my-room"

# VAD Settings (Voice Activity Detection)
MIN_VOLUME = 0.005           # Sensitivity (Lower = more sensitive)
SILENCE_DURATION = 0.6       # Seconds of silence to wait before sending to Groq

async def main():
    agent_source = rtc.AudioSource(48000, 1)
    agent_track = rtc.LocalAudioTrack.create_audio_track("denoised_output", agent_source)

    # We use a shared session for LiveKit, but Groq manages its own connection
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
            logger.info("✅ Agent Connected! Waiting for user audio...")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return

        await room.local_participant.publish_track(agent_track)
        await asyncio.Event().wait()

async def process_track(track, room, audio_source):
    # 1. Initialize Groq Client
    groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

    # 2. Setup Noise Cancellation
    clean_stream = rtc.AudioStream(
        track,
        noise_cancellation=noise_cancellation.BVC()
    )
    
    # Buffers for Audio Logic
    audio_buffer = [] 
    last_speech_time = asyncio.get_event_loop().time()
    is_speaking = False

    logger.info("🌊 Groq Audio Pipeline Started")

    try:
        async for event in clean_stream:
            # A. Play back clean audio (echo) immediately
            await audio_source.capture_frame(event.frame)

            # B. Process Audio for Groq
            frame = event.frame
            
            # Convert LiveKit Frame to Int16 Numpy Array
            data_int16 = np.frombuffer(frame.data, dtype=np.int16)
            
            # Calculate Volume (RMS)
            volume = np.sqrt(np.mean(data_int16**2)) / 32768.0
            current_time = asyncio.get_event_loop().time()

            # --- LOGIC: DETECT SPEECH ---
            if volume > MIN_VOLUME:
                if not is_speaking:
                    is_speaking = True
                    print("   (User speaking...)", end="\r")
                last_speech_time = current_time
                audio_buffer.append(data_int16)
            
            # --- LOGIC: DETECT SILENCE & SEND ---
            else:
                if is_speaking:
                    # Keep recording during brief pauses
                    audio_buffer.append(data_int16)
                    
                    # If silence is long enough, send to Groq
                    if current_time - last_speech_time > SILENCE_DURATION:
                        is_speaking = False
                        
                        if len(audio_buffer) > 0:
                            # 1. Prepare WAV file in memory
                            full_audio = np.concatenate(audio_buffer)
                            audio_buffer = [] # Clear buffer
                            
                            wav_buffer = io.BytesIO()
                            with wave.open(wav_buffer, 'wb') as wf:
                                wf.setnchannels(1)
                                wf.setsampwidth(2) # 16-bit
                                wf.setframerate(frame.sample_rate) # 48000
                                wf.writeframes(full_audio.tobytes())
                            wav_buffer.name = "audio.wav"
                            wav_buffer.seek(0)

                            # 2. Send to Groq (Async)
                            # We run this in background so audio doesn't stutter
                            asyncio.create_task(transcribe_with_groq(groq_client, wav_buffer, room))

    except Exception as e:
        logger.error(f"Error in loop: {e}")

async def transcribe_with_groq(client, audio_file, room):
    try:
        # Call Groq API (Whisper Large V3)
        transcription = await client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            response_format="json",
            language="en",
            temperature=0.0
        )
        
        text = transcription.text.strip()
        if text:
            logger.info(f"📝 FINAL: {text}")
            await room.local_participant.publish_data(text, reliable=True)
            
    except Exception as e:
        logger.error(f"Groq API Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass