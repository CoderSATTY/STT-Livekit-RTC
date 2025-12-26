import asyncio
import logging
import os
import aiohttp
from livekit import rtc, api
from livekit.agents import stt
from livekit.plugins import google, noise_cancellation
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("manual-agent")

ROOM_NAME = "my-room"

async def main():
    agent_source = rtc.AudioSource(48000, 1)
    agent_track = rtc.LocalAudioTrack.create_audio_track("denoised_output", agent_source)

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
            logger.info("Agent Connected! Waiting for user audio...")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return

        await room.local_participant.publish_track(agent_track)

        await asyncio.Event().wait()

async def process_track(track, room, audio_source):

    stt_provider = google.STT(
        languages=["en-US"],
        detect_language=False,
    )
    
    stt_stream = stt_provider.stream()
    
    clean_stream = rtc.AudioStream(
        track,
        noise_cancellation=noise_cancellation.BVC()
    )

    async def handle_stt():
        print("Google STT Listener started")
        async for event in stt_stream:
            # Google calls these "interim" results "is_final=False" internally,
            # but LiveKit maps them to INTERIM_TRANSCRIPT for us.
            if event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                print(f"   (speaking): {event.alternatives[0].text}", end="\r")
            elif event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                text = event.alternatives[0].text
                logger.info(f"FINAL: {text}")
                await room.local_participant.publish_data(text, reliable=True)

    asyncio.create_task(handle_stt())

    try:
        logger.info("Audio Pipeline Started")
        async for event in clean_stream:
            # 1. Send clean audio to Google
            stt_stream.push_frame(event.frame)
            
            # 2. Play clean audio back to room
            await audio_source.capture_frame(event.frame)
                
    except Exception as e:
        logger.error(f"Error in loop: {e}")
    finally:
        await stt_stream.aclose()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass