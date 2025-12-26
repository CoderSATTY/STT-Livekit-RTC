import os
import pyaudio
from google import genai
from google.genai import types

# Audio setup
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

long_text = """
This is a demonstration of streaming audio capabilities. Because this is a pure text-to-speech model, 
I am reading this specific text that was passed to me in the code, rather than generating a new story. 
Streaming allows you to hear the beginning of this sentence while the end is still being processed.
Deep in the silence of space, the Voyager probe continues its lonely journey, drifting past stars 
that have burned for billions of years. It carries a golden record, a message from humanity to the cosmos.
"""

response_stream = client.models.generate_content_stream(
    model="gemini-2.5-flash-preview-tts",
    contents=long_text,
    config=types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Kore')
            )
        ),
    )
)

for chunk in response_stream:
    for part in chunk.candidates[0].content.parts:
        if part.inline_data:
            stream.write(part.inline_data.data)

stream.stop_stream()
stream.close()
p.terminate()