import pyttsx3
import time

engine = pyttsx3.init()

voices = engine.getProperty('voices')
print(f"Found {len(voices)} voice(s)")
for i, voice in enumerate(voices):
    print(f"\nVoice {i}: {voice.name}")

    engine.setProperty('voice', voice.id)
    
    # Adjust other aspects
    engine.setProperty('rate', 150)    # Speed (words per minute)
    engine.setProperty('volume', 0.8)  # Volume (0.0 to 1.0)
    start = time.time()
    engine.say(f"Hello, this is voice {i}")
    engine.runAndWait()
    end = time.time()
    
    print(f"Latency: {end-start:.2f} seconds")