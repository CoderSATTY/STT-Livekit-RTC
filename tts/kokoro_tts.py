import modal
import sounddevice as sd
import soundfile as sf
from kokoro import KPipeline

print("Loading model...")
pipeline = KPipeline(lang_code='a')
text = "Kokoro is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient."

generator = pipeline(text, voice='af_heart', speed=1.0, split_pattern=r'\n+')

for i, (gs, ps, audio) in enumerate(generator):
    sd.play(audio, 24000)
    sd.wait()
    sf.write(f'{i}.wav', audio, 24000)