import modal
import time

app = modal.App("chatterbox-streaming-inference")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install("chatterbox-streaming", "soundfile")
)

@app.cls(gpu="A10G", image=image, timeout=600)
class ChatterboxTTS:
    @modal.enter()
    def load_model(self):
        from chatterbox_streaming import ChatterboxTTSStreaming
        self.tts = ChatterboxTTSStreaming()
        print("Chatterbox model loaded on GPU")

    @modal.method()
    def generate(self, text: str, voice: str = "default", speed: float = 1.0):
        start = time.time()
        audio = self.tts.synthesize(text=text, voice=voice, speed=speed)
        end = time.time()
        duration = len(audio) / 24000
        rtf = (end - start) / duration if duration > 0 else float('inf')
        print(f"Non-streaming | Inference time: {end - start:.2f}s | Audio duration: {duration:.2f}s | RTF: {rtf:.3f}")
        return audio

    @modal.method()
    def generate_stream(self, text: str, voice: str = "default", speed: float = 1.0):
        start = time.time()
        first_chunk_time = None
        full_chunks = []
        for i, chunk in enumerate(self.tts.generate(text=text, voice=voice, speed=speed)):
            full_chunks.append(chunk)
            if first_chunk_time is None:
                first_chunk_time = time.time()
                print(f"Streaming | Time to first chunk: {first_chunk_time - start:.3f}s")
        import numpy as np
        audio = np.concatenate(full_chunks)
        end = time.time()
        duration = len(audio) / 24000
        rtf = (end - start) / duration if duration > 0 else float('inf')
        print(f"Streaming | Total inference time: {end - start:.2f}s | Audio duration: {duration:.2f}s | RTF: {rtf:.3f}")
        return audio

@app.local_entrypoint()
def main():
    model = ChatterboxTTS()
    text = "Hello, this is Chatterbox running fast on a Modal GPU with inference timing checks."
    
    print("Testing non-streaming inference:")
    audio = model.generate.remote(text)
    import soundfile as sf
    sf.write("output_nonstream.wav", audio, 24000)
    print("Saved output_nonstream.wav")
    
    print("\nTesting streaming inference:")
    audio_stream = model.generate_stream.remote(text)
    sf.write("output_stream.wav", audio_stream, 24000)
    print("Saved output_stream.wav")