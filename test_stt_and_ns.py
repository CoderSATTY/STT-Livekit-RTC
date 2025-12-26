import os
import time
import shutil
import noisereduce as nr
import soundfile as sf
import jiwer
import whisper
import assemblyai as aai
from deepgram import DeepgramClient
from dotenv import load_dotenv

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

REFERENCE_TEXT = "Open Spotify and play some jazz music"
AUDIO_FILE = "test_audio.wav" 

results = []

def get_accuracy(hypothesis):
    """Calculates accuracy safely."""
    if not REFERENCE_TEXT: return "N/A"
    
    def normalize(text):
        return text.lower().replace(".", "").replace(",", "").replace("?", "").replace("!", "").strip()

    ref_clean = normalize(REFERENCE_TEXT)
    hyp_clean = normalize(hypothesis)
    
    try:
        wer = jiwer.wer(ref_clean, hyp_clean)
        return round(max(0, (1 - wer) * 100), 2)
    except Exception as e:
        return 0.0

def clean_audio_file(input_path):
    print(f"   Generating noise-suppressed version for {input_path}...")
    try:
        data, rate = sf.read(input_path)
        if len(data.shape) > 1: data = data.mean(axis=1)
        
        # Spectral Gating: Reduces noise by 90% (prop_decrease=0.9)
        reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.9)
        
        output_path = "denoised_audio.wav"
        sf.write(output_path, reduced_noise, rate)
        return output_path
    except Exception as e:
        print(f"   [Error] Noise reduction failed: {e}")
        return input_path

def test_deepgram(file_path, is_clean_run=False):
    model_name = "Deepgram Nova-3" + (" (NR)" if is_clean_run else "")
    try:
        if not DEEPGRAM_API_KEY: return

        deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)

        with open(file_path, "rb") as file:
            buffer_data = file.read()
        
        start = time.time()
        

        response = deepgram.listen.v1.media.transcribe_file(
            request=buffer_data,
            model="nova-3",
            smart_format=True,
            language="en"
        )

        latency = (time.time() - start) * 1000
        
        transcript = response.results.channels[0].alternatives[0].transcript
        acc = get_accuracy(transcript)
        
        results.append({
            "Model": model_name, 
            "Latency (ms)": round(latency, 2), 
            "Accuracy (%)": acc,
            "Transcript": transcript[:] 
        })
    except Exception as e:
        print(f"{model_name} Failed: {e}")

def test_assemblyai(file_path, is_clean_run=False):
    model_name = "AssemblyAI Best" + (" (NR)" if is_clean_run else "")
    if not ASSEMBLYAI_API_KEY: return
    try:
        aai.settings.api_key = ASSEMBLYAI_API_KEY
        transcriber = aai.Transcriber()
        start = time.time()
        transcript = transcriber.transcribe(file_path)
        latency = (time.time() - start) * 1000
        if transcript.status == aai.TranscriptStatus.error: raise Exception(transcript.error)
        results.append({ "Model": model_name, "Latency (ms)": round(latency, 2), "Accuracy (%)": get_accuracy(transcript.text), "Transcript": transcript.text[:]  })
    except Exception as e:
        print(f"{model_name} Failed: {e}")

def test_whisper(file_path, size="base", is_clean_run=False):
    model_name = f"Whisper {size}" + (" (NR)" if is_clean_run else "")
    if not shutil.which("ffmpeg"):
        print(f"{model_name} Failed: FFmpeg missing.")
        return
    try:
        model = whisper.load_model(size)
        start = time.time()
        result = model.transcribe(file_path, fp16=False)
        latency = (time.time() - start) * 1000
        results.append({ "Model": model_name, "Latency (ms)": round(latency, 2), "Accuracy (%)": get_accuracy(result["text"]), "Transcript": result["text"][:]  })
    except Exception as e:
        print(f"{model_name} Failed: {e}")

if __name__ == "__main__":
    if not os.path.exists(AUDIO_FILE):
        print(f"Error: {AUDIO_FILE} not found!")
        exit()

    print(f"Starting Benchmark on: {AUDIO_FILE}")
    print("-" * 50)
    
    test_deepgram(AUDIO_FILE)
    test_assemblyai(AUDIO_FILE)
    test_whisper(AUDIO_FILE, "base")
    
    print("\nPhase 2: Testing with Noise Suppression")
    denoised_file = clean_audio_file(AUDIO_FILE)
    test_deepgram(denoised_file, is_clean_run=True)
    test_assemblyai(denoised_file, is_clean_run=True)
    test_whisper(denoised_file, "base", is_clean_run=True)
    
    print("\n" + "="*80)
    print(f"{'Model':<30} | {'Latency (ms)':<15} | {'Accuracy (%)':<15} | {'Transcript Start'}")
    print("-" * 80)
    
    sorted_results = sorted(results, key=lambda x: (x["Accuracy (%)"] != "N/A" and x["Accuracy (%)"] or 0, -x["Latency (ms)"]), reverse=True)
    for r in sorted_results:
        print(f"{r['Model']:<30} | {r['Latency (ms)']:<15} | {r['Accuracy (%)']:<15} | {r['Transcript']}")
    # if os.path.exists(denoised_file):
    #     # try: os.remove(denoised_file)
    #     except: pass

