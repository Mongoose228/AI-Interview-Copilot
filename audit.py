import sys

print("Checking imports...")
imports_ok = True
modules_to_test = [
    "soundcard", "numpy", "soxr", "onnxruntime", "faster_whisper", 
    "deepl", "openai", "pydantic", "pydantic_settings", "PySide6", "ctranslate2"
]
for mod in modules_to_test:
    try:
        __import__(mod)
        print(f"[OK] {mod}")
    except Exception as e:
        print(f"[ERROR] failed to import {mod}: {e}")
        imports_ok = False

if not imports_ok:
    sys.exit(1)

print("\nChecking SoundCard loopback API...")
import soundcard as sc
try:
    speakers = sc.all_speakers()
    mics = sc.all_microphones(include_loopback=True)
    default_speaker = sc.default_speaker()
    print(f"[OK] Found {len(speakers)} speakers and {len(mics)} microphones (including loopbacks).")
    print(f"[OK] Default speaker: {default_speaker.name}")
except Exception as e:
    print(f"[ERROR] SoundCard loopback API failed: {e}")

print("\nChecking ONNX Runtime and Silero VAD session...")
import onnxruntime
try:
    print(f"[OK] onnxruntime version: {onnxruntime.__version__}")
    available_providers = onnxruntime.get_available_providers()
    print(f"[OK] Available providers: {available_providers}")
except Exception as e:
    print(f"[ERROR] ONNX Runtime check failed: {e}")

print("\nChecking CTranslate2 and Faster Whisper...")
import ctranslate2
try:
    print(f"[OK] ctranslate2 version: {ctranslate2.__version__}")
    print(f"[OK] CUDA available in CTranslate2: {ctranslate2.get_cuda_device_count() > 0}")
except Exception as e:
    print(f"[ERROR] CTranslate2 check failed: {e}")

print("\nChecking soxr streaming resampling...")
import soxr
import numpy as np
try:
    resampler = soxr.ResampleStream(48000, 16000, 1, 'float32')
    chunk = np.zeros(4800, dtype=np.float32)
    out = resampler.resample_chunk(chunk)
    print(f"[OK] soxr resampled 4800 samples to {len(out)} samples.")
except Exception as e:
    print(f"[ERROR] soxr check failed: {e}")
