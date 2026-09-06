import sys

print("Checking imports...")
imports_ok = True
modules_to_test = [
    "soundcard", "numpy", "onnxruntime", "faster_whisper",
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
import soundcard as sc  # noqa: E402
try:
    speakers = sc.all_speakers()
    mics = sc.all_microphones(include_loopback=True)
    default_speaker = sc.default_speaker()
    print(
        f"[OK] Found {len(speakers)} speakers and"
        f" {len(mics)} microphones (including loopbacks)."
    )
    print(f"[OK] Default speaker: {default_speaker.name}")
except Exception as e:
    print(f"[ERROR] SoundCard loopback API failed: {e}")

print("\nChecking ONNX Runtime and Silero VAD session...")
import onnxruntime  # noqa: E402
try:
    print(f"[OK] onnxruntime version: {onnxruntime.__version__}")
    available_providers = onnxruntime.get_available_providers()
    print(f"[OK] Available providers: {available_providers}")
except Exception as e:
    print(f"[ERROR] ONNX Runtime check failed: {e}")

print("\nChecking CTranslate2 and Faster Whisper...")
import ctranslate2  # noqa: E402
try:
    print(f"[OK] ctranslate2 version: {ctranslate2.__version__}")
    cuda_available = ctranslate2.get_cuda_device_count() > 0
    print(f"[OK] CUDA available in CTranslate2: {cuda_available}")
except Exception as e:
    print(f"[ERROR] CTranslate2 check failed: {e}")
