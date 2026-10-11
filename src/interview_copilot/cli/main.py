import argparse
import sys

import numpy as np

from interview_copilot.audio.soundcard_wasapi import SoundCardWASAPIBackend
from interview_copilot.config import config
from interview_copilot.pipeline import InterviewPipeline
from interview_copilot.settings_store import apply_prefs_to_config
from interview_copilot.stt.factory import build_stt_engine
from interview_copilot.vad.silero import SileroVAD


def cmd_devices(args):
    """List available loopback audio devices."""
    backend = SoundCardWASAPIBackend()
    try:
        devices = backend.list_devices()
        print("Available loopback devices:")
        for idx, d in enumerate(devices):
            default_marker = " (DEFAULT)" if d.get("is_default") else ""
            print(f"  [{idx}] {d['name']} {default_marker}")
            print(f"      ID: {d['id']}")
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        print(f"Error listing devices: {e}")
        sys.exit(1)


def cmd_capture_test(args):
    """Test audio capture and show real-time RMS (volume level)."""
    backend = SoundCardWASAPIBackend()
    device_id = args.device or config.AUDIO_DEVICE

    print("Starting audio capture test... Press Ctrl+C to stop.")
    try:
        backend.start(device_id=device_id)
        while True:
            chunk = backend.read_chunk()
            # chunk.data is float32 ndarray
            samples = chunk.data.flatten()
            if len(samples) > 0:
                rms = np.sqrt(np.mean(samples**2))
                # Print a simple bar based on RMS
                bar_length = min(50, int(rms * 200))
                bar = "#" * bar_length
                print(f"\rRMS: {rms:.4f} | {bar:<50}", end="", flush=True)
            else:
                print("\rNo data received.", end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopping capture test...")
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        print(f"\nError during capture: {e}")
        sys.exit(1)
    finally:
        backend.stop()


def cmd_vad_test(args):
    """Test audio capture and VAD processing."""
    backend = SoundCardWASAPIBackend()
    vad = SileroVAD()

    device_id = args.device or config.AUDIO_DEVICE

    print(
        "Starting VAD test... Speak into the loopback"
        " to see phrase detection. Press Ctrl+C to stop."
    )
    try:
        backend.start(device_id=device_id)
        while True:
            chunk = backend.read_chunk()
            phrases = vad.process_chunk(chunk)

            # Print a visual indicator if someone is currently speaking
            is_speaking = vad._is_speaking
            status = "[🗣️  SPEAKING]" if is_speaking else "[silence]"
            print(f"\rStatus: {status:<15}", end="", flush=True)

            for p in phrases:
                print(
                    f"\n[VAD] Phrase detected! "
                    f"Duration: {p.duration_s:.2f}s | Start: {p.captured_at:.2f}"
                )
    except KeyboardInterrupt:
        print("\nStopping VAD test...")
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        print(f"\nError during VAD test: {e}")
        sys.exit(1)
    finally:
        backend.stop()


def cmd_transcribe_test(args):
    """Test full audio capture -> VAD -> STT pipeline."""
    backend = SoundCardWASAPIBackend()
    vad = SileroVAD()
    stt = build_stt_engine()

    device_id = args.device or config.AUDIO_DEVICE

    print(
        f"Starting STT test (backend={config.STT_BACKEND})... "
        "Speak English into the loopback. Press Ctrl+C to stop."
    )
    try:
        backend.start(device_id=device_id)
        while True:
            chunk = backend.read_chunk()
            phrases = vad.process_chunk(chunk)

            is_speaking = vad._is_speaking
            status = "[🗣️  SPEAKING]" if is_speaking else "[silence]"
            print(f"\rStatus: {status:<15}", end="", flush=True)

            for p in phrases:
                print(f"\n[VAD] Phrase finished ({p.duration_s:.2f}s). Transcribing...")
                transcript = stt.transcribe(p)
                rtf = transcript.stt_duration_s / p.duration_s if p.duration_s > 0 else 0
                print(f"[STT] (RTF: {rtf:.2f}x): {transcript.text_en}")
                print("-" * 50)

    except KeyboardInterrupt:
        print("\nStopping STT test...")
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        print(f"\nError during STT test: {e}")
        sys.exit(1)
    finally:
        backend.stop()


def cmd_stt_bench(args):
    """Run golden-set STT harness (WER / latency)."""
    import json
    from pathlib import Path

    from interview_copilot.eval.harness import DEFAULT_GOLDEN, run_harness
    from interview_copilot.stt.deepgram import DeepgramEngine
    from interview_copilot.stt.groq_engine import GroqWhisperEngine
    from interview_copilot.stt.soniox import SonioxEngine
    from interview_copilot.stt.whisper_engine import WhisperEngine

    backend = (args.backend or config.STT_BACKEND or "whisper").lower()
    if backend == "deepgram":
        engine = DeepgramEngine()
    elif backend == "groq":
        engine = GroqWhisperEngine()
    elif backend == "soniox":
        engine = SonioxEngine()
    else:
        engine = WhisperEngine()

    if not engine.is_available:
        print(f"STT backend '{backend}' is not available (missing key/model).")
        sys.exit(1)

    golden = Path(args.golden) if args.golden else DEFAULT_GOLDEN
    report = run_harness(engine, golden_dir=golden)
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")


def cmd_stt_bakeoff(args):
    """Compare available STT backends on the golden set."""
    import json
    from pathlib import Path

    from interview_copilot.eval.bakeoff import write_bakeoff
    from interview_copilot.eval.harness import DEFAULT_GOLDEN

    golden = Path(args.golden) if args.golden else DEFAULT_GOLDEN
    out = Path(args.out) if args.out else Path("bench_bakeoff.json")
    result = write_bakeoff(out, golden_dir=golden)
    print(json.dumps({"ranking": result["ranking"]}, indent=2))
    print(f"Wrote {out}")


def cmd_run(args):
    """Run the full Interview Copilot pipeline."""
    import asyncio

    pipeline = InterviewPipeline()
    device_id = args.device or config.AUDIO_DEVICE

    print("Starting AI Interview Copilot Pipeline... Press Ctrl+C to stop.")
    try:
        asyncio.run(pipeline.start(device_id))
    except KeyboardInterrupt:
        print("\nStopping pipeline...")
        pipeline.stop()
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        print(f"\nPipeline error: {e}")
        sys.exit(1)


def cmd_start(args):
    """Start the GUI application."""
    try:
        from interview_copilot.gui.app import start_gui

        device_id = args.device or config.AUDIO_DEVICE
        start_gui(device_id)
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        print(f"\nGUI error: {e}")
        sys.exit(1)


def cmd_profiles(args):
    """List available candidate profiles."""
    from interview_copilot.suggestion.profile_manager import ProfileManager

    mgr = ProfileManager()
    profiles = mgr.list_profiles()
    if not profiles:
        print("No profiles found. Add a .md file to the context directory.")
        print(f"Context dir: {mgr._context_dir}")
        return
    active = None
    snap = mgr.load_active_profile()
    if snap:
        active = snap.name
    print(f"Context dir: {mgr._context_dir}")
    for name in profiles:
        marker = " (active)" if name == active else ""
        print(f"  - {name}{marker}")


def cmd_profile_set(args):
    """Set the active candidate profile."""
    from interview_copilot.suggestion.profile_manager import ProfileManager

    mgr = ProfileManager()
    snap = mgr.load_profile(args.name)
    if not snap:
        print(f"Profile '{args.name}' not found.")
        sys.exit(1)
    print(f"Active profile set to: {snap.name}")


def main():
    parser = argparse.ArgumentParser(description="Interview Copilot CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # devices command
    parser_devices = subparsers.add_parser("devices", help="List audio loopback devices")
    parser_devices.set_defaults(func=cmd_devices)

    # capture-test command
    parser_capture = subparsers.add_parser(
        "capture-test", help="Test audio capture (shows RMS volume)"
    )
    parser_capture.add_argument("--device", type=str, help="Device ID to use (default: auto)")
    parser_capture.set_defaults(func=cmd_capture_test)

    # vad-test command
    parser_vad = subparsers.add_parser("vad-test", help="Test VAD (Voice Activity Detection)")
    parser_vad.add_argument("--device", type=str, help="Device ID to use (default: auto)")
    parser_vad.set_defaults(func=cmd_vad_test)

    # transcribe-test command
    parser_stt = subparsers.add_parser(
        "transcribe-test", help="Test audio capture + VAD + STT"
    )
    parser_stt.add_argument("--device", type=str, help="Device ID to use (default: auto)")
    parser_stt.set_defaults(func=cmd_transcribe_test)

    parser_bench = subparsers.add_parser(
        "stt-bench", help="Run golden-set STT WER/latency harness"
    )
    parser_bench.add_argument(
        "--backend", type=str, default=None, help="whisper|deepgram|groq|soniox"
    )
    parser_bench.add_argument("--golden", type=str, default=None, help="Golden set dir")
    parser_bench.add_argument("--out", type=str, default=None, help="Write JSON report")
    parser_bench.set_defaults(func=cmd_stt_bench)

    parser_bake = subparsers.add_parser(
        "stt-bakeoff", help="Compare available STT backends on golden set"
    )
    parser_bake.add_argument("--golden", type=str, default=None, help="Golden set dir")
    parser_bake.add_argument(
        "--out", type=str, default="bench_bakeoff.json", help="Write JSON report"
    )
    parser_bake.set_defaults(func=cmd_stt_bakeoff)

    # run command
    parser_run = subparsers.add_parser("run", help="Run the full Interview Copilot (CLI mode)")
    parser_run.add_argument("--device", type=str, help="Device ID to use (default: auto)")
    parser_run.set_defaults(func=cmd_run)

    # start command
    parser_start = subparsers.add_parser("start", help="Start the AI Interview Copilot GUI")
    parser_start.add_argument("--device", type=str, help="Device ID to use (default: auto)")
    parser_start.set_defaults(func=cmd_start)

    # profiles
    parser_profiles = subparsers.add_parser("profiles", help="List candidate profiles")
    parser_profiles.set_defaults(func=cmd_profiles)

    parser_profile_set = subparsers.add_parser("profile-set", help="Set active candidate profile")
    parser_profile_set.add_argument("name", type=str, help="Profile name (without .md)")
    parser_profile_set.set_defaults(func=cmd_profile_set)

    args = parser.parse_args()
    apply_prefs_to_config()
    args.func(args)


if __name__ == "__main__":
    main()
