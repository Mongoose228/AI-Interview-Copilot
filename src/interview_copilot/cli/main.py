import argparse
import sys

import numpy as np

from interview_copilot.audio.soundcard_wasapi import SoundCardWASAPIBackend
from interview_copilot.pipeline import InterviewPipeline
from interview_copilot.stt.whisper_engine import WhisperEngine
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
    except Exception as e:
        print(f"Error listing devices: {e}")
        sys.exit(1)


def cmd_capture_test(args):
    """Test audio capture and show real-time RMS (volume level)."""
    backend = SoundCardWASAPIBackend()

    device_id = None
    if args.device:
        device_id = args.device

    print("Starting audio capture test... Press Ctrl+C to stop.")
    try:
        backend.start(device_id=device_id)
        while True:
            chunk = backend.read_chunk()
            # chunk.data is float32 bytes
            samples = np.frombuffer(chunk.data, dtype=np.float32)
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
    except Exception as e:
        print(f"\nError during capture: {e}")
        sys.exit(1)
    finally:
        backend.stop()


def cmd_vad_test(args):
    """Test audio capture and VAD processing."""
    backend = SoundCardWASAPIBackend()
    vad = SileroVAD()

    device_id = args.device if args.device else None

    print(
        "Starting VAD test... Speak into the loopback to see phrase detection. Press Ctrl+C to stop."
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
                    f"\n[VAD] Phrase detected! Duration: {p.duration_s:.2f}s | Start: {p.captured_at:.2f}"
                )
    except KeyboardInterrupt:
        print("\nStopping VAD test...")
    except Exception as e:
        print(f"\nError during VAD test: {e}")
        sys.exit(1)
    finally:
        backend.stop()


def cmd_transcribe_test(args):
    """Test full audio capture -> VAD -> STT pipeline."""
    backend = SoundCardWASAPIBackend()
    vad = SileroVAD()
    stt = WhisperEngine()

    device_id = args.device if args.device else None

    print("Starting STT test... Speak English into the loopback. Press Ctrl+C to stop.")
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
    except Exception as e:
        print(f"\nError during STT test: {e}")
        sys.exit(1)
    finally:
        backend.stop()


def cmd_run(args):
    """Run the full Interview Copilot pipeline."""
    import asyncio

    pipeline = InterviewPipeline()
    device_id = args.device if args.device else None

    print("Starting AI Interview Copilot Pipeline... Press Ctrl+C to stop.")
    try:
        asyncio.run(pipeline.start(device_id))
    except KeyboardInterrupt:
        print("\nStopping pipeline...")
        pipeline.stop()
    except Exception as e:
        print(f"\nPipeline error: {e}")
        sys.exit(1)


def cmd_start(args):
    """Start the GUI application."""
    try:
        from interview_copilot.gui.app import start_gui

        device_id = args.device if args.device else None
        start_gui(device_id)
    except Exception as e:
        print(f"\nGUI error: {e}")
        sys.exit(1)


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
        "transcribe-test", help="Test audio capture + VAD + Whisper STT"
    )
    parser_stt.add_argument("--device", type=str, help="Device ID to use (default: auto)")
    parser_stt.set_defaults(func=cmd_transcribe_test)

    # run command
    parser_run = subparsers.add_parser("run", help="Run the full Interview Copilot (CLI mode)")
    parser_run.add_argument("--device", type=str, help="Device ID to use (default: auto)")
    parser_run.set_defaults(func=cmd_run)

    # start command
    parser_start = subparsers.add_parser("start", help="Start the AI Interview Copilot GUI")
    parser_start.add_argument("--device", type=str, help="Device ID to use (default: auto)")
    parser_start.set_defaults(func=cmd_start)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
