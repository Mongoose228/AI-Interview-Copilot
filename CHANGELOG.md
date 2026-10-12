# Changelog

## Unreleased

### Added

- System tray icon (show/hide, Quit) and system-wide Ctrl+Shift+H hotkey (works while hidden)
- Soniox API key field in Settings
- `LLM_MAX_WAIT_FOR_SPEECH_END_S`: upper bound on waiting for VAD speech-end before the LLM call
- Stateful DSP for 20 ms chunks: smoothed auto-gain (`AutoGain`) and noise gate (`NoiseGate`)
- Deepgram streaming KeepAlive and automatic reconnect with back-off

### Changed

- Default / recommended STT path is local Whisper; Deepgram remains available but unused unless opted in (paid)
- Dual-path enabled by default for Whisper: `tiny.en` provisional UI + `base.en` final
- Dual-channel mic: separate VAD used only to suppress loopback while you speak; mic audio is not transcribed
- Escape hides the overlay instead of quitting (quit via ✕ or tray → Quit)
- Cloud STT engines raise `SttError` on network/HTTP failures, so `STT_FALLBACK` failover actually engages
- GUI errors are tracked per source (audio / OpenRouter / profile / DeepL); recovering one no longer clears the others
- Profile combo shows a "select profile" placeholder; controls stay disabled until the pipeline is ready
- CLI applies keys and settings saved in the GUI
- Warning logged when the keyring is unavailable and keys are stored in plaintext

### Fixed

- Candidate speech no longer reaches the LLM with cloud STT (`speaker` kept by Deepgram/Groq/Soniox)
- Provisional STT no longer runs on the capture thread (caused audio drops)
- Per-chunk peak normalization pumped the volume; "Light denoise" had no effect on 20 ms chunks
- Translation failure no longer marks a streaming suggestion as failed (`gather(return_exceptions=True)`)
- Prompt sanitizer no longer mangles profile text like "Operating system: Linux"
- Missing local Whisper model no longer breaks startup when it is only the fallback
- STT stage timing used a duration as the end timestamp
- Settings: Deepgram/Groq/Soniox keys now apply without restart (Deepgram stream reopens); old OpenRouter client is closed; DeepL rebuilt on key change
- Dual-channel mic recorder is opened/closed on its own thread (no cross-thread WASAPI teardown)
- Soniox reuses its HTTP client
- Deepgram: no orphaned WebSocket when the stream is closed during reconnect
- `STT_CLOUD_ENDPOINTING` marked deprecated (ignored); `LLM_MAX_WAIT_FOR_SPEECH_END_S` documented

### Refactored

- `pipeline.py` (~1050 lines) split into the `pipeline/` package: `core` (facade, lifecycle, keys/profile),
  `capture` (audio + VAD thread), `stt_worker`, `phrase_queues`, `orchestrator` (cards/debounce),
  `processing` (translation + LLM per question). Public API (`from interview_copilot.pipeline import
  InterviewPipeline`) is unchanged.

### Planned follow-ups

- Live Meet/Zoom latency with local Whisper
- Optional cloud STT later if needed
- Optionally transcribe and show candidate mic speech in the UI (today the mic is used only for echo suppression)
- Fail over from Deepgram streaming to Whisper when the key is rejected mid-session (currently it keeps reconnecting)
## 0.3.0

### Added

- STT backend abstraction (`SttEngine`) with factory: `whisper` | `deepgram` | `groq`
- Deepgram prerecorded + WebSocket streaming (Nova-3), keyterms from profile
- Groq Whisper batch STT + failover wrapper (`STT_FALLBACK`)
- Speculative dual-path (`STT_DUAL_PATH_ENABLED`): tiny local provisional + cloud final
- Audio DSP: soxr resample, energy-based mono, normalize, light denoise
- Dual-channel capture scaffold (`AUDIO_DUAL_CHANNEL` + mic device)
- STT golden harness: `interview-copilot-cli stt-bench`
- STT bake-off across available backends: `interview-copilot-cli stt-bakeoff`
- Optional Soniox backend (`SONIOX_API_KEY`)
- In-app Settings dialog (API keys via keyring when available), persisted audio device
- Privacy banner updated for cloud STT providers
- Windows SAPI TTS golden clips for local WER (no PII)
### Changed

- Default `AUDIO_CHUNK_MS=20`, `LLM_DEBOUNCE_DELAY_S=0.25`
- Cloud streaming disables Whisper-style VAD partial re-transcription
- Cloud endpointing mode: Deepgram owns utterance finals; Silero remains speaking gate

## 0.2.0

- Debounce creates a single result card after merge (no stuck "Thinking..." cards)
- Unified data directory under `%APPDATA%/interview_copilot` (context, models, logs, state)
- Clear configuration states for missing API key / profile / DeepL
- Persistent vs transient header errors; HTML-escaped GUI text
- Profile and device selectors, pause/resume, clear history, copy answer, Ctrl+Shift+H hide
- Pipeline dependency injection and real orchestrator tests
- Windows CI matrix + PyInstaller Windows artifact (without torch)
- Removed unused PyAudioWPatch backend and `TEXT_LOGGING_ENABLED` / `AUDIO_BACKEND`

## 0.1.0

- Initial MVP: WASAPI loopback, Silero VAD, faster-whisper, OpenRouter suggestions, PySide6 overlay
