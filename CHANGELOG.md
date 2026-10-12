# Changelog

## Unreleased

### Changed

- Default / recommended STT path is local Whisper; Deepgram remains available but unused unless opted in (paid)
- Dual-path enabled by default for Whisper: `tiny.en` provisional UI + `base.en` final
- Dual-channel mic path wired: separate VAD, `speaker=candidate` (no LLM), suppress loopback while mic speaks

### Fixed

- Settings: Deepgram/Groq/Soniox keys now apply without restart (Deepgram stream reopens); old OpenRouter client is closed; DeepL rebuilt on key change
- Dual-channel mic recorder is opened/closed on its own thread (no cross-thread WASAPI teardown)
- Soniox reuses its HTTP client
- Deepgram: no orphaned WebSocket when the stream is closed during reconnect
- `STT_CLOUD_ENDPOINTING` marked deprecated (ignored); `LLM_MAX_WAIT_FOR_SPEECH_END_S` documented

### Planned follow-ups

- Live Meet/Zoom latency with local Whisper
- Optional cloud STT later if needed
- Show candidate mic transcripts in UI (currently STT runs but LLM path skips)
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
