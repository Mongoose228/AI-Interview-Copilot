# Changelog

## Unreleased

- Added bounded partial Whisper transcription for faster CPU feedback
- Provisional and final transcripts now update one stable card
- Reduced defaults to 350 ms VAD silence and 400 ms LLM debounce
- Switched the default CPU model from `small.en` to `base.en`
- Added text-free latency and Whisper real-time-factor metrics

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
