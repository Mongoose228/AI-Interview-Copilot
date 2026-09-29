# Changelog

## Unreleased

### Added

- Bounded partial Whisper transcription for faster CPU feedback
- Provisional and final transcripts update one stable card
- Text-free latency and Whisper real-time-factor metrics in logs

### Changed

- Defaults: 350 ms VAD silence, 400 ms LLM debounce
- Default CPU model: `small.en` → `base.en`

### Docs

- Rewrote `PERFORMANCE_ROADMAP.md` to match shipped CPU profile and remaining validation

### Planned for 0.3.0

- Confirm latency targets on real meeting loopback audio
- In-app profile / settings UI (API keys, model, debounce, translation) so users
  do not need to edit `.env` and AppData folders by hand
- Tag and ship Windows zip as **0.3.0** after validation

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
