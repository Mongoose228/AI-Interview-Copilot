# AI Interview Copilot

Real-time interview assistant for Windows: captures system audio (WASAPI loopback),
transcribes English speech, optionally translates, and streams AI answer suggestions
via OpenRouter.

## Requirements

- **Windows 10 version 2004+** (for screen-capture exclusion of the overlay)
- Python **3.11+**
- OpenRouter API key (suggestions)
- Optional later: Deepgram / Groq / Soniox (cloud STT — paid; not required)
- Optional: DeepL API key (cloud translation)
- Interview audio in **English** (`WHISPER_MODEL=base.en` by default)

## Install

```bash
pip install -e .
# optional local NLLB translation (slow on CPU; not in the Windows release zip):
pip install -e ".[nllb]"
# development:
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and set at least `OPENROUTER_API_KEY`.

### Data directory

App data lives under `%APPDATA%\interview_copilot\`:

| Path | Purpose |
|------|---------|
| `context\` | Candidate profile `.md` files |
| `models\silero_vad.onnx` | VAD model (downloaded on first run) |
| `logs\copilot.log` | Rotating file log |
| `.copilot_state.json` | Active profile |
| `window_state.json` | Overlay geometry + last audio device |
| `user_prefs.json` | Non-secret prefs (secrets prefer OS keyring) |

`.env` is loaded from CWD, then data-dir, then the project root (editable installs).

## Speech-to-text backends

Default is **local Whisper** (`STT_BACKEND=whisper`) — free, offline audio.
Cloud backends stay in the app for later; do not enable them unless you want to pay.

| `STT_BACKEND` | Notes |
|---------------|--------|
| `whisper` (**default**) | Local faster-whisper; $0; recommended now |
| `deepgram` | Streaming Nova-3 (paid); optional later |
| `groq` | Cloud Whisper via Groq; optional |
| `soniox` | Cloud STT; optional |

Set keys in `.env` or **Settings** only if you switch off Whisper.
With Whisper, profile terms still help via `initial_prompt` when
`STT_KEYTERMS_FROM_PROFILE=true`.

Golden-set bench:

```bash
python scripts/generate_golden_audio.py
interview-copilot-cli stt-bench --backend whisper
```

## Run

```bash
# GUI (recommended)
interview-copilot
# or
python -m interview_copilot.gui.app

# CLI
interview-copilot-cli start
interview-copilot-cli devices
interview-copilot-cli profiles
interview-copilot-cli profile-set my_resume
interview-copilot-cli run
```

## Profiles

1. Create `%APPDATA%\interview_copilot\context\my_resume.md` (or set `CONTEXT_DIR`).
2. Select it in the GUI combo, or run `profile-set my_resume`.
3. `example_profile.md` is never auto-selected.

## GUI hotkeys / controls

- **Pause / Resume** — stop/start capture
- **Clear** — clear cards and transcript history
- **Copy** on a card — copy AI suggestion
- **Ctrl+Shift+H** — hide/show overlay
- **Escape** — quit
- Drag the title to move; resize via the bottom-right grip
- Overlay is excluded from screen capture on supported Windows builds

## Translation

`TRANSLATION_BACKEND` in `.env`:

- `none` — default
- `deepl` — requires `DEEPL_API_KEY`
- `nllb` — local model (`pip install -e ".[nllb]"`); slow on CPU

## Low-latency transcription

The CPU profile uses `base.en`, 350 ms VAD silence and a 400 ms LLM debounce.
While speech is active, the app takes a bounded snapshot roughly every 800 ms:

- only one Whisper inference runs at a time;
- at most one pending partial snapshot is retained;
- provisional text updates one existing card;
- only final text enters conversation history and OpenRouter;
- final phrases take priority over stale partial snapshots.

Relevant `.env` settings:

```env
WHISPER_MODEL=base.en
VAD_SILENCE_MS=350
LLM_DEBOUNCE_DELAY_S=0.4
PARTIAL_TRANSCRIPTION_ENABLED=true
PARTIAL_TRANSCRIPTION_INTERVAL_MS=800
PARTIAL_TRANSCRIPTION_MIN_AUDIO_MS=700
```

`WHISPER_CPU_THREADS=4` remains unchanged. Logs include text-free latency and
real-time-factor metrics under `[Latency]` and `[STT]`.

Details and acceptance targets: [PERFORMANCE_ROADMAP.md](PERFORMANCE_ROADMAP.md).
See [CHANGELOG.md](CHANGELOG.md) for 0.3.0 STT upgrades.

## Privacy

Transcripts/profile go to OpenRouter when suggestions are enabled.
With default **local Whisper**, interview **audio stays on your machine**.
Only if you later enable Deepgram/Groq/Soniox is audio sent to that provider.
`LOG_OBFUSCATION_ENABLED=true` only affects **local** logs (no transcript text).

A one-time banner explains this on first launch (and after privacy text updates).

## Troubleshooting

| Symptom | What to check |
|---------|----------------|
| No audio / no transcripts | `devices` — pick a **loopback** device; ensure meeting audio is playing |
| "OpenRouter API key not set" | `OPENROUTER_API_KEY` in `.env` |
| "No candidate profile" | Create a `.md` under context dir (not only `example_profile`) |
| DeepL warning | Set `DEEPL_API_KEY` or set `TRANSLATION_BACKEND=none` |
| Slow first start | Whisper model download/warmup; VAD ONNX download |
| High CPU usage | Increase `PARTIAL_TRANSCRIPTION_INTERVAL_MS` or disable partial transcription |
| Partial text changes | Expected: provisional text is replaced by the final Whisper result |

## Development

```bash
ruff check .
pytest tests/ -v
```

Windows release zip is built in CI via PyInstaller (`interview_copilot.spec`). Torch/NLLB
are excluded from the bundle; use DeepL or `none` for translation in the packaged app.

## License

MIT — see [LICENSE](LICENSE).
