# AI Interview Copilot

Real-time interview assistant for Windows: captures system audio (WASAPI loopback),
transcribes English speech, optionally translates, and streams AI answer suggestions
via OpenRouter.

## Requirements

- **Windows 10 version 2004+** (for screen-capture exclusion of the overlay)
- Python **3.11+**
- OpenRouter API key (suggestions)
- Optional: DeepL API key (cloud translation)
- Interview audio in **English** (`WHISPER_MODEL=small.en` by default)

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
| `window_state.json` | Overlay geometry |

`.env` is loaded from CWD, then data-dir, then the project root (editable installs).

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

## Privacy

Transcripts and the candidate profile are **always** sent to OpenRouter when suggestions
are enabled. `LOG_OBFUSCATION_ENABLED=true` only affects **local** logs (no transcript text).

A one-time banner explains this on first launch.

## Troubleshooting

| Symptom | What to check |
|---------|----------------|
| No audio / no transcripts | `devices` — pick a **loopback** device; ensure meeting audio is playing |
| "OpenRouter API key not set" | `OPENROUTER_API_KEY` in `.env` |
| "No candidate profile" | Create a `.md` under context dir (not only `example_profile`) |
| DeepL warning | Set `DEEPL_API_KEY` or set `TRANSLATION_BACKEND=none` |
| Slow first start | Whisper model download/warmup; VAD ONNX download |
| GPU | `WHISPER_DEVICE=cuda` when CUDA + ctranslate2 GPU build available |

## Development

```bash
ruff check .
pytest tests/ -v
```

Windows release zip is built in CI via PyInstaller (`interview_copilot.spec`). Torch/NLLB
are excluded from the bundle; use DeepL or `none` for translation in the packaged app.

## License

MIT — see [LICENSE](LICENSE).
