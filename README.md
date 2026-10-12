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

Copy `.env.example` to `.env` and set at least `OPENROUTER_API_KEY`
(or enter it later in the GUI **Settings** dialog).

### Configuration sources

1. `.env` — loaded from CWD, then the data dir, then the project root (editable installs).
2. **Settings** dialog in the GUI — API keys and a few options; saved values override `.env`
   for both the GUI and the CLI.

API keys entered in Settings go to the OS keyring (Windows Credential Manager). If the keyring
is unavailable they fall back to plaintext `user_prefs.json`, and a warning is logged.

### Data directory

App data lives under `%APPDATA%\interview_copilot\`:

| Path | Purpose |
|------|---------|
| `context\` | Candidate profile `.md` files |
| `models\silero_vad.onnx` | VAD model (downloaded on first run) |
| `logs\copilot.log` | Rotating file log |
| `.copilot_state.json` | Active profile |
| `.privacy_state.json` | Privacy banner acknowledgement |
| `window_state.json` | Overlay geometry + last audio device |
| `user_prefs.json` | Settings-dialog prefs (secrets only if the keyring is unavailable) |

## Speech-to-text backends

Default is **local Whisper** (`STT_BACKEND=whisper`) — free, offline audio.
Cloud backends stay in the app for later; do not enable them unless you want to pay.

| `STT_BACKEND` | Notes |
|---------------|--------|
| `whisper` (**default**) | Local faster-whisper; $0; recommended now |
| `deepgram` | Streaming Nova-3 (paid); optional later |
| `groq` | Cloud Whisper via Groq; optional |
| `soniox` | Cloud STT; optional |

Set keys in `.env` or **Settings** only if you switch off Whisper. Keys changed in
Settings apply immediately (a running Deepgram stream reconnects with the new key);
changing the backend itself requires a restart.
With Whisper, profile terms still help via `initial_prompt` when
`STT_KEYTERMS_FROM_PROFILE=true`.

- **Dual-path** (`STT_DUAL_PATH_ENABLED=true`, default): a fast `tiny.en` model
  (`STT_PROVISIONAL_MODEL`) shows provisional text while you listen; the main model
  produces the final text sent to the LLM. Provisional STT turns itself off if it runs
  slower than real time.
- **Failover** (`STT_FALLBACK`, default `whisper`): if a cloud backend fails (network,
  HTTP, auth), transcription switches to the fallback and retries the primary after 60 s.
- `STT_CLOUD_ENDPOINTING` is deprecated and ignored: with Deepgram streaming, Deepgram
  always decides where utterances end.

Golden-set bench:

```bash
python scripts/generate_golden_audio.py
interview-copilot-cli stt-bench --backend whisper
interview-copilot-cli stt-bakeoff --out bench_bakeoff.json
```

## Audio capture

- **Loopback** (default): records what your speakers play, i.e. the interviewer.
- **Dual-channel** (`AUDIO_DUAL_CHANNEL=true` or Settings): also opens your microphone
  (`AUDIO_MIC_DEVICE`, auto-picked if empty). The mic is **not transcribed**; a second VAD
  only detects when you speak so loopback phrases are suppressed during that time
  (`AUDIO_SUPPRESS_LOOPBACK_WHILE_MIC=true`) — your own voice echoed from the speakers
  doesn't become an "interviewer question".
- **DSP** (`DSP_ENABLED=true`): mono mix, resample to 16 kHz, a light noise gate
  (`DSP_DENOISE_ENABLED`) and smoothed auto-gain (`DSP_NORMALIZE_ENABLED`) that keep state
  across 20 ms chunks.

## Run

```bash
# GUI (recommended)
interview-copilot
# or
python -m interview_copilot.gui.app

# CLI
interview-copilot-cli start              # same GUI
interview-copilot-cli run                # full pipeline, results printed to the console
interview-copilot-cli devices            # list loopback / mic devices
interview-copilot-cli profiles
interview-copilot-cli profile-set my_resume
```

Diagnostics (all accept `--device <id>`):

```bash
interview-copilot-cli capture-test       # live RMS level of the capture device
interview-copilot-cli vad-test           # speech start/end detection
interview-copilot-cli transcribe-test    # capture + VAD + STT, no LLM
```

## Profiles

1. Create `%APPDATA%\interview_copilot\context\my_resume.md` (or set `CONTEXT_DIR`).
2. Select it in the GUI combo, or run `profile-set my_resume`.
3. `example_profile.md` is never auto-selected.

## GUI hotkeys / controls

- **Pause / Resume** — stop/start capture
- **Clear** — clear cards and transcript history
- **Copy** on a card — copy AI suggestion
- **Settings** — API keys, STT backend, mic device, dual-path / dual-channel / denoise
- **Ctrl+Shift+H** — hide/show the overlay; registered system-wide, so it works while hidden
- **Tray icon** — click to show/hide; right-click → *Quit*
- **Escape** — hide the overlay (refused if neither the hotkey nor the tray is available)
- **✕** — quit
- Drag the title to move; resize via the bottom-right grip
- Overlay is excluded from screen capture on supported Windows builds
- Profile / device / Pause controls are disabled until models finish loading

## Translation

`TRANSLATION_BACKEND` in `.env`:

- `none` — default
- `deepl` — requires `DEEPL_API_KEY`
- `nllb` — local model (`pip install -e ".[nllb]"`); slow on CPU

## Low-latency transcription

The CPU profile uses `base.en`, 350 ms VAD silence and a 250 ms LLM debounce.
While speech is active, the app takes a bounded snapshot roughly every 800 ms
and transcribes it with the provisional model (dual-path):

- only one Whisper inference runs at a time, on the STT thread (never the capture thread);
- at most one pending partial snapshot is retained;
- provisional text updates one existing card;
- only final text enters conversation history and OpenRouter;
- final phrases take priority over stale partial snapshots;
- the LLM call waits for the speaker to finish, but at most
  `LLM_MAX_WAIT_FOR_SPEECH_END_S` (constant background noise can keep VAD "speaking").

Relevant `.env` settings:

```env
WHISPER_MODEL=base.en
STT_DUAL_PATH_ENABLED=true
STT_PROVISIONAL_MODEL=tiny.en
VAD_SILENCE_MS=350
LLM_DEBOUNCE_DELAY_S=0.25
LLM_MAX_WAIT_FOR_SPEECH_END_S=4.0
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
| "OpenRouter API key not set" | Enter it in **Settings** (applies immediately) or `OPENROUTER_API_KEY` in `.env` |
| "No candidate profile" | Create a `.md` under context dir (not only `example_profile`) and pick it |
| DeepL warning | Set `DEEPL_API_KEY` (Settings or `.env`) or set `TRANSLATION_BACKEND=none` |
| Slow first start | Whisper model download/warmup; VAD ONNX download |
| High CPU usage | Increase `PARTIAL_TRANSCRIPTION_INTERVAL_MS`, disable partials or dual-path |
| Partial text changes | Expected: provisional text is replaced by the final Whisper result |
| Overlay disappeared | Press **Ctrl+Shift+H** or click the tray icon |
| Own answers show up as questions | Enable dual-channel capture so mic speech suppresses loopback |
| "System keyring unavailable" in log | API keys are stored in plaintext `user_prefs.json` |

## Architecture

```
capture thread          STT thread              asyncio event loop
──────────────          ──────────              ──────────────────
audio device → DSP      PhraseQueues            transcript_queue
  → VAD phrases  ─────▶   finals first  ──────▶   cards (partial/final)
  → partial snapshots     → SttEngine             debounce + merge segments
  → (Deepgram stream) ───────────────────────▶    → translation + LLM stream
```

`src/interview_copilot/pipeline/`:

| Module | Responsibility |
|--------|----------------|
| `core.py` | `InterviewPipeline` facade: wiring, callbacks, profile, API-key reload, start/stop/pause |
| `capture.py` | Capture thread: audio device, VAD, mic echo suppression, streaming-STT feed, retries |
| `phrase_queues.py` | Hand-off between threads: bounded finals queue + latest-only partial mailbox |
| `stt_worker.py` | STT thread: transcribes finals (main engine) and partials (provisional engine) |
| `orchestrator.py` | Event loop: card upserts, debounce, merging segments, one LLM task at a time |
| `processing.py` | Per question: translation + streamed suggestion → `PipelineResult` |

Other packages: `audio/` (WASAPI capture, DSP, dual capture), `vad/` (Silero ONNX),
`stt/` (engines, factory, failover, dual-path), `suggestion/` (OpenRouter, profiles,
prompt sanitizing), `translation/`, `gui/` (PySide6 overlay), `cli/`, `eval/` (WER harness).

## Development

```bash
ruff check .
pytest tests/ -v
```

Tests run without audio hardware or network: pipeline components are injected with
fakes from `tests/fakes.py`.

Windows release zip is built in CI via PyInstaller (`interview_copilot.spec`). Torch/NLLB
are excluded from the bundle; use DeepL or `none` for translation in the packaged app.

## License

MIT — see [LICENSE](LICENSE).
