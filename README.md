# AI Interview Copilot

Приложение для транскрибации речи в реальном времени, перевода и генерации подсказок на собеседованиях.

## Основной стек и технологии

- **GUI**: PySide6 (Qt)
- **STT (Speech-to-Text)**: Faster-Whisper
- **Audio Capture**: WASAPI Loopback (PyAudioPatch / SoundCard)
- **VAD**: Silero VAD
- **Translation**: DeepL API / NLLB (HuggingFace)
- **Suggestions**: OpenRouter API (Claude, GPT, etc.)

## Запуск

1. Установите зависимости (требуется Python 3.10+):
   ```bash
   pip install -e .
   ```
2. Настройте конфигурацию в файле `.env` (см. `.env.example`).
3. Запустите GUI:
   ```bash
   python -m interview_copilot.gui.app
   ```
