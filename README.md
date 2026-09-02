# AI Interview Copilot

Приложение для транскрибации речи в реальном времени, перевода и генерации подсказок на собеседованиях.

## Основной стек и технологии

- **GUI**: PySide6 (Qt)
- **STT (Speech-to-Text)**: Faster-Whisper
- **Audio Capture**: WASAPI Loopback (SoundCard)
- **VAD**: Silero VAD
- **Translation**: DeepL API / NLLB (HuggingFace)
- **Suggestions**: OpenRouter API (Claude, GPT, Gemini, etc.)

## Установка и Запуск

1. Установите базовые зависимости (требуется Python 3.11+):
   ```bash
   pip install -e .
   ```
   *(Опционально)* Для локального перевода через NLLB:
   ```bash
   pip install -e ".[nllb]"
   ```
   *(Опционально)* Для разработки и тестирования:
   ```bash
   pip install -e ".[dev]"
   ```

2. Настройте конфигурацию в файле `.env` (см. `.env.example`).
3. Запустите GUI:
   ```bash
   python -m interview_copilot.gui.app
   ```
   Или через CLI:
   ```bash
   python -m interview_copilot.cli.main start
   ```

## Использование CLI (Отладка и Тестирование)

Проект включает удобную CLI-утилиту для проверки и отладки отдельных компонентов (аудио, VAD, STT).

Доступные команды:
- `python -m interview_copilot.cli.main devices` — список доступных аудиоустройств (Loopback).
- `python -m interview_copilot.cli.main capture-test` — проверка захвата звука (показывает уровень громкости).
- `python -m interview_copilot.cli.main vad-test` — тест детектора активности голоса (VAD).
- `python -m interview_copilot.cli.main transcribe-test` — тест всего пайплайна распознавания (Capture -> VAD -> STT).
- `python -m interview_copilot.cli.main run` — запуск полного пайплайна в консоли (без GUI).

К любой из команд (кроме `devices`) можно добавить флаг `--device <ID>`, чтобы явно указать звуковое устройство, если автоматический выбор не срабатывает.
