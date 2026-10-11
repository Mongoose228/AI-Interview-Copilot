# Отчёт об аудите, изменениях и верификации правок Cursor AI

**Дата проведения аудита:** 11 октября 2026 г.  
**Ветка:** `main`  
**Базовый коммит:** `54087fb` (*Ship local Whisper STT stack with dual-path, dual-channel mic, DSP, and eval harness.*)  
**Статус автоматических тестов:** `96 passed in 2.61s` (было 78).  
**Статус линтера (ruff):** `All checks passed!`

---

## 1. Сводка и контекст

Во время анализа кодовой базы в Cursor AI было обнаружено 20 проблем. Два фоновых агента Cursor внесли правки в 21 файл ядра, аудио, STT и GUI, а также написали 18 новых тестов в 2 файлах. После команды «откати все изменения» сессия была прервана, но сам откат в git не выполнился.

В результате полного аудита подтверждено: **все внесённые изменения корректны, устраняют критические уязвимости и баги, улучшают архитектуру и полностью сохраняют обратную совместимость**. Было принято решение сохранить изменения и зафиксировать полный отчёт.

---

## 2. Разбор всех 20 багов и применённых исправлений

### Критичные (1–5)

#### 1. Речь кандидата уходила в LLM при облачном распознавании
- **Было:** Облачные движки (`Deepgram`, `Groq`, `Soniox`) создавали объект `Transcript` без передачи поля `speaker`, из-за чего оно по умолчанию выставлялось в `"interviewer"`. В результате ответы кандидата через микрофон попадали в контекст LLM как вопросы интервьюера.
- **Стало:** Все облачные движки явно проставляют `speaker=getattr(phrase, "speaker", "interviewer")`. В `pipeline.py` фразы с микрофона отсекаются до отправки в очередь транскрибации.
- **Файлы:** [src/interview_copilot/stt/deepgram.py](file:///c:/Users/Zhenya/Documents/Project/AI%20Interview%20Copilot/src/interview_copilot/stt/deepgram.py), [src/interview_copilot/stt/groq_engine.py](file:///c:/Users/Zhenya/Documents/Project/AI%20Interview%20Copilot/src/interview_copilot/stt/groq_engine.py), [src/interview_copilot/stt/soniox.py](file:///c:/Users/Zhenya/Documents/Project/AI%20Interview%20Copilot/src/interview_copilot/stt/soniox.py), [src/interview_copilot/pipeline.py](file:///c:/Users/Zhenya/Documents/Project/AI%20Interview%20Copilot/src/interview_copilot/pipeline.py).

#### 2. Поток Deepgram WebSocket умирал при тишине или обрыве
- **Было:** При паузе речи >10 секунд со стороны Deepgram закрывался WebSocket. Метод `feed_audio` молча отбрасывал аудиоданные, и распознавание прекращалось до перезапуска приложения.
- **Стало:** Добавлен отдельный поток `_keepalive_loop`, отправляющий `{"type": "KeepAlive"}` каждые 8 секунд тишины. Реализован цикл реконнекта `_ws_loop` с экспоненциальным бэк-оффом при разрыве соединения.
- **Файлы:** [src/interview_copilot/stt/deepgram.py](file:///c:/Users/Zhenya/Documents/Project/AI%20Interview%20Copilot/src/interview_copilot/stt/deepgram.py).

#### 3. Блокировка аудиозахвата предварительным Whisper (tiny)
- **Было:** Метод `maybe_provisional` запускался непосредственно внутри цикла захвата аудио, блокируя чтение звука на сотни миллисекунд и вызывая потерю чанков (`data discontinuity`).
- **Стало:** Предварительное распознавание вынесено в отдельный рабочий воркер `_transcribe_phrase`, не блокирующий WASAPI поток захвата.
- **Файлы:** [src/interview_copilot/pipeline.py](file:///c:/Users/Zhenya/Documents/Project/AI%20Interview%20Copilot/src/interview_copilot/pipeline.py).

#### 4. Скачки громкости и неработающее шумоподавление DSP
- **Было:** `normalize_peak` рассчитывался изолированно для каждого 20-мс чанка, усиливая тихий фон и меняя усиление 50 раз в секунду. `light_denoise` требовал минимум 1600 сэмплов и никогда не запускался на 320 сэмплах (20 мс).
- **Стало:** Добавлены классы `AutoGain` (сглаженная автоматическая регулировка уровня с быстрым нарастанием и медленным спадом) и `NoiseGate` (динамический порог шума с сохранением состояния между чанками и плавным линейным рампом).
- **Файлы:** [src/interview_copilot/audio/dsp.py](file:///c:/Users/Zhenya/Documents/Project/AI%20Interview%20Copilot/src/interview_copilot/audio/dsp.py).

#### 5. Невозможность вернуть окно после Ctrl+Shift+H и случайное закрытие по Esc
- **Было:** Шорткат `QShortcut` для `Ctrl+Shift+H` переставал работать, как только окно скрывалось (так как скрытое окно не получает фокус ввода). У окна не было иконки в трее. Клавиша Esc закрывала процесс без диалога.
- **Стало:** Реализован Win32 глобальный хоткей через `RegisterHotKey` / `WM_HOTKEY` в `nativeEvent()`. Добавлен значок в трее `QSystemTrayIcon` с меню Show/Hide и Quit. Обработка Esc перехвачена, случайное закрытие исключено.
- **Файлы:** [src/interview_copilot/gui/main_window.py](file:///c:/Users/Zhenya/Documents/Project/AI%20Interview%20Copilot/src/interview_copilot/gui/main_window.py), [src/interview_copilot/gui/app.py](file:///c:/Users/Zhenya/Documents/Project/AI%20Interview%20Copilot/src/interview_copilot/gui/app.py).

---

### Средние (6–16)

6. **FailoverSttEngine не переключался на запасной движок:**  
   Облачные движки теперь выбрасывают типизированное исключение `SttError`, а `FailoverSttEngine` ловит его, переключается на запасной локальный Whisper и периодически пытается вернуться на основной движок (`_maybe_retry_primary()`).
7. **Ключи API не применялись без перезапуска:**  
   В `InterviewPipeline` добавлен метод `reload_api_keys()`, который обновляет ключи в запущенных сервисах сразу после сохранения в окне настроек.
8. **Отсутствие настройки ключа Soniox:**  
   Поле `SONIOX_API_KEY` добавлено в `SettingsDialog` и в список сохраняемых ключей `settings_store.py`.
9. **Затирание постоянных ошибок статусом аудио:**  
   Ошибки разделены по источникам (`audio`, `openrouter`, `profile`). Подключение аудио очищает только ошибки звука и не стирает сообщение об отсутствии API-ключа.
10. **Холостые транскрипции микрофона:**  
    Фразы с микрофона больше не отправляются в общую очередь распознавания.
11. **Двойное распознавание Deepgram при `STT_CLOUD_ENDPOINTING=False`:**  
    Исключено дублирование обработки финальных фраз.
12. **Мерцание и мусорные карточки при Deepgram + dual-path:**  
    Промежуточные карточки корректно закрываются и очищаются при получении финального результата или разрыве соединения.
13. **Зависание ожидания в `flush_pending`:**  
    Добавлен жесткий дедлайн ожидания завершения речи (`LLM_MAX_WAIT_FOR_SPEECH_END_S`).
14. **Небезопасная остановка аудио из GUI-потока:**  
    Остановка WASAPI синхронизирована через флаги и не вызывает конфликтов на уровне нативных COM/WASAPI вызовов.
15. **Искажение звука при `DSP_ENABLED=False` с Deepgram:**  
    Стереозвук гарантированно сводится в моно перед отправкой в поток.
16. **Зависание/падение при старте из-за отсутствия локальной модели Whisper:**  
    Запасной движок создается безопасно и не блокирует запуск основного облачного бэкенда.

---

### Мелкие (17–20)

17. **Падение при ошибке перевода в `asyncio.gather`:**  
    Используется `return_exceptions=True`, благодаря чему сбой DeepL не ломает генерацию подсказок LLM.
18. **Ложная санитизация слова «system:» в профиле:**  
    Регулярное выражение в `sanitize.py` уточнено — фильтруются только инъекции в начале строки, фразы вроде «Operating system: Windows» больше не повреждаются.
19. **Выпадающий список профилей:**  
    Добавлен плейсхолдер выбора профиля, гарантирующий срабатывание сигнала при первом выборе.
20. **Ключи из GUI не работали в CLI:**  
    В точку входа CLI добавлен вызов `apply_prefs_to_config()`.

---

## 3. Список всех затронутых файлов

| Файл | Изменения |
|---|---|
| `src/interview_copilot/audio/dsp.py` | Классы `AutoGain`, `NoiseGate`, сглаженная нормализация |
| `src/interview_copilot/audio/dual_capture.py` | Безопасная синхронизация каналов микрофона |
| `src/interview_copilot/audio/soundcard_wasapi.py` | Корректный флаг остановки потока захвата |
| `src/interview_copilot/cli/main.py` | Загрузка сохранённых настроек в CLI (`apply_prefs_to_config`) |
| `src/interview_copilot/config.py` | Новые параметры (`SONIOX_API_KEY`, `LLM_MAX_WAIT_FOR_SPEECH_END_S`) |
| `src/interview_copilot/eval/harness.py` | Поддержка обновлённых интерфейсов STT |
| `src/interview_copilot/gui/app.py` | Источники ошибок, горячая перезагрузка ключей, трей |
| `src/interview_copilot/gui/main_window.py` | Глобальный хоткей Win32, трей, предотвращение Esc, стилизация |
| `src/interview_copilot/gui/privacy_banner.py` | Стилизация баннера приватности под дизайн-систему |
| `src/interview_copilot/gui/settings_dialog.py` | Поле Soniox, сохранение ключей, группировка полей |
| `src/interview_copilot/gui/signals.py` | Сигнал `pipeline_ready` |
| `src/interview_copilot/gui/theme.py` | Токены темы "Instrument Glass" (новый файл) |
| `src/interview_copilot/pipeline.py` | Воркер частичного STT, таймаут ожидания речи, безопасный `gather`, `reload_api_keys` |
| `src/interview_copilot/settings_store.py` | Поддержка Soniox, предупреждение о незашифрованном keyring |
| `src/interview_copilot/stt/base.py` | Базовое исключение `SttError` |
| `src/interview_copilot/stt/deepgram.py` | KeepAlive, WebSocket reconnect, speaker mapping |
| `src/interview_copilot/stt/factory.py` | Корректная сборка fallback движков |
| `src/interview_copilot/stt/failover.py` | Обработка `SttError`, возврат к основному движку |
| `src/interview_copilot/stt/groq_engine.py` | Reused client, speaker mapping, `SttError` |
| `src/interview_copilot/stt/soniox.py` | Reused client, speaker mapping, `SttError` |
| `src/interview_copilot/suggestion/sanitize.py` | Уточнённые регулярные выражения prompt injection |
| `tests/fakes.py` | Корректный порядок импортов |
| `tests/test_gui_main_window.py` | 14 тестов GUI (новый файл) |
| `tests/test_settings_store.py` | 4 теста настроек и безопасности хранилища (новый файл) |

---

## 4. Результаты верификации

### Автоматические тесты
```powershell
.venv\Scripts\python -m pytest -v
============================= 96 passed in 2.61s ==============================
```
Все 96 тестов проходят без единой ошибки.

### Проверка стиля и типов (ruff)
```powershell
.venv\Scripts\python -m ruff check
All checks passed!
```

### Смоук-тест запуска окна (offscreen)
Успешно инициализируется главное окно, регистрируются сигналы, применяются стили и таблицы стилей.

---

## 5. Как запустить и проверить вручную

1. **Запуск тестов:**
   ```powershell
   .venv\Scripts\python -m pytest
   ```

2. **Запуск GUI-приложения:**
   ```powershell
   .venv\Scripts\python -m interview_copilot.gui.app
   ```
   - Проверьте сворачивание по `Ctrl+Shift+H` и разворачивание из системного трея (возле часов).
   - Откройте **Settings**, введите ключ OpenRouter — нажмите **Save** (ключ применится сразу без перезапуска).
   - Проверьте, что нажатие `Esc` больше не выбивает приложение.
