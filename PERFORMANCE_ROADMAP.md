# Performance Roadmap

## Статус реализации

CPU-оптимизация **реализована** (defaults в `.env.example` / `config.py`):

| Параметр | Значение |
|----------|----------|
| `WHISPER_MODEL` | `base.en` |
| `VAD_SILENCE_MS` | 350 |
| `LLM_DEBOUNCE_DELAY_S` | 0.4 |
| Partial STT | вкл., интервал 800 мс, мин. аудио 700 мс |

Поведение pipeline:

- provisional и final обновляют одну карточку со стабильным ID;
- очередь partial хранит только последний snapshot; final имеет приоритет;
- в OpenRouter уходит только final-текст;
- в логах есть latency и RTF-метрики без текста (`[Latency]`, `[STT]`).

`WHISPER_CPU_THREADS=4` не менялся. GPU-путь не валидировался (нет NVIDIA GPU в среде разработки).

## Цель

Сократить задержку появления транскрипции с прежних ~3–5 с до ~1–2 с,
не ухудшая заметно качество распознавания технической речи.

## Источники задержки (до / после)

**Было (профиль 0.2.0 до latency-работ):**

- `VAD_SILENCE_MS=600` — ~0,6 с ожидания конца речи;
- Whisper `small.en` на CPU — обычно 1–3 с;
- `LLM_DEBOUNCE_DELAY_S=1.2` — карточка ждала debounce перед показом.

**Сейчас (текущие defaults):**

- VAD silence 350 мс;
- Whisper `base.en` + partial snapshots во время речи;
- debounce 400 мс только перед вызовом LLM; транскрипт показывается раньше.

Сетевой запрос OpenRouter влияет на AI-ответ, но не должен задерживать
показ уже распознанного текста.

## Этапы

### Этап 1 — быстрые настройки — сделано

Профиль в defaults:

```env
WHISPER_MODEL=base.en
VAD_SILENCE_MS=350
LLM_DEBOUNCE_DELAY_S=0.4
WHISPER_BEAM_SIZE=1
WHISPER_COMPUTE_TYPE=auto
```

Тюнинг при необходимости:

- если `base.en` всё ещё медленный — проверить `tiny.en`;
- не опускать `VAD_SILENCE_MS` сильно ниже 250–300 мс (риск резки пауз внутри вопроса);
- `WHISPER_CPU_THREADS=4` зафиксирован.

### Этап 2 — разделить транскрипцию и LLM debounce — сделано

1. Результат Whisper виден сразу после STT (в т.ч. provisional).
2. Фрагменты накапливаются во внутреннем буфере.
3. Debounce только перед merge вопроса и вызовом OpenRouter.
4. Одна предварительная карточка вместо «Thinking...» на каждый фрагмент.
5. AI-ответ стримится в ту же карточку.

### Этап 3 — ускорение на GPU — не применялось

При наличии совместимой NVIDIA GPU проверить:

```env
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
```

Нужны совместимый GPU build CTranslate2 и CUDA-библиотеки. Сравнить время STT,
VRAM и стабильность с CPU `int8`.

### Этап 4 — частичная транскрипция — сделано

- распознавание накопленного аудио ~каждые 0,5–1 с во время речи;
- provisional текст на карточке, final после VAD end;
- один inference за раз, latest-only mailbox для partial;
- в LLM только финализированный объединённый вопрос.

## Метрики и критерии готовности

Метрики в логах уже пишутся:

- `speech_end -> STT_result`;
- `STT_result -> transcript_visible`;
- `speech_end -> first_LLM_token`;
- real-time factor Whisper;
- модель, устройство и compute type.

Целевые значения (текущий путь — **local Whisper**):

- транскрипт виден ≤ 1,5 с после конца короткого вопроса (loopback);
- первый AI-токен ≤ 2,5 с при нормальной сети;
- tech-термины улучшаются через `initial_prompt` из профиля.

Опционально позже (платный cloud, не используем сейчас):

- Deepgram streaming: interim ≤ 400–600 мс; final ≤ 700–900 мс.

### Baseline table

| Setup | Platform | mean latency | mean WER | Notes |
|-------|----------|--------------|----------|-------|
| whisper base.en CPU (**current**) | SAPI TTS golden | **~0.61 s** | **~0.49** | measured 2026-10-06; keep as default |
| whisper base.en CPU | Zoom/Meet live | _TBD_ | _TBD_ | next real validation |
| deepgram / groq / soniox | — | — | — | code ready; **not in use** until you opt in |

Harness:

```bash
python scripts/generate_golden_audio.py
interview-copilot-cli stt-bench --backend whisper --out bench_whisper.json
interview-copilot-cli stt-bakeoff --out bench_bakeoff.json
```

## Что осталось проверить

1. Latency на реальном Zoom / Meet / Teams с **локальным Whisper**.
2. Тюнинг модели (`small.en` на GPU vs `base.en` на CPU) по harness.
3. A/B denoise on/off на реальных клипах.
4. Cloud STT — только если позже решите платить и смените `STT_BACKEND`.

См. также `CHANGELOG.md` (0.3.0).
