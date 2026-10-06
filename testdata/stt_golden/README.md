# STT golden set

Place WAV clips (16 kHz mono float/PCM16) here and list them in `manifest.json`.

Do **not** commit personal interview recordings with PII. Use synthetic clips
from `scripts/generate_golden_audio.py` or your own anonymized samples.

Each manifest entry:

```json
{
  "id": "tech_k8s_01",
  "audio": "tech_k8s_01.wav",
  "reference": "How would you deploy a Kubernetes service with PostgreSQL?",
  "tags": ["tech", "short"]
}
```

Run the harness:

```bash
interview-copilot-cli stt-bench --backend whisper
```
