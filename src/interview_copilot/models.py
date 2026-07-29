import uuid
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class AudioChunk:
    id: uuid.UUID
    data: bytes
    sample_rate: int
    channels: int
    captured_at: float


@dataclass(frozen=True)
class SpeechPhrase:
    id: uuid.UUID
    audio_data: np.ndarray
    duration_s: float
    captured_at: float
    vad_end_at: float


@dataclass(frozen=True)
class Transcript:
    phrase_id: uuid.UUID
    text_en: str
    language: str
    confidence: float
    stt_duration_s: float


@dataclass(frozen=True)
class SuggestionResult:
    answer_en: str
    answer_ru: str
    needs_verification: bool


@dataclass(frozen=True)
class StageTiming:
    stage_name: str
    started_at: float
    ended_at: float
    duration_s: float


@dataclass(frozen=True)
class StageError:
    stage_name: str
    error_type: str
    message: str
    recoverable: bool


@dataclass(frozen=True)
class ProfileSnapshot:
    name: str
    content_hash: str
    content: str
    version: int
    loaded_at: float


@dataclass(frozen=True)
class PipelineResult:
    id: uuid.UUID
    transcript: str
    translation_ru: str | None
    suggestion: SuggestionResult | None
    profile: ProfileSnapshot | None
    timings: list[StageTiming] = field(default_factory=list)
    errors: list[StageError] = field(default_factory=list)
    created_at: float = 0.0
