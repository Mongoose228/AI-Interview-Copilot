import uuid
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class SuggestionStatus(str, Enum):
    OK = "ok"
    SKIPPED_NO_PROFILE = "skipped_no_profile"
    SKIPPED_NO_KEY = "skipped_no_key"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    PENDING = "pending"


@dataclass(frozen=True)
class AudioChunk:
    id: uuid.UUID
    data: np.ndarray
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
    speaker: str = "interviewer"  # reserved for future diarization
    stt_started_at: float = 0.0
    stt_ended_at: float = 0.0


@dataclass(frozen=True)
class SuggestionResult:
    answer_en: str
    has_hedging: bool = False


@dataclass(frozen=True)
class StageTiming:
    stage_name: str
    started_at: float
    ended_at: float
    duration_s: float


@dataclass(frozen=True)
class ProfileSnapshot:
    name: str
    content_hash: str
    content: str
    loaded_at: float


@dataclass(frozen=True)
class PipelineResult:
    id: uuid.UUID
    transcript: str
    translation_ru: str | None
    suggestion: SuggestionResult | None
    profile: ProfileSnapshot | None
    timings: list[StageTiming] = field(default_factory=list)
    created_at: float = 0.0
    is_cancelled: bool = False
    answer_ru: str | None = None
    suggestion_status: SuggestionStatus = SuggestionStatus.PENDING
