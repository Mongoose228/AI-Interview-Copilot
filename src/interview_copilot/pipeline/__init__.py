"""Real-time interview pipeline: capture/VAD → STT → debounce → LLM suggestion."""

from ..config import config
from .core import InterviewPipeline

__all__ = ["InterviewPipeline", "config"]
