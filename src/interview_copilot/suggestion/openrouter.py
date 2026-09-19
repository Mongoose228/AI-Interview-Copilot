import re
from collections.abc import Awaitable, Callable

import httpx
from openai import AsyncOpenAI

from ..config import config
from ..logging_config import logger
from ..models import ProfileSnapshot, SuggestionResult, Transcript
from .sanitize import sanitize_for_prompt

# Explicit uncertainty phrases only (word-boundary aware)
_HEDGING_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bi'?m not sure\b",
        r"\bi am not sure\b",
        r"\bnot entirely sure\b",
        r"\bnot 100%\b",
        r"\bi think\b",
        r"\bmight be\b",
        r"\bcould be\b",
        r"\bmaybe\b",
        r"\bperhaps\b",
        r"\bdouble[- ]check\b",
        r"\bit'?s possible\b",
    ]
]


def _detect_needs_verification(text: str) -> bool:
    for pattern in _HEDGING_PATTERNS:
        if pattern.search(text):
            return True
    return False


class OpenRouterSuggester:
    def __init__(self):
        self._api_key = config.OPENROUTER_API_KEY
        self._base_url = config.OPENROUTER_BASE_URL
        self._model = config.OPENROUTER_MODEL
        self._client = None

        if not self._api_key:
            logger.warning(
                "OpenRouter API key not configured. Suggestions will be disabled."
            )
        else:
            timeout = httpx.Timeout(
                connect=config.NETWORK_CONNECT_TIMEOUT,
                read=config.NETWORK_READ_TIMEOUT,
                write=config.NETWORK_CONNECT_TIMEOUT,
                pool=config.NETWORK_CONNECT_TIMEOUT,
            )
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=timeout,
                max_retries=2,
            )
            logger.info(f"OpenRouter Suggester initialized with model {self._model}.")

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    async def warm_up(self):
        """Pre-warm the TCP+TLS connection to avoid cold-start latency on first real request."""
        if not self._client:
            return
        try:
            await self._client.models.list()
            logger.info("OpenRouter connection pre-warmed.")
        except Exception as e:
            logger.warning(f"OpenRouter warm-up failed (non-fatal): {e}")

    def _build_system_prompt(self, profile: ProfileSnapshot) -> str:
        safe_content = sanitize_for_prompt(profile.content)
        prompt = (
            "You are an AI Interview Copilot assisting a candidate"
            " during a technical interview.\n"
            "Below is the candidate's profile. Use this to provide"
            " relevant and personalized answers.\n\n"
            f"--- CANDIDATE PROFILE ---\n{safe_content}\n-------------------------\n\n"
            "STRICT RULES:\n"
            "1. NEVER invent companies, projects, achievements, or experience"
            " that are NOT in the candidate's profile above.\n"
            "2. If the profile does not contain enough information to answer"
            " confidently, say so explicitly.\n"
            "3. If a question is outside the candidate's stated expertise,"
            " provide a general technical answer and note that it's general knowledge.\n"
            "4. Keep your answer brief (2-4 sentences), professional, and accurate.\n"
            "5. Output your answer as plain text in English only.\n"
            "6. Do NOT use markdown formatting, markdown blocks, or JSON."
        )
        return prompt

    async def get_suggestion(
        self,
        transcript_history: list[Transcript],
        profile: ProfileSnapshot,
        stream_callback: Callable[[str], Awaitable[None]] | None = None
    ) -> SuggestionResult | None:
        if not self._client:
            return None

        if not transcript_history or not profile:
            return None

        recent_transcripts = transcript_history[-5:]

        context_lines = []
        for i, t in enumerate(recent_transcripts):
            safe_text = sanitize_for_prompt(t.text_en)
            if not safe_text:
                continue
            if i == len(recent_transcripts) - 1:
                context_lines.append(f'Current question (Interviewer): "{safe_text}"')
            else:
                context_lines.append(f'- Interviewer: "{safe_text}"')

        context = "\n".join(context_lines)

        if not context.strip():
            return None

        system_prompt = self._build_system_prompt(profile)
        user_prompt = (
            f"Conversation context:\n{context}\n\n"
            f"Provide a suggested response to the CURRENT question only."
        )

        try:
            response_stream = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": system_prompt,
                                "cache_control": {"type": "ephemeral"}
                            }
                        ]
                    },
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=config.MAX_LLM_TOKENS,
                stream=True,
                extra_body={"provider": {"sort": "throughput"}},
            )

            full_text = ""
            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    token = chunk.choices[0].delta.content
                    full_text += token
                    if stream_callback:
                        await stream_callback(token)

            full_text = full_text.strip()
            if not full_text:
                return None

            return SuggestionResult(
                answer_en=full_text,
                has_hedging=_detect_needs_verification(full_text)
            )

        except Exception as e:
            logger.error(f"OpenRouter Suggestion failed: {e}")
            return None
