from collections.abc import Awaitable, Callable

import httpx
from openai import AsyncOpenAI

from ..config import config
from ..logging_config import logger
from ..models import ProfileSnapshot, SuggestionResult, Transcript

_HEDGING_TERMS = {
    "i'm not sure", "i am not sure", "i think", "might be", "could be", 
    "maybe", "perhaps", "verify", "double check", "double-check", 
    "not 100%", "not entirely sure", "it's possible"
}

def _detect_needs_verification(text: str) -> bool:
    lower_text = text.lower()
    for term in _HEDGING_TERMS:
        if term in lower_text:
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

    def _build_system_prompt(self, profile: ProfileSnapshot) -> str:
        prompt = (
            "You are an AI Interview Copilot assisting a candidate during a technical interview.\n"
            "Below is the candidate's profile. Use this to provide relevant and personalized answers.\n\n"
            f"--- CANDIDATE PROFILE ---\n{profile.content}\n-------------------------\n\n"
            "Your task is to provide a brief, professional, and accurate response to the interviewer's question.\n"
            "Output your answer as plain text in English. Keep the answer concise (2-3 sentences max).\n"
            "Do NOT use markdown formatting, markdown blocks, or JSON."
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

        # Build context from the last N transcripts
        if not transcript_history:
            return None

        recent_transcripts = transcript_history[-5:]  # Last 5 phrases
        
        context_lines = []
        for i, t in enumerate(recent_transcripts):
            if not t.text_en:
                continue
            if i == len(recent_transcripts) - 1:
                context_lines.append(f'Current question: "{t.text_en}"')
            else:
                context_lines.append(f'- Interviewer: "{t.text_en}"')
        
        context = "\n".join(context_lines)

        if not context.strip():
            return None

        system_prompt = self._build_system_prompt(profile)
        user_prompt = f'Context of conversation:\n{context}\n\nSuggest a response.'

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
                max_tokens=1000,
                stream=True,
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

            needs_verify = _detect_needs_verification(full_text)

            return SuggestionResult(
                answer_en=full_text,
                needs_verification=needs_verify
            )

        except (RuntimeError, ValueError, TypeError, OSError) as e:
            logger.error(f"OpenRouter Suggestion failed: {e}")
            return None

