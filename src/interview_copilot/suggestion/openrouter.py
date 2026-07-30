import json

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from ..config import config
from ..logging_config import logger
from ..models import ProfileSnapshot, SuggestionResult, Transcript


class OpenRouterResponseFormat(BaseModel):
    answer_en: str = Field(description="Suggested answer in English")
    answer_ru: str = Field(description="Translation of the suggested answer in Russian")
    needs_verification: bool = Field(
        description="True if the AI is not confident and user should double check"
    )


def _build_strict_schema() -> dict:
    """Build JSON schema with additionalProperties: false for strict mode."""
    schema = OpenRouterResponseFormat.model_json_schema()
    schema["additionalProperties"] = False
    return schema


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
            "Output your answer ONLY in the requested JSON format. Keep the answer concise (2-3 sentences max)."
        )
        return prompt

    async def get_suggestion(
        self, transcript_history: list[Transcript], profile: ProfileSnapshot
    ) -> SuggestionResult | None:
        if not self._client:
            return None

        # Build context from the last N transcripts
        # If there is no history, or the latest is empty, skip
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
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=1000,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "suggestion_schema",
                        "schema": _build_strict_schema(),
                        # Removed strict=True for better model compatibility
                    },
                },
            )

            result_text = response.choices[0].message.content
            if result_text:
                # Some models might wrap JSON in markdown block even with response_format
                result_text = result_text.strip()
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                if result_text.startswith("```"):
                    result_text = result_text[3:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                result_text = result_text.strip()

                data = json.loads(result_text)
                return SuggestionResult(
                    answer_en=data.get("answer_en", ""),
                    answer_ru=data.get("answer_ru", ""),
                    needs_verification=data.get("needs_verification", False),
                )

            return None

        except Exception as e:
            logger.error(f"OpenRouter Suggestion failed: {e}")
            return SuggestionResult(
                answer_en="[Error fetching AI suggestion]",
                answer_ru="[Ошибка получения подсказки]",
                needs_verification=True
            )
