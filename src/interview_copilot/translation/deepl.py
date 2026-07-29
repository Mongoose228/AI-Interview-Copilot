import deepl

from ..config import config
from ..logging_config import logger
from .base import Translator


class DeepLTranslator(Translator):
    def __init__(self):
        self._enabled = config.LOCAL_TRANSLATION_ENABLED
        self._api_key = config.DEEPL_API_KEY
        self._translator = None

        if self._enabled:
            if not self._api_key:
                logger.warning("DeepL is enabled but DEEPL_API_KEY is not set.")
            else:
                try:
                    self._translator = deepl.Translator(self._api_key)
                    logger.info("DeepL Translator initialized successfully.")
                except Exception as e:
                    logger.error(f"Failed to initialize DeepL: {e}")
                    self._translator = None

    def translate(self, text: str, source_lang: str = "EN", target_lang: str = "RU") -> str | None:
        if not self._enabled or not self._translator or not text:
            return None

        try:
            # DeepL uses 'EN' and 'RU'
            result = self._translator.translate_text(
                text, source_lang=source_lang.upper(), target_lang=target_lang.upper()
            )
            return result.text
        except deepl.exceptions.DeepLException as e:
            logger.error(f"DeepL translation error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected translation error: {e}")
            return None
