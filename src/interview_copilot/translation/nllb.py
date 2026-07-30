from ..config import config
from ..logging_config import logger
from .base import Translator


class NLLBTranslator(Translator):
    def __init__(self):
        self._model_name = config.NLLB_MODEL
        self._tokenizer = None
        self._model = None
        self._ready = False

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            logger.info(f"Initializing NLLB model '{self._model_name}'...")

            # NLLB translation (EN -> RU)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, src_lang="eng_Latn"
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
            self._ready = True

            logger.info("NLLB Translator initialized successfully.")
        except ImportError:
            logger.error("transformers or torch not installed. Cannot use NLLB.")
        except Exception as e:
            logger.error(f"Failed to initialize NLLB: {e}")

    def translate(self, text: str, source_lang: str = "EN", target_lang: str = "RU") -> str | None:
        if not self._ready or not self._model or not self._tokenizer or not text:
            return None

        # DeepL used EN/RU. NLLB uses eng_Latn/rus_Cyrl.
        # We only really support EN -> RU in this MVP
        if target_lang != "RU" and target_lang != "rus_Cyrl":
            logger.warning(
                f"NLLB currently hardcoded for EN->RU. Ignoring request for {target_lang}"
            )
            return None

        try:
            inputs = self._tokenizer(text, return_tensors="pt")

            # target language token for RU
            forced_bos_token_id = self._tokenizer.lang_code_to_id["rus_Cyrl"]

            translated_tokens = self._model.generate(
                **inputs, forced_bos_token_id=forced_bos_token_id, max_length=200
            )

            result = self._tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            return result
        except Exception as e:
            logger.error(f"NLLB translation error: {e}")
            return None
