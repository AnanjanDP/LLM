from abc import ABC, abstractmethod
from typing import List, Dict, Generator, Any
import time
from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import LLMProviderError

try:
    from groq import Groq
except ImportError:
    Groq = None


class BaseLLMProvider(ABC):
    """Abstract interface for LLM provider adapters."""

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        pass

    @abstractmethod
    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> Generator[str, None, None]:
        pass


class MockLLMProvider(BaseLLMProvider):
    """Fallback LLM provider used when external API keys are invalid/missing or for offline testing."""

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        last_msg = messages[-1]["content"] if messages else ""
        return (
            f"[AI Assistant Platform Response]\n\n"
            f"I analyzed your request regarding: '{last_msg[:120]}...'\n\n"
            "Here is the context-grounded explanation based on retrieved knowledge base records. "
            "To connect live Groq API completions, ensure your `GROQ_API_KEY` is configured in `backend/app/.env`."
        )

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> Generator[str, None, None]:
        full_text = self.generate(messages, model, temperature, max_tokens)
        words = full_text.split(" ")
        for word in words:
            yield word + " "
            time.sleep(0.03)


class GroqProvider(BaseLLMProvider):
    """Groq API provider adapter with automatic model detection & fallback."""

    FALLBACK_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]

    def __init__(self, api_key: str = None):
        key = api_key or settings.GROQ_API_KEY
        if not key:
            raise LLMProviderError("Groq API Key missing.")
        if Groq is None:
            raise LLMProviderError("groq python package not installed.")
        self.client = Groq(api_key=key)
        self.default_model = settings.DEFAULT_LLM_MODEL
        self.fallback = MockLLMProvider()

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        requested = model or self.default_model
        models_to_try = [requested] + [m for m in self.FALLBACK_MODELS if m != requested]
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
        max_tok = max_tokens or settings.LLM_MAX_TOKENS

        for target_model in models_to_try:
            try:
                logger.info(f"Invoking Groq API model '{target_model}'")
                response = self.client.chat.completions.create(
                    model=target_model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tok,
                    timeout=settings.LLM_TIMEOUT,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                err_str = str(e).lower()
                if "invalid_api_key" in err_str or "401" in err_str or "authenticationerror" in err_str:
                    logger.warning("Groq API Key invalid/expired. Falling back to Mock provider.")
                    return self.fallback.generate(messages, model, temperature, max_tokens)
                elif any(k in err_str for k in ("model_decommissioned", "model_not_found", "404", "400", "does not exist", "decommissioned")):
                    logger.warning(f"Groq model '{target_model}' is decommissioned or unavailable ({e}). Trying next fallback model...")
                    continue
                else:
                    logger.error(f"Groq generation error with '{target_model}': {e}")
                    # Try next model before throwing error
                    continue

        # If all model attempts fail, return fallback response gracefully
        return self.fallback.generate(messages, model, temperature, max_tokens)

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> Generator[str, None, None]:
        requested = model or self.default_model
        models_to_try = [requested] + [m for m in self.FALLBACK_MODELS if m != requested]
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
        max_tok = max_tokens or settings.LLM_MAX_TOKENS

        for target_model in models_to_try:
            try:
                logger.info(f"Invoking Groq stream model '{target_model}'")
                stream = self.client.chat.completions.create(
                    model=target_model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tok,
                    stream=True,
                    timeout=settings.LLM_TIMEOUT,
                )
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return
            except Exception as e:
                err_str = str(e).lower()
                if "invalid_api_key" in err_str or "401" in err_str or "authenticationerror" in err_str:
                    logger.warning("Groq API Key invalid/expired. Streaming fallback.")
                    for token in self.fallback.generate_stream(messages, model, temperature, max_tokens):
                        yield token
                    return
                elif any(k in err_str for k in ("model_decommissioned", "model_not_found", "404", "400", "does not exist", "decommissioned")):
                    logger.warning(f"Groq stream model '{target_model}' decommissioned/unavailable. Trying next fallback...")
                    continue
                else:
                    logger.error(f"Groq stream error with '{target_model}': {e}")
                    continue

        # Fallback to mock streaming if all models fail
        for token in self.fallback.generate_stream(messages, model, temperature, max_tokens):
            yield token


class LLMProviderFactory:
    """Factory for selecting and instantiating active LLM provider."""

    @staticmethod
    def get_provider(provider_name: str = None) -> BaseLLMProvider:
        name = (provider_name or settings.DEFAULT_LLM_PROVIDER).lower()

        if name == "mock":
            return MockLLMProvider()

        if name == "groq" and settings.GROQ_API_KEY and not settings.GROQ_API_KEY.startswith("your_"):
            try:
                return GroqProvider()
            except Exception as e:
                logger.warning(f"Failed to initialize GroqProvider ({e}), falling back to MockLLMProvider")
                return MockLLMProvider()

        return MockLLMProvider()


llm_factory = LLMProviderFactory()
