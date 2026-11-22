from abc import ABC, abstractmethod
from typing import List, Dict, Generator
import time
from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import LLMProviderError

try:
    import groq
    from groq import Groq
except ImportError:
    groq = None
    Groq = None


class BaseLLMProvider(ABC):
    """Abstract interface for LLM provider adapters."""

    name = "base"

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
    """Deterministic offline provider. Used only when asked for ("mock") or when no API key is configured."""

    name = "mock"

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        last_msg = messages[-1]["content"] if messages else ""
        return (
            f"[Mock provider: no language model was called]\n\n"
            f"Your request: '{last_msg[:120]}...'\n\n"
            "Set GROQ_API_KEY in backend/app/.env to get real answers."
        )

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> Generator[str, None, None]:
        for word in self.generate(messages, model, temperature, max_tokens).split(" "):
            yield word + " "
            time.sleep(0.01)


def _is_model_unavailable(exc: Exception) -> bool:
    """The requested model does not exist, was retired, or the request was rejected for that model."""
    if groq is None:
        return False
    return isinstance(exc, (groq.NotFoundError, groq.BadRequestError))


def _is_retryable_on_other_model(exc: Exception) -> bool:
    """A different model may succeed: it is unavailable, or this model's own quota is exhausted."""
    return _is_model_unavailable(exc) or (groq is not None and isinstance(exc, groq.RateLimitError))


class GroqProvider(BaseLLMProvider):
    """Groq API provider with fallback across models.

    Failures are surfaced, never disguised: an invalid key, a network error, or every model failing raises an
    LLMProviderError (HTTP 502) instead of quietly returning a fake answer.
    """

    name = "groq"

    def __init__(self, api_key: str = None):
        key = api_key or settings.GROQ_API_KEY
        if not key:
            raise LLMProviderError("Groq API Key missing.")
        if Groq is None:
            raise LLMProviderError("groq python package not installed.")
        self.client = Groq(api_key=key)
        self.default_model = settings.DEFAULT_LLM_MODEL

    def _models_to_try(self, requested: str = None) -> List[str]:
        first = requested or self.default_model
        return [first] + [m for m in settings.llm_fallback_models if m != first]

    def _explain(self, exc: Exception) -> LLMProviderError:
        if groq is not None and isinstance(exc, groq.AuthenticationError):
            return LLMProviderError("the Groq API key was rejected. Check GROQ_API_KEY.")
        if groq is not None and isinstance(exc, groq.RateLimitError):
            return LLMProviderError("Groq rate limit reached on every available model. Try again shortly.")
        if groq is not None and isinstance(exc, (groq.APIConnectionError, groq.APITimeoutError)):
            return LLMProviderError("could not reach Groq (network error or timeout).")
        return LLMProviderError("the model request failed.")

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
        max_tok = max_tokens or settings.LLM_MAX_TOKENS
        last_error: Exception = None

        for target_model in self._models_to_try(model):
            try:
                logger.info(f"Invoking Groq model '{target_model}'")
                response = self.client.chat.completions.create(
                    model=target_model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tok,
                    timeout=settings.LLM_TIMEOUT,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                if _is_retryable_on_other_model(e):
                    logger.warning(f"Groq model '{target_model}' unavailable ({type(e).__name__}); trying the next model.")
                    continue
                logger.error(f"Groq generation error with '{target_model}': {type(e).__name__}: {e}")
                raise self._explain(e) from e

        raise self._explain(last_error)

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> Generator[str, None, None]:
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
        max_tok = max_tokens or settings.LLM_MAX_TOKENS
        last_error: Exception = None

        for target_model in self._models_to_try(model):
            yielded = False
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
                        yielded = True
                        yield chunk.choices[0].delta.content
                return
            except Exception as e:
                last_error = e
                # Once text has reached the user, switching model would splice two answers together.
                if not yielded and _is_retryable_on_other_model(e):
                    logger.warning(f"Groq stream model '{target_model}' unavailable ({type(e).__name__}); trying the next model.")
                    continue
                logger.error(f"Groq stream error with '{target_model}': {type(e).__name__}: {e}")
                raise self._explain(e) from e

        raise self._explain(last_error)


class LLMProviderFactory:
    """Factory for selecting and instantiating the active LLM provider."""

    @staticmethod
    def get_provider(provider_name: str = None) -> BaseLLMProvider:
        name = (provider_name or settings.DEFAULT_LLM_PROVIDER).lower()

        if name == "mock":
            return MockLLMProvider()

        if name == "groq":
            if not settings.GROQ_API_KEY or settings.GROQ_API_KEY.startswith("your_"):
                logger.warning("GROQ_API_KEY is not set; using the mock provider.")
                return MockLLMProvider()
            return GroqProvider()

        raise LLMProviderError(f"unknown provider '{name}'. Use 'groq' or 'mock'.")


llm_factory = LLMProviderFactory()
