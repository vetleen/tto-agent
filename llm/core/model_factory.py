"""Model factory using LangChain's init_chat_model for provider detection and instantiation.

Replaces the custom ModelRegistry with LangChain's built-in provider detection,
automatic max_retries/timeout, and rate_limiter support.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from llm.service.errors import LLMConfigurationError

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised via mocks in unit tests
    from langchain.chat_models import init_chat_model
    from langchain_core.rate_limiters import InMemoryRateLimiter
except Exception as exc:
    init_chat_model = None  # type: ignore[assignment]
    InMemoryRateLimiter = None  # type: ignore[assignment]
    _import_error: Exception | None = exc
else:
    _import_error = None

# Explicit prefix -> provider mapping
_EXPLICIT_PREFIXES: dict[str, str] = {
    "openai/": "openai",
    "anthropic/": "anthropic",
    "gemini/": "google_genai",
}

# Auto-detect by model name prefix
_AUTO_DETECT: dict[str, str] = {
    "gpt-": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "claude-": "anthropic",
    "gemini-": "google_genai",
}

# Models requiring OpenAI's Responses API
_RESPONSES_API_PREFIXES = ("gpt-5.4", "gpt-5.2-pro")


def _parse_provider(model_name: str) -> tuple[str, str]:
    """Parse provider and API model name from a model string.

    Returns (provider, api_model) tuple.
    """
    for prefix, provider in _EXPLICIT_PREFIXES.items():
        if model_name.startswith(prefix):
            return provider, model_name[len(prefix):]

    for prefix, provider in _AUTO_DETECT.items():
        if model_name.startswith(prefix):
            return provider, model_name

    raise LLMConfigurationError(
        f"Cannot determine provider for model_name='{model_name}'. "
        f"Use an explicit prefix ({', '.join(_EXPLICIT_PREFIXES.keys())}) "
        f"or a recognized model name."
    )


def _get_provider_kwargs(provider: str, api_model: str) -> dict[str, Any]:
    """Return provider-specific default kwargs."""
    kwargs: dict[str, Any] = {"stream_usage": True}
    if provider == "openai":
        if any(api_model.startswith(p) for p in _RESPONSES_API_PREFIXES):
            kwargs["use_responses_api"] = True
    return kwargs


# Shared per-provider rate limiters (thread-safe singletons)
_rate_limiters: dict[str, object] = {}
_rate_limiter_lock = threading.Lock()


def _get_rate_limiter(provider: str):
    """Return a shared rate limiter for the provider, or None if not configured.

    Rate limiting is opt-in via env vars (e.g. LLM_RATE_LIMIT_OPENAI_RPS=10).
    """
    env_key = f"LLM_RATE_LIMIT_{provider.upper()}_RPS"
    rps_str = os.environ.get(env_key)
    if not rps_str:
        return None

    if provider not in _rate_limiters:
        with _rate_limiter_lock:
            if provider not in _rate_limiters:
                rps = float(rps_str)
                _rate_limiters[provider] = InMemoryRateLimiter(
                    requests_per_second=rps,
                )
    return _rate_limiters[provider]


def create_chat_model(model_name: str, *, fallback_models: list[str] | None = None, **overrides):
    """Create a ChatModel instance using init_chat_model for provider detection.

    Args:
        model_name: Model name with optional provider prefix
            (e.g. "gpt-5-mini", "anthropic/claude-sonnet-4-6").
        fallback_models: Optional list of fallback model names (not yet implemented).
        **overrides: Additional kwargs passed to init_chat_model.

    Returns:
        A provider-specific ChatModel wrapper.
    """
    if _import_error is not None or init_chat_model is None:
        raise LLMConfigurationError(
            "langchain is not installed or failed to import. "
            "Install with `pip install langchain`."
        ) from _import_error

    # Import providers here to avoid circular imports
    from llm.core.providers.anthropic import AnthropicChatModel
    from llm.core.providers.base import BaseLangChainChatModel
    from llm.core.providers.gemini import GeminiChatModel
    from llm.core.providers.openai import OpenAIChatModel

    provider_wrappers = {
        "openai": OpenAIChatModel,
        "anthropic": AnthropicChatModel,
        "google_genai": GeminiChatModel,
    }

    provider, api_model = _parse_provider(model_name)
    provider_kwargs = _get_provider_kwargs(provider, api_model)
    provider_kwargs.update(overrides)

    # TODO: when fallback_models is provided, chain via .with_fallbacks()

    rate_limiter = _get_rate_limiter(provider)

    lc_client = init_chat_model(
        api_model,
        model_provider=provider,
        max_retries=3,
        timeout=120,
        **({"rate_limiter": rate_limiter} if rate_limiter else {}),
        **provider_kwargs,
    )

    wrapper_cls = provider_wrappers.get(provider, BaseLangChainChatModel)
    return wrapper_cls(model_name=model_name, client=lc_client)


def create_variant_client(api_model: str, provider: str, **extra_kwargs):
    """Create a variant LangChain client for the same model with different settings.

    Used by provider wrappers for thinking-enabled model creation
    (e.g. reasoning_effort for OpenAI, extended thinking for Anthropic).

    Args:
        api_model: The API model name (without provider prefix).
        provider: The provider string (e.g. "openai", "anthropic").
        **extra_kwargs: Additional kwargs merged with provider defaults.

    Returns:
        A raw LangChain chat model client.
    """
    if _import_error is not None or init_chat_model is None:
        raise LLMConfigurationError(
            "langchain is not installed or failed to import."
        ) from _import_error

    provider_kwargs = _get_provider_kwargs(provider, api_model)
    provider_kwargs.update(extra_kwargs)

    rate_limiter = _get_rate_limiter(provider)

    return init_chat_model(
        api_model,
        model_provider=provider,
        max_retries=3,
        timeout=120,
        **({"rate_limiter": rate_limiter} if rate_limiter else {}),
        **provider_kwargs,
    )


__all__ = ["create_chat_model", "create_variant_client"]
