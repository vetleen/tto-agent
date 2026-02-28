from __future__ import annotations


class LLMError(Exception):
    """Base error type for all LLM service failures."""


class LLMPolicyDenied(LLMError):
    """Request violates LLM policy (e.g. disallowed model)."""


class LLMConfigurationError(LLMError):
    """Misconfiguration of LLM settings, models, or environment."""


class LLMProviderError(LLMError):
    """Error raised from a concrete model/provider integration."""


class LLMTimeoutError(LLMError):
    """Timeout while waiting for a model or tool."""


__all__ = [
    "LLMError",
    "LLMPolicyDenied",
    "LLMConfigurationError",
    "LLMProviderError",
    "LLMTimeoutError",
]

