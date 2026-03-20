from __future__ import annotations


class LLMError(Exception):
    """Base error type for all LLM service failures."""


class LLMPolicyDenied(LLMError):
    """Request violates LLM policy (e.g. disallowed model)."""


class LLMConfigurationError(LLMError):
    """Misconfiguration of LLM settings, models, or environment."""


class LLMProviderError(LLMError):
    """Error raised from a concrete model/provider integration."""

    error_code: str = "unknown"

    def __init__(self, *args, error_code: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if error_code is not None:
            self.error_code = error_code


class LLMTimeoutError(LLMProviderError):
    """Timeout while waiting for a model or tool."""

    error_code: str = "timeout"


class LLMRateLimitError(LLMProviderError):
    """Rate limit (429) from an LLM provider after retries exhausted."""

    error_code: str = "rate_limited"


class LLMOverloadedError(LLMProviderError):
    """Provider returned 503/529 (overloaded)."""

    error_code: str = "overloaded"


class LLMAuthError(LLMProviderError):
    """Authentication/authorization failure (401/403)."""

    error_code: str = "auth_error"


class LLMRequestTooLargeError(LLMProviderError):
    """Request too large for the model (400 with token-related keywords)."""

    error_code: str = "request_too_large"


class LLMConnectionError(LLMProviderError):
    """Network connectivity failure reaching the provider."""

    error_code: str = "connection_error"


__all__ = [
    "LLMError",
    "LLMPolicyDenied",
    "LLMConfigurationError",
    "LLMProviderError",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "LLMOverloadedError",
    "LLMAuthError",
    "LLMRequestTooLargeError",
    "LLMConnectionError",
]

