"""Model resolution and policy helpers (allowed models, default model)."""

from __future__ import annotations

import os
from typing import List, Optional

from llm.service.errors import LLMConfigurationError, LLMPolicyDenied


def _get_env_allowed_models() -> List[str]:
    """Raw env parse — no registry cross-check."""
    raw = os.environ.get("LLM_ALLOWED_MODELS", "").strip()
    if not raw:
        return []
    return [m.strip() for m in raw.split(",") if m.strip()]


def _get_default_model_env() -> Optional[str]:
    v = os.environ.get("DEFAULT_LLM_MODEL") or os.environ.get("LLM_DEFAULT_MODEL")
    return v.strip() if v else None


def get_allowed_models() -> List[str]:
    """Env allow-list intersected with the model registry.

    Models in ``LLM_ALLOWED_MODELS`` without a registry entry are dropped
    silently at runtime (pricing/display/capabilities would be unknown for
    them). Startup emits a warning for each mismatch via ``LlmConfig.ready``
    so ops can see dropped entries in Sentry.
    """
    from llm.model_registry import get_model_info

    return [m for m in _get_env_allowed_models() if get_model_info(m) is not None]


def get_env_unregistered_models() -> List[str]:
    """Env-listed models not found in the registry. Drives the startup warning."""
    from llm.model_registry import get_model_info

    return [m for m in _get_env_allowed_models() if get_model_info(m) is None]


def resolve_model(requested: Optional[str] = None) -> str:
    """
    Resolve the model name to use: validate requested against allowed list,
    or choose default (DEFAULT_LLM_MODEL if allowed, else first allowed).
    Raises LLMConfigurationError if no allowed models configured.
    Raises LLMPolicyDenied if requested is not in the allowed list.
    """
    allowed = get_allowed_models()
    if not allowed:
        raise LLMConfigurationError(
            "LLM_ALLOWED_MODELS is empty or not set, or no listed models "
            "are in the model registry. "
            "Set a comma-separated list of allowed model names."
        )

    if requested is not None:
        if requested not in allowed:
            raise LLMPolicyDenied(
                f"Model '{requested}' is not in LLM_ALLOWED_MODELS. Allowed: {allowed}"
            )
        return requested

    default = _get_default_model_env()
    if default and default in allowed:
        return default
    return allowed[0]


def get_fallback_models() -> list[str]:
    """Return fallback model names from LLM_FALLBACK_MODELS env var.

    Not wired up yet -- reserved for future .with_fallbacks() support.
    """
    raw = os.environ.get("LLM_FALLBACK_MODELS", "").strip()
    if not raw:
        return []
    return [m.strip() for m in raw.split(",") if m.strip()]


__all__ = [
    "get_allowed_models",
    "get_env_unregistered_models",
    "resolve_model",
    "get_fallback_models",
]
