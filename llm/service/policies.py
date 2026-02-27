"""Model resolution and policy helpers (allowed models, default model)."""

from __future__ import annotations

import os
from typing import List, Optional

from llm.service.errors import LLMConfigurationError, LLMPolicyDenied


def _get_allowed_models() -> List[str]:
    raw = os.environ.get("LLM_ALLOWED_MODELS", "").strip()
    if not raw:
        return []
    return [m.strip() for m in raw.split(",") if m.strip()]


def _get_default_model_env() -> Optional[str]:
    v = os.environ.get("DEFAULT_LLM_MODEL") or os.environ.get("LLM_DEFAULT_MODEL")
    return v.strip() if v else None


def get_allowed_models() -> List[str]:
    """Return the list of allowed model names from LLM_ALLOWED_MODELS."""
    return _get_allowed_models()


def resolve_model(requested: Optional[str] = None) -> str:
    """
    Resolve the model name to use: validate requested against allowed list,
    or choose default (DEFAULT_LLM_MODEL if allowed, else first allowed).
    Raises LLMConfigurationError if no allowed models configured.
    Raises LLMPolicyDenied if requested is not in the allowed list.
    """
    allowed = _get_allowed_models()
    if not allowed:
        raise LLMConfigurationError(
            "LLM_ALLOWED_MODELS is empty or not set. "
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


__all__ = ["get_allowed_models", "resolve_model"]
