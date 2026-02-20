"""
LLM service configuration from Django settings.
"""
from django.conf import settings


def get_default_model() -> str:
    return getattr(settings, "LLM_DEFAULT_MODEL", "openai/gpt-5.2")


def get_allowed_models() -> list[str]:
    """List of model strings that are explicitly allowed. Empty = no restriction (dev)."""
    return getattr(settings, "LLM_ALLOWED_MODELS", [])


def is_model_allowed(model: str | None) -> bool:
    allowed = get_allowed_models()
    if not allowed:
        return True
    return model in allowed if model else False


def get_request_timeout() -> float:
    """Request timeout in seconds."""
    return float(getattr(settings, "LLM_REQUEST_TIMEOUT", 60.0))


def get_max_retries() -> int:
    return int(getattr(settings, "LLM_MAX_RETRIES", 2))


def get_log_write_timeout() -> float:
    """DB write timeout for LLMCallLog in seconds."""
    return float(getattr(settings, "LLM_LOG_WRITE_TIMEOUT", 5.0))


def get_pre_call_hooks():
    """List of callables(request: LLMRequest) -> None; raise to block."""
    return list(getattr(settings, "LLM_PRE_CALL_HOOKS", []))


def get_post_call_hooks():
    """List of callables(result: LLMResult) -> None; raise to block."""
    return list(getattr(settings, "LLM_POST_CALL_HOOKS", []))


# Model prefixes that do not support reasoning_effort; omit the param for these.
# Extend when a provider rejects or ignores reasoning_effort.
REASONING_EFFORT_DISALLOWED_PREFIXES = ("moonshot/",)


def should_send_reasoning_effort(model: str | None) -> bool:
    """True if we should include reasoning_effort in the request for this model."""
    if not model:
        return True
    return not any(model.startswith(prefix) for prefix in REASONING_EFFORT_DISALLOWED_PREFIXES)
