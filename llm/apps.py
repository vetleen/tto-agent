import logging

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class LlmConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "llm"
    verbose_name = "LLM Service"

    def ready(self) -> None:  # pragma: no cover - import side effects only
        # Import pipelines so they register; import providers so model prefixes are registered.
        try:
            from .pipelines import simple_chat  # noqa: F401
            from .core import providers  # noqa: F401
        except Exception:
            logger.error(
                "Failed to import LLM pipelines/providers during startup. "
                "LLM features will be unavailable until the issue is resolved.",
                exc_info=True,
            )

