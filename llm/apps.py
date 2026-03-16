import logging

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class LlmConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "llm"
    verbose_name = "LLM Service"

    def ready(self) -> None:  # pragma: no cover - import side effects only
        # Import pipelines so they register; import tools so they register.
        # Providers no longer need to self-register (model factory handles creation).
        try:
            from .pipelines import simple_chat  # noqa: F401
            from .pipelines import structured_output  # noqa: F401
            from .tools import builtins  # noqa: F401
        except Exception:
            logger.error(
                "Failed to import LLM pipelines/tools during startup. "
                "LLM features will be unavailable until the issue is resolved.",
                exc_info=True,
            )
