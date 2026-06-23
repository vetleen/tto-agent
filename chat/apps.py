from django.apps import AppConfig


class ChatConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "chat"

    def ready(self):
        import chat.signals  # noqa: F401 — retention + file cleanup signals
        import chat.tools  # noqa: F401 — register tools on startup
        import chat.canvas_tools  # noqa: F401 — register canvas tools on startup
        import chat.subagent_tool  # noqa: F401 — register sub-agent tools on startup
        import chat.task_tools  # noqa: F401 — register task tools on startup
        import chat.tool_loops  # noqa: F401 — register loop tools on startup
        import chat.image_tools  # noqa: F401 — register image generation tool on startup
