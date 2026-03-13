from django.apps import AppConfig


class ChatConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "chat"

    def ready(self):
        import chat.tools  # noqa: F401 — register tools on startup
        import chat.canvas_tools  # noqa: F401 — register canvas tools on startup
        import chat.subagent_tool  # noqa: F401 — register sub-agent tools on startup
