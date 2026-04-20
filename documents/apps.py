from django.apps import AppConfig


class DocumentsConfig(AppConfig):
    name = "documents"
    verbose_name = "Documents"

    def ready(self):
        from documents import signals  # noqa: F401
