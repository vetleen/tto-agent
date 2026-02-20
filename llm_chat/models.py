import uuid

from django.conf import settings
from django.db import models
from django.db.models import Q


class ChatThread(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="chat_threads",
    )

    title = models.CharField(max_length=255, blank=True, default="")
    is_archived = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_message_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-last_message_at", "-created_at"]
        indexes = [
            models.Index(fields=["user", "-last_message_at", "-created_at"]),
            models.Index(fields=["user", "-created_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.title or 'Untitled chat'} ({self.id})"


class ChatMessage(models.Model):
    class Role(models.TextChoices):
        USER = "user", "User"
        ASSISTANT = "assistant", "Assistant"
        SYSTEM = "system", "System"
        TOOL = "tool", "Tool"

    class Status(models.TextChoices):
        FINAL = "final", "Final"
        STREAMING = "streaming", "Streaming"
        ERROR = "error", "Error"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    thread = models.ForeignKey(
        ChatThread,
        on_delete=models.CASCADE,
        related_name="messages",
    )

    role = models.CharField(max_length=20, choices=Role.choices)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.FINAL)

    content = models.TextField(blank=True, default="")
    error = models.TextField(blank=True, default="")

    token_count = models.PositiveIntegerField(default=0)

    # Optional link to the detailed OpenAI call log (primarily for assistant messages)
    llm_call_log = models.ForeignKey(
        "llm_service.LLMCallLog",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="chat_messages",
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at", "id"]
        indexes = [
            models.Index(fields=["thread", "created_at", "id"]),
            models.Index(fields=["thread", "role", "created_at", "id"]),
            models.Index(fields=["thread", "status", "created_at", "id"]),
        ]
        constraints = [
            # Ensure only one assistant message is "in-flight" per thread.
            models.UniqueConstraint(
                fields=["thread"],
                condition=Q(role="assistant", status="streaming"),
                name="uniq_streaming_assistant_per_thread",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.role} ({self.status}) in {self.thread_id}"

    @staticmethod
    def _encoding_name_for_model(model_name: str | None) -> str:
        """
        Return a tiktoken encoding name for a given model.

        Notes (2026-01 docs):
        - gpt-4 / gpt-4-turbo / gpt-3.5-turbo → cl100k_base
        - gpt-4o / gpt-4o-mini → o200k_base
        - Future gpt-5* models are assumed to use o200k_base unless
          OpenAI publishes a different encoding.
        """
        if not model_name:
            return "cl100k_base"

        m = model_name.lower()

        # Newer models: 4o/4o-mini and we assume 5.x families use o200k_base
        if "gpt-4o" in m or "4o-mini" in m or m.startswith("gpt-5"):
            return "o200k_base"

        # Classic GPT-4 / 3.5 turbo families
        if "gpt-4" in m or "gpt-3.5" in m:
            return "cl100k_base"

        # Fallback
        return "cl100k_base"

    @classmethod
    def count_tokens(cls, text: str, *, model_name: str | None = None) -> int:
        """
        Count tokens using tiktoken.

        If tiktoken isn't installed yet, returns 0 (keeps app bootable).
        """
        if not text:
            return 0

        try:
            import tiktoken  # type: ignore
        except Exception:
            return 0

        try:
            # Prefer encoding_for_model when possible.
            if model_name:
                enc = tiktoken.encoding_for_model(model_name)
            else:
                enc = tiktoken.get_encoding(cls._encoding_name_for_model(model_name))
        except Exception:
            # Fallback to a safe default.
            enc = tiktoken.get_encoding("cl100k_base")

        return len(enc.encode(text))

    def save(self, *args, **kwargs):
        # Always keep token_count in sync with content for now.
        model_name: str | None = None
        if self.llm_call_log_id and getattr(self.llm_call_log, "model", None):
            model_name = self.llm_call_log.model

        self.token_count = self.count_tokens(self.content, model_name=model_name)
        super().save(*args, **kwargs)
