import uuid
from decimal import Decimal

from django.conf import settings
from django.db import models


class LLMCallLog(models.Model):
    """Per-call log for observability and debugging. Not for persisting conversations."""

    class Status(models.TextChoices):
        SUCCESS = "success", "Success"
        ERROR = "error", "Error"

    # Identity
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    duration_ms = models.PositiveIntegerField(null=True, blank=True)

    # Attribution
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="llm_call_logs",
    )
    run_id = models.CharField(max_length=255, blank=True, db_index=True)

    # Request
    model = models.CharField(max_length=255)
    is_stream = models.BooleanField(default=False)
    prompt = models.JSONField()  # full messages list: [{role, content}, ...]

    # Response
    raw_output = models.TextField(blank=True)

    # Usage
    input_tokens = models.PositiveIntegerField(null=True, blank=True)
    output_tokens = models.PositiveIntegerField(null=True, blank=True)
    total_tokens = models.PositiveIntegerField(null=True, blank=True)
    cost_usd = models.DecimalField(max_digits=12, decimal_places=8, null=True, blank=True)

    # Status / errors
    status = models.CharField(
        max_length=32,
        choices=Status.choices,
        default=Status.SUCCESS,
        db_index=True,
    )
    error_type = models.CharField(max_length=255, blank=True, null=True)
    error_message = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["created_at"], name="llm_calllog_created_idx"),
            models.Index(fields=["user", "created_at"], name="llm_calllog_user_created_idx"),
            models.Index(fields=["model", "created_at"], name="llm_calllog_model_created_idx"),
        ]
        verbose_name = "LLM Call Log"
        verbose_name_plural = "LLM Call Logs"

    def __str__(self):
        return f"{self.model} @ {self.created_at} ({self.status})"


__all__ = ["LLMCallLog"]
