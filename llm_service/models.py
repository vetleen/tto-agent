import uuid
from decimal import Decimal

from django.conf import settings
from django.db import models


class LLMCallLog(models.Model):
    """Per-call log for observability, cost, and debugging."""

    class Status(models.TextChoices):
        SUCCESS = "success", "Success"
        ERROR = "error", "Error"
        CANCELLED = "cancelled", "Cancelled"
        BLOCKED = "blocked", "Blocked"
        LOGGING_FAILED = "logging_failed", "Logging failed"

    # Identity
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    duration_ms = models.PositiveIntegerField(null=True, blank=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="llm_call_logs",
    )
    metadata = models.JSONField(default=dict, blank=True)
    request_id = models.CharField(max_length=255, blank=True, db_index=True)

    # Request
    model = models.CharField(max_length=255)
    is_stream = models.BooleanField(default=False)
    request_kwargs = models.JSONField(default=dict, blank=True)
    prompt_hash = models.CharField(max_length=64, blank=True)
    prompt_preview = models.TextField(blank=True)
    user_message_preview = models.CharField(max_length=300, blank=True)

    # Response
    provider_response_id = models.CharField(max_length=255, blank=True, null=True)
    response_model = models.CharField(max_length=255, blank=True, null=True)
    response_preview = models.TextField(blank=True)
    response_hash = models.CharField(max_length=64, blank=True, null=True)
    raw_response_payload = models.TextField(blank=True)

    # Usage / cost
    input_tokens = models.PositiveIntegerField(default=0)
    output_tokens = models.PositiveIntegerField(default=0)
    total_tokens = models.PositiveIntegerField(default=0)
    cost_usd = models.DecimalField(
        max_digits=12, decimal_places=8, null=True, blank=True
    )
    cost_source = models.CharField(max_length=64, blank=True, null=True)

    # Errors
    status = models.CharField(
        max_length=32, choices=Status.choices, default=Status.SUCCESS, db_index=True
    )
    error_type = models.CharField(max_length=255, blank=True, null=True)
    error_message = models.TextField(blank=True)
    http_status = models.PositiveIntegerField(null=True, blank=True)
    retry_count = models.PositiveIntegerField(default=0)
    provider_request_id = models.CharField(max_length=255, blank=True, null=True)

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
