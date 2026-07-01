import uuid
from decimal import Decimal

from django.conf import settings
from django.db import models


class LLMCallLog(models.Model):
    """Per-call log for observability and debugging. Not for persisting conversations."""

    class Status(models.TextChoices):
        SUCCESS = "success", "Success"
        ERROR = "error", "Error"
        CANCELLED = "cancelled", "Cancelled"

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
    tools = models.JSONField(null=True, blank=True)  # tool schemas: [{"name": "...", "description": "..."}]

    # Response
    raw_output = models.TextField(blank=True)

    # Usage
    input_tokens = models.PositiveIntegerField(null=True, blank=True)
    output_tokens = models.PositiveIntegerField(null=True, blank=True)
    total_tokens = models.PositiveIntegerField(null=True, blank=True)
    cost_usd = models.DecimalField(max_digits=12, decimal_places=8, null=True, blank=True)

    # Tracing
    trace_id = models.CharField(max_length=255, blank=True, db_index=True)
    conversation_id = models.CharField(max_length=255, blank=True, db_index=True)

    # Response metadata
    response_metadata = models.JSONField(null=True, blank=True)
    stop_reason = models.CharField(max_length=64, blank=True)
    provider_model_id = models.CharField(max_length=255, blank=True)

    # Extended usage
    cached_tokens = models.PositiveIntegerField(null=True, blank=True)
    cache_write_tokens = models.PositiveIntegerField(null=True, blank=True)
    reasoning_tokens = models.PositiveIntegerField(null=True, blank=True)

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


class OpsUsageLog(models.Model):
    """Lightweight per-call log of EPO OPS API usage, for observing per-org
    consumption of the shared fair-use quota.

    Deliberately minimal and decoupled: ``org_id``/``user_id`` are plain ints
    (denormalized, not FKs) so the write is cheap, never cascades, and a per-org
    ``GROUP BY`` needs no join and survives a user later changing orgs. Written
    best-effort by ``llm.tools.epo_ops._log_ops_usage`` — a failure here must
    never break a tool call.
    """

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    org_id = models.PositiveIntegerField(null=True, blank=True, db_index=True)
    user_id = models.PositiveIntegerField(null=True, blank=True)
    tool_name = models.CharField(max_length=64, db_index=True)
    response_bytes = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["org_id", "created_at"], name="ops_usage_org_created_idx"),
        ]
        verbose_name = "OPS Usage Log"
        verbose_name_plural = "OPS Usage Logs"

    def __str__(self):
        return f"{self.tool_name} org={self.org_id} @ {self.created_at}"


__all__ = ["LLMCallLog", "OpsUsageLog"]
