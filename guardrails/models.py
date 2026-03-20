from __future__ import annotations

import uuid

from django.conf import settings
from django.db import models


class GuardrailEvent(models.Model):
    """Immutable audit record for guardrail checks (heuristic, classifier, LLM review)."""

    class TriggerSource(models.TextChoices):
        USER_MESSAGE = "user_message", "User Message"
        DOCUMENT_CHUNK = "document_chunk", "Document Chunk"
        WEB_CONTENT = "web_content", "Web Content"
        TOOL_RESULT = "tool_result", "Tool Result"

    class CheckType(models.TextChoices):
        HEURISTIC = "heuristic", "Heuristic"
        CLASSIFIER = "classifier", "Classifier"
        LLM_REVIEW = "llm_review", "LLM Review"

    class Severity(models.TextChoices):
        LOW = "low", "Low"
        MEDIUM = "medium", "Medium"
        HIGH = "high", "High"
        CRITICAL = "critical", "Critical"

    class ActionTaken(models.TextChoices):
        LOGGED = "logged", "Logged"
        WARNED = "warned", "Warned"
        BLOCKED = "blocked", "Blocked"
        ESCALATED = "escalated", "Escalated"
        SUSPENDED = "suspended", "Suspended"
        DISMISSED = "dismissed", "Dismissed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="guardrail_events",
    )
    organization = models.ForeignKey(
        "accounts.Organization",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="guardrail_events",
    )
    thread = models.ForeignKey(
        "chat.ChatThread",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="guardrail_events",
    )
    llm_call_log = models.ForeignKey(
        "llm.LLMCallLog",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="guardrail_events",
    )

    trigger_source = models.CharField(max_length=20, choices=TriggerSource.choices)
    check_type = models.CharField(max_length=20, choices=CheckType.choices)
    tags = models.JSONField(default=list, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    severity = models.CharField(max_length=10, choices=Severity.choices)
    action_taken = models.CharField(max_length=20, choices=ActionTaken.choices)
    raw_input = models.TextField()
    reviewer_output = models.TextField(null=True, blank=True)
    related_event = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="escalations",
    )

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "-created_at"]),
            models.Index(fields=["organization", "-created_at"]),
            models.Index(fields=["action_taken", "-created_at"]),
        ]

    def __str__(self) -> str:
        return f"GuardrailEvent {self.id} ({self.check_type}/{self.action_taken})"
