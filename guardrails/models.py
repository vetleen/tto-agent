from __future__ import annotations

import uuid

from django.conf import settings
from django.db import models
from django.utils import timezone

from core.retention import RETENTION_PERIODS


class GuardrailEvent(models.Model):
    """Immutable audit record for guardrail checks (heuristic, classifier, LLM review)."""

    class TriggerSource(models.TextChoices):
        USER_MESSAGE = "user_message", "User Message"
        DOCUMENT_CHUNK = "document_chunk", "Document Chunk"
        WEB_CONTENT = "web_content", "Web Content"
        TOOL_RESULT = "tool_result", "Tool Result"
        IDENTITY_FIELD = "identity_field", "Identity Field"
        SKILL_RESOURCE = "skill_resource", "Skill Resource"

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
    # Which chunk a document_chunk event is about. Plain ints, not FKs, so the
    # audit row outlives a deleted chunk/version (precedent: PIIReviewEvent.org_id).
    # Lets per-chunk verdicts be joined back (guardrails/shadow.py).
    chunk_id = models.BigIntegerField(null=True, blank=True, db_index=True)
    version_id = models.BigIntegerField(null=True, blank=True)
    retain_until = models.DateTimeField(null=True, blank=True, db_index=True)

    def save(self, *args, **kwargs):
        if self._state.adding:
            self.retain_until = timezone.now() + RETENTION_PERIODS["guardrails.GuardrailEvent"]
        super().save(*args, **kwargs)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "-created_at"]),
            models.Index(fields=["organization", "-created_at"]),
            models.Index(fields=["action_taken", "-created_at"]),
        ]

    def __str__(self) -> str:
        return f"GuardrailEvent {self.id} ({self.check_type}/{self.action_taken})"


class ChunkScanShadowRun(models.Model):
    """One nightly (or manual) shadow comparison run (guardrails/shadow.py).

    ``trace_id`` is the ``LLMCallLog.trace_id`` of every decisions and reviewer
    call the run made, so its cost can be summed; ``summary`` holds the counters
    the run logged at the end (the report reads them from here).
    """

    run_date = models.DateField(db_index=True)
    trace_id = models.CharField(max_length=64)
    started_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    summary = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self) -> str:
        return f"ChunkScanShadowRun {self.run_date} ({self.trace_id})"


class ChunkScanShadowResult(models.Model):
    """One chunk's shadow comparison: the production classifier's verdict next to
    the decisions quick pathway's, plus the blind reviewer verdict for a positive.

    Ids and numbers only — never chunk text (the report fetches previews live).
    ``chunk_id``/``version_id`` are plain ints so rows outlive deleted chunks.
    """

    run_date = models.DateField(db_index=True)
    trace_id = models.CharField(max_length=64, blank=True, default="")
    org_id = models.IntegerField(null=True, blank=True)
    version_id = models.BigIntegerField()
    chunk_id = models.BigIntegerField()
    chunk_index = models.PositiveIntegerField()
    chunk_chars = models.PositiveIntegerField()
    # Production (from GuardrailEvents carrying chunk_id).
    classifier_flagged = models.BooleanField()
    classifier_confidence = models.FloatField(null=True, blank=True)
    classifier_tags = models.JSONField(default=list, blank=True)
    # allow | quarantine (reviewer) | threshold_quarantine | threshold_logged | "" (not flagged)
    prod_review = models.CharField(max_length=24, blank=True, default="")
    # Decisions quick pathway. probability None = the endpoint refused the question.
    quick_probability = models.FloatField(null=True, blank=True)
    quick_concern = models.CharField(max_length=32, blank=True, default="")
    quick_flagged = models.BooleanField(default=False)  # probability >= SHADOW_FLAG_THRESHOLD
    quick_pathway = models.CharField(max_length=12)  # quick | structured | mixed
    # Blind reviewer verdict for a positive under either system:
    # allow | quarantine | error | no_model | not_reviewed | "" (not a positive)
    blind_review = models.CharField(max_length=24, blank=True, default="")
    # Whose tag + number framed the review: classifier | quick.
    blind_review_source = models.CharField(max_length=12, blank=True, default="")
    blind_review_confidence = models.FloatField(null=True, blank=True)
    blind_review_severity = models.CharField(max_length=10, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["run_date", "chunk_id"], name="guardrails_shadow_unique_day_chunk"
            ),
        ]

    def __str__(self) -> str:
        return f"ChunkScanShadowResult {self.run_date} chunk={self.chunk_id}"


class ClassifierShadowResult(models.Model):
    """One inline shadow comparison of a text classifier call
    (guardrails/shadow_classifier.py): the production classifier's verdict and its
    reviewer's, next to the decisions quick pathway's, plus the blind reviewer
    verdict for a positive under either system.

    Ids, a hash and numbers only — never the text. ``subject`` holds the ids that
    identify what was classified (message/thread, source label, resource/skill,
    field); the ``family`` says which classifier.
    """

    FAMILIES = ("chat_message", "web_content", "skill_content", "identity_field")

    family = models.CharField(max_length=20)
    created_at = models.DateTimeField(auto_now_add=True)
    org_id = models.IntegerField(null=True, blank=True)
    user_id = models.IntegerField(null=True, blank=True)
    conversation_id = models.CharField(max_length=64, blank=True, default="")
    subject = models.JSONField(default=dict, blank=True)
    text_chars = models.PositiveIntegerField()
    text_sha256 = models.CharField(max_length=64)
    # Production: the classifier's verdict and what its reviewer decided.
    prod_flagged = models.BooleanField()
    prod_confidence = models.FloatField(null=True, blank=True)
    prod_tags = models.JSONField(default=list, blank=True)
    # web: allow | withhold | no_model | error; skill: allow | quarantine |
    # threshold_quarantine | threshold_logged | no_model; chat: dismiss | warn |
    # block | suspend | no_model | error; identity: blocked; "" = not flagged.
    prod_review = models.CharField(max_length=24, blank=True, default="")
    # Decisions quick pathway. probability None = refusal or decide error.
    quick_probability = models.FloatField(null=True, blank=True)
    quick_concern = models.CharField(max_length=32, blank=True, default="")
    quick_flagged = models.BooleanField(default=False)  # probability >= 0.5
    quick_pathway = models.CharField(max_length=12, blank=True, default="")  # quick | structured | mixed | "" (error)
    # Blind reviewer verdict for a positive under either system: the family's
    # reviewer action | error | no_model | not_reviewed | "" (not a positive / identity).
    blind_review = models.CharField(max_length=24, blank=True, default="")
    blind_review_source = models.CharField(max_length=12, blank=True, default="")  # classifier | quick
    blind_review_confidence = models.FloatField(null=True, blank=True)
    blind_review_severity = models.CharField(max_length=10, blank=True, default="")
    trace_id = models.CharField(max_length=64, blank=True, default="")
    decide_error = models.CharField(max_length=64, blank=True, default="")  # exception class name

    class Meta:
        indexes = [
            models.Index(fields=["family", "created_at"], name="guardrails_clfshadow_fam_crt"),
        ]

    def __str__(self) -> str:
        return f"ClassifierShadowResult {self.family} {self.created_at:%Y-%m-%d}"
