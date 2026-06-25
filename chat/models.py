from __future__ import annotations

import uuid

from django.conf import settings
from django.db import models
from django.utils import timezone

from core.retention import RETENTION_PERIODS


class ChatThread(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    data_rooms = models.ManyToManyField(
        "documents.DataRoom",
        through="ChatThreadDataRoom",
        related_name="chat_threads",
        blank=True,
    )
    skills = models.ManyToManyField(
        "agent_skills.AgentSkill",
        through="ChatThreadSkill",
        related_name="chat_threads",
        blank=True,
    )
    active_canvas = models.ForeignKey(
        "ChatCanvas",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="+",
    )
    title = models.CharField(max_length=255, blank=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="chat_threads",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    is_archived = models.BooleanField(default=False)
    emoji = models.CharField(max_length=8, blank=True, default="")

    # Per-thread LLM model choice. Empty → the user's preferred chat model.
    # A stored model that's no longer allowed falls back by tier; see
    # core.preferences.resolve_thread_model.
    model = models.CharField(max_length=128, blank=True, default="")

    # Generic per-thread context bag for features like edit-in-chat.
    # Known keys:
    #   - source_skill_id (str UUID): the skill being edited (or its fork)
    #   - pending_initial_turn (bool): trigger an assistant turn on next load
    metadata = models.JSONField(default=dict, blank=True)

    # Rolling summary of older messages
    summary = models.TextField(blank=True, default="")
    summary_token_count = models.PositiveIntegerField(default=0)
    summary_up_to_message_id = models.UUIDField(null=True, blank=True)
    summary_message_count = models.PositiveIntegerField(default=0)

    retain_until = models.DateTimeField(null=True, blank=True, db_index=True)

    def save(self, *args, **kwargs):
        self.retain_until = timezone.now() + RETENTION_PERIODS["chat.ChatThread"]
        update_fields = kwargs.get("update_fields")
        if update_fields is not None:
            if "retain_until" not in update_fields:
                kwargs["update_fields"] = list(update_fields) + ["retain_until"]
        super().save(*args, **kwargs)

    class Meta:
        ordering = ["-updated_at"]
        indexes = [
            models.Index(fields=["created_by", "-updated_at"]),
        ]

    def __str__(self) -> str:
        return self.title or f"Thread {self.id}"


class ChatThreadDataRoom(models.Model):
    thread = models.ForeignKey(
        ChatThread,
        on_delete=models.CASCADE,
        related_name="thread_data_rooms",
    )
    data_room = models.ForeignKey(
        "documents.DataRoom",
        on_delete=models.CASCADE,
        related_name="thread_links",
    )
    attached_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [("thread", "data_room")]
        ordering = ["attached_at"]

    def __str__(self) -> str:
        return f"{self.thread_id} ↔ {self.data_room_id}"


class ChatThreadSkill(models.Model):
    thread = models.ForeignKey(
        ChatThread,
        on_delete=models.CASCADE,
        related_name="thread_skills",
    )
    skill = models.ForeignKey(
        "agent_skills.AgentSkill",
        on_delete=models.CASCADE,
        related_name="thread_skill_links",
    )
    attached_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [("thread", "skill")]
        # Composite ordering: attach order, with id as a deterministic
        # tie-break when several rows are bulk-created in the same request
        # (so prompt order, tool-union, and template collision resolution
        # are stable).
        ordering = ["attached_at", "id"]

    def __str__(self) -> str:
        return f"{self.thread_id} ↔ {self.skill_id}"


class Loop(models.Model):
    """A recurring, scheduled chat turn bound 1:1 to a ChatThread.

    The same ``prompt`` fires on a cadence; each fire runs a full agent turn
    headlessly (see ``chat/loop_service.py``) and persists to the thread, so the
    loop is also a normal, browsable chat. The loop is "stupid": it stores only
    ``next_run`` (when to fire next), recomputed from the fire time after each
    run, never when it last ran.
    """

    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        PAUSED = "paused", "Paused"

    class HistoryMode(models.TextChoices):
        FRESH = "fresh", "Fresh each run"
        CONVERSATIONAL = "conversational", "Conversational"

    class Cadence(models.TextChoices):
        INTERVAL = "interval", "Interval"
        CLOCK = "clock", "Clock"

    class ClockFrequency(models.TextChoices):
        DAILY = "daily", "Daily"
        WEEKDAYS = "weekdays", "Weekdays"
        WEEKLY = "weekly", "Weekly"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    thread = models.OneToOneField(
        ChatThread, on_delete=models.CASCADE, related_name="loop",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="loops",
    )

    prompt = models.TextField()
    history_mode = models.CharField(
        max_length=16, choices=HistoryMode.choices, default=HistoryMode.FRESH,
    )

    # --- Schedule: the cadence kind drives next_run; the loop stores only when
    # it should next fire. ---
    cadence_kind = models.CharField(
        max_length=16, choices=Cadence.choices, default=Cadence.INTERVAL,
    )
    interval_seconds = models.PositiveIntegerField(null=True, blank=True)
    clock_time = models.TimeField(null=True, blank=True)
    clock_frequency = models.CharField(
        max_length=16, choices=ClockFrequency.choices, blank=True, default="",
    )
    clock_weekday = models.PositiveSmallIntegerField(
        null=True, blank=True,
    )  # 0=Mon … 6=Sun, used when clock_frequency=weekly
    tz = models.CharField(max_length=64, default="UTC")  # resolves clock times

    next_run = models.DateTimeField(db_index=True)

    # --- Run limits / state ---
    # Optional run cap. NULL means "unlimited" — the loop runs until the user
    # pauses it (the default for new loops). A positive integer makes the loop
    # auto-pause once ``runs_completed`` reaches it.
    max_runs = models.PositiveIntegerField(null=True, blank=True, default=None)
    runs_completed = models.PositiveIntegerField(default=0)
    status = models.CharField(
        max_length=8, choices=Status.choices, default=Status.ACTIVE, db_index=True,
    )

    # Reentrancy lock — the tick-and-scan task claims a loop via an atomic CAS
    # on (running=False → True) so a still-running turn is never double-fired.
    running = models.BooleanField(default=False)
    locked_at = models.DateTimeField(null=True, blank=True)

    # Output / unread tracking
    last_result_at = models.DateTimeField(null=True, blank=True)
    last_seen_at = models.DateTimeField(null=True, blank=True)

    consecutive_errors = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["status", "next_run"]),
            models.Index(fields=["created_by", "-last_result_at"]),
        ]

    def __str__(self) -> str:
        return f"Loop {self.id} ({self.status})"

    @property
    def is_unread(self) -> bool:
        """A loop is unread when it produced a result the owner hasn't opened."""
        if not self.last_result_at:
            return False
        return self.last_seen_at is None or self.last_result_at > self.last_seen_at

    @property
    def schedule_label(self) -> str:
        """Human-readable cadence, e.g. 'Every 6 hours' or 'Weekdays at 09:00'."""
        if self.cadence_kind == self.Cadence.INTERVAL and self.interval_seconds:
            secs = self.interval_seconds
            if secs % 86400 == 0:
                n = secs // 86400
                unit = "day" if n == 1 else "days"
            elif secs % 3600 == 0:
                n = secs // 3600
                unit = "hour" if n == 1 else "hours"
            else:
                n = max(1, secs // 60)
                unit = "minute" if n == 1 else "minutes"
            return f"Every {n} {unit}"
        if self.cadence_kind == self.Cadence.CLOCK and self.clock_time:
            t = self.clock_time.strftime("%H:%M")
            if self.clock_frequency == self.ClockFrequency.WEEKLY:
                days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                day = days[self.clock_weekday] if self.clock_weekday is not None else "weekly"
                return f"{day}s at {t}"
            if self.clock_frequency == self.ClockFrequency.WEEKDAYS:
                return f"Weekdays at {t}"
            return f"Daily at {t}"
        return "—"


class ChatMessage(models.Model):
    class Role(models.TextChoices):
        SYSTEM = "system", "System"
        USER = "user", "User"
        ASSISTANT = "assistant", "Assistant"
        TOOL = "tool", "Tool"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    thread = models.ForeignKey(
        ChatThread,
        on_delete=models.CASCADE,
        related_name="messages",
    )
    role = models.CharField(max_length=10, choices=Role.choices)
    content = models.TextField()
    tool_call_id = models.CharField(max_length=255, null=True, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    token_count = models.PositiveIntegerField(default=0)
    is_redacted = models.BooleanField(default=False)
    # Server-injected messages that should be sent to the LLM but not rendered
    # in the chat UI (e.g. seed messages from "edit skill in chat").
    is_hidden_from_user = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
        indexes = [
            models.Index(fields=["thread", "created_at"]),
        ]

    def save(self, *args, **kwargs):
        if not self.token_count and self.content and not self.is_redacted:
            from core.tokens import count_tokens

            self.token_count = count_tokens(self.content)
        super().save(*args, **kwargs)

    def __str__(self) -> str:
        return f"{self.role}: {self.content[:50]}"


class ChatAttachment(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    message = models.ForeignKey(
        ChatMessage,
        on_delete=models.CASCADE,
        related_name="attachments",
        null=True,
        blank=True,
    )
    thread = models.ForeignKey(
        ChatThread,
        on_delete=models.CASCADE,
        related_name="attachments",
    )
    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
    )
    file = models.FileField(upload_to="chat_attachments/%Y/%m/")
    original_filename = models.CharField(max_length=255)
    content_type = models.CharField(max_length=100)
    size_bytes = models.PositiveIntegerField()
    # Cached text extracted from a docx/pdf attachment (with inline
    # [[image:uuid|...]] tokens for any embedded images persisted as
    # message-scoped Assets). Populated once, lazily, the first time the
    # attachment is enriched into an LLM request — so per-turn replay reuses it
    # instead of re-extracting and recreating assets. Empty for images/text.
    extracted_content = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
        indexes = [
            models.Index(fields=["thread", "created_at"]),
            models.Index(fields=["message"]),
        ]

    def __str__(self) -> str:
        return f"Attachment {self.original_filename} ({self.id})"


class ChatCanvas(models.Model):
    thread = models.ForeignKey(
        ChatThread, on_delete=models.CASCADE, related_name="canvases"
    )
    title = models.CharField(max_length=255, blank=True, default="Untitled document")
    content = models.TextField(blank=True, default="")
    accepted_checkpoint = models.ForeignKey(
        "CanvasCheckpoint",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="+",
    )
    is_active = models.BooleanField(default=False)
    last_activated_at = models.DateTimeField(null=True, blank=True)
    # Consecutive blocked (GDPR Art 9/10) sync saves of this canvas to a data room.
    # Reset on a successful save and after the agent's retry budget is exhausted (then
    # the doc is deferred to a quarantined draft and the user is warned). See
    # CanvasSaveToDocumentTool / documents.services.sync_scan.
    dr_save_attempts = models.PositiveSmallIntegerField(default=0)
    # Soft delete: a non-null timestamp hides the canvas from the agent (prompt +
    # tool lookups) and the UI tabs, as if deleted, while preserving content and
    # version history for an Undo/restore. Enumeration/lookup sites filter on
    # deleted_at__isnull=True; see chat.services.soft_delete_canvas / restore_canvas.
    deleted_at = models.DateTimeField(null=True, blank=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["created_at"]
        constraints = [
            # Titles are unique only among *live* canvases, so a soft-deleted
            # canvas keeps its title without blocking a new one that reuses it.
            models.UniqueConstraint(
                fields=["thread", "title"],
                condition=models.Q(deleted_at__isnull=True),
                name="unique_canvas_title_per_thread",
            ),
        ]
        indexes = [
            models.Index(fields=["thread", "created_at"]),
            models.Index(fields=["thread", "is_active", "-last_activated_at"]),
        ]

    def __str__(self):
        return f"Canvas for thread {self.thread_id}: {self.title}"


class CanvasCheckpoint(models.Model):
    class Source(models.TextChoices):
        ORIGINAL = "original", "Original"
        AI_EDIT = "ai_edit", "AI Edit"
        USER_SAVE = "user_save", "User Save"
        IMPORT = "import", "Import"
        RESTORE = "restore", "Restore"
        REDACTED = "redacted", "Redacted"

    canvas = models.ForeignKey(
        ChatCanvas, on_delete=models.CASCADE, related_name="checkpoints"
    )
    title = models.CharField(max_length=255, blank=True, default="")
    content = models.TextField(blank=True, default="")
    source = models.CharField(max_length=20, choices=Source.choices)
    description = models.CharField(max_length=255, blank=True, default="")
    order = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["order"]
        indexes = [models.Index(fields=["canvas", "order"])]

    def __str__(self):
        return f"Checkpoint #{self.order} ({self.source}) for canvas {self.canvas_id}"


class SubAgentRun(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    thread = models.ForeignKey(ChatThread, on_delete=models.CASCADE, related_name="subagent_runs")
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="subagent_runs")
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.PENDING)

    # Task spec
    prompt = models.TextField()
    skill_slug = models.CharField(max_length=64, blank=True)
    model_tier = models.CharField(max_length=10, default="mid")
    model_used = models.CharField(max_length=128, blank=True)
    timeout = models.PositiveIntegerField(
        default=0,
        help_text="Seconds the tool waited for the result (0=fire-and-forget)",
    )

    # Context (copied at creation time)
    data_room_ids = models.JSONField(default=list)
    tool_names = models.JSONField(default=list)

    # Result
    result = models.TextField(blank=True)
    error = models.TextField(blank=True)
    celery_task_id = models.CharField(max_length=255, blank=True)

    # Metrics
    tokens_used = models.PositiveIntegerField(default=0)
    cost_usd = models.FloatField(default=0.0)

    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    # When the orchestrator claimed this run's result for reporting (see
    # ChatConsumer._claim_unreported_subagents). Acts as a lease: a claim
    # older than the lease window with no assistant response is re-claimable.
    reported_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "status"]),
            models.Index(fields=["thread", "-created_at"]),
        ]

    def __str__(self) -> str:
        return f"SubAgentRun {self.id} ({self.status})"


class ThreadTask(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        IN_PROGRESS = "in_progress", "In Progress"
        COMPLETED = "completed", "Completed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    thread = models.ForeignKey(ChatThread, on_delete=models.CASCADE, related_name="tasks")
    title = models.CharField(max_length=512)
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.PENDING)
    order = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["order", "created_at"]

    def __str__(self):
        return f"[{self.status}] {self.title[:60]}"


class ThreadChunkUsage(models.Model):
    """Records that a chat thread consumed a specific document chunk during RAG retrieval."""

    thread = models.ForeignKey(
        ChatThread,
        on_delete=models.CASCADE,
        related_name="chunk_usages",
    )
    chunk = models.ForeignKey(
        "documents.DataRoomDocumentChunk",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="thread_usages",
    )
    document = models.ForeignKey(
        "documents.DataRoomDocument",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="thread_usages",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["thread", "chunk"],
                name="unique_thread_chunk_usage",
                condition=models.Q(chunk__isnull=False),
            ),
        ]
        indexes = [
            models.Index(fields=["document"], name="threadchunkusage_doc_idx"),
        ]

    def __str__(self) -> str:
        return f"Thread {self.thread_id} used chunk {self.chunk_id}"


class Asset(models.Model):
    """A persisted asset: either binary bytes (an image) or a blob-less reference
    to a data-room document version. Backs both inline-image tokens
    (``[[image:uuid]]``) and file-download tokens (``[[file:uuid]]``); ``kind``
    distinguishes the two reference flavours so they never share a row.

    Scoped to exactly one owner — a data-room document version, a canvas, a
    chat message, or a chat thread — via the nullable FKs below (enforced by a
    CheckConstraint). Thread ownership exists for tool-generated images: the
    assistant message isn't persisted until after the turn streams, but a tool
    must mint a stable asset id mid-run (so the model can embed its token), and
    the thread always exists at that point.
    Lives in the chat app (not documents) so every FK points chat -> documents
    or within chat, avoiding a migration cycle (documents must not depend on chat).
    """

    KIND_IMAGE = "image"
    KIND_FILE = "file"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Exactly one owner is set (see Meta.constraints).
    version = models.ForeignKey(
        "documents.DataRoomDocumentVersion",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="assets",
    )
    canvas = models.ForeignKey(
        "ChatCanvas",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="assets",
    )
    message = models.ForeignKey(
        "ChatMessage",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="assets",
    )
    thread = models.ForeignKey(
        "ChatThread",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="assets",
    )

    # Empty for a *reference* asset (version-owned): the bytes live on the
    # data-room version's native file (native_blob / the document's
    # original_file) and are resolved on serve — see image_asset_source /
    # file_asset_source.
    blob = models.FileField(upload_to="image_assets/%Y/%m/", max_length=500, blank=True)
    content_type = models.CharField(max_length=100)
    # Render/serve mode, only meaningful for reference assets: an inline image
    # (``[[image:]]``) vs a file download (``[[file:]]``). Blob-owning assets are
    # always images. Keeps an image-ref and a file-ref for the SAME version as
    # distinct rows (their dedup queries filter on this).
    kind = models.CharField(
        max_length=8,
        choices=[(KIND_IMAGE, KIND_IMAGE), (KIND_FILE, KIND_FILE)],
        default=KIND_IMAGE,
        db_index=True,
    )
    size_bytes = models.PositiveIntegerField(default=0)
    width = models.PositiveIntegerField(null=True, blank=True)
    height = models.PositiveIntegerField(null=True, blank=True)
    # Hex SHA-256 of the bytes — lets the same image dedupe across owners later.
    sha256 = models.CharField(max_length=64, blank=True, default="", db_index=True)
    description = models.TextField(blank=True, default="")
    alt_text = models.CharField(max_length=1024, blank=True, default="")
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["version"]),
            models.Index(fields=["canvas"]),
            models.Index(fields=["thread"]),
        ]
        constraints = [
            models.CheckConstraint(
                name="imageasset_exactly_one_owner",
                condition=(
                    models.Q(version__isnull=False, canvas__isnull=True, message__isnull=True, thread__isnull=True)
                    | models.Q(version__isnull=True, canvas__isnull=False, message__isnull=True, thread__isnull=True)
                    | models.Q(version__isnull=True, canvas__isnull=True, message__isnull=False, thread__isnull=True)
                    | models.Q(version__isnull=True, canvas__isnull=True, message__isnull=True, thread__isnull=False)
                ),
            ),
        ]

    def __str__(self) -> str:
        return f"Asset {self.id} ({self.content_type})"
