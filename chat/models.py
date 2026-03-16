from __future__ import annotations

import uuid

from django.conf import settings
from django.db import models


class ChatThread(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    data_rooms = models.ManyToManyField(
        "documents.DataRoom",
        through="ChatThreadDataRoom",
        related_name="chat_threads",
        blank=True,
    )
    skill = models.ForeignKey(
        "agent_skills.AgentSkill",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="chat_threads",
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

    # Rolling summary of older messages
    summary = models.TextField(blank=True, default="")
    summary_token_count = models.PositiveIntegerField(default=0)
    summary_up_to_message_id = models.UUIDField(null=True, blank=True)
    summary_message_count = models.PositiveIntegerField(default=0)

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
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
        indexes = [
            models.Index(fields=["thread", "created_at"]),
        ]

    def save(self, *args, **kwargs):
        if not self.token_count and self.content:
            from core.tokens import count_tokens

            self.token_count = count_tokens(self.content)
        super().save(*args, **kwargs)

    def __str__(self) -> str:
        return f"{self.role}: {self.content[:50]}"


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
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["created_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["thread", "title"],
                name="unique_canvas_title_per_thread",
            ),
        ]
        indexes = [models.Index(fields=["thread", "created_at"])]

    def __str__(self):
        return f"Canvas for thread {self.thread_id}: {self.title}"


class CanvasCheckpoint(models.Model):
    class Source(models.TextChoices):
        ORIGINAL = "original", "Original"
        AI_EDIT = "ai_edit", "AI Edit"
        USER_SAVE = "user_save", "User Save"
        IMPORT = "import", "Import"
        RESTORE = "restore", "Restore"

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
    blocking = models.BooleanField(default=False)

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
    completed_at = models.DateTimeField(null=True, blank=True)

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
