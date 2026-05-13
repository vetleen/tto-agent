from __future__ import annotations

import uuid

from django.conf import settings
from django.db import models


class Meeting(models.Model):
    """A meeting tracked by Wilfred.

    A meeting is a first-class object distinct from data rooms. It owns its
    transcript (live or uploaded) and zero or more attachments (slides,
    agendas). Users can open a "Summarize meeting in chat" thread to draft
    minutes with Wilfred, and can export the raw transcript to a data room.

    Status transitions:
        DRAFT             -> LIVE_TRANSCRIBING (user clicks Transcribe)
        LIVE_TRANSCRIBING -> READY              (user clicks Stop)
        LIVE_TRANSCRIBING -> INTERRUPTED        (WS disconnect/tab close)
        INTERRUPTED       -> LIVE_TRANSCRIBING  (user clicks Resume)
        DRAFT             -> READY              (transcript text uploaded)
        DRAFT             -> READY              (audio uploaded + transcribed)
        any               -> FAILED             (audio upload transcription error)
    """

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        LIVE_TRANSCRIBING = "live_transcribing", "Live transcribing"
        INTERRUPTED = "interrupted", "Interrupted"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    class TranscriptSource(models.TextChoices):
        LIVE = "live", "Live transcription"
        AUDIO_UPLOAD = "audio_upload", "Audio upload"
        TEXT_UPLOAD = "text_upload", "Text upload"

    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, db_index=True)
    name = models.CharField(max_length=255)
    slug = models.SlugField(max_length=100, unique=True)
    agenda = models.TextField(blank=True, default="")
    participants = models.TextField(blank=True, default="")
    description = models.TextField(blank=True, default="")
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.DRAFT,
        db_index=True,
    )
    transcript = models.TextField(blank=True, default="")
    transcript_updated_at = models.DateTimeField(null=True, blank=True)
    transcript_source = models.CharField(
        max_length=16,
        choices=TranscriptSource.choices,
        blank=True,
        default="",
    )
    transcription_model = models.CharField(max_length=128, blank=True, default="")
    transcription_error = models.TextField(blank=True, default="")
    transcription_chunks_total = models.PositiveIntegerField(default=0)
    transcription_chunks_done = models.PositiveIntegerField(default=0)
    # Optional BCP-47 / ISO-639-1 language code (e.g. "en", "no"). When blank,
    # the transcription model auto-detects per call. Set by the user from a
    # dropdown on the meeting page when they know the meeting language and
    # want to skip detection.
    forced_language = models.CharField(max_length=8, blank=True, default="")
    started_at = models.DateTimeField(null=True, blank=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    duration_seconds = models.PositiveIntegerField(null=True, blank=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="meetings",
    )
    is_archived = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at", "name"]
        indexes = [
            models.Index(fields=["created_by", "-updated_at"]),
        ]

    def __str__(self) -> str:
        return self.name


class MeetingTranscriptSegment(models.Model):
    """A single segment of a live-transcribed meeting.

    Live transcription chunks the audio client-side (one fully-formed
    container file every ~30 s). Each chunk is uploaded over the WebSocket,
    transcribed by ``transcribe_meeting_chunk_task``, and persisted as a
    segment row. Segments are the source of truth *during* transcription;
    after each segment lands, ``Meeting.transcript`` is recomputed under
    ``select_for_update`` so it stays the source of truth for read views.
    """

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    meeting = models.ForeignKey(
        Meeting,
        on_delete=models.CASCADE,
        related_name="segments",
    )
    segment_index = models.PositiveIntegerField()
    start_offset_seconds = models.FloatField(default=0.0)
    duration_seconds = models.FloatField(null=True, blank=True)
    text = models.TextField(blank=True, default="")
    transcription_model = models.CharField(max_length=128, blank=True, default="")
    status = models.CharField(
        max_length=10,
        choices=Status.choices,
        default=Status.PENDING,
        db_index=True,
    )
    error = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    transcribed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["meeting", "segment_index"]
        constraints = [
            models.UniqueConstraint(
                fields=["meeting", "segment_index"],
                name="meetings_segment_unique_per_meeting",
            ),
        ]

    def __str__(self) -> str:
        return f"Segment {self.segment_index} of meeting {self.meeting_id} ({self.status})"


class MeetingAttachment(models.Model):
    """A supporting file uploaded to a meeting (slides, agenda PDF, etc.).

    Distinct from the transcript and from any chat-thread attachments. When
    the user opens a "Summarize meeting in chat" thread, supported files are
    copied into fresh ChatAttachment rows on that thread; the originals stay
    on the meeting page. See ``meetings.services.minutes``.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    meeting = models.ForeignKey(
        Meeting,
        on_delete=models.CASCADE,
        related_name="attachments",
    )
    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="meeting_attachments",
    )
    file = models.FileField(upload_to="meeting_attachments/%Y/%m/", max_length=500)
    original_filename = models.CharField(max_length=255)
    content_type = models.CharField(max_length=128, blank=True, default="")
    size_bytes = models.PositiveIntegerField(default=0)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-uploaded_at"]
        indexes = [
            models.Index(fields=["meeting", "-uploaded_at"]),
        ]

    def __str__(self) -> str:
        return self.original_filename or f"Attachment {self.id}"
