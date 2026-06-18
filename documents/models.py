from __future__ import annotations

import uuid

from django.conf import settings
from django.contrib.postgres.indexes import GinIndex
from django.contrib.postgres.search import SearchVectorField
from django.db import models
from django.utils import timezone

from core.retention import RETENTION_PERIODS


class DataRoom(models.Model):
    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, db_index=True)
    name = models.CharField(max_length=255)
    slug = models.SlugField(max_length=100)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="data_rooms",
    )
    is_archived = models.BooleanField(default=False)
    description = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    retain_until = models.DateTimeField(null=True, blank=True, db_index=True)

    def save(self, *args, **kwargs):
        self.retain_until = timezone.now() + RETENTION_PERIODS["documents.DataRoom"]
        update_fields = kwargs.get("update_fields")
        if update_fields is not None:
            if "retain_until" not in update_fields:
                kwargs["update_fields"] = list(update_fields) + ["retain_until"]
        super().save(*args, **kwargs)

    class Meta:
        ordering = ["-updated_at", "name"]
        constraints = [
            # Slugs are display/identity values scoped to their owner; global
            # uniqueness would leak data room names across tenants (a collision
            # suffix reveals another user has a room with that name).
            models.UniqueConstraint(
                fields=["created_by", "slug"],
                name="documents_unique_slug_per_user",
            ),
        ]

    def __str__(self) -> str:
        return self.name


class DataRoomDocument(models.Model):
    class Status(models.TextChoices):
        UPLOADED = "uploaded", "Uploaded"
        PROCESSING = "processing", "Processing"
        # Chunked and embedded, but the GDPR PII scan hasn't finished — the
        # document must not surface in retrieval until the scan clears it.
        SCANNING = "scanning", "Scanning"
        SCAN_FAILED = "scan_failed", "Scan failed"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    data_room = models.ForeignKey(
        DataRoom,
        on_delete=models.CASCADE,
        related_name="documents",
    )
    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="uploaded_documents",
    )
    original_file = models.FileField(
        upload_to="documents/%Y/%m/",
        blank=True,
        max_length=500,
    )
    # Immutable provenance: what the file was literally called when uploaded.
    original_filename = models.CharField(max_length=255)
    # Mutable display name (renaming sets this; original_filename is preserved).
    name = models.CharField(max_length=255, blank=True, default="")
    mime_type = models.CharField(max_length=128, blank=True)
    size_bytes = models.PositiveIntegerField(null=True, blank=True)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.UPLOADED,
        db_index=True,
    )
    processing_error = models.TextField(null=True, blank=True)
    # Times the stale-document sweeper has re-enqueued processing for this
    # document (documents.tasks.requeue_stale_documents). Caps crash-loops: a
    # poison document that OOMs the worker must not be re-enqueued forever.
    requeue_count = models.PositiveSmallIntegerField(default=0)
    token_count = models.PositiveIntegerField(null=True, blank=True)
    parser_type = models.CharField(max_length=64, blank=True)
    chunking_strategy = models.CharField(max_length=64, blank=True)
    embedding_model = models.CharField(max_length=128, blank=True)
    description = models.TextField(blank=True, default="")
    transcript = models.TextField(blank=True, default="")
    transcription_model = models.CharField(max_length=128, blank=True)
    is_archived = models.BooleanField(default=False)
    is_quarantined = models.BooleanField(default=False, db_index=True)
    quarantine_reason = models.TextField(blank=True, default="")
    is_partially_quarantined = models.BooleanField(default=False, db_index=True)
    doc_index = models.PositiveIntegerField(default=0)
    file_metadata_date = models.DateField(null=True, blank=True)
    document_date = models.DateField(null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Two-pointer versioning. ``current_version`` is the working/editing head
    # (advances immediately on save); ``active_searchable_version`` is what
    # retrieval reads (advances only when a version finishes processing+scan as
    # READY and not quarantined). Both nullable for migration safety.
    current_version = models.ForeignKey(
        "DataRoomDocumentVersion",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="+",
    )
    active_searchable_version = models.ForeignKey(
        "DataRoomDocumentVersion",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="+",
    )

    @property
    def display_name(self) -> str:
        return self.name or self.original_filename

    class Meta:
        ordering = ["-uploaded_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["data_room", "doc_index"],
                name="documents_unique_doc_index_per_data_room",
                condition=models.Q(doc_index__gt=0),
            ),
        ]

    def save(self, *args, **kwargs):
        if not self.doc_index:
            from django.db import transaction
            from django.db.models import Max

            with transaction.atomic():
                # Lock the data room row to serialize concurrent doc_index assignment
                DataRoom.objects.filter(pk=self.data_room_id).select_for_update().first()
                max_idx = DataRoomDocument.objects.filter(
                    data_room=self.data_room
                ).aggregate(Max("doc_index"))["doc_index__max"] or 0
                self.doc_index = max_idx + 1
                super().save(*args, **kwargs)
        else:
            super().save(*args, **kwargs)

    def __str__(self) -> str:
        return self.original_filename or f"Document {self.pk}"


class DataRoomDocumentVersion(models.Model):
    """An immutable point-in-time version of a document's working content.

    v0 is the original upload (native bytes + extracted markdown); later
    versions are markdown edits. Each *processed* version owns its own chunks
    and embeddings, so a rollback is a cheap pointer flip (no re-processing).
    Mirrors the canvas checkpoint pattern (chat.CanvasCheckpoint).

    Version lifecycle is owned by the prune task (documents.tasks); there is no
    per-version retention sweep — GDPR erasure happens by CASCADE when the parent
    document or data room is deleted.
    """

    class Origin(models.TextChoices):
        UPLOADED = "uploaded", "Uploaded"
        AGENT_CREATED = "agent_created", "Agent created"
        CANVAS_EXPORT = "canvas_export", "Canvas export"
        USER_EDIT = "user_edit", "User edit"
        RESTORE = "restore", "Restore"

    document = models.ForeignKey(
        DataRoomDocument,
        on_delete=models.CASCADE,
        related_name="versions",
    )
    # 0 == the original upload; later versions increment. Assigned by the
    # versioning service under a parent-row lock (see services/versioning.py).
    version_index = models.PositiveIntegerField(default=0)
    origin = models.CharField(
        max_length=20, choices=Origin.choices, default=Origin.UPLOADED,
    )
    status = models.CharField(
        max_length=20,
        choices=DataRoomDocument.Status.choices,
        default=DataRoomDocument.Status.UPLOADED,
        db_index=True,
    )
    # Working content (markdown). Empty for legacy v0 backfilled from existing
    # chunks — readers fall back to joined chunk text in that case.
    content = models.TextField(blank=True, default="")
    # Native source bytes — only set for upload-origin versions (e.g. v0).
    native_blob = models.FileField(
        upload_to="documents/%Y/%m/", blank=True, max_length=500,
    )
    native_filename = models.CharField(max_length=255, blank=True)
    mime_type = models.CharField(max_length=128, blank=True)
    size_bytes = models.PositiveIntegerField(null=True, blank=True)

    # Retrieval pointer mirror — only the document's active_searchable_version
    # has is_searchable=True. The app-DB pointer is authoritative; this flag and
    # the pgvector cmetadata mirror it for fast filtering.
    is_searchable = models.BooleanField(default=False, db_index=True)
    is_quarantined = models.BooleanField(default=False, db_index=True)
    is_partially_quarantined = models.BooleanField(default=False, db_index=True)
    quarantine_reason = models.TextField(blank=True, default="")

    processing_error = models.TextField(null=True, blank=True)
    requeue_count = models.PositiveSmallIntegerField(default=0)
    token_count = models.PositiveIntegerField(null=True, blank=True)
    parser_type = models.CharField(max_length=64, blank=True)
    chunking_strategy = models.CharField(max_length=64, blank=True)
    embedding_model = models.CharField(max_length=128, blank=True)

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="+",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["document", "version_index"]
        constraints = [
            models.UniqueConstraint(
                fields=["document", "version_index"],
                name="documents_version_unique_per_document",
            ),
        ]
        indexes = [
            models.Index(fields=["document", "version_index"]),
            models.Index(fields=["document", "is_searchable"]),
        ]

    def __str__(self) -> str:
        return f"v{self.version_index} of document {self.document_id}"


class DataRoomDocumentChunk(models.Model):
    class GuardrailScanState(models.TextChoices):
        # Progress marker for the adversarial-content scan (guardrails.tasks).
        # Lets a retried scan resume: chunks past a phase are not re-processed,
        # so events are not duplicated and classifier calls are not re-paid.
        PENDING = "pending", "Pending"  # not yet heuristic-scanned
        HEURISTIC_DONE = "heuristic_done", "Heuristic Done"  # awaiting classifier
        DONE = "done", "Done"  # fully scanned (or heuristic-blocked)

    # Chunks belong to a version. Reach the document via ``version.document``.
    version = models.ForeignKey(
        "DataRoomDocumentVersion",
        on_delete=models.CASCADE,
        related_name="chunks",
    )
    chunk_index = models.PositiveIntegerField()
    heading = models.CharField(max_length=512, null=True, blank=True)
    text = models.TextField()
    token_count = models.PositiveIntegerField()
    source_page_start = models.PositiveIntegerField(null=True, blank=True)
    source_page_end = models.PositiveIntegerField(null=True, blank=True)
    source_offset_start = models.PositiveIntegerField(null=True, blank=True)
    source_offset_end = models.PositiveIntegerField(null=True, blank=True)
    search_vector = SearchVectorField(null=True)
    is_quarantined = models.BooleanField(default=False, db_index=True)
    quarantine_reason = models.TextField(blank=True, default="")
    guardrail_scan_state = models.CharField(
        max_length=20,
        choices=GuardrailScanState.choices,
        default=GuardrailScanState.PENDING,
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["version", "chunk_index"]
        constraints = [
            models.UniqueConstraint(
                fields=["version", "chunk_index"],
                name="documents_chunk_unique_per_version",
            ),
        ]
        indexes = [
            models.Index(fields=["version", "chunk_index"], name="documents_chunk_version_idx"),
            GinIndex(fields=["search_vector"], name="chunk_search_vector_gin"),
        ]

    def __str__(self) -> str:
        return f"Chunk {self.chunk_index} of version {self.version_id}"


class DataRoomDocumentTag(models.Model):
    # Classifications attach to the version that was scanned. Reach the document
    # via ``version.document``. Document-level sensitivity is the union over
    # retained versions (see services/versioning.recompute_document_sensitivity).
    version = models.ForeignKey(
        "DataRoomDocumentVersion",
        on_delete=models.CASCADE,
        related_name="tags",
    )
    key = models.CharField(max_length=100)
    value = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["version", "key"],
                name="documents_tag_unique_per_version",
            ),
        ]
        ordering = ["key"]

    def __str__(self) -> str:
        return f"{self.key}={self.value} (version={self.version_id})"
