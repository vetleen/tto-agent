from __future__ import annotations

import uuid

from django.conf import settings
from django.contrib.postgres.indexes import GinIndex
from django.contrib.postgres.search import SearchVectorField
from django.db import models


class DataRoom(models.Model):
    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, db_index=True)
    name = models.CharField(max_length=255)
    slug = models.SlugField(max_length=100, unique=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="data_rooms",
    )
    is_archived = models.BooleanField(default=False)
    is_shared = models.BooleanField(default=False)
    description = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at", "name"]

    def __str__(self) -> str:
        return self.name


class DataRoomDocument(models.Model):
    class Status(models.TextChoices):
        UPLOADED = "uploaded", "Uploaded"
        PROCESSING = "processing", "Processing"
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
    original_filename = models.CharField(max_length=255)
    mime_type = models.CharField(max_length=128, blank=True)
    size_bytes = models.PositiveIntegerField(null=True, blank=True)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.UPLOADED,
        db_index=True,
    )
    processing_error = models.TextField(null=True, blank=True)
    token_count = models.PositiveIntegerField(null=True, blank=True)
    parser_type = models.CharField(max_length=64, blank=True)
    chunking_strategy = models.CharField(max_length=64, blank=True)
    embedding_model = models.CharField(max_length=128, blank=True)
    description = models.TextField(blank=True, default="")
    transcript = models.TextField(blank=True, default="")
    transcription_model = models.CharField(max_length=128, blank=True)
    is_archived = models.BooleanField(default=False)
    doc_index = models.PositiveIntegerField(default=0)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

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


class DataRoomDocumentChunk(models.Model):
    document = models.ForeignKey(
        DataRoomDocument,
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
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["document", "chunk_index"]
        constraints = [
            models.UniqueConstraint(
                fields=["document", "chunk_index"],
                name="documents_chunk_unique_per_document",
            ),
        ]
        indexes = [
            models.Index(fields=["document", "chunk_index"]),
            GinIndex(fields=["search_vector"], name="chunk_search_vector_gin"),
        ]

    def __str__(self) -> str:
        return f"Chunk {self.chunk_index} of {self.document_id}"


class DataRoomDocumentTag(models.Model):
    document = models.ForeignKey(DataRoomDocument, on_delete=models.CASCADE, related_name="tags")
    key = models.CharField(max_length=100)
    value = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["document", "key"],
                name="documents_tag_unique_per_document",
            ),
        ]
        ordering = ["key"]

    def __str__(self) -> str:
        return f"{self.key}={self.value} (doc={self.document_id})"
