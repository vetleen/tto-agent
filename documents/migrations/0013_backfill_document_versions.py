"""Backfill a v0 version for every existing document and repoint chunks/tags.

For each existing DataRoomDocument we create a single ``version_index=0``
(origin=uploaded) version that represents the original, copy its status and
quarantine flags onto the version, repoint that document's chunks and tags to
it, and set both version pointers to v0.

The v0 ``content`` is left empty — we do NOT re-extract markdown here; readers
fall back to joined chunk text for legacy v0s. The original bytes stay on
``DataRoomDocument.original_file`` (no S3 copy into ``native_blob``).
"""
from __future__ import annotations

from django.db import migrations


def backfill_versions(apps, schema_editor):
    DataRoomDocument = apps.get_model("documents", "DataRoomDocument")
    DataRoomDocumentVersion = apps.get_model("documents", "DataRoomDocumentVersion")
    DataRoomDocumentChunk = apps.get_model("documents", "DataRoomDocumentChunk")
    DataRoomDocumentTag = apps.get_model("documents", "DataRoomDocumentTag")

    qs = DataRoomDocument.objects.all().order_by("pk")
    for doc in qs.iterator(chunk_size=500):
        is_searchable = (
            doc.status == "ready"
            and not doc.is_quarantined
            and not doc.is_archived
        )
        v0 = DataRoomDocumentVersion.objects.create(
            document=doc,
            version_index=0,
            origin="uploaded",
            status=doc.status,
            content="",
            native_filename=doc.original_filename,
            mime_type=doc.mime_type or "",
            size_bytes=doc.size_bytes,
            is_searchable=is_searchable,
            is_quarantined=doc.is_quarantined,
            is_partially_quarantined=doc.is_partially_quarantined,
            quarantine_reason=doc.quarantine_reason or "",
            token_count=doc.token_count,
            parser_type=doc.parser_type or "",
            chunking_strategy=doc.chunking_strategy or "",
            embedding_model=doc.embedding_model or "",
            created_by=doc.uploaded_by,
            processed_at=doc.processed_at,
        )
        DataRoomDocumentChunk.objects.filter(document_id=doc.pk).update(version=v0)
        DataRoomDocumentTag.objects.filter(document_id=doc.pk).update(version=v0)
        DataRoomDocument.objects.filter(pk=doc.pk).update(
            current_version=v0, active_searchable_version=v0,
        )


def reverse(apps, schema_editor):
    # Detach pointers and delete v0s; chunks/tags keep their document FK so the
    # forward op can re-run. Safe because 0012 fields are still nullable here.
    DataRoomDocument = apps.get_model("documents", "DataRoomDocument")
    DataRoomDocumentVersion = apps.get_model("documents", "DataRoomDocumentVersion")
    DataRoomDocumentChunk = apps.get_model("documents", "DataRoomDocumentChunk")
    DataRoomDocumentTag = apps.get_model("documents", "DataRoomDocumentTag")

    DataRoomDocument.objects.update(current_version=None, active_searchable_version=None)
    DataRoomDocumentChunk.objects.update(version=None)
    DataRoomDocumentTag.objects.update(version=None)
    DataRoomDocumentVersion.objects.all().delete()


class Migration(migrations.Migration):

    dependencies = [
        ("documents", "0012_dataroomdocument_name_dataroomdocumentversion_and_more"),
    ]

    operations = [
        migrations.RunPython(backfill_versions, reverse),
    ]
