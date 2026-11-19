from __future__ import annotations

import logging

from django.db.models.signals import post_delete, pre_delete
from django.dispatch import receiver

from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentVersion

logger = logging.getLogger(__name__)


@receiver(post_delete, sender=DataRoomDocumentVersion)
def delete_native_blob_on_version_delete(sender, instance, **kwargs):
    """Remove a version's native source bytes from storage when the version is deleted.

    Fires for prune drops and for the document/data-room CASCADE — without it a
    pruned or erased version's native file would linger in S3.
    """
    file_field = instance.native_blob
    if not file_field:
        return
    name = file_field.name
    if not name:
        return
    try:
        file_field.storage.delete(name)
    except Exception:
        logger.exception(
            "Failed to delete native blob for DataRoomDocumentVersion id=%s path=%s",
            instance.pk, name,
        )


@receiver(pre_delete, sender=DataRoomDocument)
def redact_llm_call_logs_on_document_delete(sender, instance, **kwargs):
    """GDPR Art. 17: deleting a document scrubs the content (prompt, raw_output,
    tool schemas) of the LLMCallLog rows its processing wrote (description, PII
    scan, chunk scan, picture descriptions — ``LLMCallLog.document_id``). The
    rows stay for cost/usage analytics. Fires per document for direct deletes,
    queryset bulk deletes and the DataRoom cascade alike."""
    from llm.models import LLMCallLog
    from llm.retention import redact_llm_call_logs

    redact_llm_call_logs(LLMCallLog.objects.filter(document_id=instance.pk))


@receiver(pre_delete, sender=DataRoom)
def redact_llm_call_logs_on_data_room_delete(sender, instance, **kwargs):
    """As above for the rows tagged with the data room itself (its description,
    and any document row whose document is already gone)."""
    from llm.models import LLMCallLog
    from llm.retention import redact_llm_call_logs

    redact_llm_call_logs(LLMCallLog.objects.filter(data_room_id=instance.pk))


@receiver(post_delete, sender=DataRoomDocument)
def delete_original_file_on_document_delete(sender, instance, **kwargs):
    """Remove the stored binary from storage (S3 or local) when a document row is deleted.

    Django's FileField does not delete the underlying file on model delete; without this
    the original upload would persist in S3 after a user deletes a document or data room,
    defeating GDPR erasure.
    """
    file_field = instance.original_file
    if not file_field:
        return
    name = file_field.name
    if not name:
        return
    try:
        file_field.storage.delete(name)
    except Exception:
        logger.exception(
            "Failed to delete original file for DataRoomDocument id=%s path=%s",
            instance.pk,
            name,
        )


@receiver(post_delete, sender=DataRoomDocument)
def delete_vectors_on_document_delete(sender, instance, **kwargs):
    """Remove the document's embedding rows from the vector store on delete.

    The pgvector table (langchain_pg_embedding) stores the full chunk text
    alongside each embedding; without this the deleted document's content would
    persist there indefinitely, defeating GDPR erasure. Fires per instance for
    direct deletes, queryset bulk deletes, and the DataRoom cascade alike.
    """
    from documents.services.vector_store import delete_vectors_for_document

    try:
        delete_vectors_for_document(instance.pk)
    except Exception:
        logger.exception(
            "Failed to delete vectors for DataRoomDocument id=%s", instance.pk
        )
