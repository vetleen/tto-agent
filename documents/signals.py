from __future__ import annotations

import logging

from django.db.models.signals import post_delete
from django.dispatch import receiver

from documents.models import DataRoomDocument

logger = logging.getLogger(__name__)


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
