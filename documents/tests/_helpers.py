"""Shared test helpers for the versioned document model.

A document's chunks now belong to a ``DataRoomDocumentVersion``, and retrieval
keys off the document's ``active_searchable_version``. ``make_document`` builds a
document with a v0 version, sets both pointers, and attaches chunks — so tests can
create a retrievable document in one call.
"""
from __future__ import annotations

from documents.models import (
    DataRoomDocument,
    DataRoomDocumentChunk,
    DataRoomDocumentVersion,
)

READY = DataRoomDocument.Status.READY


def make_version(doc, *, version_index=0, status=READY, searchable=None,
                 is_quarantined=False, is_partially_quarantined=False,
                 origin=DataRoomDocumentVersion.Origin.UPLOADED, chunks=None,
                 make_active=True, make_current=True):
    """Create a version on ``doc`` with optional chunks.

    ``make_current`` points doc.current_version at it (the working head);
    ``make_active`` points doc.active_searchable_version at it (live/retrievable).
    """
    if searchable is None:
        searchable = (
            status == READY
            and not is_quarantined
            and not doc.is_quarantined
            and not doc.is_archived
        )
    version = DataRoomDocumentVersion.objects.create(
        document=doc, version_index=version_index, status=status,
        origin=origin, is_searchable=searchable, is_quarantined=is_quarantined,
        is_partially_quarantined=is_partially_quarantined,
    )
    if chunks:
        _add_chunks(version, chunks)
    updates = {}
    if make_current:
        updates["current_version"] = version
        doc.current_version = version
    if make_active:
        updates["active_searchable_version"] = version
        doc.active_searchable_version = version
    if updates:
        DataRoomDocument.objects.filter(pk=doc.pk).update(**updates)
    return version


def _add_chunks(version, chunks):
    objs = []
    for i, spec in enumerate(chunks):
        if isinstance(spec, str):
            spec = {"text": spec}
        text = spec["text"]
        objs.append(DataRoomDocumentChunk(
            version=version,
            chunk_index=spec.get("chunk_index", i),
            heading=spec.get("heading"),
            text=text,
            token_count=spec.get("token_count", max(1, len(text.split()))),
            is_quarantined=spec.get("is_quarantined", False),
        ))
    DataRoomDocumentChunk.objects.bulk_create(objs)


def make_document(data_room, uploaded_by, *, status=READY, chunks=None,
                  is_archived=False, is_quarantined=False, is_partially_quarantined=False,
                  searchable=None, original_filename="doc.md", **kwargs):
    """Create a document with a v0 version (+ chunks) and both pointers set."""
    doc = DataRoomDocument.objects.create(
        data_room=data_room, uploaded_by=uploaded_by, status=status,
        is_archived=is_archived, is_quarantined=is_quarantined,
        is_partially_quarantined=is_partially_quarantined,
        original_filename=original_filename, **kwargs,
    )
    make_version(
        doc, status=status, searchable=searchable, is_quarantined=is_quarantined,
        is_partially_quarantined=is_partially_quarantined, chunks=chunks,
    )
    return doc


def add_chunks(doc, chunks):
    """Attach chunks to a document's current (working) version."""
    version = doc.current_version or doc.active_searchable_version
    _add_chunks(version, chunks)
    return version
