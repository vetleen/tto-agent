"""Split a data-room email's attachments into documents of their own.

When an .eml/.msg is processed, every non-image attachment is handed to
``EmailAttachmentSplitter.handle`` (via ``load_documents(attachment_handler=…)``)
instead of being flattened into the email's text. Each one becomes an ordinary
data-room document — same checks, same pipeline as a direct upload, so a PDF is
viewable natively with page numbers, a deck gets slide renders, a workbook the
spreadsheet pipeline. The email's own text lists each attachment with a note on
where it went (or why it wasn't added).

Children are documents in their own right: no FK, no cascade. Provenance rides
on their display name (``"<email> › <file>"``) and v0 tags. A nested email is
just another child, which splits its own attachments when it is processed
(bounded by ``MAX_EMAIL_NESTING_DEPTH``). The parent never waits on its
children — with a handful of worker slots that could deadlock.

Images are not split: the loader describes them inline (description + token).
"""
from __future__ import annotations

import logging

from django.conf import settings
from django.core.files.base import ContentFile

from core.files import sha256_of_bytes
from documents.models import DataRoomDocument, DataRoomDocumentTag
from documents.services import uploads

logger = logging.getLogger(__name__)

SOURCE_TAG = "email_attachment"
TAG_PARENT_DOC = "email_parent_doc"
TAG_PARENT_VERSION = "email_parent_version"
TAG_PARENT_FILENAME = "email_parent_filename"
TAG_ORDINAL = "email_attachment_ordinal"
TAG_DEPTH = "email_depth"

DISPLAY_NAME_MAX = 75
_SEP = " › "


def split_display_name(parent_name: str, filename: str, max_length: int = DISPLAY_NAME_MAX) -> str:
    """``"<parent> › <filename>"`` capped at *max_length*, shortening the parent
    part (never the attachment's own filename) with an ellipsis."""
    full = f"{parent_name}{_SEP}{filename}"
    if len(full) <= max_length:
        return full
    keep = max_length - len(_SEP) - len(filename) - 1
    if keep < 8:
        return filename[:max_length]
    return f"{parent_name[:keep].rstrip()}…{_SEP}{filename}"


def email_depth_of(doc) -> int:
    """How deep in an email-forward chain *doc* sits (0 = uploaded directly)."""
    value = (
        DataRoomDocumentTag.objects.filter(version__document=doc, key=TAG_DEPTH)
        .values_list("value", flat=True).first()
    )
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


class EmailAttachmentSplitter:
    """Attachment handler for one email version (see module docstring).

    ``outcomes`` collects one record per handled attachment for
    ``processing_metadata["email_attachments"]`` (the UI's warning badge reads it).
    """

    def __init__(self, doc, version, *, depth: int = 0):
        self.doc = doc
        self.version = version
        self.depth = depth
        self.outcomes: list[dict] = []
        self.created = 0
        self._remaining_slots: int | None = None

    # -- helpers ----------------------------------------------------------
    def _record(self, part, outcome: str, *, reason: str = "", child=None) -> None:
        self.outcomes.append({
            "ordinal": part.ordinal,
            "filename": part.filename,
            "size": part.size,
            "outcome": outcome,
            "reason": reason,
            "doc_id": child.id if child is not None else None,
            "doc_index": child.doc_index if child is not None else None,
        })

    def _skip(self, part, reason: str) -> str:
        self._record(part, "skipped", reason=reason)
        return f"not added: {reason.rstrip('.')}"

    def _existing_child(self, part):
        return (
            DataRoomDocument.objects.filter(
                versions__tags__key=TAG_PARENT_VERSION,
                versions__tags__value=str(self.version.id),
            )
            .filter(versions__tags__key=TAG_ORDINAL, versions__tags__value=str(part.ordinal))
            .order_by("id").first()
        )

    @staticmethod
    def _ref(doc) -> str:
        return f'#{doc.doc_index} "{doc.display_name}"'

    # -- handler protocol -------------------------------------------------
    def handle(self, part) -> str:
        """Split *part* out of the email. Returns the note listed beside it."""
        from core.file_types import KIND_EMAIL, canonical_mime_for_extension, kind_for_extension

        # A retry of this same version (task retry, stale-sweeper requeue)
        # re-runs extraction: reuse the child it already made.
        existing = self._existing_child(part)
        if existing is not None:
            self._record(part, "split", child=existing)
            return f"added to this data room as {self._ref(existing)}"

        if not part.data:
            return self._skip(part, "the attachment could not be read.")

        filename = uploads.safe_original_filename(part.filename, max_length=DISPLAY_NAME_MAX)
        mime = canonical_mime_for_extension(part.ext) or ""
        reason = uploads.check_file(self.doc.uploaded_by, filename, part.size, mime)
        if reason:
            return self._skip(part, reason)
        if kind_for_extension(part.ext) == KIND_EMAIL and self.depth + 1 >= _max_depth():
            return self._skip(part, "too many levels of forwarded emails.")
        max_split = getattr(settings, "DOCUMENT_EMAIL_MAX_SPLIT_ATTACHMENTS", 25)
        if self.created >= max_split:
            return self._skip(part, f"this email has more than {max_split} attachments; upload the rest directly.")

        sha = sha256_of_bytes(part.data)
        dup = uploads.duplicate_in_data_room(self.doc.data_room, sha)
        if dup is not None:
            self._record(part, "duplicate", child=dup)
            return f"identical to {self._ref(dup)}, already in this data room"

        if self._remaining_slots is None:
            self._remaining_slots = uploads.in_flight_remaining(self.doc.uploaded_by)
        if self._remaining_slots <= 0:
            return self._skip(part, "too many files are processing right now; upload it directly later.")

        child = DataRoomDocument.objects.create(
            data_room=self.doc.data_room,
            uploaded_by=self.doc.uploaded_by,
            original_file=ContentFile(
                part.data, name=uploads.safe_original_filename(part.filename, max_length=180),
            ),
            original_filename=filename,
            name=split_display_name(self.doc.display_name, filename),
            mime_type=mime,
            size_bytes=part.size,
            content_sha256=sha,
            status=DataRoomDocument.Status.UPLOADED,
        )
        self.created += 1
        # Tags go on before the child is queued: a nested email reads its depth
        # from them when it is processed.
        queued = uploads.queue_new_document(child, tags={
            "source": SOURCE_TAG,
            TAG_PARENT_DOC: self.doc.id,
            TAG_PARENT_VERSION: self.version.id,
            TAG_PARENT_FILENAME: self.doc.original_filename,
            TAG_ORDINAL: part.ordinal,
            TAG_DEPTH: self.depth + 1,
        })
        if queued is None:
            self._record(part, "failed", reason="processing could not be started.", child=child)
            return f"added to this data room as {self._ref(child)}, but its processing could not be started"
        self._remaining_slots -= 1
        logger.info(
            "email_attachments: split document_id=%s from version_id=%s ordinal=%s",
            child.id, self.version.id, part.ordinal,
        )
        self._record(part, "split", child=child)
        return f"added to this data room as {self._ref(child)}"


def _max_depth() -> int:
    from documents.services.chunking import MAX_EMAIL_NESTING_DEPTH

    return MAX_EMAIL_NESTING_DEPTH
