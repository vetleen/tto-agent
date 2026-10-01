"""Emails (.eml/.msg) attached in chat: split into the email plus its files.

At upload (and on the meeting → "minutes with Wilfred" copy) an email row is
created as usual, and every file it carries becomes an attachment row of its own
(``parent`` = the email, ``email_ordinal`` = its position in the email), held to
the same rules as if the user had attached it directly: allowed kinds, per-type
size caps, the thread's byte budget. A PDF inside an email is therefore shown
natively on the upload turn and paged with ``chat_attachment_view`` like any
other PDF; a deck gets slide renders.

The email row itself is processed like a Word document: its text (headers, body,
attachment list) becomes ``extracted_content``. Image attachments are described
inline there (description + ``[[image:…]]`` token) rather than split; each split
file is listed as "attached separately as #N", and anything not attached carries
its reason. A forwarded email is just another child, split recursively (bounded
by ``MAX_EMAIL_NESTING_DEPTH``).

The split happens synchronously at upload — parsing an email takes milliseconds —
so the client gets every id back at once and the consumer's linking and
turn-hold (``_link_attachments`` / ``_wait_for_attachments``) need no changes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from django.conf import settings
from django.core.files.base import ContentFile

logger = logging.getLogger(__name__)

_KIND_REFUSALS = {
    "spreadsheet": "spreadsheets can't be attached in chat — upload it to a data room instead",
    "audio": "audio files can't be attached in chat — upload it to a data room instead",
}


@dataclass
class SplitResult:
    children: list = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    bytes_used: int = 0


def email_extension(att) -> str:
    """``eml`` or ``msg`` for an email attachment row."""
    from core.file_types import extension_for_mime

    name = (att.original_filename or "").lower()
    if name.endswith(".msg"):
        return "msg"
    if name.endswith(".eml"):
        return "eml"
    return extension_for_mime(att.content_type) or "eml"


def part_content_type(part) -> str | None:
    """The canonical chat content type for an email part, or None if unknown."""
    from core.file_types import canonical_extension, canonical_mime_for_extension

    return canonical_mime_for_extension(part.ext) or canonical_mime_for_extension(canonical_extension(part.ext))


def part_refusal(part) -> str | None:
    """Why *part* can't become a chat attachment (None when it can). Pure — the
    email's text recomputes it to explain a missing file."""
    from chat.services import max_size_for_content_type
    from core.file_types import CHAT_KINDS, kind_for_extension

    if not part.data:
        return "the attachment could not be read"
    kind = kind_for_extension(part.ext)
    if kind not in CHAT_KINDS:
        return _KIND_REFUSALS.get(kind, "unsupported file type")
    max_size = max_size_for_content_type(part_content_type(part) or "")
    if part.size > max_size:
        return f"too large (max {max_size // (1024 * 1024)} MB)"
    return None


def _max_depth() -> int:
    from documents.services.chunking import MAX_EMAIL_NESTING_DEPTH

    return MAX_EMAIL_NESTING_DEPTH


def split_email_attachment(email_att, data: bytes, *, byte_budget: int, depth: int = 0) -> SplitResult:
    """Create one child attachment row per file in *email_att* (bytes *data*).

    *byte_budget* is what the thread may still store; children beyond it — or
    beyond ``DOCUMENT_EMAIL_MAX_SPLIT_ATTACHMENTS`` — are reported in
    ``warnings`` instead. Rows are created in order (email first), so the
    thread's ``#N`` numbering keeps an email's files together.
    """
    from chat.attachment_processing import dispatch_after_commit, initial_processing_state
    from chat.models import ChatAttachment
    from core.file_types import KIND_EMAIL, is_image_extension, kind_for_extension
    from documents.services.email_split import parse_email

    result = SplitResult()
    try:
        parsed = parse_email(data, email_extension(email_att))
    except Exception:
        logger.info("chat email split: attachment_id=%s could not be parsed", email_att.id, exc_info=True)
        result.warnings.append(f"{email_att.original_filename}: the email could not be read.")
        return result

    max_split = getattr(settings, "DOCUMENT_EMAIL_MAX_SPLIT_ATTACHMENTS", 25)
    created = 0
    for part in parsed.parts:
        if is_image_extension(part.ext):
            continue  # described inline in the email's text
        reason = part_refusal(part)
        is_email = kind_for_extension(part.ext) == KIND_EMAIL
        if reason is None and is_email and depth + 1 >= _max_depth():
            reason = "too many levels of forwarded emails"
        if reason is None and created >= max_split:
            reason = f"the email has more than {max_split} attachments"
        if reason is None and part.size > byte_budget - result.bytes_used:
            reason = "this chat has reached its attachment limit"
        if reason:
            result.warnings.append(f"{part.filename}: not attached — {reason}.")
            continue

        ct = part_content_type(part)
        child = ChatAttachment.objects.create(
            thread_id=email_att.thread_id,
            uploaded_by_id=email_att.uploaded_by_id,
            file=ContentFile(part.data, name=part.filename[:255]),
            original_filename=part.filename[:255],
            content_type=ct,
            size_bytes=part.size,
            processing_state=initial_processing_state(ct),
            parent=email_att,
            email_ordinal=part.ordinal,
        )
        dispatch_after_commit(child)
        created += 1
        result.children.append(child)
        result.bytes_used += part.size
        if is_email:
            nested = split_email_attachment(
                child, part.data, byte_budget=byte_budget - result.bytes_used, depth=depth + 1,
            )
            result.children.extend(nested.children)
            result.warnings.extend(nested.warnings)
            result.bytes_used += nested.bytes_used
    return result


def attachment_json(att) -> dict:
    """The upload/reattach response entry for one attachment row."""
    return {
        "id": str(att.id),
        "filename": att.original_filename,
        "content_type": att.content_type,
        "size_bytes": att.size_bytes,
        "processing_state": att.processing_state,
        "parent_id": str(att.parent_id) if att.parent_id else None,
    }


class _ChatAttachmentNotes:
    """``attachment_handler`` for an email row's text: names where each file went."""

    def __init__(self, email_att):
        from chat.services import list_thread_attachments

        self.children = {c.email_ordinal: c for c in email_att.children.all() if c.email_ordinal}
        self.numbers = {a.id: n for n, a in list_thread_attachments(email_att.thread_id)}

    def handle(self, part) -> str:
        child = self.children.get(part.ordinal)
        if child is not None:
            n = self.numbers.get(child.id)
            return f"attached separately as #{n}" if n else "attached separately"
        reason = part_refusal(part) or "not attached by the upload (attachment limit)"
        return f"not attached: {reason}"


def email_attachment_text(att, file_bytes: bytes, *, image_sink=None) -> str:
    """The email row's text: headers, body, and its attachment list (image
    attachments described inline through *image_sink*)."""
    from documents.services.chunking import email_to_markdown

    return email_to_markdown(
        file_bytes, email_extension(att), image_sink=image_sink, attachment_handler=_ChatAttachmentNotes(att),
    )
