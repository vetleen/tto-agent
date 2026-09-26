"""Slide renders for ``.pptx`` chat attachments.

The rendering machinery (optimize → split into batches → Gotenberg → rasterize →
store) lives in :mod:`documents.services.page_render` and is owner-agnostic: it
talks to a ``RenderTarget``. This module is the chat-attachment implementation of
that interface, so a deck attached to a chat (or copied in from a meeting) gets the
same per-slide JPEGs a data-room deck gets — stored as ``chat.Asset`` rows owned by
the ``ChatAttachment`` (``role=page_render``, unique per ``(attachment, page_number)``).

State lives on ``ChatAttachment.page_render_state`` (same values as the data-room
version's ``PageRenderState``); ``page_count`` and ``processing_error`` are written
alongside. There is no attempt counter here — Celery's ``max_retries`` on the
processing task bounds retries — and no heartbeat, because attachments have no
stale sweeper.

Upload names stay opaque: the trace id is ``a-<attachment uuid>``, so the render
service's logs never see a filename.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from documents.services.page_render import _int_setting, render_pages, store_render_asset

logger = logging.getLogger(__name__)


@dataclass
class AttachmentTarget:
    """``RenderTarget`` over a ``chat.ChatAttachment`` instance."""

    attachment: object  # chat.models.ChatAttachment (lazy import: chat models load after documents)

    @property
    def label(self) -> str:
        return f"attachment_id={self.attachment.id}"

    @property
    def trace_id(self) -> str:
        return f"a-{self.attachment.id}"

    @property
    def max_slides(self) -> int:
        return _int_setting("CHAT_ATTACHMENT_RENDER_MAX_SLIDES", 50)

    def bump_attempt(self) -> None:
        # Celery ``max_retries`` on process_chat_attachment bounds attempts.
        return None

    def qualifies(self) -> bool:
        from chat.services import attachment_kind
        from core.file_types import KIND_PPTX

        return attachment_kind(self.attachment) == KIND_PPTX

    def read_bytes(self) -> bytes | None:
        source = self.attachment.file
        if not source:
            return None
        try:
            with source.open("rb") as fh:
                return fh.read()
        except Exception:  # noqa: BLE001
            logger.warning(
                "attachment_render: could not read bytes for attachment_id=%s",
                self.attachment.id, exc_info=True,
            )
            return None

    def existing_pages(self) -> set[int]:
        from chat.models import Asset

        return set(
            Asset.objects.filter(attachment=self.attachment, role=Asset.ROLE_PAGE_RENDER)
            .exclude(page_number__isnull=True)
            .values_list("page_number", flat=True)
        )

    def store(self, page_number: int, jpeg: bytes) -> None:
        store_render_asset({"attachment": self.attachment}, page_number, jpeg)

    def set_state(self, state: str, *, error: str | None = None, page_count: int | None = None) -> None:
        from chat.models import ChatAttachment

        fields = {"page_render_state": state}
        if error is not None:
            fields["processing_error"] = error
        if page_count is not None:
            fields["page_count"] = page_count
        ChatAttachment.objects.filter(pk=self.attachment.pk).update(**fields)

    def touch(self) -> None:
        # No stale sweeper for attachments; nothing to heartbeat.
        return None


def render_attachment_pages(attachment, *, count_attempt: bool = True) -> str:
    """Render every missing slide of a pptx chat attachment; returns the final state.

    Raises ``RenderBusy`` / ``RenderUnavailable`` like the data-room path so the
    processing task can retry with backoff and resume at the first missing slide.
    """
    return render_pages(AttachmentTarget(attachment), count_attempt=count_attempt)


__all__ = ["AttachmentTarget", "render_attachment_pages"]
