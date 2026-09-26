"""``chat_attachment_view`` — let the agent (re-)view a file the user attached.

Attachments are only sent to the model automatically on the turn they were
uploaded (see ``ChatConsumer._enrich_with_attachments``); on later turns the
user's messages carry ``[Attached: #N …]`` markers and the agent views a file on
demand with this tool. Images and PDFs are queued as native assets (pathway
``"attachment"``) that the pipeline drains into the next request — visible for
the rest of this reply only; a presentation's slides are queued the same way from
the per-slide renders the worker stored (``chat/attachment_render.py``), a few at
a time via ``pages``. Everything else comes back as extracted text. The result
always states which ``representation`` the agent received.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from pydantic import BaseModel, Field

from llm.tools.interfaces import ContextAwareTool, ReasonBaseModel

logger = logging.getLogger(__name__)

# Extracted-text page size (matches document_read's per-call output cap).
ATTACHMENT_VIEW_TEXT_CAP = 32_000


class AttachmentViewInput(ReasonBaseModel):
    attachment_number: int = Field(
        description=(
            "Which attachment to view, by its number in the '# Attachments' list in "
            "your context (the same numbers appear in the [Attached: #N …] markers "
            "on the user's messages)."
        ),
    )
    mode: Literal["auto", "native", "extracted"] = Field(
        default="auto",
        description=(
            "'auto' shows images, PDFs and presentation slides natively and everything "
            "else as extracted text. 'native' insists on the native form (falls back to "
            "extracted text, with a reason, when that's impossible). 'extracted' returns "
            "text only — use it to read a PDF's or deck's text cheaply, or to page "
            "through a long file."
        ),
    )
    pages: str | None = Field(
        default=None,
        description=(
            "Optional 1-based page (PDF) or slide (presentation) selection, e.g. "
            "'3-5,12'. PDF native: only those pages are attached (as a smaller PDF), "
            "saving context. Presentation: those slides are attached as images (the "
            "first slides when omitted; a cap applies per call, and the result says "
            "which pages= to pass next). Extracted: only those pages'/slides' text is "
            "returned. Omit to view the whole PDF or the first slides."
        ),
    )
    char_offset: int = Field(
        default=0,
        ge=0,
        description=(
            "Extracted text only: character position to start from. A result that "
            "was cut off tells you the next_char_offset to continue with."
        ),
    )


def _supports(model_id: str | None, modality: str) -> bool:
    """Whether the run's model takes ``modality`` natively (unknown model → yes;
    the pipeline drain still degrades if it can't)."""
    if not model_id:
        return True
    from llm.display import supports_modality

    return supports_modality(model_id, modality)


class AttachmentViewTool(ContextAwareTool):
    """View a chat attachment natively (images/PDFs) or as extracted text."""

    name: str = "chat_attachment_view"
    section: str = "chat"  # always-on (gated per thread on having a sent attachment)
    # Attachments belong to the user's thread; sub-agents never see them.
    audience: str = "main"
    start_label: str = "Viewing attachment..."
    end_label: str = "Viewed attachment"
    description: str = (
        "View a file the user attached to this chat, by its number from the "
        "'# Attachments' list in your context (the same numbers appear in the "
        "[Attached: #N …] markers on the user's messages). Images, PDFs and "
        "presentations (.pptx) can be shown to you natively — you see the page, "
        "image or slide as it actually looks (layout, charts, visuals); a deck's "
        "slides come as images, a few per call, selected with `pages`. Word "
        "documents and text files can only be returned as extracted text, without "
        "their visual design. The result states which form you received "
        "(`representation`) and why, if it was degraded. Use `pages` to view part of "
        "a PDF or deck and mode='extracted' to read its text (long text is paged "
        "with `char_offset`). A native view is only visible to you during the reply "
        "in which you requested it — view the file again in a later reply if you "
        "need to look at it again."
    )
    args_schema: type[BaseModel] = AttachmentViewInput

    def end_label_for_result(self, result: dict) -> str | None:
        if not result:
            return None
        if result.get("status") != "ok":
            return "Couldn't view attachment"
        filename = result.get("filename") or "attachment"
        pages = result.get("pages")
        page_part = f", p. {str(pages).replace('-', '–')}" if pages else ""
        rep = result.get("representation")
        if rep == "slides":
            from chat.slide_view import format_numbers

            slides = [int(n) for n in (result.get("slides") or [])]
            noun = "slide" if len(slides) == 1 else "slides"
            return f"Viewed {filename} ({noun} {format_numbers(slides)} of {result.get('total_slides')})"
        if rep == "native":
            return f"Viewed {filename} (native{page_part})"
        if rep == "native_pages_as_images":
            return (
                f"Viewed {filename} (first {result.get('pages_attached')} of "
                f"{result.get('total_pages')} pages as images)"
            )
        if rep == "extracted":
            return f"Read {filename} (extracted text{page_part})"
        return None

    def _run(
        self,
        attachment_number: int,
        mode: str = "auto",
        pages: str | None = None,
        char_offset: int = 0,
        **kwargs,
    ) -> str:
        from core.file_types import KIND_DOCX, KIND_IMAGE, KIND_PDF, KIND_PPTX, KIND_TEXT

        from chat.services import attachment_kind, list_thread_attachments

        context = self.context
        thread_id = context.conversation_id if context else None
        if not thread_id:
            return _error("No thread context available.")

        pairs = list_thread_attachments(thread_id)
        if not pairs:
            return _error("There are no attachments on this chat.")
        att = dict(pairs).get(attachment_number)
        if att is None:
            return json.dumps({
                "status": "error",
                "message": f"No attachment #{attachment_number}.",
                "available_attachments": [
                    {"number": n, "filename": a.original_filename} for n, a in pairs
                ],
            })
        filename = att.original_filename
        if att.message_id is None:
            return _error(
                f"Attachment #{attachment_number} ('{filename}') hasn't been sent yet — "
                "it is still a draft in the message composer."
            )
        if att.message.is_redacted:
            return _error(
                f"Attachment #{attachment_number} ('{filename}') was removed by the "
                "content safety system and can't be viewed."
            )

        kind = attachment_kind(att)
        if kind not in (KIND_IMAGE, KIND_PDF, KIND_PPTX, KIND_DOCX, KIND_TEXT):
            return _error(f"Can't view '{filename}': unsupported file type.")

        try:
            with att.file.open("rb") as fh:
                file_bytes = fh.read()
        except Exception:
            logger.exception("chat_attachment_view: failed to read attachment %s", att.id)
            return _error("Could not read the attachment file.")

        base = {
            "status": "ok",
            "attachment_number": attachment_number,
            "filename": filename,
            "kind": kind,
        }
        model_id = getattr(context, "model_id", None)

        if kind == KIND_IMAGE:
            return self._view_image(att, file_bytes, base, mode, model_id)
        if kind == KIND_PDF:
            return self._view_pdf(att, file_bytes, base, mode, pages, char_offset, model_id)
        if kind == KIND_PPTX:
            return self._view_pptx(att, file_bytes, base, mode, pages, char_offset, model_id)

        reason = ""
        if mode == "native":
            reason = (
                "Word documents and text files can't be shown natively; "
                "extracted text returned."
            )
        text = self._extracted_text(att, file_bytes, kind)
        return self._extracted_result(base, text, char_offset, reason=reason)

    # --- per-kind paths -------------------------------------------------

    def _view_image(self, att, file_bytes, base, mode, model_id) -> str:
        import base64

        n = base["attachment_number"]
        if not _supports(model_id, "image"):
            return _error(
                f"The current model cannot view images; attachment #{n} is an image "
                "and has no text form."
            )
        if not self.context.try_add_native_asset({
            "kind": "image",
            "asset_id": "",
            "b64": base64.b64encode(file_bytes).decode("ascii"),
            "media_type": att.content_type or "image/png",
            "description": f"Attachment #{n} '{att.original_filename}'",
        }, pathway="attachment"):
            return _error(
                f"Attachment #{n} could not be attached — the attachment budget for "
                "this reply is exhausted."
            )
        result = {**base, "representation": "native", "note": "The image is attached below for you to view."}
        if mode == "extracted":
            result["note"] += " (Images are always shown natively.)"
        return json.dumps(result)

    def _view_pdf(self, att, file_bytes, base, mode, pages, char_offset, model_id) -> str:
        from chat.pdf_attach import attach_pdf_to_context, pdf_page_count
        from chat.services import get_or_extract_attachment_text

        n_pages = pdf_page_count(file_bytes)
        if n_pages and att.page_count != n_pages:
            att.page_count = n_pages
            att.save(update_fields=["page_count"])

        reason = ""
        if mode != "extracted" and not _supports(model_id, "pdf"):
            reason = "the current model cannot view PDFs natively"
        if mode != "extracted" and not reason:
            extracted = get_or_extract_attachment_text(
                att, file_bytes, user=_user(self.context)
            )
            outcome = attach_pdf_to_context(
                self.context, file_bytes,
                pathway="attachment",
                filename=att.original_filename,
                description=f"Attachment #{base['attachment_number']} '{att.original_filename}'",
                extracted_text=extracted[:ATTACHMENT_VIEW_TEXT_CAP],
                pages=pages,
            )
            if outcome.representation == "native":
                result = {
                    **base, "representation": "native",
                    "total_pages": n_pages or outcome.total_pages,
                    "pages_attached": outcome.pages_attached,
                    "note": "The PDF is attached below for you to view.",
                }
                if outcome.page_note:
                    result["pages"] = pages
                if outcome.compressed:
                    result["compressed"] = True
                    result["note"] = "The PDF is attached below for you to view (losslessly compressed to fit)."
                return json.dumps(result)
            if outcome.representation == "native_pages_as_images":
                text, _ = self._pdf_text(att, file_bytes, pages)
                result = self._extracted_payload(
                    {**base, "representation": "native_pages_as_images"}, text, char_offset
                )
                if result.get("status") != "ok":
                    return json.dumps(result)
                result.update({
                    "pages_attached": outcome.pages_attached,
                    "total_pages": n_pages or outcome.total_pages,
                    "reason": f"PDF {outcome.reason}",
                    "note": (
                        f"You are seeing a TRUNCATED view: the first {outcome.pages_attached} "
                        f"of {outcome.total_pages} pages are attached below as images. The "
                        "extracted text is in `content`. " + result.get("note", "")
                    ).strip(),
                })
                if outcome.page_note:
                    result["pages"] = pages
                return json.dumps(result)
            reason = (
                f"PDF {outcome.reason}" if outcome.over_page_cap
                else "the PDF is too large for the remaining attachment budget"
            )

        text, page_scoped = self._pdf_text(att, file_bytes, pages)
        payload_base = {**base, "representation": "extracted"}
        if page_scoped:
            payload_base["pages"] = pages
        result = self._extracted_payload(payload_base, text, char_offset)
        if result.get("status") == "ok":
            if reason:
                result["reason"] = reason
            if page_scoped:
                result["note"] = " ".join(filter(None, [
                    result.get("note"), "Page-scoped text omits embedded-image descriptions.",
                ]))
        return json.dumps(result)

    def _view_pptx(self, att, file_bytes, base, mode, pages, char_offset, model_id) -> str:
        """A deck: its worker-rendered slides as images (``representation``
        ``"slides"``), else the slide-by-slide extracted text with a reason."""
        from django.conf import settings

        from chat.models import Asset, ChatAttachment
        from chat.services import get_or_extract_attachment_text
        from chat.slide_view import format_numbers, queue_slide_images

        Render = ChatAttachment.PageRenderState
        n = base["attachment_number"]
        total = int(att.page_count or 0)
        if total <= 0:
            try:
                from documents.services.page_render import count_slides

                total = count_slides(file_bytes)
            except Exception:  # noqa: BLE001 — a broken deck still has its cached text
                logger.warning("chat_attachment_view: could not count slides of attachment %s", att.id, exc_info=True)
                total = 0
            if total:
                ChatAttachment.objects.filter(pk=att.pk).update(page_count=total)
                att.page_count = total

        reason = ""
        if mode != "extracted":
            state = att.page_render_state
            still_processing = (
                att.processing_state == ChatAttachment.ProcessingState.PENDING
                or state == Render.PENDING
            )
            if not _supports(model_id, "image"):
                reason = "the current model cannot view images, so the slides can't be shown"
            elif state in (Render.READY, Render.PARTIAL) and total > 0:
                outcome = queue_slide_images(
                    self.context,
                    assets=Asset.objects.filter(attachment=att, role=Asset.ROLE_PAGE_RENDER),
                    filename=att.original_filename, total=total, pages=pages,
                    pathway="attachment", log_ref=f"attachment {att.id}",
                )
                if outcome.shown:
                    noun = "Slide" if len(outcome.shown) == 1 else "Slides"
                    verb = "is" if len(outcome.shown) == 1 else "are"
                    result = {
                        **base,
                        "representation": "slides",
                        "slides": outcome.shown,
                        "total_slides": total,
                        "note": (
                            f"{noun} {format_numbers(outcome.shown)} of {total} {verb} attached "
                            f"below as images{outcome.selection_note}."
                            + outcome.missing_note() + outcome.budget_note()
                            + outcome.cap_note() + outcome.more_hint()
                        ),
                    }
                    if pages:
                        result["pages"] = pages
                    return json.dumps(result)
                if outcome.budget_hit:
                    reason = (
                        "the attachment budget for this reply is exhausted, so no slides "
                        "could be attached"
                    )
                else:
                    reason = "no slide previews are stored for the selected slides"
            elif still_processing:
                reason = (
                    "slide previews are still being prepared — view this attachment again "
                    "shortly to see the slides"
                )
            elif state == Render.SKIPPED:
                cap = getattr(settings, "CHAT_ATTACHMENT_RENDER_MAX_SLIDES", 50)
                reason = (
                    att.processing_error
                    or f"slide previews are only rendered for decks up to {cap} slides"
                )
            elif state == Render.FAILED:
                reason = "the slide previews could not be rendered"
            else:
                reason = "no slide previews exist for this deck"
            if reason:
                reason = f"Attachment #{n}: {reason}; extracted text returned."

        text = get_or_extract_attachment_text(att, file_bytes, user=_user(self.context))
        payload_base = {**base, "representation": "extracted"}
        scoped = _slides_text(text, pages, total) if pages else ""
        if scoped:
            payload_base["pages"] = pages
            text = scoped
        return self._extracted_result(payload_base, text, char_offset, reason=reason)

    # --- helpers -------------------------------------------------------

    def _pdf_text(self, att, file_bytes, pages) -> tuple[str, bool]:
        """``(text, page_scoped)`` — the selected pages' text when ``pages`` is
        given and readable, else the full cached extraction."""
        from chat.pdf_attach import extract_pdf_pages_text, parse_page_ranges, pdf_page_count
        from chat.services import get_or_extract_attachment_text

        if pages:
            indices = parse_page_ranges(pages, pdf_page_count(file_bytes))
            page_texts = extract_pdf_pages_text(file_bytes, indices)
            if page_texts:
                return "\n\n".join(f"--- page {p} ---\n{t}" for p, t in page_texts), True
        return get_or_extract_attachment_text(att, file_bytes, user=_user(self.context)), False

    def _extracted_text(self, att, file_bytes, kind) -> str:
        from core.file_types import KIND_TEXT

        from chat.services import get_or_extract_attachment_text

        if kind == KIND_TEXT:
            return file_bytes.decode("utf-8", errors="replace")
        return get_or_extract_attachment_text(att, file_bytes, user=_user(self.context))

    def _extracted_payload(self, base: dict, text: str, char_offset: int) -> dict:
        text = text or ""
        total = len(text)
        if total and char_offset >= total:
            return {
                "status": "error",
                "message": f"char_offset {char_offset} is past the end (total_chars={total}).",
            }
        chunk = text[char_offset:char_offset + ATTACHMENT_VIEW_TEXT_CAP]
        result = {
            **base,
            "content": chunk,
            "char_offset": char_offset,
            "total_chars": total,
            "truncated": char_offset + len(chunk) < total,
        }
        if result["truncated"]:
            nxt = char_offset + len(chunk)
            result["next_char_offset"] = nxt
            result["note"] = f"Text was cut off; continue with char_offset={nxt} to read more."
        if not total:
            result["note"] = "No readable text was found in this file."
        return result

    def _extracted_result(self, base: dict, text: str, char_offset: int, *, reason: str = "") -> str:
        result = self._extracted_payload({**base, "representation": "extracted"}, text, char_offset)
        if reason and result.get("status") == "ok":
            result["reason"] = reason
        return json.dumps(result)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message})


def _slides_text(markdown: str, pages: str | None, total: int) -> str:
    """The ``## Slide N`` sections of a deck's extracted markdown that ``pages``
    selects, in deck order; ``""`` when nothing matches (caller keeps the whole
    text). ``total`` bounds the selection like a PDF's page count."""
    import re

    from chat.pdf_attach import parse_page_ranges

    if not pages or not markdown:
        return ""
    wanted = {i + 1 for i in parse_page_ranges(pages, total or 10_000)}
    if not wanted:
        return ""
    parts = re.split(r"(?m)^(?=## Slide \d+\b)", markdown)
    picked = []
    for part in parts:
        m = re.match(r"## Slide (\d+)\b", part)
        if m and int(m.group(1)) in wanted:
            picked.append(part.strip())
    return "\n\n".join(picked)


def _user(context):
    from chat.tools import _get_user_ctx

    return _get_user_ctx(context)


_registry = None
try:
    from llm.tools.registry import get_tool_registry

    _registry = get_tool_registry()
    _registry.register_tool(AttachmentViewTool())
except Exception:  # pragma: no cover - registration best-effort at import
    logger.exception("Failed to register chat_attachment_view tool")
