"""Chat services — summarisation, image description, and docx import helpers."""

from __future__ import annotations

import base64
import logging
import re
from typing import TYPE_CHECKING

from core.file_types import (
    CHAT_KINDS,
    KIND_DOCX,
    KIND_IMAGE,
    KIND_PDF,
    KIND_TEXT,
    canonical_mimes_for_kinds,
)
from core.styles import FOOTNOTE_MARKER
from llm.core.model_factory import detect_provider

if TYPE_CHECKING:
    from django.core.files.uploadedfile import UploadedFile

    from chat.models import ChatMessage

logger = logging.getLogger(__name__)

CANVAS_MAX_CHARS = 75_000
MAX_CANVASES_PER_THREAD = 10
MAX_ACTIVE_CANVASES = 3
EMAIL_BLOCK_RE = re.compile(r"```email\s*\n(.*?)```", re.DOTALL)

# ATX heading (up to 3 leading spaces, 1-6 hashes, then a space or end-of-line).
# `#hashtag` / `###Bad` (no space) are intentionally not headings.
_ATX_HEADING_RE = re.compile(r"^( {0,3})(#{1,6})(\s.*|)$")
# Fenced code delimiter: 3+ backticks or tildes, optionally indented/with info.
_CODE_FENCE_RE = re.compile(r"^( {0,3})(`{3,}|~{3,})(.*)$")


def _iter_markdown_lines_skipping_code(lines):
    """Yield ``(line, in_code)`` tracking fenced code blocks.

    A fence opens on the first ``` ``` ``` / ``~~~`` run and closes on a later
    run of the same character that is at least as long, mirroring CommonMark.
    """
    fence_char = None
    fence_len = 0
    for line in lines:
        fence = _CODE_FENCE_RE.match(line)
        if fence:
            marker = fence.group(2)
            if fence_char is None:
                fence_char, fence_len = marker[0], len(marker)
                yield line, True  # the fence line itself is "in code"
                continue
            if marker[0] == fence_char and len(marker) >= fence_len:
                fence_char, fence_len = None, 0
            yield line, True
            continue
        yield line, fence_char is not None


def normalize_heading_levels(markdown: str) -> str:
    """Promote ATX headings so the shallowest level present becomes ``#`` (H1).

    The whole document is shifted by the same delta, preserving the relative
    hierarchy (``## / ###`` → ``# / ##``). Fenced code blocks are skipped so a
    ``#`` comment inside code is never mistaken for a heading. Returns the input
    unchanged when there are no headings or the top level is already ``#``.
    """
    if not markdown or "#" not in markdown:
        return markdown

    lines = markdown.split("\n")

    min_level = 7
    for line, in_code in _iter_markdown_lines_skipping_code(lines):
        if in_code:
            continue
        m = _ATX_HEADING_RE.match(line)
        if m:
            min_level = min(min_level, len(m.group(2)))

    if min_level >= 7 or min_level == 1:
        return markdown

    shift = min_level - 1
    out = []
    for line, in_code in _iter_markdown_lines_skipping_code(lines):
        m = None if in_code else _ATX_HEADING_RE.match(line)
        if m:
            indent, hashes, rest = m.group(1), m.group(2), m.group(3)
            new_level = max(1, len(hashes) - shift)
            out.append(f"{indent}{'#' * new_level}{rest}")
        else:
            out.append(line)
    return "\n".join(out)


# Markdown footnote citations: inline reference `[^id]` and definition `[^id]: text`.
_FOOTNOTE_DEF_RE = re.compile(r"^(\s*)\[\^([^\]]+)\]:\s?(.*)$")
_FOOTNOTE_REF_RE = re.compile(r"\[\^([^\]]+)\]")


def render_citations(markdown: str) -> str:
    """Turn Markdown footnote citations into export-friendly markup.

    LLM canvases use footnote syntax (``text.[^1]`` plus a ``[^1]: source``
    list), which the stock ``markdown`` render leaves as literal ``[^1]`` text.
    We can't use the ``footnotes`` extension: these docs reuse labels per
    section (each section has its own Sources list), which the extension would
    collide and relocate. Instead, line by line: inline references become
    ``<sup>`` superscripts and definitions become numbered list items, keeping
    each Sources list where it is. Fenced code blocks are skipped.
    """
    if not markdown or "[^" not in markdown:
        return markdown

    out = []
    for line, in_code in _iter_markdown_lines_skipping_code(markdown.split("\n")):
        if in_code:
            out.append(line)
            continue
        definition = _FOOTNOTE_DEF_RE.match(line)
        if definition:
            label, rest = definition.group(2), definition.group(3)
            # Numeric labels → a real ordered-list item (clean references list);
            # non-numeric labels can't start an <ol>, so keep a bracketed marker.
            # The marker after the number tags the item so the export can shrink
            # the sources list (see core.styles.FOOTNOTE_MARKER).
            out.append(f"{label}. {FOOTNOTE_MARKER}{rest}" if label.isdigit() else f"\\[{label}\\] {rest}")
        else:
            out.append(_FOOTNOTE_REF_RE.sub(lambda m: f"<sup>{m.group(1)}</sup>", line))
    return "\n".join(out)


def resolve_canvas(thread_id, canvas_name=None):
    """Resolve a canvas by name or fall back to the most recently activated canvas.

    Returns (canvas, error_msg). One of the two is always None.
    """
    from chat.models import ChatCanvas

    if canvas_name:
        try:
            canvas = ChatCanvas.objects.select_related("accepted_checkpoint").get(
                thread_id=thread_id, title=canvas_name,
            )
            return canvas, None
        except ChatCanvas.DoesNotExist:
            return None, f"No canvas named '{canvas_name}' in this thread."

    canvas = (
        ChatCanvas.objects.filter(thread_id=thread_id, is_active=True)
        .select_related("accepted_checkpoint")
        .order_by("-last_activated_at")
        .first()
    )
    if canvas:
        return canvas, None
    return None, "No active canvas in this thread."


def activate_canvas(thread_id, canvas):
    """Mark a canvas as active (LLM-visible) and update the UI tab pointer.

    If already active, bumps last_activated_at. Enforces the 3-canvas cap
    by deactivating the oldest-activated canvas when needed.
    """
    from django.utils import timezone

    from chat.models import ChatCanvas, ChatThread

    now = timezone.now()

    if canvas.is_active:
        canvas.last_activated_at = now
        canvas.save(update_fields=["last_activated_at"])
    else:
        active_count = ChatCanvas.objects.filter(
            thread_id=thread_id, is_active=True,
        ).count()

        if active_count >= MAX_ACTIVE_CANVASES:
            excess = active_count - MAX_ACTIVE_CANVASES + 1
            oldest_pks = list(
                ChatCanvas.objects.filter(thread_id=thread_id, is_active=True)
                .order_by("last_activated_at")
                .values_list("pk", flat=True)[:excess]
            )
            ChatCanvas.objects.filter(pk__in=oldest_pks).update(is_active=False)

        canvas.is_active = True
        canvas.last_activated_at = now
        canvas.save(update_fields=["is_active", "last_activated_at"])

    ChatThread.objects.filter(pk=thread_id).update(active_canvas=canvas)


def set_active_canvases(thread_id, canvas_names):
    """Deactivate ALL canvases for the thread, then activate the named ones.

    Returns (activated_list, errors).
    """
    from django.utils import timezone

    from chat.models import ChatCanvas

    now = timezone.now()
    ChatCanvas.objects.filter(thread_id=thread_id).update(is_active=False)

    activated = []
    errors = []
    for name in canvas_names[:MAX_ACTIVE_CANVASES]:
        try:
            canvas = ChatCanvas.objects.get(thread_id=thread_id, title=name)
            canvas.is_active = True
            canvas.last_activated_at = now
            canvas.save(update_fields=["is_active", "last_activated_at"])
            activated.append(canvas)
        except ChatCanvas.DoesNotExist:
            errors.append(f"No canvas named '{name}' in this thread.")

    return activated, errors


def set_active_canvas(thread_id, canvas):
    """Activate a single canvas and update the UI tab pointer."""
    activate_canvas(thread_id, canvas)


def snapshot_user_edits(canvas):
    """Checkpoint uncommitted user edits before the AI mutates a canvas.

    If the canvas content has drifted from its latest checkpoint (the user
    edited in the UI but no checkpoint captured it yet), save a ``user_save``
    checkpoint first so an AI write/edit can never silently clobber user work
    — it stays recoverable from version history.
    """
    from chat.models import CanvasCheckpoint

    latest = (
        CanvasCheckpoint.objects.filter(canvas=canvas)
        .order_by("-order")
        .first()
    )
    if latest is None:
        drifted = bool(canvas.content)  # nothing to preserve on a blank canvas
    else:
        drifted = latest.content != canvas.content or latest.title != canvas.title
    if drifted:
        create_canvas_checkpoint(
            canvas, source="user_save",
            description="Auto-saved before AI edit",
        )


def create_canvas_checkpoint(canvas, source, description=""):
    """Create a new checkpoint for the given canvas, using its current title/content.

    For AI edits, if the latest checkpoint is already an ai_edit, update it
    in-place so that multiple tool calls in one assistant turn produce only
    one checkpoint.
    """
    from chat.models import CanvasCheckpoint

    latest = (
        CanvasCheckpoint.objects.filter(canvas=canvas)
        .order_by("-order")
        .first()
    )

    # Coalesce consecutive AI edits into a single checkpoint, but never
    # update the accepted checkpoint (that's the diff baseline).
    if (
        source == "ai_edit"
        and latest
        and latest.source == "ai_edit"
        and latest.pk != canvas.accepted_checkpoint_id
    ):
        latest.title = canvas.title
        latest.content = canvas.content
        latest.description = description
        latest.save(update_fields=["title", "content", "description"])
        return latest

    last_order = latest.order if latest else 0
    return CanvasCheckpoint.objects.create(
        canvas=canvas,
        title=canvas.title,
        content=canvas.content,
        source=source,
        description=description,
        order=last_order + 1,
    )


def save_canvas_to_data_room(canvas, data_room, user):
    """Save a canvas's content as a markdown document in *data_room* and enqueue processing.

    The created document goes through the normal upload pipeline (chunk → embed →
    guardrails → PII scan). The caller is responsible for verifying that *user*
    may write to *data_room*. Returns the created ``DataRoomDocument``.
    """
    from django.core.files.base import ContentFile

    from documents.models import DataRoomDocument, DataRoomDocumentTag, DataRoomDocumentVersion
    from documents.services.versioning import create_version

    title = canvas.title or "Untitled document"
    content = canvas.content or ""

    safe_title = (
        "".join(c for c in title if c.isalnum() or c in " _-").strip() or "document"
    )
    filename = f"{safe_title}.md"
    file_bytes = content.encode("utf-8")
    file_content = ContentFile(file_bytes, name=filename)

    doc = DataRoomDocument.objects.create(
        data_room=data_room,
        uploaded_by=user,
        original_file=file_content,
        original_filename=filename,
        name=title,
        mime_type="text/markdown",
        size_bytes=len(file_bytes),
        status=DataRoomDocument.Status.UPLOADED,
    )
    # v0 carries the canvas markdown as working content; create_version advances
    # current_version and enqueues processing (chunk → embed → guardrails → PII).
    version = create_version(
        doc,
        content=content,
        origin=DataRoomDocumentVersion.Origin.CANVAS_EXPORT,
        created_by=user,
        native_filename=filename,
        mime_type="text/markdown",
        size_bytes=len(file_bytes),
        enqueue=True,
    )
    DataRoomDocumentTag.objects.create(
        version=version, key="source", value="canvas_export"
    )

    return doc


CANVAS_MAX_IMAGES = 25
SUMMARY_TARGET_TOKENS = 2_000

# Derived from the unified capability table (core/file_types.py). Chat accepts
# image/pdf/docx/text kinds (no audio, no .msg/.eml email formats). Uses the
# canonical (official) MIME per type — not the broader browser-variant set used
# for data-room cross-checks — so e.g. a real .xls labelled application/vnd.ms-
# excel isn't silently accepted as text.
SUPPORTED_IMAGE_TYPES = frozenset(canonical_mimes_for_kinds({KIND_IMAGE}))
SUPPORTED_PDF_TYPES = frozenset(canonical_mimes_for_kinds({KIND_PDF}))
SUPPORTED_TEXT_TYPES = frozenset(canonical_mimes_for_kinds({KIND_TEXT}))
SUPPORTED_DOCX_TYPES = frozenset(canonical_mimes_for_kinds({KIND_DOCX}))
SUPPORTED_ATTACHMENT_TYPES = frozenset(canonical_mimes_for_kinds(CHAT_KINDS))

MAX_ATTACHMENT_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_PDF_ATTACHMENT_SIZE = 30 * 1024 * 1024  # 30 MB


def max_size_for_content_type(content_type: str) -> int:
    """Return the maximum upload size in bytes for a given content type."""
    if content_type in SUPPORTED_PDF_TYPES:
        return MAX_PDF_ATTACHMENT_SIZE
    return MAX_ATTACHMENT_SIZE


def build_image_content_block(b64_data: str, media_type: str, provider: str) -> dict:
    """Build a provider-specific image content block for multimodal messages."""
    if provider == "anthropic":
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64_data,
            },
        }
    else:
        # OpenAI / Gemini style
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64_data}"},
        }


def build_pdf_content_block(b64_data: str, filename: str, provider: str) -> dict:
    """Build a provider-specific content block for a PDF attachment."""
    if provider == "anthropic":
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": b64_data,
            },
        }
    elif provider == "openai":
        return {
            "type": "file",
            "file": {
                "filename": filename,
                "file_data": f"data:application/pdf;base64,{b64_data}",
            },
        }
    else:
        # Gemini / fallback — data URI style
        return {
            "type": "image_url",
            "image_url": {"url": f"data:application/pdf;base64,{b64_data}"},
        }


def build_text_content_block(text: str, filename: str) -> dict:
    """Wrap plain text from an attached file into a content block."""
    return {
        "type": "text",
        "text": f"[Attached file: {filename}]\n\n{text}",
    }


DOCX_MAX_DESCRIBED_IMAGES = 10


def extract_docx_text(file_bytes: bytes, *, user=None) -> str:
    """Extract text from a .docx file as markdown.

    Without *user*, images become simple ``[Image N]`` placeholders. With a
    user, the first :data:`DOCX_MAX_DESCRIBED_IMAGES` images are described via a
    vision-capable model and the rest get a format-only label.
    """
    from core.docx import docx_to_markdown, placeholder_image_sink

    if user is None:
        sink = placeholder_image_sink
    else:
        sink = describe_image_sink(user, max_described=DOCX_MAX_DESCRIBED_IMAGES)
    return docx_to_markdown(file_bytes, image_sink=sink)


async def generate_summary(
    messages: list[ChatMessage],
    existing_summary: str = "",
    *,
    user_id: int,
    conversation_id,
    model: str | None = None,
) -> str:
    """Summarise *messages* into a concise rolling summary.

    When *existing_summary* is provided it is folded into the new summary
    so that the LLM produces a single coherent summary covering all prior
    history.
    """
    from django.conf import settings as django_settings

    from llm import get_llm_service
    from llm.types import ChatRequest, Message, RunContext

    parts: list[str] = []
    if existing_summary:
        parts.append(
            f"Previous conversation summary:\n{existing_summary}\n"
        )
    parts.append("Messages to summarise:")
    for msg in messages:
        parts.append(f"[{msg.role}]: {msg.content}")

    user_prompt = "\n".join(parts)

    system_prompt = (
        "Produce a concise summary for an LLM chatbot "
        f"of its conversation with a user. Target ~{SUMMARY_TARGET_TOKENS} tokens. "
        "Preserve key facts, decisions, important points as well as "
        "any context the assistant would need to continue the conversation "
        "coherently. Do NOT include any preamble — output only the summary."
    )

    context = RunContext.create(
        user_id=user_id,
        conversation_id=conversation_id,
    )
    request = ChatRequest(
        messages=[
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ],
        model=model or django_settings.LLM_DEFAULT_MID_MODEL,
        stream=False,
        tools=[],
        context=context,
    )

    service = get_llm_service()
    response = await service.arun("simple_chat", request)
    return response.message.content.strip()


def describe_image(
    image_bytes: bytes,
    content_type: str,
    user,
    alt_text: str | None = None,
    model: str | None = None,
) -> str | None:
    """Use a vision-capable model to describe an image.

    When *model* is given it is used directly (the caller must have picked a
    vision-capable model, e.g. via ``resolve_org_feature_model`` for data-room
    ingestion). Otherwise this cascades through the user's cheap → mid → primary
    models, picking the first that supports vision. Returns the description text,
    or None on failure / when no vision model is available.
    """
    from core.preferences import get_preferences
    from llm import get_llm_service
    from llm.display import supports_vision
    from llm.types import ChatRequest, Message, RunContext

    if model is not None:
        if not supports_vision(model):
            return None
    else:
        prefs = get_preferences(user)
        preferred = prefs.feature_models.get("image_description", prefs.cheap_model)
        for candidate in [preferred, prefs.cheap_model, prefs.mid_model, prefs.top_model]:
            if supports_vision(candidate):
                model = candidate
                break

    if model is None:
        return None

    media_type = content_type or "image/png"
    if media_type not in SUPPORTED_IMAGE_TYPES:
        return None

    b64 = base64.b64encode(image_bytes).decode("ascii")

    prompt = (
        "Describe this image in one concise sentence suitable as an alt-text placeholder "
        "in a document. Focus on what the image depicts (chart type, diagram subject, "
        "photo content). Do NOT include any preamble — output only the description."
    )
    if alt_text:
        prompt += f"\nThe original alt text was: {alt_text}"

    # Determine provider to use the right content block format
    provider = detect_provider(model)
    image_block = build_image_content_block(b64, media_type, provider)

    content_blocks: list = [
        {"type": "text", "text": prompt},
        image_block,
    ]

    context = RunContext.create(user_id=user.pk)
    request = ChatRequest(
        messages=[Message(role="user", content=content_blocks)],
        model=model,
        stream=False,
        tools=[],
        context=context,
    )

    try:
        service = get_llm_service()
        response = service.run("simple_chat", request)
        return response.message.content.strip()
    except Exception:
        logger.exception("Failed to describe image")
        return None


def describe_image_sink(user, *, max_described=None, model=None, over_limit_label=None):
    """Return a docx ``image_sink`` that describes embedded images.

    The first *max_described* images (all when ``None``) are described via a
    vision-capable model, yielding ``[Image N: <description>]``. Beyond the
    limit, *over_limit_label* is used if given, else a format-only label. On a
    failed/empty description the format-only label is used too.
    """

    def sink(image, idx: int) -> str:
        over_limit = max_described is not None and idx > max_described
        if over_limit and over_limit_label is not None:
            return f"[Image {idx}: {over_limit_label}]"
        if not over_limit:
            with image.open() as img_file:
                img_bytes = img_file.read()
            description = describe_image(
                img_bytes, image.content_type, user, alt_text=image.alt_text, model=model
            )
            if description:
                return f"[Image {idx}: {description}]"
        # Fallback: label with the image format (e.g. "TIFF image", "EMF image").
        fmt = (image.content_type or "").split("/")[-1].upper().lstrip("X-")
        label = f"{fmt} image" if fmt else "image"
        return f"[Image {idx}: {label}]"

    return sink


def generate_canvas_title(doc_title: str, doc_content: str, user) -> str | None:
    """Generate a short title for an imported canvas document using the cheap LLM.

    Returns the title string or None on failure (never raises).
    """
    from core.preferences import get_preferences
    from llm import get_llm_service
    from llm.types import ChatRequest, Message, RunContext

    prefs = get_preferences(user)
    prompt = (
        f"Generate a short title (max 5 words) for a document titled: {doc_title}. "
        f"Document starts with: {doc_content[:1000]}. "
        "Reply with ONLY the title."
    )

    context = RunContext.create(user_id=user.pk)
    request = ChatRequest(
        messages=[Message(role="user", content=prompt)],
        model=prefs.feature_models.get("canvas_title", prefs.cheap_model),
        stream=False,
        tools=[],
        context=context,
    )

    try:
        service = get_llm_service()
        response = service.run("simple_chat", request)
        title = response.message.content.strip().strip("\"'")
        return title[:255] if title else None
    except Exception:
        logger.exception("Failed to generate canvas title")
        return None


def import_docx_to_canvas(uploaded_file: UploadedFile, user) -> tuple[str, str, bool]:
    """Convert a .docx upload to markdown with LLM-described image placeholders.

    Returns (title, content, truncated).
    """
    from core.docx import docx_to_markdown

    sink = describe_image_sink(
        user,
        max_described=CANVAS_MAX_IMAGES,
        over_limit_label="image skipped – import limit reached",
    )
    content = docx_to_markdown(uploaded_file, image_sink=sink)

    # Truncate to character limit
    truncated = len(content) > CANVAS_MAX_CHARS
    if truncated:
        content = content[:CANVAS_MAX_CHARS]

    # Derive title from filename
    original_name = uploaded_file.name or "document"
    title = original_name.rsplit(".", 1)[0][:255] or "Untitled document"

    return title, content, truncated


# ---------------------------------------------------------------------------
# Email block rendering (for .docx export)
# ---------------------------------------------------------------------------

_EMAIL_HEADER_FIELDS = ("To", "Cc", "Bcc", "Subject")


def _parse_email_block(raw: str) -> tuple[dict[str, str], str]:
    """Parse an email block into (headers_dict, body_text).

    Headers are lines at the top matching ``Key: value`` for known fields.
    The body starts after the first blank line or the first non-header line.
    """
    lines = raw.split("\n")
    headers: dict[str, str] = {}
    body_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            body_start = i + 1
            break
        found = False
        for field in _EMAIL_HEADER_FIELDS:
            if stripped.lower().startswith(field.lower() + ":"):
                headers[field] = stripped[len(field) + 1 :].strip()
                found = True
                break
        if not found:
            body_start = i
            break
    else:
        # All lines were headers, no body
        body_start = len(lines)
    body = "\n".join(lines[body_start:]).strip()
    return headers, body


def replace_email_with_html(content: str) -> str:
    """Replace ```email blocks with styled HTML tables for docx export."""
    matches = list(EMAIL_BLOCK_RE.finditer(content))
    if not matches:
        return content

    for m in reversed(matches):
        headers, body = _parse_email_block(m.group(1))

        rows = ""
        for field in _EMAIL_HEADER_FIELDS:
            if field in headers:
                from html import escape
                rows += (
                    f'<tr><td style="padding:4px 8px;font-weight:bold;'
                    f'vertical-align:top;">{field}:</td>'
                    f'<td style="padding:4px 8px;">'
                    f'{escape(headers[field])}</td></tr>'
                )

        escaped_body = ""
        if body:
            from html import escape
            escaped_body = escape(body).replace("\n", "<br/>")

        replacement = (
            '<table style="border:1px solid #ccc;border-collapse:collapse;'
            'width:100%;margin:1em 0;">'
            f"{rows}"
            "</table>"
        )
        if escaped_body:
            replacement += (
                '<div style="padding:8px;border:1px solid #ccc;'
                f'border-top:none;margin-bottom:1em;">{escaped_body}</div>'
            )

        content = content[: m.start()] + replacement + content[m.end() :]

    return content
