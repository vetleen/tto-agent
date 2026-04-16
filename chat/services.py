"""Chat services — summarisation, image description, and docx import helpers."""

from __future__ import annotations

import base64
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from django.core.files.uploadedfile import UploadedFile

    from chat.models import ChatMessage

logger = logging.getLogger(__name__)

CANVAS_MAX_CHARS = 75_000
MAX_CANVASES_PER_THREAD = 10
MAX_ACTIVE_CANVASES = 3
MERMAID_BLOCK_RE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)
EMAIL_BLOCK_RE = re.compile(r"```email\s*\n(.*?)```", re.DOTALL)


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

CANVAS_MAX_IMAGES = 25
SUMMARY_TARGET_TOKENS = 2_000

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
SUPPORTED_PDF_TYPES = {"application/pdf"}
SUPPORTED_TEXT_TYPES = {
    "text/plain", "text/markdown", "text/csv", "text/html",
    "application/json", "text/xml", "application/xml",
}
SUPPORTED_DOCX_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
SUPPORTED_ATTACHMENT_TYPES = (
    SUPPORTED_IMAGE_TYPES | SUPPORTED_PDF_TYPES | SUPPORTED_TEXT_TYPES | SUPPORTED_DOCX_TYPES
)

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
    """Extract text from a .docx file as markdown using mammoth + markdownify.

    Images are replaced with ``[Image N: <description>]`` placeholders.  When
    *user* is provided, the first :data:`DOCX_MAX_DESCRIBED_IMAGES` images are
    described via a vision-capable LLM; remaining images get a format-only
    label.  Without a user, all images get a simple ``[Image N]`` placeholder.
    """
    import io

    import mammoth
    from markdownify import markdownify as md

    image_counter = 0

    def convert_image(image):
        nonlocal image_counter
        image_counter += 1
        idx = image_counter

        if user is not None and idx <= DOCX_MAX_DESCRIBED_IMAGES:
            with image.open() as img_file:
                img_bytes = img_file.read()
            description = describe_image(
                img_bytes, image.content_type, user, alt_text=image.alt_text,
            )
            if description:
                return {"alt": f"[Image {idx}: {description}]", "src": "#"}

        # Fallback: format-only label or simple placeholder
        if user is not None:
            fmt = (image.content_type or "").split("/")[-1].upper().lstrip("X-")
            label = f"{fmt} image" if fmt else "image"
            return {"alt": f"[Image {idx}: {label}]", "src": "#"}
        return {"alt": f"[Image {idx}]", "src": "#"}

    result = mammoth.convert_to_html(
        io.BytesIO(file_bytes),
        convert_image=mammoth.images.img_element(convert_image),
    )
    content = md(result.value, heading_style="ATX").strip()

    # Clean up markdown image syntax: ![placeholder](#) → placeholder
    content = re.sub(r"!\[(\[Image \d+[^\]]*\])\]\([^)]*\)", r"\1", content)
    return content


async def generate_summary(
    messages: list[ChatMessage],
    existing_summary: str = "",
    *,
    user_id: int,
    conversation_id,
) -> str:
    """Summarise *messages* into a concise rolling summary.

    When *existing_summary* is provided it is folded into the new summary
    so that the LLM produces a single coherent summary covering all prior
    history.

    Uses the mid-tier model (same cheap model used for title generation).
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
        model=django_settings.LLM_DEFAULT_MID_MODEL,
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
) -> str | None:
    """Use a vision-capable model to describe an image.

    Cascades through cheap → mid → primary model tiers, picking the first
    that supports vision.  Returns description text, or None on failure.
    """
    from core.preferences import get_preferences
    from llm import get_llm_service
    from llm.display import supports_vision
    from llm.types import ChatRequest, Message, RunContext

    prefs = get_preferences(user)
    model = None
    for candidate in [prefs.cheap_model, prefs.mid_model, prefs.top_model]:
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
    provider = model.split("/", 1)[0].lower() if "/" in model else ""
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
        model=prefs.cheap_model,
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
    import mammoth

    image_counter = 0

    def convert_image(image):
        nonlocal image_counter
        image_counter += 1
        idx = image_counter

        if idx > CANVAS_MAX_IMAGES:
            placeholder = f"[Image {idx}: image skipped – import limit reached]"
            return {"alt": placeholder, "src": "#"}

        with image.open() as img_file:
            img_bytes = img_file.read()

        description = describe_image(
            img_bytes, image.content_type, user, alt_text=image.alt_text
        )
        if description:
            placeholder = f"[Image {idx}: {description}]"
        else:
            # Fallback: label with the image format (e.g. "TIFF image", "EMF image")
            fmt = (image.content_type or "").split("/")[-1].upper().lstrip("X-")
            label = f"{fmt} image" if fmt else "unsupported format"
            placeholder = f"[Image {idx}: {label}]"

        # mammoth.images.img_element uses these as <img> HTML attributes;
        # markdownify converts <img alt="placeholder" src="#"> to ![placeholder](#).
        return {"alt": placeholder, "src": "#"}

    from markdownify import markdownify as md

    result = mammoth.convert_to_html(
        uploaded_file, convert_image=mammoth.images.img_element(convert_image)
    )
    html = result.value

    # Convert HTML (with proper tables) to markdown
    content = md(html, heading_style="ATX")

    # Replace image HTML remnants with our placeholders
    content = re.sub(
        r"!\[(\[Image \d+[^\]]*\])\]\([^)]*\)",
        r"\1",
        content,
    )

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


# ---------------------------------------------------------------------------
# Mermaid diagram rendering (for .docx export)
# ---------------------------------------------------------------------------

MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"


def _render_mermaid_pngs(sources: list[str]) -> list[bytes | None]:
    """Render multiple mermaid diagrams to PNG using a separate subprocess.

    Playwright needs ProactorEventLoop on Windows but Daphne uses
    SelectorEventLoop.  Changing the policy in-process would break the
    WebSocket connections, so we isolate the rendering in a child process.
    """
    import json
    import subprocess
    import sys
    import tempfile

    results: list[bytes | None] = [None] * len(sources)
    tmp_input = None

    # Write sources to a temp file for the subprocess to read.
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(sources, f)
            tmp_input = f.name

        # The subprocess sets ProactorEventLoop itself, rendering each
        # diagram and writing base64-encoded results as JSON to stdout.
        script = _MERMAID_SUBPROCESS_SCRIPT
        proc = subprocess.run(
            [sys.executable, "-c", script, tmp_input],
            capture_output=True,
            text=True,
            timeout=60 + 20 * len(sources),
        )
        if proc.returncode == 0 and proc.stdout.strip():
            encoded_list = json.loads(proc.stdout)
            for i, b64 in enumerate(encoded_list):
                if b64:
                    results[i] = base64.b64decode(b64)
        else:
            logger.error(
                "Mermaid subprocess failed (rc=%d): %s",
                proc.returncode,
                proc.stderr[:500],
            )
    except Exception:
        logger.exception("Failed to run mermaid rendering subprocess")
    finally:
        if tmp_input:
            import os
            try:
                os.unlink(tmp_input)
            except Exception:
                pass
    return results


_MERMAID_SUBPROCESS_SCRIPT = r'''
import asyncio, base64, json, sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
from html import escape
from playwright.sync_api import sync_playwright

MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"
sources = json.load(open(sys.argv[1], encoding="utf-8"))
results = [None] * len(sources)
try:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        for i, source in enumerate(sources):
            esc = escape(source)
            html = (
                "<!DOCTYPE html><html><head>"
                '<script src="' + MERMAID_CDN + '"></script>'
                "</head><body>"
                '<pre class="mermaid">' + esc + "</pre>"
                "<script>mermaid.initialize({startOnLoad:true});</script>"
                "</body></html>"
            )
            try:
                page.set_content(html, wait_until="networkidle")
                page.wait_for_selector(".mermaid svg", timeout=15000)
                el = page.query_selector(".mermaid")
                if el:
                    results[i] = base64.b64encode(el.screenshot(type="png")).decode()
            except Exception as e:
                print(f"Diagram {i} failed: {e}", file=sys.stderr)
        browser.close()
except Exception as e:
    print(f"Mermaid subprocess error: {e}", file=sys.stderr)
print(json.dumps(results))
'''


def replace_mermaid_with_images(content: str) -> str:
    """Replace ```mermaid blocks with base64 <img> tags for docx export."""
    matches = list(MERMAID_BLOCK_RE.finditer(content))
    if not matches:
        return content

    sources = [m.group(1) for m in matches]
    rendered_pngs = _render_mermaid_pngs(sources)

    # Replace in reverse order to preserve string offsets
    for m, png in reversed(list(zip(matches, rendered_pngs))):
        if png:
            b64 = base64.b64encode(png).decode()
            replacement = f'<img src="data:image/png;base64,{b64}" alt="Diagram" />'
        else:
            # Keep original code block on render failure
            replacement = m.group(0)
        content = content[: m.start()] + replacement + content[m.end() :]

    return content
