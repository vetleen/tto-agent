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
MERMAID_BLOCK_RE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)
EMAIL_BLOCK_RE = re.compile(r"```email\s*\n(.*?)```", re.DOTALL)


def resolve_canvas(thread_id, canvas_name=None):
    """Resolve a canvas by name or fall back to active canvas.

    Returns (canvas, error_msg). One of the two is always None.
    """
    from chat.models import ChatCanvas, ChatThread

    if canvas_name:
        try:
            canvas = ChatCanvas.objects.select_related("accepted_checkpoint").get(
                thread_id=thread_id, title=canvas_name,
            )
            return canvas, None
        except ChatCanvas.DoesNotExist:
            return None, f"No canvas named '{canvas_name}' in this thread."

    try:
        thread = ChatThread.objects.select_related(
            "active_canvas__accepted_checkpoint",
        ).get(pk=thread_id)
    except ChatThread.DoesNotExist:
        return None, "Thread not found."

    if thread.active_canvas:
        return thread.active_canvas, None
    return None, "No active canvas in this thread."


def set_active_canvas(thread_id, canvas):
    """Update the thread's active_canvas pointer."""
    from chat.models import ChatThread

    ChatThread.objects.filter(pk=thread_id).update(active_canvas=canvas)


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

    SUPPORTED_MEDIA_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
    media_type = content_type or "image/png"
    if media_type not in SUPPORTED_MEDIA_TYPES:
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

    if provider == "anthropic":
        image_block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64,
            },
        }
    else:
        # OpenAI / Gemini style
        image_block = {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64}"},
        }

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
from urllib.parse import quote
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
                data_url = "data:text/html;charset=utf-8," + quote(html)
                page.goto(data_url, wait_until="networkidle")
                page.wait_for_selector(".mermaid svg", timeout=15000)
                el = page.query_selector(".mermaid")
                if el:
                    results[i] = base64.b64encode(el.screenshot(type="png")).decode()
            except Exception:
                pass
        browser.close()
except Exception:
    pass
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
