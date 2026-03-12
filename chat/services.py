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
MERMAID_BLOCK_RE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)


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
        "Produce a concise summary for an LLM chatbot"
        f"of its conversation with a user. Target ~{SUMMARY_TARGET_TOKENS} tokens.  "
        "Preserve key facts, decisions, important points as well as"
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
    for candidate in [prefs.cheap_model, prefs.mid_model, prefs.primary_model]:
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
# Mermaid diagram rendering (for .docx export)
# ---------------------------------------------------------------------------

MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"


async def _render_mermaid_png(source: str) -> bytes | None:
    """Render a single mermaid diagram to PNG using Playwright."""
    from html import escape

    from playwright.async_api import async_playwright

    escaped = escape(source)
    html = (
        "<!DOCTYPE html><html><head>"
        '<script src="' + MERMAID_CDN + '"></script>'
        "</head><body>"
        '<div class="mermaid">' + escaped + "</div>"
        "<script>mermaid.initialize({startOnLoad:true});</script>"
        "</body></html>"
    )
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.set_content(html)
            await page.wait_for_selector(".mermaid svg", timeout=15_000)
            el = await page.query_selector(".mermaid")
            png_bytes = await el.screenshot(type="png") if el else None
            await browser.close()
            return png_bytes
    except Exception:
        logger.exception("Failed to render mermaid diagram")
        return None


def replace_mermaid_with_images(content: str) -> str:
    """Replace ```mermaid blocks with base64 <img> tags for docx export."""
    import asyncio

    matches = list(MERMAID_BLOCK_RE.finditer(content))
    if not matches:
        return content

    async def _render_all():
        results = []
        for m in matches:
            png = await _render_mermaid_png(m.group(1))
            results.append((m, png))
        return results

    rendered = asyncio.run(_render_all())

    # Replace in reverse order to preserve offsets
    for m, png in reversed(rendered):
        if png:
            b64 = base64.b64encode(png).decode()
            replacement = f'<img src="data:image/png;base64,{b64}" alt="Diagram" />'
        else:
            # Keep original code block on failure
            replacement = m.group(0)
        content = content[: m.start()] + replacement + content[m.end() :]

    return content
