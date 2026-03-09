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


def import_docx_to_canvas(uploaded_file: UploadedFile, user) -> tuple[str, str]:
    """Convert a .docx upload to markdown with LLM-described image placeholders.

    Returns (title, content).
    """
    import mammoth

    image_counter = 0

    def convert_image(image):
        nonlocal image_counter
        image_counter += 1
        idx = image_counter

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
        # with src="#" mammoth produces ![placeholder](#) in markdown.
        return {"alt": placeholder, "src": "#"}

    result = mammoth.convert_to_markdown(
        uploaded_file, convert_image=mammoth.images.img_element(convert_image)
    )
    content = result.value

    # mammoth wraps alt text in markdown image syntax: ![placeholder]()
    # Replace ![...Image N...]() with just the placeholder text
    content = re.sub(
        r"!\[(\[Image \d+[^\]]*\])\]\([^)]*\)",
        r"\1",
        content,
    )

    # Derive title from filename
    original_name = uploaded_file.name or "document"
    title = original_name.rsplit(".", 1)[0][:255] or "Untitled document"

    return title, content
