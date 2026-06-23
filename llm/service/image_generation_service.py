"""Image generation service with cost tracking and observability.

Generates (and edits) images via the Google Gemini API, calling the
``google-genai`` SDK directly. Image generation uses a different request shape
(``generate_content`` with an image response modality) than the chat pipeline,
so — like the transcription service — this is a standalone capability service
that calls the provider SDK directly and writes its own ``LLMCallLog`` row for
unified cost tracking.

Editing is the same call: pass one or more input images alongside the prompt
and Gemini returns an edited/derived image.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from llm.image_generation_registry import get_image_generation_model_info
from llm.service.errors import LLMProviderError
from llm.service.logger import log_image_generation, log_image_generation_error
from llm.service.pricing import calculate_image_generation_cost
from llm.types.context import RunContext

logger = logging.getLogger(__name__)


@dataclass
class InputImage:
    """An image fed into generation for editing or as a style/content reference."""

    data: bytes
    mime_type: str


@dataclass
class ImageGenerationResult:
    """Result of an image-generation call."""

    img_bytes: bytes
    media_type: str
    model: str
    width: Optional[int]
    height: Optional[int]
    cost_usd: Optional[Decimal]
    is_edit: bool


class ImageGenerationError(LLMProviderError):
    """The provider returned no usable image — a safety block, refusal, or
    empty response. Carries a user-facing message the tool can surface."""


class ImageGenerationService:
    """Image generation + editing via Gemini, with cost tracking."""

    def generate(
        self,
        prompt: str,
        model_id: str,
        context: RunContext | None = None,
        *,
        input_images: list[InputImage] | None = None,
        aspect_ratio: str | None = None,
    ) -> ImageGenerationResult:
        """Generate (or edit) a single image from a text prompt.

        ``input_images`` makes this an edit/reference call (Gemini accepts the
        image(s) alongside the prompt). ``aspect_ratio`` is a provider-neutral
        hint (e.g. ``"16:9"``); it's passed through to the model when the SDK
        supports it, else folded into the prompt.

        Logs the call to ``LLMCallLog`` and returns an ``ImageGenerationResult``.
        Raises ``ImageGenerationError`` when the model returns no image, or
        ``LLMProviderError`` on transport/API failure.
        """
        if context is None:
            context = RunContext.create()

        info = get_image_generation_model_info(model_id)
        if info is None:
            raise ValueError(f"Unknown image generation model: {model_id}")
        if info.provider != "google_genai":
            # v1 only wires Gemini. Other providers slot in here later.
            raise ValueError(f"Unsupported image generation provider: {info.provider}")

        is_edit = bool(input_images)
        if is_edit and not info.supports_editing:
            raise ValueError(f"Model {model_id} does not support image editing")

        logger.info(
            "ImageGenerationService.generate: run_id=%s model=%s prompt_len=%d inputs=%d edit=%s",
            context.run_id, model_id, len(prompt), len(input_images or []), is_edit,
        )

        t0 = time.perf_counter()
        try:
            img_bytes, media_type, usage = self._call_gemini(
                info, prompt, input_images or [], aspect_ratio
            )
        except ImageGenerationError as exc:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            log_image_generation_error(
                model_id, context, exc, duration_ms, prompt=prompt, is_edit=is_edit
            )
            raise
        except Exception as exc:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            log_image_generation_error(
                model_id, context, exc, duration_ms, prompt=prompt, is_edit=is_edit
            )
            raise LLMProviderError(f"Image generation failed: {exc}") from exc

        duration_ms = int((time.perf_counter() - t0) * 1000)
        width, height = _image_dimensions(img_bytes)
        input_tokens, output_tokens, total_tokens = _extract_usage(usage)
        cost = calculate_image_generation_cost(
            model_id, n_images=1, input_tokens=input_tokens, output_tokens=output_tokens
        )

        log_image_generation(
            model=model_id,
            context=context,
            prompt=prompt,
            cost_usd=cost,
            duration_ms=duration_ms,
            width=width,
            height=height,
            n_images=1,
            is_edit=is_edit,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

        logger.info(
            "ImageGenerationService.generate: run_id=%s model=%s size=%sx%s bytes=%d wall_ms=%d cost=%s",
            context.run_id, model_id, width, height, len(img_bytes), duration_ms, cost,
        )

        return ImageGenerationResult(
            img_bytes=img_bytes,
            media_type=media_type,
            model=model_id,
            width=width,
            height=height,
            cost_usd=cost,
            is_edit=is_edit,
        )

    def _client(self):
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise LLMProviderError("GEMINI_API_KEY is not configured")
        return genai.Client(api_key=api_key)

    def _call_gemini(self, info, prompt, input_images, aspect_ratio):
        """Call Gemini generate_content with the image response modality.

        Returns ``(img_bytes, media_type, usage_metadata)``.
        """
        from google.genai import types

        client = self._client()

        contents: list = [prompt]
        for img in input_images:
            contents.append(types.Part.from_bytes(data=img.data, mime_type=img.mime_type))

        config_kwargs: dict = {"response_modalities": ["TEXT", "IMAGE"]}
        if aspect_ratio:
            try:
                config_kwargs["image_config"] = types.ImageConfig(aspect_ratio=aspect_ratio)
            except Exception:
                # Older SDKs lack ImageConfig — fold the hint into the prompt.
                contents[0] = f"{prompt}\n\n(Aspect ratio: {aspect_ratio})"

        config = types.GenerateContentConfig(**config_kwargs)
        response = client.models.generate_content(
            model=info.api_model, contents=contents, config=config
        )

        img_bytes, media_type = _extract_image(response)
        if img_bytes is None:
            raise ImageGenerationError(_no_image_reason(response))
        return img_bytes, media_type or f"image/{info.output_format}", getattr(
            response, "usage_metadata", None
        )


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _extract_image(response):
    """Return ``(bytes, mime_type)`` for the first inline image part, else
    ``(None, None)``."""
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            data = getattr(inline, "data", None) if inline is not None else None
            if data:
                return data, getattr(inline, "mime_type", None)
    return None, None


def _no_image_reason(response) -> str:
    """Build a user-facing reason when the model returned no image — usually a
    safety block or content-policy refusal."""
    feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(feedback, "block_reason", None) if feedback else None
    if block_reason:
        return (
            "The image couldn't be generated because the request was blocked "
            f"by the provider's safety filters ({block_reason})."
        )
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        finish = getattr(candidates[0], "finish_reason", None)
        if finish and str(finish).upper() not in ("STOP", "FINISH_REASON_STOP"):
            return (
                "The image couldn't be generated "
                f"(the model stopped early: {finish})."
            )
    return "The model did not return an image. Try rephrasing the prompt."


def _extract_usage(usage):
    """Pull ``(input_tokens, output_tokens, total_tokens)`` from Gemini's
    ``usage_metadata``, or a 3-tuple of ``None`` when absent."""
    if usage is None:
        return None, None, None

    def _int_or_none(value):
        return value if isinstance(value, int) else None

    return (
        _int_or_none(getattr(usage, "prompt_token_count", None)),
        _int_or_none(getattr(usage, "candidates_token_count", None)),
        _int_or_none(getattr(usage, "total_token_count", None)),
    )


def _image_dimensions(img_bytes: bytes):
    """Best-effort ``(width, height)`` via Pillow; ``(None, None)`` on failure."""
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(img_bytes)) as im:
            return im.width, im.height
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_global_service: ImageGenerationService | None = None
_global_service_lock = threading.Lock()


def get_image_generation_service() -> ImageGenerationService:
    """Return the process-wide ImageGenerationService singleton (thread-safe)."""
    global _global_service
    if _global_service is None:
        with _global_service_lock:
            if _global_service is None:
                _global_service = ImageGenerationService()
    return _global_service
