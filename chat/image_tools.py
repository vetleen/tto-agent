"""Image generation tool: let the assistant create/edit images via an image model.

``chat_generate_image`` is a chat-domain tool (it doesn't touch data rooms). It
calls the image-generation service, persists the result as a thread-owned
Asset, and returns a ``[[image:uuid|]]`` token the model embeds in its reply
to show the user. The same tool edits/restyles existing images when given
``input_images`` (image tokens it already knows about).
"""

from __future__ import annotations

import base64
import json
import logging
import re
import uuid as _uuid

from pydantic import BaseModel, Field

from llm.tools import ContextAwareTool, ReasonBaseModel, get_tool_registry

logger = logging.getLogger(__name__)

_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


class ChatGenerateImageInput(ReasonBaseModel):
    prompt: str = Field(
        description="A vivid, detailed description of the image to generate. "
        "When editing, describe the change you want applied to the input image(s)."
    )
    input_images: list[str] = Field(
        default_factory=list,
        description="Optional image tokens to edit or use as references — paste the "
        "[[image:<uuid>|...]] token(s) of image(s) already in this conversation "
        "(ones you generated, or data-room images you viewed). Leave empty to "
        "generate from scratch.",
    )
    aspect_ratio: str | None = Field(
        default=None,
        description="Optional aspect ratio, e.g. '1:1', '16:9', '4:3', '9:16', '3:4'.",
    )


class ChatGenerateImageTool(ContextAwareTool):
    """Generate (or edit) an image from a text prompt using the configured image model."""

    name: str = "chat_generate_image"
    description: str = (
        "Generate an image from a text prompt. Can also EDIT or restyle an existing "
        "image: pass its [[image:<uuid>|...]] token in input_images along with a prompt "
        "describing the change (you can pass several as references). The generated image "
        "is saved and you get back a token — embed that token in your reply to show it to "
        "the user. Generate ONCE, present the result, and wait for the user's feedback; do "
        "not regenerate or iterate on your own unless the user asks."
    )
    args_schema: type[BaseModel] = ChatGenerateImageInput

    def _run(
        self,
        prompt: str,
        input_images: list[str] | None = None,
        aspect_ratio: str | None = None,
        **kwargs,
    ) -> str:
        from chat.assets import image_token, store_thread_image
        from chat.models import ChatThread
        from core.preferences import get_preferences
        from llm.service.image_generation_service import (
            ImageGenerationError,
            get_image_generation_service,
        )

        if not prompt or not prompt.strip():
            raise ValueError("chat_generate_image requires a non-empty 'prompt'")

        context = self.context
        user = _resolve_user(context)
        if user is None:
            return json.dumps({"status": "error", "message": "No user in context."})

        prefs = get_preferences(user)
        model_id = prefs.image_model
        if not model_id:
            return json.dumps(
                {"status": "error", "message": "Image generation is not enabled for this account."}
            )

        thread = _resolve_thread(context)
        if thread is None:
            return json.dumps({"status": "error", "message": "Could not resolve the conversation."})

        # Resolve any input images (edit / reference). Skip refs the user can't
        # access or that don't resolve, rather than failing the whole call.
        resolved_inputs = []
        for ref in input_images or []:
            img = _resolve_input_image(ref, user)
            if img is not None:
                resolved_inputs.append(img)

        service = get_image_generation_service()
        try:
            result = service.generate(
                prompt,
                model_id,
                context=context,
                input_images=resolved_inputs or None,
                aspect_ratio=aspect_ratio,
            )
        except ImageGenerationError as exc:
            return json.dumps({"status": "error", "message": str(exc)})
        except Exception as exc:  # transport / API failure
            logger.exception("chat_generate_image failed")
            return json.dumps(
                {"status": "error", "message": f"Image generation failed: {exc}"}
            )

        asset = store_thread_image(
            thread,
            img_bytes=result.img_bytes,
            content_type=result.media_type,
            description=prompt[:500],
            created_by=user,
        )
        token = image_token(asset.id, "")

        # Surface the image to the model too, so it can describe/caption it.
        if context is not None:
            context.pending_image_assets.append(
                {
                    "asset_id": token,
                    "b64": base64.b64encode(result.img_bytes).decode("ascii"),
                    "media_type": result.media_type,
                    "description": prompt[:500],
                }
            )

        return json.dumps(
            {
                "status": "ok",
                "is_edit": result.is_edit,
                "token": token,
                "width": result.width,
                "height": result.height,
                "message": (
                    "Image ready. Embed this token in your reply to show it to the user, "
                    "then briefly present it and ask if they'd like any changes — do not "
                    f"regenerate unless asked: {token}"
                ),
            }
        )


def _resolve_user(context):
    if context is None or not getattr(context, "user_id", None):
        return None
    from django.contrib.auth import get_user_model

    User = get_user_model()
    try:
        return User.objects.get(pk=context.user_id)
    except (User.DoesNotExist, ValueError, TypeError):
        return None


def _resolve_thread(context):
    cid = getattr(context, "conversation_id", None) if context else None
    if not cid:
        return None
    from chat.models import ChatThread

    try:
        return ChatThread.objects.get(pk=_uuid.UUID(str(cid)))
    except (ChatThread.DoesNotExist, ValueError, TypeError):
        return None


def _resolve_input_image(ref: str, user):
    """Resolve an [[image:uuid|...]] token (or bare uuid) to an InputImage the
    user may access, or None."""
    m = _UUID_RE.search(ref or "")
    if not m:
        return None
    from chat.assets import image_asset_source
    from chat.models import Asset
    from chat.views import _user_can_access_asset
    from llm.service.image_generation_service import InputImage

    try:
        asset = Asset.objects.get(id=m.group(0))
    except (Asset.DoesNotExist, ValueError):
        return None
    if not _user_can_access_asset(user, asset):
        return None
    source, ct = image_asset_source(asset)
    if source is None:
        return None
    try:
        with source.open("rb") as f:
            return InputImage(data=f.read(), mime_type=ct or "image/png")
    except Exception:
        return None


_registry = get_tool_registry()
_registry.register_tool(ChatGenerateImageTool())
