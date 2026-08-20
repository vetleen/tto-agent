"""web_image_view tool — fetch, sanitize, and show web images to the assistant.

Discovery happens in ``web_fetch(include_images=True)``, which lists a page's
content images with ``img-N`` handles registered on the run context. This tool
resolves those handles, SSRF-fetches the bytes (following redirects, reusing the
web_fetch pinning), sanitizes them (Pillow re-encode — never trust the served
Content-Type), persists a thread-scoped Asset carrying web provenance, shows the
image(s) to the model, and hands back an embeddable ``[[image:uuid]]`` token.

Chat-domain (not ``llm/tools``) because minting an embeddable token needs
``store_thread_image`` + a resolved thread. ``audience="shared"`` so research
sub-agents can view too; a sub-agent must return the durable token to its parent,
never an ``img-N`` handle (handles are meaningless outside the run that minted them).
"""

from __future__ import annotations

import base64
import logging

from pydantic import BaseModel, Field

from llm.tools import ContextAwareTool, ReasonBaseModel, get_tool_registry

logger = logging.getLogger(__name__)

# Per-image download cap (smaller than the page cap — a single content image).
_WEB_IMAGE_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
# Cap images attached per call, mirroring document_view_image.
_MAX_ATTACH_PER_CALL = 4


class WebImageViewInput(ReasonBaseModel):
    handles: list[str] = Field(
        description=(
            "Image handles to view, from a prior web_fetch(include_images=true) "
            'result — e.g. ["img-1", "img-4"].'
        ),
    )


class WebImageViewTool(ContextAwareTool):
    """View web images (by ``img-N`` handle) and get embeddable tokens for them."""

    name: str = "web_image_view"
    section: str = "skills"
    subagent_section: str = "chat"
    audience: str = "shared"
    start_label: str = "Viewing image..."
    end_label: str = "Viewed image"
    description: str = (
        "View one or more images from a page you fetched with "
        "web_fetch(include_images=true), by their img-N handles. Each image is "
        "downloaded and shown to you, and you get back an [[image:uuid]] token you "
        "can embed in your reply or a canvas. Keep the source attribution when you "
        "embed a web image."
    )
    args_schema: type[BaseModel] = WebImageViewInput

    def _run(self, handles: list[str], **kwargs) -> str:
        from django.conf import settings as dj_settings

        from chat.assets import image_token, store_thread_image
        from chat.image_tools import _resolve_thread, _resolve_user
        from core.images import sanitize_raster_image
        from llm.tools.web_fetch import (
            _RedirectBlocked,
            _ResponseTooLarge,
            _SSRFBlocked,
            _pinned_get_following_redirects,
        )

        if not handles or not isinstance(handles, list):
            raise ValueError("web_image_view requires a non-empty 'handles' list")

        context = self.context
        manifest = getattr(context, "web_image_manifest", None) or {}
        user = _resolve_user(context)
        thread = _resolve_thread(context)
        if thread is None:
            return "Could not resolve the conversation to attach images to."

        headers = {
            "User-Agent": f"Mozilla/5.0 (compatible; {dj_settings.ASSISTANT_NAME}Bot/1.0)",
            "Accept": "image/*,*/*;q=0.8",
        }
        results: list[str] = []
        attached = 0
        for handle in handles:
            if attached >= _MAX_ATTACH_PER_CALL:
                results.append(f"{handle}: skipped — max {_MAX_ATTACH_PER_CALL} images per call.")
                continue
            entry = manifest.get(handle)
            if not entry:
                results.append(
                    f"{handle}: unknown handle. Fetch the page with include_images=true first."
                )
                continue
            url = entry.get("url", "")
            page_url = entry.get("page_url", "")
            try:
                response, _final_url = _pinned_get_following_redirects(
                    url, headers=headers, max_bytes=_WEB_IMAGE_MAX_BYTES, timeout=15
                )
                response.raise_for_status()
            except (_SSRFBlocked, _ResponseTooLarge, _RedirectBlocked) as exc:
                results.append(f"{handle}: could not fetch — {getattr(exc, 'message', str(exc))}.")
                continue
            except Exception:
                logger.debug("web_image_view: fetch failed for handle=%s", handle)
                results.append(f"{handle}: could not fetch the image.")
                continue

            # Content-Type is a cheap pre-filter only — never trusted to ACCEPT;
            # the sanitizer (Pillow) is the authoritative gate below.
            ctype = response.headers.get("Content-Type", "")
            if ctype and not ctype.lower().startswith("image/"):
                results.append(f"{handle}: not an image ({ctype}).")
                continue

            safe = sanitize_raster_image(response.content)
            if safe is None:
                results.append(f"{handle}: not a usable image (rejected during sanitization).")
                continue
            safe_bytes, media_type = safe

            alt = entry.get("alt", "")
            description = alt or entry.get("filename", "")
            asset = store_thread_image(
                thread,
                img_bytes=safe_bytes,
                content_type=media_type,
                description=description,
                alt_text=alt,
                source_url=url,
                source_page_url=page_url,
                created_by=user,
            )
            token = image_token(asset.id, "")
            context.pending_image_assets.append({
                "asset_id": token,
                "b64": base64.b64encode(safe_bytes).decode("ascii"),
                "media_type": media_type,
                "description": description,
            })
            attached += 1
            results.append(f"{handle}: attached. Embed token {token} to use it. Source: {url}")

        if attached == 0:
            return "\n".join(results) or "No images were attached."
        return (
            "\n".join(results)
            + "\n\nKeep the source attribution when you embed a web image."
            + "\n\n(The image(s) are now visible to you below.)"
        )


_registry = get_tool_registry()
_registry.register_tool(WebImageViewTool())
