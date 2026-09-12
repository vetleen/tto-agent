"""Provider-aware ceiling on native-asset (PDF/image) bytes in one LLM request,
enforced by skill-priority pruning of the assembled message list.

The per-run add-time budget (:class:`llm.types.context.RunContext`) is a
fair-share + memory-bail guard applied when a tool or attachment *adds* an asset.
This module enforces the TRUE per-request ceiling — ``min(pool, provider limit)``
— on the fully assembled messages right before they go to the provider (called
from :class:`llm.core.providers.base.BaseLangChainChatModel`), evicting the
lowest-priority (non-skill) native blocks oldest-first when a request overflows.
Skill assets are evicted last, matching the "skills bump earlier files" policy.

Native blocks carry private ``_wf_*`` markers (pathway, base64 length, label)
added at drain/enrichment time; this module reads them for classification and
strips them from the outgoing blocks, so a provider never sees them and the
prompt cache is preserved on the no-overflow path.
"""

from __future__ import annotations

import logging

from llm.types.context import PATHWAY_SKILL, native_asset_pool_bytes

logger = logging.getLogger(__name__)

# Content-block types that can carry native (base64) file bytes.
_NATIVE_TYPES = {"image", "image_url", "document", "file"}

_ANTHROPIC_DEFAULT_CEILING = 33_554_432  # 32 MB


def provider_native_b64_ceiling(provider_id: str | None) -> int:
    """Effective ceiling (base64 chars) on native bytes for one request to this
    provider: ``min(pool, provider-specific limit)``. Providers without a
    specific limit are bounded only by the pool."""
    pool = native_asset_pool_bytes()
    if "anthropic" in (provider_id or "").lower():
        try:
            from django.conf import settings

            limit = int(getattr(
                settings, "NATIVE_REQUEST_MAX_B64_BYTES_ANTHROPIC",
                _ANTHROPIC_DEFAULT_CEILING,
            ))
        except Exception:
            limit = _ANTHROPIC_DEFAULT_CEILING
        return min(pool, limit)
    return pool


def _len_after_b64(data_uri) -> int:
    if not data_uri:
        return 0
    marker = "base64,"
    idx = data_uri.find(marker)
    return (len(data_uri) - (idx + len(marker))) if idx != -1 else 0


def _is_native_block(blk: dict) -> bool:
    return isinstance(blk, dict) and blk.get("type") in _NATIVE_TYPES


def _has_marker(blk: dict) -> bool:
    return isinstance(blk, dict) and any(k.startswith("_wf") for k in blk)


def _block_b64_len(blk: dict) -> int:
    """Base64 length of a native block, from its marker or derived from the payload."""
    if "_wf_b64len" in blk:
        try:
            return int(blk["_wf_b64len"])
        except (TypeError, ValueError):
            pass
    src = blk.get("source")
    if isinstance(src, dict) and src.get("type") == "base64":
        return len(src.get("data") or "")
    image_url = blk.get("image_url")
    if isinstance(image_url, dict):
        return _len_after_b64(image_url.get("url"))
    file_obj = blk.get("file")
    if isinstance(file_obj, dict):
        return _len_after_b64(file_obj.get("file_data"))
    return 0


def _block_pathway(blk: dict) -> str:
    # Untagged native blocks (e.g. pre-existing history) count as non-skill.
    return blk.get("_wf_pathway") or "dataroom"


def _strip_markers(blk: dict) -> dict:
    return {k: v for k, v in blk.items() if not k.startswith("_wf")}


def _make_stub(blk: dict) -> dict:
    label = blk.get("_wf_label") or "file"
    return {
        "type": "text",
        "text": f"[Earlier file '{label}' omitted to fit the request size limit.]",
    }


def enforce_native_request_limits(messages, provider_id):
    """Return a message list whose native base64 total is within the provider
    ceiling, pruning lowest-priority native blocks oldest-first.

    Non-skill blocks are evicted before skill blocks; within a tier the oldest
    (earliest message, then earliest block) go first. Evicted blocks become a
    short text stub. ``_wf_*`` markers are stripped from every returned native
    block. Messages with no native/marked blocks are reused by reference, so the
    common (text-only) path returns the original list unchanged.
    """
    native = []  # (msg_idx, blk_idx, size, pathway)
    total = 0
    for mi, msg in enumerate(messages):
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for bi, blk in enumerate(content):
            if _is_native_block(blk):
                size = _block_b64_len(blk)
                native.append((mi, bi, size, _block_pathway(blk)))
                total += size

    if not native:
        return messages

    ceiling = provider_native_b64_ceiling(provider_id)

    evict: set = set()
    if total > ceiling:
        # Eviction order: non-skill first (priority key 0), then skill (1);
        # within a tier, oldest-first (message index, then block index).
        order = sorted(
            range(len(native)),
            key=lambda i: (
                1 if native[i][3] == PATHWAY_SKILL else 0,
                native[i][0],
                native[i][1],
            ),
        )
        remaining = total
        for i in order:
            if remaining <= ceiling:
                break
            evict.add((native[i][0], native[i][1]))
            remaining -= native[i][2]
        logger.warning(
            "native-asset request over ceiling: total_b64=%d ceiling=%d "
            "provider=%s evicting=%d/%d native blocks",
            total, ceiling, provider_id, len(evict), len(native),
        )

    touched = {mi for (mi, _bi, _s, _p) in native}
    out = list(messages)
    for mi in touched:
        msg = messages[mi]
        new_content = []
        for bi, blk in enumerate(msg.content):
            if isinstance(blk, dict) and (mi, bi) in evict:
                new_content.append(_make_stub(blk))
            elif isinstance(blk, dict) and _has_marker(blk):
                new_content.append(_strip_markers(blk))
            else:
                new_content.append(blk)
        out[mi] = msg.model_copy(update={"content": new_content})
    return out


__all__ = ["enforce_native_request_limits", "provider_native_b64_ceiling"]
