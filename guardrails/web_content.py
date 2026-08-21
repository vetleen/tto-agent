"""Observability-only guardrail scan for web content (search results, fetched pages).

Enqueues a classifier → reviewer scan of untrusted web content and logs a
GuardrailEvent for anything the classifier flags. **Never blocks or alters the
content** — the tool result has already been returned to the assistant by the time
the scan runs. The goal is to measure prompt-injection exposure from web sources,
and what the classifier→reviewer chain decides, before deciding on enforcement.

The actual scan runs on the Celery worker (``guardrails.tasks.scan_web_content_task``)
so the LLM calls stay off the latency- and memory-sensitive web dyno. This module
only resolves attribution and enqueues; it never raises into the calling tool.

Replaces the previous Layer-0 regex heuristic, which production data showed had 0/21
precision on web content (every hit was an editorial AI-news page mentioning
"jailbreak"), so it was removed rather than kept as a signal.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def scan_web_content_from_tool(text: str, context, *, source_label: str) -> None:
    """Fire-and-forget web-content scan for tool call sites (web_fetch, brave_search).

    Derives user/thread attribution from a RunContext-like object, caps the text,
    and enqueues the worker scan. Never raises — one shared guard for every call
    site. ``context`` may be None (no attribution → nothing is enqueued). A broker
    hiccup drops the observation rather than breaking the fetch.
    """
    try:
        if not text or not text.strip():
            return
        user_id = getattr(context, "user_id", None) if context is not None else None
        if user_id is None:
            return
        thread_id = getattr(context, "conversation_id", None) if context is not None else None

        from guardrails.tasks import _MAX_WEB_SCAN_CHARS, scan_web_content_task

        scan_web_content_task.delay(
            text[:_MAX_WEB_SCAN_CHARS], user_id, thread_id, source_label,
        )
    except Exception:
        logger.debug("web content scan enqueue failed (non-fatal) source=%s", source_label)


def _resolve_org_id(user_id_int: int) -> int | None:
    """Look up the user's first org membership. Returns None if not found.

    Used by ``scan_web_content_task`` to attribute the GuardrailEvent to an org.
    """
    try:
        from accounts.models import Membership

        return (
            Membership.objects.filter(user_id=user_id_int)
            .values_list("org_id", flat=True)
            .first()
        )
    except Exception:
        return None
