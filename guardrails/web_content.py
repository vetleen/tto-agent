"""Observability-only guardrail scan for web content (search results, fetched pages).

Runs the heuristic scanner on web content and logs a GuardrailEvent when
something suspicious is detected. **Never blocks or alters the content** —
the tool result always passes through unchanged. The goal is to measure
prompt-injection exposure from web sources before deciding on enforcement.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def scan_web_content(
    text: str,
    *,
    user_id: str | None,
    thread_id: str | None,
    org_id: int | None,
    source_label: str,
) -> None:
    """Heuristic-scan web content and log a GuardrailEvent if suspicious.

    This is a fire-and-forget helper: it catches all exceptions so it never
    crashes the calling tool, and it never blocks or modifies the content.

    Called from sync context (tools run in ThreadPoolExecutor), so uses
    ``_create_event_sync`` directly.

    Parameters
    ----------
    text:
        The web content to scan.
    user_id:
        String user PK from RunContext (converted to int for the FK).
        If None the event is skipped (user FK is non-nullable).
    thread_id:
        Chat thread UUID string, if available.
    org_id:
        Organization PK, if known. Resolved from user_id when None and
        the heuristic fires (rare path).
    source_label:
        Freeform label for logging, e.g. ``"brave_search"`` or ``"web_fetch"``.
    """
    try:
        if not text or not text.strip():
            return

        if user_id is None:
            return

        from guardrails.heuristics import heuristic_scan

        result = heuristic_scan(text)
        if not result.is_suspicious:
            return

        # Resolve org_id lazily (only on the rare suspicious path)
        if org_id is None:
            org_id = _resolve_org_id(int(user_id))

        # Map confidence → severity
        if result.confidence >= 0.9:
            severity = "high"
        elif result.confidence >= 0.7:
            severity = "medium"
        else:
            severity = "low"

        from guardrails.models import GuardrailEvent

        GuardrailEvent.objects.create(
            user_id=int(user_id),
            organization_id=org_id,
            thread_id=thread_id,
            trigger_source="web_content",
            check_type="heuristic",
            tags=result.tags,
            confidence=result.confidence,
            severity=severity,
            action_taken="logged",
            raw_input=text[:2000],
        )

        logger.info(
            "guardrail: web_content scan source=%s user_id=%s tags=%s confidence=%.2f severity=%s",
            source_label,
            user_id,
            result.tags,
            result.confidence,
            severity,
        )

    except Exception:
        logger.exception(
            "guardrail: web_content scan failed (non-fatal) source=%s user_id=%s",
            source_label,
            user_id,
        )


def _resolve_org_id(user_id_int: int) -> int | None:
    """Look up the user's first org membership. Returns None if not found."""
    try:
        from accounts.models import Membership

        return (
            Membership.objects.filter(user_id=user_id_int)
            .values_list("org_id", flat=True)
            .first()
        )
    except Exception:
        return None
