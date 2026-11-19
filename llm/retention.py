"""Redaction and anonymisation of ``LLMCallLog`` rows (GDPR).

Two levels, both keeping the row for cost/usage analytics:

- **Content** (``redact_llm_call_logs``): prompt, raw_output and tool schemas
  are scrubbed. Runs after ``LLM_LOG_CONTENT_RETENTION_DAYS`` (enforce_retention)
  and when the user, chat thread, data-room document or data room the row
  belongs to is deleted (pre_delete signals in accounts/chat/documents).
- **Identity** (``anonymize_llm_call_logs``): after
  ``LLM_LOG_IDENTITY_RETENTION_DAYS`` the user and every object reference are
  removed, with the user's organization stamped on ``org_id`` first.
"""
from __future__ import annotations

from datetime import timedelta

from django.conf import settings
from django.db.models import F, OuterRef, PositiveBigIntegerField, Q, Subquery
from django.db.models.functions import Coalesce

REDACTED_PROMPT = {"redacted": True}


def document_log_ids(doc) -> dict:
    """``RunContext.create`` kwargs tagging a call with the data-room document
    it is made for, so deleting the document (or its data room) redacts the
    call's log row. Anything that is not a saved document yields no ids."""
    doc_id = getattr(doc, "pk", None)
    room_id = getattr(doc, "data_room_id", None)
    return {
        "document_id": doc_id if isinstance(doc_id, int) else None,
        "data_room_id": room_id if isinstance(room_id, int) else None,
    }


def content_cutoff(now):
    return now - timedelta(days=settings.LLM_LOG_CONTENT_RETENTION_DAYS)


def identity_cutoff(now):
    return now - timedelta(days=settings.LLM_LOG_IDENTITY_RETENTION_DAYS)


def unredacted(qs):
    """Rows of ``qs`` whose content has not been redacted yet."""
    return qs.exclude(prompt=REDACTED_PROMPT)


def redact_llm_call_logs(qs) -> int:
    """Scrub the content fields of ``qs``; metadata stays. Idempotent."""
    return unredacted(qs).update(prompt=REDACTED_PROMPT, raw_output="", tools=None)


def _org_of_user():
    from accounts.models import Membership

    return Subquery(
        Membership.objects.filter(user_id=OuterRef("user_id")).values("org_id")[:1],
        output_field=PositiveBigIntegerField(),
    )


def stamp_org(qs) -> int:
    """Record the user's organization on rows that are about to lose their user."""
    return qs.filter(org_id__isnull=True, user__isnull=False).update(org_id=_org_of_user())


def identified(qs):
    """Rows of ``qs`` that still reference a user or an object."""
    return qs.filter(
        Q(user__isnull=False)
        | ~Q(run_id="")
        | ~Q(trace_id="")
        | ~Q(conversation_id="")
        | Q(document_id__isnull=False)
        | Q(data_room_id__isnull=False)
    )


def anonymize_llm_call_logs(qs) -> int:
    """Drop the user and object references of ``qs``, keeping the organization.

    One UPDATE: the ``org_id`` subquery reads the row's old ``user_id``.
    Idempotent (an anonymised row has nothing left to match ``identified``).
    """
    return qs.update(
        org_id=Coalesce(
            F("org_id"), _org_of_user(), output_field=PositiveBigIntegerField()
        ),
        user=None,
        run_id="",
        trace_id="",
        conversation_id="",
        document_id=None,
        data_room_id=None,
    )
