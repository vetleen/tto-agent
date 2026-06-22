"""Synchronous scan-at-save: run the same scan sink inline and return a verdict.

The async pipeline (``create_version`` → ``process_document_version`` →
``scan_document_version`` → ``finalize_document_metadata``) only reaches its
quarantine verdict minutes later, via a Celery chain — long after the agent's turn is
dead. For agent-authored saves we want the verdict NOW, in the same tool call, so the
agent (or a user clicking *Save to data room*) can remediate in loop.

``scan_version_synchronously`` runs the IDENTICAL scan code as the async path
(``guardrails.tasks._scan_chunks_for_version`` + ``documents.tasks.finalize_version``)
inline against a version created with ``enqueue=False``, then reads the resulting state
back as a :class:`Verdict`. There is one scan "sink"; only the trigger differs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Verdict:
    """Outcome of a synchronous scan of a single document version.

    ``status``:
      - ``clean``       — released READY and became the active searchable version.
      - ``warn``        — some chunks were quarantined and excluded from retrieval, but
                          the version is NOT blocked: it still became active. Surfaced
                          to the caller; does not consume a remediation retry.
      - ``blocked``     — GDPR Article 9/10 quarantine; the version did NOT become
                          active and is invisible to retrieval.
      - ``scan_failed`` — extraction/chunking/scan error; no verdict could be reached.
    """

    status: Literal["clean", "warn", "blocked", "scan_failed"]
    is_quarantined: bool
    is_partially_quarantined: bool
    reasons: list[str]
    reviewer_reasoning: str | None
    version_index: int
    became_active: bool

    @property
    def ok(self) -> bool:
        """True when the content was saved and went live (``warn`` included)."""
        return self.status in ("clean", "warn")

    @property
    def detail(self) -> str:
        return self.reasons[0] if self.reasons else ""

    def to_tool_json(self) -> dict:
        """Agent-facing result. Callers merge in doc-specific fields (doc_index, etc.)."""
        if self.status == "clean":
            return {"status": "ok", "verdict": "clean", "version": self.version_index,
                    "became_searchable": self.became_active}
        if self.status == "warn":
            return {
                "status": "ok", "verdict": "warn", "version": self.version_index,
                "became_searchable": self.became_active, "reasons": self.reasons,
                "note": ("Some sections were flagged as sensitive or unsafe and excluded "
                         "from search; the document was saved and is live without them. "
                         "Review and edit if that is not what you intended."),
            }
        if self.status == "blocked":
            return {
                "status": "blocked", "verdict": "blocked", "reasons": self.reasons,
                "detail": self.detail,
                "note": ("Saved content was rejected. Edit the canvas to remove the "
                         "flagged content and save again."),
            }
        return {
            "status": "error", "verdict": "scan_failed", "reasons": self.reasons,
            "detail": self.detail,
            "note": "The safety scan could not complete; try saving again.",
        }

    def to_http_json(self) -> dict:
        """Button-facing result. Callers merge in doc-specific fields."""
        return {"ok": self.ok, "verdict": self.status, "reason": self.detail}


def scan_version_synchronously(version_id: int) -> Verdict:
    """Process + scan a version inline and return its :class:`Verdict`.

    Runs chunk → embed → guardrail scan → PII scan → release synchronously (the same
    functions the async pipeline uses), then reads the resulting DB state back. The
    version must already exist (create it with ``create_version(..., enqueue=False)``).
    """
    from documents.models import DataRoomDocument, DataRoomDocumentVersion
    from documents.services.process_document import process_document_version
    from documents.tasks import finalize_version

    Status = DataRoomDocument.Status

    # 1. Chunk + embed, hold in SCANNING. dispatch_scan=False suppresses the async
    #    hand-off so we can run the scan inline below.
    process_document_version(version_id, dispatch_scan=False)

    version = (
        DataRoomDocumentVersion.objects.select_related("document")
        .filter(pk=version_id)
        .first()
    )
    if version is None:
        return _scan_failed_verdict(0)
    # Extraction/chunking failed (e.g. empty/too-large) — already terminal.
    if version.status in (Status.FAILED, Status.SCAN_FAILED):
        return _build_verdict(version_id)

    # 2. Guardrail chunk scan — the SAME sink as the async task. Fail closed: a scan
    #    error marks the version SCAN_FAILED rather than releasing it unclassified.
    from guardrails.tasks import _mark_scan_failed, _scan_chunks_for_version

    try:
        _scan_chunks_for_version(version)
    except Exception:
        logger.exception("scan_version_synchronously: chunk scan failed version_id=%s", version_id)
        _mark_scan_failed(version_id)
        return _build_verdict(version_id)

    # 3. PII scan + Article 9/10 quarantine + release — the SAME sink, eager (no Celery
    #    retry; a gated PII LLM failure is terminal SCAN_FAILED).
    finalize_version(version_id, eager=True)

    # 4. Read the resulting state back as the verdict.
    return _build_verdict(version_id)


def _build_verdict(version_id: int) -> Verdict:
    from documents.models import DataRoomDocument, DataRoomDocumentVersion

    Status = DataRoomDocument.Status
    version = (
        DataRoomDocumentVersion.objects.select_related("document")
        .filter(pk=version_id)
        .first()
    )
    if version is None:
        return _scan_failed_verdict(0)

    became_active = version.document.active_searchable_version_id == version.id

    if version.status in (Status.FAILED, Status.SCAN_FAILED):
        reason = version.processing_error or "The document could not be scanned."
        return Verdict(
            status="scan_failed", is_quarantined=False, is_partially_quarantined=False,
            reasons=[reason], reviewer_reasoning=None,
            version_index=version.version_index, became_active=False,
        )

    if version.is_quarantined:
        reason = version.quarantine_reason or "Contains restricted personal data."
        return Verdict(
            status="blocked", is_quarantined=True,
            is_partially_quarantined=version.is_partially_quarantined,
            reasons=[reason], reviewer_reasoning=None,
            version_index=version.version_index, became_active=False,
        )

    if version.is_partially_quarantined:
        chunk_reasons = [
            r for r in version.chunks.filter(is_quarantined=True)
            .order_by("chunk_index").values_list("quarantine_reason", flat=True) if r
        ]
        return Verdict(
            status="warn", is_quarantined=False, is_partially_quarantined=True,
            reasons=chunk_reasons or ["Some sections were flagged and excluded from search."],
            reviewer_reasoning=chunk_reasons[0] if chunk_reasons else None,
            version_index=version.version_index, became_active=became_active,
        )

    return Verdict(
        status="clean", is_quarantined=False, is_partially_quarantined=False,
        reasons=[], reviewer_reasoning=None,
        version_index=version.version_index, became_active=became_active,
    )


def _scan_failed_verdict(version_index: int) -> Verdict:
    return Verdict(
        status="scan_failed", is_quarantined=False, is_partially_quarantined=False,
        reasons=["The document version could not be found after processing."],
        reviewer_reasoning=None, version_index=version_index, became_active=False,
    )
