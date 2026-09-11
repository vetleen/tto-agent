"""Ingest + guardrail/PII scan + approval-gate logic for skill resources.

Design (see plan): resources are read *whole, by name* — no chunking/embedding.
We reuse the ``documents`` extraction and the ``guardrails``/``documents`` scan
*primitives*, composing them here so those apps stay untouched (the only change
elsewhere is a new ``skill_resource`` GuardrailEvent trigger source).

Scan timing (confirmed with user):
- Uploaded files (image/pdf/text-extractable) are scanned **on upload**
  (``ingest_file`` -> ``scan_resource``), setting a per-resource quarantine.
  An upload always sets ``original_filename``.
- Typed text (instructions, description, and text resources created via the
  modal — ``create_text_resource``, ``original_filename == ""``) is scanned at
  the **enable gate** (``scan_and_approve_skill``), hash-cached — never on every
  draft save.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile

from django.core.files.base import ContentFile

from core import file_types as ft
from core.tokens import count_tokens

from .models import MAX_RESOURCE_CHARS, AgentSkill, SkillResource

logger = logging.getLogger(__name__)

# Scan text in windows so a large resource still gets fully covered without one
# giant LLM call. A resource is capped at MAX_RESOURCE_CHARS (75k), so at most
# two windows.
_SCAN_WINDOW_CHARS = 40_000

# Max resources per skill (authoring-time guard; mirrors the spirit of the
# Data Room in-flight cap).
RESOURCE_COUNT_CAP = 50


class UnsupportedResourceType(ValueError):
    """Raised when an uploaded file's type is not a valid skill resource."""


def _ext_of(filename: str) -> str:
    """Dotless, lowercased, canonicalized extension of a filename."""
    return ft.canonical_extension(os.path.splitext(filename or "")[1])


# --- file-type detection ---------------------------------------------------

def detect_file_type(filename: str) -> str:
    """Map an upload's filename to one of the three resource file-types.

    Raises :class:`UnsupportedResourceType` for audio/unknown extensions.
    """
    ext = _ext_of(filename)
    kind = ft.kind_for_extension(ext)
    if kind is None or kind == ft.KIND_AUDIO:
        raise UnsupportedResourceType(ext or filename)
    if kind == ft.KIND_IMAGE:
        return SkillResource.FileType.IMAGE
    if kind == ft.KIND_PDF:
        return SkillResource.FileType.PDF
    # docx/dotx/pptx/xlsx/xlsm/msg/eml/txt/md/... all extract to text.
    return SkillResource.FileType.TEXT


def _org_id_for_skill(skill: AgentSkill, user) -> int | None:
    """Org used to resolve scan models: the skill's org, else the user's."""
    if skill.organization_id:
        return skill.organization_id
    if user is not None and getattr(user, "pk", None):
        from accounts.models import Membership

        return (
            Membership.objects.filter(user_id=user.pk)
            .values_list("org_id", flat=True)
            .first()
        )
    return None


def _windows(text: str, size: int) -> list[str]:
    text = text or ""
    if len(text) <= size:
        return [text] if text.strip() else []
    return [text[i : i + size] for i in range(0, len(text), size)]


# --- content hashing / approval -------------------------------------------

def resource_content_hash(content: str) -> str:
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()


def compute_skill_content_hash(skill: AgentSkill) -> str:
    """Stable hash over the skill's authored text + its resource set.

    MUST match the algorithm the 0006 grandfather migration used, so a
    grandfathered skill stays approved until it is actually edited.
    """
    parts = [skill.instructions or "", skill.description or ""]
    for res in skill.templates.order_by("name"):
        parts.append(
            "\x1f".join([res.name or "", res.kind or "", res.content_sha256 or ""])
        )
    return hashlib.sha256("\x1e".join(parts).encode("utf-8")).hexdigest()


def _scanning_configured(skill: AgentSkill) -> bool:
    """Whether this skill's org has any content scanning configured. When it has
    none, there is nothing to gate on, so an unscanned skill is trivially usable
    (also the case in tests / unconfigured orgs)."""
    from core.preferences import resolve_org_feature_model
    from documents.services.pii_scan import pii_gate_applies

    org_id = _org_id_for_skill(skill, None)
    if skill.organization_id is None and skill.created_by_id:
        from accounts.models import Membership

        org_id = (
            Membership.objects.filter(user_id=skill.created_by_id)
            .values_list("org_id", flat=True)
            .first()
        )
    return bool(
        pii_gate_applies(org_id)
        or resolve_org_feature_model(org_id, "guardrail_web_scan")
    )


def skill_is_approved(skill: AgentSkill) -> bool:
    """Whether a skill may be attached to a thread. System skills are trusted; a
    BLOCKED skill never passes; an APPROVED skill passes while its content is
    unchanged; an unscanned skill passes only when the org has no scanning
    configured (nothing to scan)."""
    if skill.level == AgentSkill.Level.SYSTEM:
        return True
    if skill.scan_state == AgentSkill.ScanState.BLOCKED:
        return False
    if (
        skill.scan_state == AgentSkill.ScanState.APPROVED
        and bool(skill.approved_content_hash)
        and skill.approved_content_hash == compute_skill_content_hash(skill)
    ):
        return True
    return not _scanning_configured(skill)


def recompute_standing_tokens(skill: AgentSkill) -> int:
    """Standing prompt cost consulted by the attach token budget: the skill's
    instructions plus a one-line manifest entry per resource (name + kind).
    Cheap; called once on save, not per attach."""
    manifest = "\n".join(
        f"- {r.name} ({r.kind})" for r in skill.templates.order_by("name")
    )
    total = count_tokens(skill.instructions or "") + count_tokens(manifest)
    if skill.standing_token_count != total:
        skill.standing_token_count = total
        skill.save(update_fields=["standing_token_count"])
    return total


def attach_token_budget() -> int:
    from django.conf import settings

    return int(getattr(settings, "SKILL_ATTACH_TOKEN_BUDGET", 40_000))


def _standing(skill: AgentSkill) -> int:
    return skill.standing_token_count or recompute_standing_tokens(skill)


def skills_within_budget(skills, budget: int | None = None):
    """Greedy in attach order: return ``(kept, dropped)`` so the summed standing
    cost stays within ``budget``. At least one skill is always kept (a single
    skill's instructions are capped well under the budget)."""
    budget = attach_token_budget() if budget is None else budget
    kept, dropped, total = [], [], 0
    for skill in skills:
        cost = _standing(skill)
        if not kept or total + cost <= budget:
            kept.append(skill)
            total += cost
        else:
            dropped.append(skill)
    return kept, dropped


def trim_ids_to_budget(skill_ids, budget: int | None = None) -> list[str]:
    """Budget-trim an ordered list of skill ids (load-path backstop)."""
    ids = [str(i) for i in skill_ids]
    by_id = {str(s.id): s for s in AgentSkill.objects.filter(id__in=ids)}
    ordered = [by_id[i] for i in ids if i in by_id]
    kept, _ = skills_within_budget(ordered, budget)
    return [str(s.id) for s in kept]


# --- text extraction -------------------------------------------------------

def extract_text(data: bytes, ext: str) -> str:
    """Extract a file's text/markdown using the documents pipeline (standalone,
    no DataRoomDocumentVersion). Chunking/embedding are deliberately skipped."""
    from documents.services.chunking import clean_extracted_text, load_documents

    canonical = ft.canonical_extension(ext)
    suffix = f".{canonical}" if canonical else ""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        docs = load_documents(tmp_path, canonical)
        text = "\n\n".join(getattr(d, "page_content", "") for d in docs)
        return clean_extracted_text(text)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# --- scanning --------------------------------------------------------------

def _scan_text_guardrail(text, user, org_id, label) -> tuple[str, str, list]:
    """Adversarial/prompt-injection scan of a text blob. Returns
    ``(action, detail, tags)`` where action is ``"allow"`` or ``"quarantine"``.
    Logs GuardrailEvents (trigger_source=skill_resource); NEVER suspends a user.
    """
    from guardrails.classifier import (
        GuardrailModelUnavailableError,
        classify_web_content_sync,
    )
    from guardrails.heuristics import heuristic_scan
    from guardrails.reviewer import review_flagged_chunk
    from guardrails.service import _create_event_sync

    detail = "Contains content flagged as adversarial (possible prompt injection)."
    for window in _windows(text, _SCAN_WINDOW_CHARS):
        hres = heuristic_scan(window)
        if hres.should_block:
            _log_guardrail(
                _create_event_sync, user, org_id, "heuristic", hres.tags,
                hres.confidence, "high", "blocked", window,
            )
            return "quarantine", detail, list(hres.tags)

        try:
            cres = classify_web_content_sync(window, user.pk, org_id)
        except GuardrailModelUnavailableError:
            logger.warning(
                "skill guardrail: no classifier model for org_id=%s; skipping", org_id,
            )
            continue
        if not cres.is_suspicious:
            continue

        decision = review_flagged_chunk(
            window, cres, document_title=label, neighbor_context="",
            org_id=org_id, user_id=user.pk,
        )
        if decision is None:
            quarantine = cres.confidence >= 0.9
            severity, reasoning, check = "medium", cres.reasoning, "classifier"
            confidence = cres.confidence
        else:
            quarantine = decision.action == "quarantine"
            severity, reasoning, check = decision.severity, decision.reasoning, "llm_review"
            confidence = decision.confidence
        _log_guardrail(
            _create_event_sync, user, org_id, check, cres.concern_tags,
            confidence, severity, "blocked" if quarantine else "dismissed",
            window, reviewer_output=reasoning,
        )
        if quarantine:
            return "quarantine", detail, list(cres.concern_tags)
    return "allow", "", []


def _log_guardrail(
    create_fn, user, org_id, check_type, tags, confidence, severity,
    action_taken, window, reviewer_output=None,
):
    try:
        create_fn(
            user=user, org_id=org_id, thread_id=None,
            trigger_source="skill_resource", check_type=check_type,
            tags=list(tags or []), confidence=confidence, severity=severity,
            action_taken=action_taken, raw_input=window[:2000],
            reviewer_output=reviewer_output,
        )
    except Exception:
        logger.exception("skill guardrail: failed to write GuardrailEvent")


_ARTICLE_BY_CATEGORY = {
    "pii_special_category": ("article_9", "Article 9 (special category)"),
    "pii_criminal_offence": ("article_10", "Article 10 (criminal offence)"),
}


def _scan_text_pii(text, user, org_id, label) -> tuple[dict, bool, str, str]:
    """PII scan of a text blob. Returns
    ``(pii_categories, is_quarantined, reason, detail)``. Only reviewer-confirmed
    Article 9 / criminal-offence data quarantines (same rule as Data Rooms)."""
    from documents.services.pii_scan import (
        GATED_CATEGORIES,
        pii_gate_applies,
        scan_pii_categories,
    )
    from documents.services.pii_review import review_flagged_pii_window

    if not pii_gate_applies(org_id):
        return {}, False, "", ""

    confirmed: dict[str, bool] = {}
    detail_parts: list[str] = []
    uid = user.pk if user is not None else None
    for window in _windows(text, _SCAN_WINDOW_CHARS):
        cats = scan_pii_categories(window, user_id=uid, org_id=org_id)
        for cat, present in cats.items():
            if present and cat not in GATED_CATEGORIES:
                confirmed[cat] = True
        gated = [c for c in GATED_CATEGORIES if cats.get(c)]
        if not gated:
            continue
        decision = review_flagged_pii_window(window, gated, label, org_id, uid)
        if decision is None:
            for cat in gated:
                confirmed[cat] = True
            continue
        for cat in gated:
            attr = _ARTICLE_BY_CATEGORY[cat][0]
            if getattr(decision, attr, False):
                confirmed[cat] = True
        finding = (getattr(decision, "findings", "") or "").strip()
        if finding:
            detail_parts.append(finding)

    articles = [
        _ARTICLE_BY_CATEGORY[c][1] for c in GATED_CATEGORIES if confirmed.get(c)
    ]
    if articles:
        reason = "Contains GDPR " + " and ".join(articles) + " personal data."
        return confirmed, True, reason, " ".join(detail_parts)
    return confirmed, False, "", ""


def scan_resource(resource: SkillResource, user) -> None:
    """Guardrail + PII scan one resource's text; set status/quarantine. Fails
    closed (SCAN_FAILED) on an LLM error so unscanned content is never READY."""
    text = resource.content or ""
    org_id = _org_id_for_skill(resource.skill, user)

    resource.status = SkillResource.Status.SCANNING
    resource.save(update_fields=["status"])

    try:
        if text.strip():
            g_action, g_detail, _ = _scan_text_guardrail(text, user, org_id, resource.name)
            pii_cats, pii_quar, pii_reason, pii_detail = _scan_text_pii(
                text, user, org_id, resource.name
            )
        else:
            g_action, g_detail = "allow", ""
            pii_cats, pii_quar, pii_reason, pii_detail = {}, False, "", ""
    except Exception:
        logger.exception("scan_resource: scan failed for resource_id=%s", resource.pk)
        resource.status = SkillResource.Status.SCAN_FAILED
        resource.error = "The safety scan could not be completed. Try again."
        resource.save(update_fields=["status", "error"])
        return

    resource.pii_categories = pii_cats
    resource.error = ""
    if g_action == "quarantine" or pii_quar:
        resource.is_quarantined = True
        resource.status = SkillResource.Status.QUARANTINED
        resource.quarantine_reason = (pii_reason or g_detail)[:255]
        resource.quarantine_detail = pii_detail or g_detail
    else:
        resource.is_quarantined = False
        resource.status = SkillResource.Status.READY
        resource.quarantine_reason = ""
        resource.quarantine_detail = ""
    resource.save(
        update_fields=[
            "pii_categories", "is_quarantined", "status",
            "quarantine_reason", "quarantine_detail", "error",
        ]
    )


# --- creation --------------------------------------------------------------

def create_pending_upload(skill: AgentSkill, *, data: bytes, filename: str, user,
                          kind: str = SkillResource.Kind.REFERENCE) -> SkillResource:
    """Store an uploaded file and return a PROCESSING resource — fast, no
    extraction or scanning (those run off the request in ``process_upload``, so
    heavy PDF/Office extraction never ties up or OOMs the web dyno)."""
    file_type = detect_file_type(filename)  # raises UnsupportedResourceType
    ext = _ext_of(filename)
    resource = SkillResource(
        skill=skill,
        name=_unique_name(skill, filename),
        kind=kind,
        file_type=file_type,
        original_filename=filename,
        media_type=ft.canonical_mime_for_extension(ext) or "",
        status=SkillResource.Status.PROCESSING,
    )
    resource.original_file.save(filename, ContentFile(data), save=False)
    resource.save()
    recompute_standing_tokens(skill)
    return resource


def process_upload(resource: SkillResource, user) -> None:
    """Extract text from a stored upload then guardrail/PII scan it. Runs on the
    worker (see tasks.py). Fails closed (SCAN_FAILED) on any extraction error."""
    try:
        with resource.original_file.open("rb") as fh:
            data = fh.read()
        if resource.file_type == SkillResource.FileType.IMAGE:
            resource.content = ""
            resource.content_sha256 = hashlib.sha256(data).hexdigest()
        elif resource.file_type == SkillResource.FileType.PDF:
            resource.content = extract_text(data, "pdf")[:MAX_RESOURCE_CHARS]
            resource.content_sha256 = hashlib.sha256(data).hexdigest()
        else:  # text-extractable
            ext = _ext_of(resource.original_filename)
            resource.content = extract_text(data, ext)[:MAX_RESOURCE_CHARS]
            resource.content_sha256 = resource_content_hash(resource.content)
        resource.token_count = count_tokens(resource.content) if resource.content else 0
        resource.save(update_fields=["content", "content_sha256", "token_count"])
    except Exception:
        logger.exception(
            "process_upload: extraction failed for %s", resource.original_filename
        )
        resource.status = SkillResource.Status.SCAN_FAILED
        resource.error = "Could not read this file."
        resource.save(update_fields=["status", "error"])
        return

    scan_resource(resource, user)


def ingest_file(skill: AgentSkill, *, data: bytes, filename: str, user,
                kind: str = SkillResource.Kind.REFERENCE) -> SkillResource:
    """Synchronous store + extract + scan (used by tests and any sync caller)."""
    resource = create_pending_upload(
        skill, data=data, filename=filename, user=user, kind=kind
    )
    process_upload(resource, user)
    return resource


def create_text_resource(skill: AgentSkill, *, name: str, content: str, user,
                         kind: str = SkillResource.Kind.REFERENCE) -> SkillResource:
    """Create a typed text resource. NOT scanned here — its content is covered by
    the skill's enable-gate scan (it carries no ``original_filename``)."""
    content = (content or "")[:MAX_RESOURCE_CHARS]
    resource = SkillResource.objects.create(
        skill=skill,
        name=name,
        kind=kind,
        file_type=SkillResource.FileType.TEXT,
        content=content,
        content_sha256=resource_content_hash(content),
        token_count=count_tokens(content),
        status=SkillResource.Status.READY,
    )
    recompute_standing_tokens(skill)
    return resource


def update_resource(resource: SkillResource, *, name=None, content=None) -> None:
    """Rename a resource and/or edit a typed text resource's content.

    Content is editable only for typed text resources (no original file);
    uploaded files are rename-only. Typed text is (re)scanned at the enable gate,
    not here, so this makes no LLM calls. Recomputes the skill standing count.
    """
    fields: list[str] = []
    if name is not None:
        new_name = name.strip()[:255]
        if new_name and new_name != resource.name:
            resource.name = new_name
            fields.append("name")

    content_editable = (
        resource.file_type == SkillResource.FileType.TEXT
        and not resource.original_filename
    )
    if content is not None and content_editable:
        content = content[:MAX_RESOURCE_CHARS]
        if content != resource.content:
            resource.content = content
            resource.content_sha256 = resource_content_hash(content)
            resource.token_count = count_tokens(content)
            fields += ["content", "content_sha256", "token_count"]

    if fields:
        resource.save(update_fields=fields + ["updated_at"])
        recompute_standing_tokens(resource.skill)


def copy_resource(src: SkillResource, dest_skill: AgentSkill) -> SkillResource:
    """Duplicate a resource (including its native file + scan state) onto another
    skill. Used by fork / promote so a copied skill keeps its whole resource set."""
    dup = SkillResource(
        skill=dest_skill,
        name=src.name,
        kind=src.kind,
        file_type=src.file_type,
        content=src.content,
        original_filename=src.original_filename,
        media_type=src.media_type,
        content_sha256=src.content_sha256,
        token_count=src.token_count,
        status=src.status,
        is_quarantined=src.is_quarantined,
        quarantine_reason=src.quarantine_reason,
        quarantine_detail=src.quarantine_detail,
        pii_categories=dict(src.pii_categories or {}),
    )
    if src.original_file:
        try:
            with src.original_file.open("rb") as fh:
                data = fh.read()
            dup.original_file.save(
                src.original_filename or src.name, ContentFile(data), save=False
            )
        except Exception:
            logger.exception("copy_resource: failed to copy file for %s", src.pk)
    dup.save()
    return dup


def _unique_name(skill: AgentSkill, filename: str) -> str:
    """A resource name unique within the skill (unique_resource_per_skill)."""
    base = os.path.basename(filename) or "resource"
    existing = set(skill.templates.values_list("name", flat=True))
    if base not in existing:
        return base[:255]
    stem, dot, ext = base.rpartition(".")
    stem = stem or base
    n = 2
    while True:
        candidate = f"{stem} ({n}){dot}{ext}"[:255]
        if candidate not in existing:
            return candidate
        n += 1


# --- enable gate -----------------------------------------------------------

def scan_and_approve_skill(skill: AgentSkill, user) -> bool:
    """The enable/publish gate. Scans the skill's authored text (instructions,
    description, typed text resources) and verifies no bundled resource is
    quarantined, then stamps APPROVED (with the content hash) or BLOCKED.

    System skills are auto-approved. Returns True when approved."""
    if skill.level == AgentSkill.Level.SYSTEM:
        return True

    skill.scan_state = AgentSkill.ScanState.PENDING
    skill.save(update_fields=["scan_state"])

    org_id = _org_id_for_skill(skill, user)
    blocked: list[str] = []

    quarantined = list(
        skill.templates.filter(is_quarantined=True).values_list("name", flat=True)
    )
    if quarantined:
        blocked.append(
            "Remove or replace quarantined file(s): " + ", ".join(quarantined[:5])
        )

    blobs = [("instructions", skill.instructions), ("description", skill.description)]
    for res in skill.templates.filter(
        file_type=SkillResource.FileType.TEXT, original_filename=""
    ):
        blobs.append((res.name, res.content))

    try:
        for label, blob in blobs:
            if not (blob or "").strip():
                continue
            g_action, _, _ = _scan_text_guardrail(blob, user, org_id, f"{skill.name}: {label}")
            if g_action == "quarantine":
                blocked.append(f"“{label}” was flagged as adversarial content.")
                continue
            _, pii_quar, pii_reason, _ = _scan_text_pii(blob, user, org_id, f"{skill.name}: {label}")
            if pii_quar:
                blocked.append(f"“{label}”: {pii_reason}")
    except Exception:
        logger.exception("scan_and_approve_skill: scan failed for skill_id=%s", skill.pk)
        skill.scan_state = AgentSkill.ScanState.UNSCANNED
        skill.scan_detail = "The safety scan could not be completed. Try again."
        skill.save(update_fields=["scan_state", "scan_detail"])
        return False

    if blocked:
        skill.scan_state = AgentSkill.ScanState.BLOCKED
        skill.scan_detail = " ".join(blocked)[:2000]
        skill.save(update_fields=["scan_state", "scan_detail"])
        return False

    skill.scan_state = AgentSkill.ScanState.APPROVED
    skill.approved_content_hash = compute_skill_content_hash(skill)
    skill.scan_detail = ""
    skill.save(update_fields=["scan_state", "approved_content_hash", "scan_detail"])
    return True
