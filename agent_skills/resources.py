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
  modal — ``create_text_resource``, ``original_filename == ""``) is scanned by
  the **approval gate** (``scan_and_approve_skill``), which every content write
  re-queues through ``request_skill_rescan`` (asynchronously, on the worker —
  the gate makes LLM calls and must never run inside a web request). Approval
  is hash-cached on the skill, and each clean text blob's verdict is cached by
  content, so a save only re-scans what actually changed.
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

# Resource statuses whose content hash is not final yet (an upload still
# extracting/scanning on the worker). The approval gate defers while any
# resource is in flight; ``process_upload`` re-runs it when the upload lands.
IN_FLIGHT_STATUSES = (
    SkillResource.Status.PENDING,
    SkillResource.Status.PROCESSING,
    SkillResource.Status.SCANNING,
)

# Memo attribute (on a user instance) for the per-org "is scanning configured"
# answer — the same request-scoped pattern as ``services._active_system_skill_slugs``.
_SCAN_CONFIG_CACHE_ATTR = "_cached_skill_scanning_configured"


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


def compute_skill_content_hash(skill: AgentSkill, resources=None) -> str:
    """Stable hash over the skill's authored text + its resource set.

    MUST match the algorithm the 0006 grandfather migration used, so a
    grandfathered skill stays approved until it is actually edited.

    ``resources`` lets bulk callers pass a pre-fetched, name-ordered list (see
    :func:`bulk_skill_approval`). Without it ``skill.templates.all()`` is used;
    ``SkillResource.Meta.ordering`` is ``["name"]``, so both paths keep the
    database's collation order. Never sort in Python here — codepoint order
    differs from the Postgres collation for mixed-case names and would silently
    change (and un-approve) every existing hash.
    """
    if resources is None:
        resources = skill.templates.all()
    parts = [skill.instructions or "", skill.description or ""]
    for res in resources:
        parts.append(
            "\x1f".join([res.name or "", res.kind or "", res.content_sha256 or ""])
        )
    return hashlib.sha256("\x1e".join(parts).encode("utf-8")).hexdigest()


def _org_id_for_scanning(skill: AgentSkill, user=None) -> int | None:
    """Org whose scan configuration governs ``skill``: the skill's own org, else
    its creator's org (a user skill). When ``user`` *is* the creator, the memoized
    ``get_membership`` answers without a query."""
    if skill.organization_id:
        return skill.organization_id
    if not skill.created_by_id:
        return None
    if user is not None and getattr(user, "pk", None) == skill.created_by_id:
        from accounts.models import get_membership

        membership = get_membership(user)
        return membership.org_id if membership else None
    from accounts.models import Membership

    return (
        Membership.objects.filter(user_id=skill.created_by_id)
        .values_list("org_id", flat=True)
        .first()
    )


def _scanning_configured_for_org(org_id) -> bool:
    from core.preferences import resolve_org_feature_model
    from documents.services.pii_scan import pii_gate_applies

    return bool(
        pii_gate_applies(org_id)
        or resolve_org_feature_model(org_id, "guardrail_web_scan")
    )


def _scanning_configured(skill: AgentSkill, user=None) -> bool:
    """Whether this skill's org has any content scanning configured. When it has
    none, there is nothing to gate on, so an unscanned skill is trivially usable
    (also the case in tests / unconfigured orgs).

    The per-org answer costs several queries (feature-model + PII-gate
    resolution), so it is memoized on ``user`` for the life of the request or
    connection — every skill in a list resolves against the same one or two orgs.
    Long-lived holders (WebSocket consumers) call
    :func:`invalidate_scan_config_cache` before re-reading org state.
    """
    org_id = _org_id_for_scanning(skill, user)
    memo = getattr(user, _SCAN_CONFIG_CACHE_ATTR, None) if user is not None else None
    if isinstance(memo, dict) and org_id in memo:
        return memo[org_id]
    configured = _scanning_configured_for_org(org_id)
    if user is not None:
        try:
            if not isinstance(memo, dict):
                memo = {}
                setattr(user, _SCAN_CONFIG_CACHE_ATTR, memo)
            memo[org_id] = configured
        except (AttributeError, TypeError):
            # A user object that refuses attributes (not a real Django User) —
            # degrade to the uncached result rather than blow up the gate.
            pass
    return configured


def invalidate_scan_config_cache(user) -> None:
    """Drop the memoized per-org scan configuration so the next read re-resolves."""
    if hasattr(user, _SCAN_CONFIG_CACHE_ATTR):
        delattr(user, _SCAN_CONFIG_CACHE_ATTR)


def skill_is_approved(skill: AgentSkill, *, user=None, resources=None) -> bool:
    """Whether a skill may be attached to a thread. System skills are trusted; a
    BLOCKED skill never passes; an APPROVED skill passes while its content is
    unchanged; an unscanned skill passes only when the org has no scanning
    configured (nothing to scan).

    ``user`` enables the per-org config memo; ``resources`` is a pre-fetched,
    name-ordered resource list (bulk callers). Both are optional."""
    if skill.level == AgentSkill.Level.SYSTEM:
        return True
    if skill.scan_state == AgentSkill.ScanState.BLOCKED:
        return False
    if (
        skill.scan_state == AgentSkill.ScanState.APPROVED
        and bool(skill.approved_content_hash)
        and skill.approved_content_hash == compute_skill_content_hash(skill, resources)
    ):
        return True
    return not _scanning_configured(skill, user)


def approval_info(skill: AgentSkill, *, user=None, resources=None) -> dict:
    """``{"approved", "scan_state", "detail"}`` for one skill — the shape the
    Skills list, the chat catalogue and the status endpoints all render from.
    ``approved`` is the attachability verdict (:func:`skill_is_approved`);
    ``scan_state`` only flavours it (pending → "Scanning…", blocked → why)."""
    return {
        "approved": skill_is_approved(skill, user=user, resources=resources),
        "scan_state": skill.scan_state,
        "detail": skill.scan_detail or "",
    }


def bulk_skill_approval(user, skills) -> dict[str, dict]:
    """:func:`approval_info` for many skills with a bounded query count: one
    resource prefetch for every non-system skill plus one scan-config resolve per
    org (memoized on ``user``). Skills that already carry a resource prefetch are
    not re-queried. Keyed by ``str(skill.id)``."""
    from django.db.models import Prefetch, prefetch_related_objects

    skills = list(skills)
    non_system = [s for s in skills if s.level != AgentSkill.Level.SYSTEM]
    if non_system:
        prefetch_related_objects(
            non_system,
            Prefetch("templates", queryset=SkillResource.objects.order_by("name")),
        )
    out: dict[str, dict] = {}
    for skill in skills:
        resources = (
            None if skill.level == AgentSkill.Level.SYSTEM else list(skill.templates.all())
        )
        out[str(skill.id)] = approval_info(skill, user=user, resources=resources)
    return out


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


def attach_token_budget(max_context_tokens: int | None = None) -> int:
    """Standing-token budget for the skills attached to one chat thread.

    The MIN of the fixed ceiling (``SKILL_ATTACH_TOKEN_BUDGET``, 60k) and a
    fraction (``SKILL_ATTACH_BUDGET_FRACTION``, 0.55) of the turn's input ceiling.
    So a small ``max_context_tokens`` caps skills to leave room for conversation
    history + the aim, while large budgets get the full 60k. Model-agnostic — it
    uses a nominal medium-effort output reservation so the budget can be computed
    at skill-attach time (before a model/turn is chosen); the exact per-turn
    input ceiling still governs the actual history window.
    """
    from django.conf import settings

    fixed = int(getattr(settings, "SKILL_ATTACH_TOKEN_BUDGET", 60_000))
    if not max_context_tokens:
        return fixed
    fraction = float(getattr(settings, "SKILL_ATTACH_BUDGET_FRACTION", 0.55))
    margin = int(getattr(settings, "CONTEXT_SAFETY_MARGIN_TOKENS", 8_000))
    # Nominal medium-effort output reservation (see llm.context_budget); avoids a
    # model dependency at attach time.
    ceiling_est = max(int(max_context_tokens) - 16_384 - margin, 0)
    return max(1, min(fixed, int(fraction * ceiling_est)))


def _standing(skill: AgentSkill) -> int:
    return skill.standing_token_count or recompute_standing_tokens(skill)


def skills_within_budget(skills, budget: int | None = None, max_context_tokens: int | None = None):
    """Greedy in attach order: return ``(kept, dropped)`` so the summed standing
    cost stays within ``budget``. At least one skill is always kept (a single
    skill's instructions are capped well under the budget). When ``budget`` is
    None it's derived from ``max_context_tokens`` (aim-relative)."""
    budget = attach_token_budget(max_context_tokens) if budget is None else budget
    kept, dropped, total = [], [], 0
    for skill in skills:
        cost = _standing(skill)
        if not kept or total + cost <= budget:
            kept.append(skill)
            total += cost
        else:
            dropped.append(skill)
    return kept, dropped


def trim_ids_to_budget_verbose(
    skill_ids, budget: int | None = None, max_context_tokens: int | None = None
):
    """Budget-trim an ordered list of skill ids; return ``(kept_ids, dropped)``
    where ``dropped`` is the list of AgentSkill objects that didn't fit (for
    surfacing a graceful notice to the user)."""
    ids = [str(i) for i in skill_ids]
    by_id = {str(s.id): s for s in AgentSkill.objects.filter(id__in=ids)}
    ordered = [by_id[i] for i in ids if i in by_id]
    kept, dropped = skills_within_budget(ordered, budget, max_context_tokens)
    return [str(s.id) for s in kept], dropped


def trim_ids_to_budget(
    skill_ids, budget: int | None = None, max_context_tokens: int | None = None
) -> list[str]:
    """Budget-trim an ordered list of skill ids (load-path backstop)."""
    kept, _ = trim_ids_to_budget_verbose(skill_ids, budget, max_context_tokens)
    return kept


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
    # ``user`` may be None (an org skill with no creator, scanned by the worker or
    # a management command): the scan still runs; only the event log is skipped.
    uid = getattr(user, "pk", None)
    for window in _windows(text, _SCAN_WINDOW_CHARS):
        hres = heuristic_scan(window)
        if hres.should_block:
            _log_guardrail(
                _create_event_sync, user, org_id, "heuristic", hres.tags,
                hres.confidence, "high", "blocked", window,
            )
            return "quarantine", detail, list(hres.tags)

        try:
            cres = classify_web_content_sync(window, uid, org_id)
        except GuardrailModelUnavailableError:
            logger.warning(
                "skill guardrail: no classifier model for org_id=%s; skipping", org_id,
            )
            continue
        if not cres.is_suspicious:
            continue

        decision = review_flagged_chunk(
            window, cres, document_title=label, neighbor_context="",
            org_id=org_id, user_id=uid,
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
    if user is None:
        # GuardrailEvent.user is required; a creator-less org skill scanned on
        # the worker has nobody to attribute the event to. The verdict itself is
        # unaffected — only the audit row is skipped.
        logger.info(
            "skill guardrail: no user to attribute a %s/%s event to; skipping event",
            check_type, action_taken,
        )
        return
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
                          kind: str = SkillResource.Kind.REFERENCE,
                          name: str | None = None) -> SkillResource:
    """Store an uploaded file and return a PROCESSING resource — fast, no
    extraction or scanning (those run off the request in ``process_upload``, so
    heavy PDF/Office extraction never ties up or OOMs the web dyno).

    ``name`` overrides the display name (deduped within the skill); when omitted
    the name is derived from ``filename`` (the upload form's behavior). The
    agent's file-attach tool passes an explicit name."""
    file_type = detect_file_type(filename)  # raises UnsupportedResourceType
    ext = _ext_of(filename)
    resource = SkillResource(
        skill=skill,
        name=_unique_name(skill, name or filename),
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


def _store_optimized_image(resource: SkillResource, data: bytes) -> bool:
    """Save a downscaled/transcoded copy to ``resource.optimized_file`` (unsaved).

    Returns True when a genuinely smaller derivative was produced and set on the
    field (caller then persists ``optimized_file``/``media_type``); False when
    the image can't be decoded or optimizing wouldn't shrink it (the view then
    falls back to ``original_file``).
    """
    try:
        from django.core.files.base import ContentFile

        from core.images import optimize_for_vision

        opt = optimize_for_vision(data)
        if opt is None:
            return False
        opt_bytes, media = opt
        if len(opt_bytes) >= len(data):
            return False
        ext = {"image/jpeg": "jpg", "image/png": "png"}.get(media, "img")
        base = (resource.original_filename or resource.name or "resource").rsplit(".", 1)[0]
        resource.optimized_file.save(f"{base}.{ext}"[:255], ContentFile(opt_bytes), save=False)
        resource.media_type = media
        return True
    except Exception:
        logger.info("skill resource image optimize failed", exc_info=True)
        return False


def process_upload(resource: SkillResource, user) -> None:
    """Extract text from a stored upload then guardrail/PII scan it. Runs on the
    worker (see tasks.py). Fails closed (SCAN_FAILED) on any extraction error."""
    try:
        with resource.original_file.open("rb") as fh:
            data = fh.read()
        extra_fields: list[str] = []
        if resource.file_type == SkillResource.FileType.IMAGE:
            resource.content = ""
            resource.content_sha256 = hashlib.sha256(data).hexdigest()
            # Downscale/transcode to the vision cap for the model to read; keep
            # original_file pristine (downloadable) — see optimized_file.
            if _store_optimized_image(resource, data):
                extra_fields += ["optimized_file", "media_type"]
        elif resource.file_type == SkillResource.FileType.PDF:
            resource.content = extract_text(data, "pdf")[:MAX_RESOURCE_CHARS]
            resource.content_sha256 = hashlib.sha256(data).hexdigest()
        else:  # text-extractable
            ext = _ext_of(resource.original_filename)
            resource.content = extract_text(data, ext)[:MAX_RESOURCE_CHARS]
            resource.content_sha256 = resource_content_hash(resource.content)
        resource.token_count = count_tokens(resource.content) if resource.content else 0
        resource.save(update_fields=["content", "content_sha256", "token_count", *extra_fields])
    except Exception:
        logger.exception(
            "process_upload: extraction failed for %s", resource.original_filename
        )
        resource.status = SkillResource.Status.SCAN_FAILED
        resource.error = "Could not read this file."
        resource.save(update_fields=["status", "error"])
        _regate_after_upload(resource, user)
        return

    scan_resource(resource, user)
    _regate_after_upload(resource, user)


def _regate_after_upload(resource: SkillResource, user) -> None:
    """Re-run the skill's approval gate now that an upload's content hash is
    final — it changed after the request that created the row returned, so the
    in-request hook (``request_skill_rescan``) deliberately left the skill
    PENDING without queueing a scan. Never masks the resource's own outcome."""
    try:
        skill = AgentSkill.objects.filter(pk=resource.skill_id).first()
        if skill is None or skill.level == AgentSkill.Level.SYSTEM:
            return
        if skill_is_approved(skill, user=user):
            return
        scan_and_approve_skill(skill, user)
    except Exception:  # noqa: BLE001 — the resource outcome is already persisted
        logger.exception(
            "process_upload: approval re-scan failed for skill_id=%s", resource.skill_id
        )


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
    the skill's approval-gate scan (it carries no ``original_filename``); the
    calling view/tool queues that scan via ``request_skill_rescan``."""
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


def seed_file_resource(
    skill: AgentSkill,
    *,
    data: bytes,
    filename: str,
    kind: str = SkillResource.Kind.REFERENCE,
) -> SkillResource:
    """Create/refresh a file-backed resource from seed bytes — synchronous,
    UNSCANNED, ``status=READY``.

    Used by ``seed_system_skills`` to bundle first-party files (image/PDF/text)
    with a *system* skill. System-skill content is trusted (``skill_is_approved``
    short-circuits for ``level=system``) and no user exists at ``post_migrate``,
    so the guardrail/PII scan that ``process_upload`` runs is deliberately
    skipped here.

    Idempotent on the content hash: a same-named resource whose bytes are
    unchanged (and whose file is present) is returned untouched, so re-seeding on
    every migrate does NOT re-write storage or orphan the previous blob (each
    ``FileField.save`` mints a fresh UUID path).
    """
    name = os.path.basename(filename) or "resource"
    file_type = detect_file_type(filename)  # raises UnsupportedResourceType
    ext = _ext_of(filename)
    is_native = file_type in (SkillResource.FileType.IMAGE, SkillResource.FileType.PDF)

    if file_type == SkillResource.FileType.IMAGE:
        text = ""
        new_hash = hashlib.sha256(data).hexdigest()
    elif file_type == SkillResource.FileType.PDF:
        text = extract_text(data, "pdf")[:MAX_RESOURCE_CHARS]
        new_hash = hashlib.sha256(data).hexdigest()
    else:  # text-extractable
        text = extract_text(data, ext)[:MAX_RESOURCE_CHARS]
        new_hash = resource_content_hash(text)

    existing = skill.templates.filter(name=name).first()
    if (
        existing is not None
        and existing.content_sha256 == new_hash
        and (existing.original_file if is_native else True)
    ):
        return existing  # unchanged — skip the storage write

    resource = existing or SkillResource(skill=skill, name=name)
    resource.kind = kind
    resource.file_type = file_type
    resource.original_filename = name
    resource.content = text
    resource.content_sha256 = new_hash
    resource.token_count = count_tokens(text) if text else 0
    resource.status = SkillResource.Status.READY
    resource.is_quarantined = False
    resource.quarantine_reason = ""
    resource.quarantine_detail = ""
    resource.error = ""

    if is_native:
        resource.original_file.save(name, ContentFile(data), save=False)
        # Fallback media type from the extension; overridden below if we produce
        # a vision-optimized derivative.
        resource.media_type = ft.canonical_mime_for_extension(ext) or ""
        if file_type == SkillResource.FileType.IMAGE:
            # Sets optimized_file + media_type when a smaller derivative exists;
            # otherwise the read path falls back to original_file.
            _store_optimized_image(resource, data)

    resource.save()
    recompute_standing_tokens(skill)
    return resource


def update_resource(
    resource: SkillResource, *, name=None, content=None, kind=None
) -> None:
    """Rename a resource, retype its kind, and/or edit a typed text resource's
    content.

    Content is editable only for typed text resources (no original file);
    uploaded files are rename-only. ``kind`` (reference/template) is editable for
    any resource. Typed text is (re)scanned by the approval gate, not here, so
    this makes no LLM calls — the caller queues the gate via
    ``request_skill_rescan``. Recomputes the skill standing count.
    """
    fields: list[str] = []
    if name is not None:
        new_name = name.strip()[:255]
        if new_name and new_name != resource.name:
            resource.name = new_name
            fields.append("name")

    if kind is not None and kind in SkillResource.Kind.values and kind != resource.kind:
        # ``kind`` is part of compute_skill_content_hash, so changing it
        # un-approves the skill → the caller's request_skill_rescan re-queues it.
        resource.kind = kind
        fields.append("kind")

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


def replace_resource_file(resource: SkillResource, *, data: bytes, filename: str) -> None:
    """Swap the underlying file of an uploaded resource in place, keeping its
    display ``name`` and ``kind``.

    Resets the row to PROCESSING with cleared extraction/scan state; the caller
    enqueues ``process_skill_resource_upload_task`` to re-extract + re-scan (so a
    heavy PDF/Office file never ties up the web dyno). Raises
    ``UnsupportedResourceType`` for a file type we can't ingest.
    """
    file_type = detect_file_type(filename)  # raises UnsupportedResourceType
    ext = _ext_of(filename)
    # Drop the previous blobs so we don't orphan storage (FileField.save mints a
    # fresh UUID path each time).
    if resource.original_file:
        resource.original_file.delete(save=False)
    if resource.optimized_file:
        resource.optimized_file.delete(save=False)
    resource.file_type = file_type
    resource.original_filename = filename
    resource.media_type = ft.canonical_mime_for_extension(ext) or ""
    resource.content = ""
    resource.content_sha256 = ""
    resource.token_count = 0
    resource.status = SkillResource.Status.PROCESSING
    resource.is_quarantined = False
    resource.quarantine_reason = ""
    resource.quarantine_detail = ""
    resource.pii_categories = {}
    resource.error = ""
    resource.original_file.save(filename, ContentFile(data), save=False)
    resource.save()
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


# --- approval gate ---------------------------------------------------------

def _scan_user_for_skill(skill: AgentSkill, user):
    """The user a scan runs as. GuardrailEvent rows need one: the caller when
    known, else the skill's creator, else an admin of the skill's org, else
    None (the scan still runs; only the audit rows are skipped)."""
    if user is not None and getattr(user, "pk", None):
        return user
    if skill.created_by_id:
        return skill.created_by
    if skill.organization_id:
        from accounts.models import Membership

        membership = (
            Membership.objects.filter(
                org_id=skill.organization_id, role=Membership.Role.ADMIN
            )
            .select_related("user")
            .order_by("pk")
            .first()
        )
        return membership.user if membership else None
    return None


def _scan_config_fingerprint(org_id) -> str:
    """Short hash of the scan configuration a verdict depends on (guardrail
    model, PII model and gate switches), so a config change invalidates cached
    clean verdicts instead of letting a blob the new model never saw through."""
    from core.preferences import resolve_org_feature_model
    from documents.services.pii_scan import resolve_pii_gate

    parts = [resolve_org_feature_model(org_id, "guardrail_web_scan") or ""]
    parts.extend(str(p) for p in resolve_pii_gate(org_id))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _blob_cache_key(skill_id, fingerprint: str, blob: str) -> str:
    digest = hashlib.sha256((blob or "").encode("utf-8")).hexdigest()
    return f"skillscan:v1:{skill_id}:{fingerprint}:{digest}"


def _scan_blob(
    skill: AgentSkill, label: str, blob: str, user, org_id, fingerprint: str
) -> list[str]:
    """Guardrail + PII scan one authored text blob; returns the block reasons
    (empty = clean).

    A clean verdict is cached (Django cache, fail-open) under the skill, the
    blob's content hash and the scan config, so a later save of the same skill
    re-scans only the blobs that actually changed — an unchanged 20k-char
    instructions field costs nothing when the user fixes a typo in the
    description. Blocks are never cached: the author's fix is re-scanned for
    real."""
    from django.conf import settings
    from django.core.cache import cache

    key = _blob_cache_key(skill.pk, fingerprint, blob)
    try:
        if cache.get(key):
            return []
    except Exception:  # noqa: BLE001 — a backend that raises degrades to a miss
        logger.info("skill scan: verdict cache read failed; scanning", exc_info=True)

    title = f"{skill.name}: {label}"
    reasons: list[str] = []
    g_action, _, _ = _scan_text_guardrail(blob, user, org_id, title)
    if g_action == "quarantine":
        reasons.append(f"“{label}” was flagged as adversarial content.")
    else:
        _, pii_quar, pii_reason, _ = _scan_text_pii(blob, user, org_id, title)
        if pii_quar:
            reasons.append(f"“{label}”: {pii_reason}")

    if not reasons:
        ttl = int(getattr(settings, "SKILL_SCAN_VERDICT_TTL_SECONDS", 30 * 86400))
        try:
            cache.set(key, 1, timeout=ttl)
        except Exception:  # noqa: BLE001
            logger.info("skill scan: verdict cache write failed", exc_info=True)
    return reasons


def _live_content_hash(skill_pk) -> str | None:
    """The skill's content hash as committed right now (fresh rows, no instance
    state), or None when the skill is gone."""
    fresh = (
        AgentSkill.objects.filter(pk=skill_pk).only("instructions", "description").first()
    )
    if fresh is None:
        return None
    resources = SkillResource.objects.filter(skill_id=skill_pk).order_by("name")
    return compute_skill_content_hash(fresh, resources)


def _stamp_scan_outcome(skill: AgentSkill, snapshot_hash: str, **fields) -> bool:
    """Write a scan outcome only if the skill's content is still what was
    scanned. A save that landed mid-scan has queued its own scan and owns the
    outcome — stamping here would approve content that was never scanned (or
    block content that was already fixed). Returns whether the write happened."""
    if _live_content_hash(skill.pk) != snapshot_hash:
        logger.info(
            "skill scan: outcome %s discarded for skill_id=%s (content changed mid-scan)",
            fields.get("scan_state"), skill.pk,
        )
        return False
    AgentSkill.objects.filter(pk=skill.pk).update(**fields)
    for name, value in fields.items():
        setattr(skill, name, value)
    return True


def scan_and_approve_skill(skill: AgentSkill, user) -> bool:
    """The approval gate. Scans the skill's authored text (instructions,
    description, typed text resources) and verifies no bundled resource is
    quarantined or failed its own scan, then stamps APPROVED (with the content
    hash) or BLOCKED (with the reasons). Makes LLM calls — runs on the worker
    (``scan_and_approve_skill_task``), never inside a web request.

    System skills are auto-approved. Defers (stays PENDING, stamps nothing)
    while an upload is still in flight; ``process_upload`` re-runs the gate when
    the upload lands. Returns True only when APPROVED was written."""
    if skill.level == AgentSkill.Level.SYSTEM:
        return True

    user = _scan_user_for_skill(skill, user)

    # Snapshot what is scanned and hash it up front; the outcome is stamped only
    # if the content is still identical at the end (see _stamp_scan_outcome).
    skill.refresh_from_db(fields=["name", "instructions", "description", "scan_state"])
    resources = list(SkillResource.objects.filter(skill=skill).order_by("name"))
    snapshot_hash = compute_skill_content_hash(skill, resources)

    pending = AgentSkill.ScanState.PENDING
    if any(r.status in IN_FLIGHT_STATUSES for r in resources):
        if skill.scan_state != pending:
            AgentSkill.objects.filter(pk=skill.pk).update(scan_state=pending, scan_detail="")
            skill.scan_state, skill.scan_detail = pending, ""
        return False

    if skill.scan_state != pending:
        AgentSkill.objects.filter(pk=skill.pk).update(scan_state=pending)
        skill.scan_state = pending

    org_id = _org_id_for_skill(skill, user)
    blocked: list[str] = []

    quarantined = [r.name for r in resources if r.is_quarantined]
    if quarantined:
        blocked.append(
            "Remove or replace quarantined file(s): " + ", ".join(quarantined[:5])
        )
    failed = [
        r.name for r in resources
        if not r.is_quarantined and r.status == SkillResource.Status.SCAN_FAILED
    ]
    if failed:
        blocked.append(
            "Re-upload file(s) whose safety scan failed: " + ", ".join(failed[:5])
        )

    blobs = [("instructions", skill.instructions), ("description", skill.description)]
    for res in resources:
        if res.file_type == SkillResource.FileType.TEXT and not res.original_filename:
            blobs.append((res.name, res.content))

    try:
        fingerprint = _scan_config_fingerprint(org_id)
        for label, blob in blobs:
            if not (blob or "").strip():
                continue
            blocked.extend(_scan_blob(skill, label, blob, user, org_id, fingerprint))
    except Exception:
        logger.exception("scan_and_approve_skill: scan failed for skill_id=%s", skill.pk)
        _stamp_scan_outcome(
            skill, snapshot_hash,
            scan_state=AgentSkill.ScanState.UNSCANNED,
            scan_detail="The safety scan could not be completed. Try again.",
        )
        return False

    if blocked:
        _stamp_scan_outcome(
            skill, snapshot_hash,
            scan_state=AgentSkill.ScanState.BLOCKED,
            scan_detail=" ".join(blocked)[:2000],
        )
        return False

    return _stamp_scan_outcome(
        skill, snapshot_hash,
        scan_state=AgentSkill.ScanState.APPROVED,
        approved_content_hash=snapshot_hash,
        scan_detail="",
    )


def _dispatch_scan(skill_id, user_id) -> bool:
    """Enqueue the approval scan on the worker. A failed publish must not strand
    the skill in PENDING: the state reverts to UNSCANNED with a retry hint (the
    toggle or the next save re-dispatches)."""
    from .tasks import scan_and_approve_skill_task

    try:
        scan_and_approve_skill_task.delay(str(skill_id), user_id)
        return True
    except Exception:  # noqa: BLE001 — broker down, publish retries exhausted, …
        logger.exception("skill scan: could not enqueue scan for skill_id=%s", skill_id)
        AgentSkill.objects.filter(
            pk=skill_id, scan_state=AgentSkill.ScanState.PENDING
        ).update(
            scan_state=AgentSkill.ScanState.UNSCANNED,
            scan_detail="The safety scan could not be scheduled. Try again.",
        )
        return False


def request_skill_rescan(skill: AgentSkill, user, *, dispatch: bool = True) -> dict:
    """Re-run the approval gate after a write that may have changed the skill's
    content — THE hook every writer calls (form save, resource endpoints, the
    Skill Creator's tools, copy/import/create, tier moves).

    * System skill, or still approved (hash unchanged, or no scanning configured
      for the org) → no-op. Rename / emoji / tool-list saves cost nothing.
    * Nothing to scan (no authored text, no blocked resource, nothing in flight)
      → APPROVED inline: a brand-new skill is usable at birth, no LLM calls.
    * Otherwise the skill goes PENDING and the scan is queued on the worker once
      the surrounding transaction commits. With an upload still in flight
      nothing is queued — ``process_upload`` runs the gate when it lands.

    Returns ``{"scan_state", "approved", "dispatched"}``.
    """
    from django.db import transaction

    if skill.level == AgentSkill.Level.SYSTEM:
        return {"scan_state": skill.scan_state, "approved": True, "dispatched": False}

    resources = list(SkillResource.objects.filter(skill=skill).order_by("name"))
    if skill_is_approved(skill, user=user, resources=resources):
        return {"scan_state": skill.scan_state, "approved": True, "dispatched": False}

    in_flight = any(r.status in IN_FLIGHT_STATUSES for r in resources)
    blocking = any(
        r.is_quarantined or r.status == SkillResource.Status.SCAN_FAILED
        for r in resources
    )
    texts = [skill.instructions, skill.description] + [
        r.content for r in resources
        if r.file_type == SkillResource.FileType.TEXT and not r.original_filename
    ]
    # str(): an import payload can hand a non-string through to the instance
    # (the column stores its text form); anything non-blank is worth scanning.
    has_text = any(str(t or "").strip() for t in texts)

    if not (in_flight or blocking or has_text):
        skill.scan_state = AgentSkill.ScanState.APPROVED
        skill.approved_content_hash = compute_skill_content_hash(skill, resources)
        skill.scan_detail = ""
        skill.save(update_fields=["scan_state", "approved_content_hash", "scan_detail"])
        return {"scan_state": skill.scan_state, "approved": True, "dispatched": False}

    skill.scan_state = AgentSkill.ScanState.PENDING
    skill.scan_detail = ""
    skill.save(update_fields=["scan_state", "scan_detail"])

    dispatched = False
    if dispatch and not in_flight:
        skill_id, user_id = str(skill.pk), getattr(user, "pk", None)
        transaction.on_commit(lambda: _dispatch_scan(skill_id, user_id))
        dispatched = True
    return {"scan_state": skill.scan_state, "approved": False, "dispatched": dispatched}
