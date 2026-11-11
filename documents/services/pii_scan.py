"""Classify document text by GDPR personal data categories.

Two-stage design for the gated Article 9/10 categories: the cheap classifier
(``scan_pii_categories``) only raises recall-tuned *candidate* flags; the
primary-model reviewer (``documents.services.pii_review``) makes the final
quarantine decision per flagged window. Every escalation — confirmed or
dismissed — is recorded as a ``PIIReviewEvent`` for calibration review.
Ordinary (Article 6) categories never escalate; the classifier's output goes
straight to tags.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_PII_SYSTEM_PROMPT = """\
You are a GDPR personal data classifier. Your task is to determine which \
categories of personal data are clearly present in a document.

## Background

Under the EU General Data Protection Regulation (GDPR), personal data is any \
information relating to an identified or identifiable natural person \
(Article 4(1)). Certain categories receive heightened protection:

- **Ordinary personal data** (Article 6): processed under a lawful basis.
- **Special category data** (Article 9): processing is prohibited unless an \
  explicit exception applies. Includes health data, racial/ethnic origin, \
  political opinions, religious beliefs, trade union membership, genetic data, \
  biometric data used for identification, and data on sex life or orientation.
- **Criminal offence data** (Article 10): requires specific legal authority.

## Categories to assess

For each category below, return `true` if the document **clearly contains** \
that type of personal data relating to identifiable individuals. Return \
`false` if the category is absent or only mentioned abstractly (e.g. a policy \
*about* health data does not constitute health data itself).

### pii_ordinary_identity
Personal identity information that directly or indirectly identifies a \
natural person. Examples: full names, email addresses, telephone numbers, \
physical/postal addresses, official identifiers (national ID numbers, \
passport numbers, organisation numbers linked to a natural person), \
photographs or portraits, biometric data that is NOT used for identification \
purposes (probably not relevant for most documents).

### pii_ordinary_professional
Information related to a person's education, employment, and professional \
life. Examples: job titles or roles, organisational affiliations, education \
and qualifications, work history or CV information, professional evaluations \
or performance reviews, salary or compensation details, professional \
relationships (co-authors, supervisors, collaborators), group or committee \
memberships, career history.

### pii_ordinary_communication
Content of communications between or about persons. Examples: meeting minutes \
or transcripts, email body content (not just headers), chat or conversation \
content, voice recordings. Note: this covers the *content*, not metadata like \
sender addresses (which fall under identity).

### pii_ordinary_contact
Digital contact and location data used to reach or locate a person. \
Examples: IP addresses, geolocation data, device identifiers or fingerprints.

### pii_ordinary_security
Authentication and account security data. Examples: password hashes, session \
tokens, authentication logs or login history.

### pii_ordinary_preferences
User preferences and configuration choices. Examples: system settings or \
display preferences, work-related tool or workflow preferences.

### pii_ordinary_financial
Financial and business data linked to identifiable natural persons. Examples: \
business information tied to sole proprietorships or identifiable founders, \
account or payment information, ownership stakes or intellectual property \
rights linked to named inventors or authors.

### pii_ordinary_social
Social and family information. Examples: family relationships (spouse, \
children), personal life history or biographical details beyond professional \
career.

### pii_special_category
GDPR Article 9 special categories — processing is generally prohibited. \
Examples: biometric data processed for the purpose of uniquely identifying a \
person, trade union membership, health data (medical conditions, diagnoses, \
treatments, disability status), racial or ethnic origin, political opinions, \
religious or philosophical beliefs, genetic data, data concerning sex life or \
sexual orientation.

Note: this category is a **candidate flag**. If it is plausibly or arguably \
present, mark it `true` — a second-stage reviewer with more context makes the \
final determination. Prefer recall over precision for this category only.

### pii_criminal_offence
GDPR Article 10 data — requires specific legal authority. Examples: criminal \
convictions, charges, or offences.

Note: this category is a **candidate flag**. If it is plausibly or arguably \
present, mark it `true` — a second-stage reviewer with more context makes the \
final determination. Prefer recall over precision for this category only.

## Decision guidance

- Mark a category `true` only when the document clearly contains personal \
  data of that type relating to identifiable natural persons.
- Generic data does not count: "A patient was treated" is not personal data alone.
- Anonymized data counts: "Patient 42 received treatment" is personal health data.
- A document *about* a data category (e.g. a privacy policy discussing health \
  data) does not itself contain that category of personal data.
- For the eight ordinary categories: mark `true` only when the document clearly \
  contains personal data of that type relating to identifiable natural persons. \
  When in doubt, err on the side of `false`.
- For `pii_special_category` and `pii_criminal_offence`: these are candidate \
  flags for a downstream reviewer. Mark `true` when such data is plausibly \
  present; when in doubt, err on the side of `true`.
"""

# Shown as processing_error when a gated document's PII scan can't run/complete.
SCAN_FAILED_MESSAGE = "Couldn't check this document for sensitive data — retry the scan."

# processing_error marker for a scan that failed only because the Celery broker was
# briefly unreachable at dispatch (not a real scan failure). requeue_stale_documents
# auto-retries versions carrying this message; exhausting the retries converts it to
# the terminal SCAN_FAILED_MESSAGE.
SCAN_DISPATCH_RETRY_MESSAGE = "Couldn't reach the scanner — retrying automatically."

# Presentation-only status (NOT a DB Status value): a scan_failed version carrying the
# transient retry marker renders in the UI as "queued/retrying" rather than "failed".
# See DataRoomDocument.presentation_status / display_status.
SCAN_RETRYING_STATUS = "scan_retrying"

# The two gated categories: an Article 9/10 classifier flag is only a candidate
# until the reviewer confirms it (or no reviewer model is configured).
GATED_CATEGORIES = ("pii_special_category", "pii_criminal_offence")

# Maps each gated category to its article flag on PIIReviewDecision.
_CATEGORY_TO_ARTICLE_ATTR = {
    "pii_special_category": "article_9",
    "pii_criminal_offence": "article_10",
}

# Cap on the window excerpt stored on a PIIReviewEvent (same as the guardrails
# chunk-event raw_input cap).
_EVENT_EXCERPT_CHARS = 2000

# All category field names, in schema order
PII_CATEGORIES = [
    "pii_ordinary_identity",
    "pii_ordinary_professional",
    "pii_ordinary_communication",
    "pii_ordinary_contact",
    "pii_ordinary_security",
    "pii_ordinary_preferences",
    "pii_ordinary_financial",
    "pii_ordinary_social",
    "pii_special_category",
    "pii_criminal_offence",
]


def org_id_for_document(doc) -> int | None:
    """Resolve the uploading user's organization id (None when unaffiliated)."""
    from accounts.models import Membership

    if not doc.uploaded_by_id:
        return None
    return (
        Membership.objects.filter(user_id=doc.uploaded_by_id)
        .values_list("org_id", flat=True)
        .first()
    )


def resolve_pii_gate(org_id) -> tuple[str, bool, bool]:
    """Return ``(pii_model, pii_scan_enabled, pii_quarantine_enabled)`` for an org.

    Single source of truth for the PII-scan configuration used by both
    ``process_document`` (to decide whether to hold a document in SCANNING)
    and ``finalize_document_metadata`` (to run the scan and quarantine).
    """
    from accounts.models import Organization
    from core.preferences import resolve_org_feature_model

    pii_model = resolve_org_feature_model(org_id, "pii_scan")
    pii_enabled = True
    pii_quarantine_enabled = True
    if org_id:
        try:
            prefs = Organization.objects.get(pk=org_id).preferences or {}
            pii_enabled = prefs.get("pii_scan_enabled", True)
            pii_quarantine_enabled = prefs.get("pii_quarantine_enabled", True)
        except Organization.DoesNotExist:
            pass
    return pii_model, pii_enabled, pii_quarantine_enabled


def pii_gate_applies(org_id) -> bool:
    """Whether documents must be held from retrieval until the PII scan completes.

    True only when a scan model is resolved AND the org has both the scan and
    quarantine enabled — without quarantine the scan is informational only, so
    there is nothing to gate on.
    """
    pii_model, pii_enabled, pii_quarantine_enabled = resolve_pii_gate(org_id)
    return bool(pii_model and pii_enabled and pii_quarantine_enabled)


# ---------------------------------------------------------------------------
# The Decisions service's questions (production and the PII shadow share them)
# ---------------------------------------------------------------------------

# Probability at/above which a category counts as present.
PII_FLAG_THRESHOLD = 0.5
# No "clearly" and no "when in doubt, answer no": against an Opus 5.5 reference on 300
# windows (2026-11-11) that wording cost recall (.877 vs .912 at 0.5). The Art. 9/10
# reviewer absorbs the extra candidates; ordinary tags are only a user aid.
PII_QUESTION_STEM = "The text contains personal data relating to identifiable natural persons, of this kind: "
PII_QUESTION_TAIL = " Generic, organisational or hypothetical data does not count."


def build_pii_questions():
    """One predicate per GDPR category, worded from the production schema's own
    field descriptions (``PIICategoryOutput``) so the Decisions service and the
    structured classifier judge the same thing.

    The descriptions alone ("Financial and business data: business information, …")
    leave out the system prompt's "relating to identifiable natural persons", so every
    question states it, with one bar for all ten categories."""
    from llm.types.decisions import Question
    from llm.types.structured import PIICategoryOutput

    questions = []
    for category in PII_CATEGORIES:
        description = (PIICategoryOutput.model_fields[category].description or category).strip().rstrip(".")
        # Continue the sentence: lowercase a leading capitalised word ("Personal identity
        # data" → "personal …") but leave acronyms alone ("GDPR Article 9 …").
        if len(description) > 1 and description[0].isupper() and description[1].islower():
            description = description[0].lower() + description[1:]
        questions.append(Question(
            type="predicate", name=category,
            instructions=f"{PII_QUESTION_STEM}{description}.{PII_QUESTION_TAIL}",
        ))
    return questions


def build_pii_input(window_text: str):
    """The PII system prompt as the developer message + the window text with the
    structured classifier's head/tail truncation (``_prepare_document_text``)."""
    from documents.services.description import _prepare_document_text
    from llm.types.decisions import InputMessage, TextPart

    return [
        InputMessage(role="developer", content=[TextPart(text=_PII_SYSTEM_PROMPT)]),
        InputMessage(role="user", content=[TextPart(text=_prepare_document_text(window_text))]),
    ]


@dataclass(frozen=True)
class PIIWindowVerdict:
    """One window's Layer-1 PII verdict.

    ``categories`` holds only the categories present (``True``), as the legacy
    ``scan_pii_categories`` returned; ``probabilities`` are the Decisions answers
    (``None`` when the legacy classifier answered); ``source`` says which did.
    """

    categories: dict[str, bool] = field(default_factory=dict)
    probabilities: dict[str, float] | None = None
    pathway: str = ""
    source: str = "decisions"
    decide_error: str = ""


def _get_decisions_service():
    """The Decisions service — extracted for testability."""
    from llm.service.decisions_service import get_decisions_service

    return get_decisions_service()


def _interpret_pii_result(result) -> PIIWindowVerdict:
    from llm.service.decisions_fallback import DecisionsRefused

    probabilities = {}
    for category in PII_CATEGORIES:
        answer = result[category]
        if answer.type != "predicate":
            raise DecisionsRefused(f"no probability for {category}")
        probabilities[category] = answer.probability
    return PIIWindowVerdict(
        categories={c: True for c, p in probabilities.items() if p >= PII_FLAG_THRESHOLD},
        probabilities=probabilities, pathway=result.pathway, source="decisions",
    )


def _scan_pii_categories_legacy(
    document_text: str,
    user_id: int | None,
    org_id: int | None,
    *,
    context=None,
    log_metadata: dict | None = None,
) -> dict[str, bool]:
    """The legacy structured classifier over an already-truncated text."""
    from core.preferences import org_feature_reasoning_params, resolve_org_feature_model
    from llm import get_llm_service
    from llm.types import ChatRequest, Message, RunContext
    from llm.types.structured import PIICategoryOutput

    model = resolve_org_feature_model(org_id, "pii_scan")
    params = org_feature_reasoning_params(org_id, "pii_scan", model)
    if log_metadata:
        params = {**(params or {}), "_log_metadata": dict(log_metadata)}
    request = ChatRequest(
        messages=[
            Message(role="system", content=_PII_SYSTEM_PROMPT),
            Message(role="user", content=document_text),
        ],
        model=model,
        stream=False,
        tools=[],
        context=context or RunContext.create(user_id=user_id),
        params=params,
    )

    service = get_llm_service()
    parsed, usage = service.run_structured(request, PIICategoryOutput)
    return {category: True for category in PII_CATEGORIES if getattr(parsed, category, False)}


def scan_pii_window(
    text: str,
    user_id: int | None = None,
    data_room_id: int | None = None,
    org_id: int | None = None,
) -> PIIWindowVerdict:
    """Classify one window into GDPR PII categories: the Decisions service (ten
    predicates, present at probability ≥ 0.5), or the legacy structured classifier
    when Decisions fails (``llm.service.decisions_fallback``)."""
    from llm.service.decisions_fallback import decide_with_fallback
    from llm.types import RunContext

    if not text.strip():
        return PIIWindowVerdict(source="empty")

    from documents.services.description import _prepare_document_text

    context = RunContext.create(user_id=user_id)

    def legacy(meta: dict) -> PIIWindowVerdict:
        categories = _scan_pii_categories_legacy(
            _prepare_document_text(text), user_id, org_id, context=context, log_metadata=meta,
        )
        return PIIWindowVerdict(
            categories=categories, source="legacy", decide_error=str(meta.get("decide_error") or ""),
        )

    verdict = decide_with_fallback(
        build_pii_questions(), build_pii_input(text),
        feature_key="pii_scan", org_id=org_id, context=context,
        prompt_cache_key="decisions:pii_scan",
        interpret=_interpret_pii_result, fallback=legacy, service=_get_decisions_service(),
    )
    logger.info(
        "scan_pii_window: user_id=%s source=%s detected=%s",
        user_id, verdict.source, list(verdict.categories),
    )
    return verdict


def scan_pii_categories(
    text: str,
    user_id: int | None = None,
    data_room_id: int | None = None,
    org_id: int | None = None,
) -> dict[str, bool]:
    """Classify document text into GDPR PII categories.

    Returns a dict of only the categories detected as ``True`` (``scan_pii_window``'s
    categories; kept for the skill PII scan and ``backfill_pii_tags``).
    """
    return scan_pii_window(text, user_id=user_id, data_room_id=data_room_id, org_id=org_id).categories


@dataclass(frozen=True)
class PIIScanResult:
    """Outcome of a full-version PII scan.

    ``categories`` holds only the categories present as ``True`` — for the two
    gated Article 9/10 categories that means reviewer-confirmed (or classifier-
    flagged when no reviewer model is configured). ``detail`` is the joined
    user-facing reviewer findings ("" when nothing was confirmed).
    """

    categories: dict[str, bool] = field(default_factory=dict)
    detail: str = ""


@dataclass
class _ScanState:
    """Mutable accumulator threaded through the per-window scans of one version."""

    detected: dict[str, bool] = field(default_factory=dict)
    # Gated categories settled for this version: reviewer-confirmed (or kept via
    # fallback). Later windows skip review for these — the union cannot change.
    settled: set = field(default_factory=set)
    findings: list = field(default_factory=list)
    window_index: int = 0


def _review_gated_candidates(
    window_text, candidates, state, *,
    document, document_title, version_id, user_id, data_room_id, org_id,
) -> None:
    """Escalate a window's Art. 9/10 candidate flags to the Layer 2 reviewer.

    Applies the reviewer's per-article verdict to ``state.detected`` (candidates
    only — the classifier gates which categories are in play), collects the
    user-facing findings, and always records a ``PIIReviewEvent`` (hit, near-hit,
    or fallback). A reviewer call failure propagates — the same contract as a
    classifier window failure (the caller retries, then marks SCAN_FAILED).
    """
    from documents.models import PIIReviewEvent
    from documents.services.pii_review import review_flagged_pii_window

    decision = review_flagged_pii_window(
        window_text, candidates, document_title, org_id, user_id=user_id,
    )

    if decision is None:
        # No reviewer model configured — keep the classifier's verdict (the
        # pre-reviewer behavior) and settle so later windows don't re-log.
        for category in candidates:
            state.detected[category] = True
            state.settled.add(category)
        action = PIIReviewEvent.Action.FALLBACK
        article_9 = "pii_special_category" in candidates
        article_10 = "pii_criminal_offence" in candidates
        confidence = None
        reasoning = ""
        findings = ""
    else:
        confirmed = [
            c for c in candidates
            if getattr(decision, _CATEGORY_TO_ARTICLE_ATTR[c], False)
        ]
        for category in confirmed:
            state.detected[category] = True
            state.settled.add(category)
        if confirmed:
            action = PIIReviewEvent.Action.CONFIRMED
            if decision.findings:
                state.findings.append(decision.findings)
        else:
            action = PIIReviewEvent.Action.DISMISSED
        article_9 = decision.article_9
        article_10 = decision.article_10
        confidence = decision.confidence
        reasoning = decision.reasoning
        findings = decision.findings

    try:
        PIIReviewEvent.objects.create(
            document_id=document.id if document else None,
            data_room_id=data_room_id,
            version_id=version_id,
            user_id=user_id,
            org_id=org_id,
            window_index=state.window_index,
            document_title=(document_title or "")[:255],
            candidate_categories=list(candidates),
            action=action,
            article_9=article_9,
            article_10=article_10,
            confidence=confidence,
            reasoning=reasoning,
            findings=findings,
            excerpt=window_text[:_EVENT_EXCERPT_CHARS],
        )
    except Exception:
        # The audit row must never take down a scan that already has a verdict.
        logger.exception(
            "PII review event logging failed version_id=%s window=%s",
            version_id, state.window_index,
        )

    logger.info(
        "pii_review: version_id=%s window=%s candidates=%s action=%s",
        version_id, state.window_index, list(candidates), action,
    )


# The chunk fields one PII window is built from (shared with the PII shadow).
PII_WINDOW_FIELDS = ("text", "heading", "token_count", "chunk_index")


def pii_window_text(window: list[dict]) -> str:
    """The text one PII window is classified on: each chunk as ``heading\\ntext``
    (the heading only when present), chunks joined by blank lines — the same join
    every other reader of a version's chunks uses."""
    from documents.services.chunk_access import _join_chunks

    return _join_chunks(window)


def iter_pii_windows(version_id: int, budget: int | None = None):
    """Yield ``(window_index, chunks)`` exactly as ``scan_pii_categories_for_version``
    batches a version: chunks accumulate in chunk_index order until their
    ``token_count`` reaches ``budget`` (``PII_SCAN_WINDOW_TOKENS``); a trailing
    partial window comes last. Shared with the PII shadow
    (``documents/services/pii_shadow.py``) so both see identical windows.
    """
    from django.conf import settings

    from documents.services.chunk_access import iter_version_chunks

    if budget is None:
        budget = getattr(settings, "PII_SCAN_WINDOW_TOKENS", 6000)
    window: list[dict] = []
    window_tokens = 0
    index = 0
    for chunk in iter_version_chunks(version_id, fields=PII_WINDOW_FIELDS):
        window.append(chunk)
        window_tokens += chunk.get("token_count") or 0
        if window_tokens >= budget:
            yield index, window
            window, window_tokens = [], 0
            index += 1
    if window:
        yield index, window


def _scan_window(
    window, state, *,
    document, document_title, version_id, user_id, data_room_id, org_id,
) -> None:
    """Scan one window of chunks and union any detected PII categories into ``state``.

    Ordinary categories go straight into ``state.detected``; gated Article 9/10
    flags are candidates escalated to the reviewer (unless already settled for
    this version). A window failure (e.g. a transient LLM error) propagates:
    documents are held from retrieval until the scan completes, so a silently
    skipped window would flip a document to READY without it ever being fully
    scanned. The caller (``finalize_document_metadata``) retries and marks the
    document SCAN_FAILED when retries are exhausted.
    """
    if not window:
        return
    window_text = pii_window_text(window)
    verdict = scan_pii_window(
        window_text, user_id=user_id, data_room_id=data_room_id, org_id=org_id,
    )
    candidates = []
    for category, present in verdict.categories.items():
        if not present:
            continue
        if category in GATED_CATEGORIES:
            if category not in state.settled:
                candidates.append(category)
        else:
            state.detected[category] = True
    if candidates:
        _review_gated_candidates(
            window_text, candidates, state,
            document=document, document_title=document_title,
            version_id=version_id, user_id=user_id,
            data_room_id=data_room_id, org_id=org_id,
        )


def scan_pii_categories_for_version(
    version_id: int,
    user_id: int | None = None,
    data_room_id: int | None = None,
    org_id: int | None = None,
) -> PIIScanResult:
    """Classify an entire version into GDPR PII categories, scanning all of it.

    Reads the version's chunks in memory-safe windows (so a long document never
    materializes all its text at once) and unions the categories detected in each
    window. Gated Article 9/10 classifier flags are escalated per window to the
    Layer 2 reviewer (``documents.services.pii_review``); only confirmed articles
    end up in the result, with the reviewer's user-facing findings in ``detail``.
    Returns early once every category has been found — further scanning cannot
    change the result.
    """
    from documents.models import DataRoomDocumentVersion

    version = (
        DataRoomDocumentVersion.objects.select_related("document")
        .filter(pk=version_id)
        .first()
    )
    document = version.document if version else None
    document_title = document.display_name if document else ""

    state = _ScanState()
    scan_kwargs = dict(
        document=document, document_title=document_title,
        version_id=version_id, user_id=user_id,
        data_room_id=data_room_id, org_id=org_id,
    )

    for index, window in iter_pii_windows(version_id):
        state.window_index = index
        _scan_window(window, state, **scan_kwargs)
        if len(state.detected) == len(PII_CATEGORIES):  # all categories found — stop early (lossless)
            break
    return PIIScanResult(state.detected, "\n".join(state.findings))
