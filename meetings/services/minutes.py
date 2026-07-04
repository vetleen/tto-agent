"""Service that prepares a fresh chat thread for the "Create meeting minutes
with Wilfred" flow.

Mirrors ``agent_skills/views.py:476-544`` (skills_edit_in_chat). The chat
consumer's existing ``pending_initial_turn`` machinery handles the auto-fire
of the first assistant turn — there is no consumer change required.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Sentinel: create_minutes_thread(summarizer_skill=_UNSET) resolves the user's
# default; passing None explicitly means "no skill" (a valid choice).
_UNSET = object()


def get_eligible_summarizer_skills(user):
    """Return the skills offered in the summarizer picker.

    Uses ``get_available_skills`` so only *main-audience* skills appear — the
    minutes thread is a main thread, and sub-agent specializations must never be
    attachable there. Also org-gated and per-user-selection-aware.
    """
    from agent_skills.services import get_available_skills

    return get_available_skills(user)


def _default_summarizer_skill(user):
    """The org-gated, selection-aware default summarizer skill, or None.

    Routes through ``get_available_skills`` (same visibility as the picker) so an
    org that hasn't enabled the system meeting-summarizer — or a user who
    disabled it — gets no default instead of a silently-dropped skill.
    """
    from agent_skills.services import get_available_skills

    return next(
        (s for s in get_available_skills(user) if s.slug == "meeting-summarizer"),
        None,
    )


def resolve_default_summarizer_skill(user):
    """Resolve which summarizer skill to default the picker/thread to.

    Order: the user's saved preference (a still-accessible skill, or an explicit
    "no skill"), else the org-enabled meeting-summarizer, else None. ``None`` is
    a valid outcome meaning "create the thread without a skill".
    """
    from agent_skills.services import get_skill_for_user
    from core.preferences import _get_user_preferences

    meetings_prefs = _get_user_preferences(user).get("meetings") or {}
    if "summarizer_skill_id" in meetings_prefs:
        saved = meetings_prefs["summarizer_skill_id"]
        if saved is None:
            return None  # explicit "no skill"
        skill = get_skill_for_user(user, saved)  # gated; None if gone/inaccessible
        if skill is not None:
            return skill
        # Stale/inaccessible saved choice → fall through to the org default.
    return _default_summarizer_skill(user)


def set_default_summarizer_preference(user, skill_id_or_none):
    """Persist the user's summarizer choice.

    ``skill_id_or_none`` is a skill-id string, or ``None`` for an explicit
    "no skill". Stored in ``UserSettings.preferences["meetings"]`` (no migration).
    """
    from accounts.services import update_user_preferences

    def mutate(prefs):
        prefs.setdefault("meetings", {})["summarizer_skill_id"] = skill_id_or_none

    update_user_preferences(user, mutate)


def _format_duration_minutes(seconds) -> str:
    if not seconds:
        return "unknown"
    minutes = max(1, int(seconds) // 60)
    return f"~{minutes} minutes"


def _build_seed_message(
    meeting,
    canvas_title: str,
    model_label: str,
    skill_name: str | None = None,
    attachment_count: int = 0,
) -> str:
    from chat.services import CANVAS_MAX_CHARS

    transcript = meeting.transcript or ""
    n_chars = len(transcript)
    duration = _format_duration_minutes(meeting.duration_seconds)
    parts: list[str] = []
    if n_chars <= CANVAS_MAX_CHARS:
        parts.append(
            f"The user opened this thread to create meeting minutes (or a summary) "
            f"for **{meeting.name}**. A transcript of the meeting "
            f"({n_chars} characters, {duration} of audio, transcribed via "
            f'`{model_label}`) is preloaded in the canvas titled "{canvas_title}".'
        )
    else:
        # The canvas only holds the first CANVAS_MAX_CHARS characters; tell the
        # model explicitly rather than claiming the full transcript is preloaded.
        parts.append(
            f"The user opened this thread to create meeting minutes (or a summary) "
            f"for **{meeting.name}**. The meeting transcript is {n_chars} characters "
            f"({duration} of audio, transcribed via `{model_label}`); only the first "
            f"{CANVAS_MAX_CHARS:,} characters are preloaded in the canvas titled "
            f'"{canvas_title}" — the remainder was truncated to fit, so the end of '
            f"the meeting is not shown in the canvas."
        )
    if meeting.agenda and meeting.agenda.strip():
        parts.append(f"Agenda: {meeting.agenda.strip()}")
    if meeting.participants and meeting.participants.strip():
        parts.append(f"Participants: {meeting.participants.strip()}")
    if meeting.description and meeting.description.strip():
        parts.append(f"Description: {meeting.description.strip()}")
    if attachment_count:
        noun = "file" if attachment_count == 1 else "files"
        parts.append(
            f"The user has also attached {attachment_count} supporting {noun} "
            "from the meeting (e.g. slides, agendas, notes). Use them alongside "
            "the transcript when drafting the minutes."
        )
    final = (
        "Your job is to produce well-structured meeting minutes (or a summary) "
        "and draft them into a new canvas. "
        "If important context is missing — attendees, meeting purpose, the boundary "
        "between decisions and action items — greet the user briefly and ask one "
        "focused question before drafting. "
    )
    if skill_name:
        final += f"Use the attached {skill_name} skill to complete the task. "
    final += "Iterate with the user until they are satisfied."
    parts.append(final)
    return " ".join(parts)


def _copy_meeting_attachments_to_thread(meeting, thread, user):
    """Copy MeetingAttachment files into fresh ChatAttachment rows on *thread*.

    Validation mirrors the chat ``+`` menu upload flow
    (``chat/views.py:upload_attachments``): unsupported content types and
    oversized files are skipped rather than raising. Bytes are copied into a
    fresh storage path so deleting the meeting attachment doesn't affect the
    chat copy (and vice versa).

    Returns ``(accepted, skipped)`` where *accepted* is a list of
    ``ChatAttachment`` instances and *skipped* is a list of
    ``(original_filename, reason)`` tuples.
    """
    from django.core.files import File
    from chat.models import ChatAttachment
    from chat.services import (
        SUPPORTED_ATTACHMENT_TYPES,
        SUPPORTED_DOCX_TYPES,
        max_size_for_content_type,
    )

    accepted: list = []
    skipped: list[tuple[str, str]] = []

    for ma in meeting.attachments.all().order_by("uploaded_at"):
        ct = ma.content_type or ""
        # Meeting-side upload accepts any type with no content_type validation;
        # browsers also sometimes report .docx as application/octet-stream.
        if ct not in SUPPORTED_ATTACHMENT_TYPES and (ma.original_filename or "").lower().endswith(".docx"):
            ct = next(iter(SUPPORTED_DOCX_TYPES))
        if ct not in SUPPORTED_ATTACHMENT_TYPES:
            skipped.append((ma.original_filename, "unsupported file type"))
            continue
        if ma.size_bytes and ma.size_bytes > max_size_for_content_type(ct):
            skipped.append((ma.original_filename, "too large"))
            continue
        try:
            # Stream the copy through storage rather than reading the whole
            # attachment into memory (up to 25 files x tens of MB per request on
            # a 512MB dyno). File(fh) lets the backend copy in chunks; create
            # inside the `with` so the handle stays open during the copy.
            with ma.file.open("rb") as fh:
                att = ChatAttachment.objects.create(
                    thread=thread,
                    message=None,
                    uploaded_by=user,
                    file=File(fh, name=ma.original_filename),
                    original_filename=(ma.original_filename or "")[:255],
                    content_type=ct,
                    size_bytes=ma.size_bytes or 0,
                )
            accepted.append(att)
        except Exception:
            logger.exception(
                "create_minutes_thread: failed to copy meeting attachment %s (%s)",
                ma.id, ma.original_filename,
            )
            skipped.append((ma.original_filename, "copy failed"))

    return accepted, skipped


def _build_attachments_disclaimer(accepted_count: int, skipped: list[tuple[str, str]]) -> str:
    """Build the visible user-message text that explains the auto-attached files."""
    parts: list[str] = []
    if accepted_count:
        parts.append(
            "These files were uploaded to the meeting and are automatically "
            "included in this thread."
        )
    if skipped:
        desc = ", ".join(f"{name} ({reason})" for name, reason in skipped)
        if accepted_count:
            parts.append(f"Skipped: {desc}.")
        else:
            parts.append(
                "Files uploaded to the meeting couldn't be included in this "
                f"thread. Skipped: {desc}."
            )
    return " ".join(parts)


def create_minutes_thread(user, meeting, summarizer_skill=_UNSET):
    """Create a ChatThread for drafting minutes, optionally pre-loaded with a skill.

    ``summarizer_skill``: pass ``_UNSET`` (default) to resolve the user's default
    via :func:`resolve_default_summarizer_skill`; pass a skill to attach it; pass
    ``None`` to create the thread with **no** skill (a valid choice — the model
    drafts the minutes itself). Returns ``(thread, error_message)``.
    """
    from chat.models import ChatCanvas, ChatMessage, ChatThread, ChatThreadSkill
    from chat.services import (
        CANVAS_MAX_CHARS,
        create_canvas_checkpoint,
        set_active_canvas,
    )

    if summarizer_skill is _UNSET:
        summarizer_skill = resolve_default_summarizer_skill(user)
    # summarizer_skill is now an AgentSkill or None (None = no-skill, valid).

    if not (meeting.transcript or "").strip():
        return None, "This meeting has no transcript yet."

    thread = ChatThread.objects.create(
        created_by=user,
        # ChatThread.title is CharField(max_length=255) and meeting.name can be
        # a full 255 chars, so cap (matches the canvas_title truncation below).
        title=f"Minutes for {meeting.name}"[:255],
        metadata={
            "source_meeting_id": str(meeting.uuid),
            "pending_initial_turn": True,
        },
    )
    if summarizer_skill is not None:
        ChatThreadSkill.objects.create(thread=thread, skill=summarizer_skill)

    # Preload the transcript into a canvas so Wilfred sees it as the active
    # canvas (its content is injected into the per-turn prompt). Truncate to
    # the canvas character cap if the transcript is unusually long.
    canvas_title = f"Meeting transcript — {meeting.name}"[:255]
    canvas_content = (meeting.transcript or "")[:CANVAS_MAX_CHARS]
    canvas = ChatCanvas.objects.create(
        thread=thread,
        title=canvas_title,
        content=canvas_content,
    )
    checkpoint = create_canvas_checkpoint(
        canvas, source="original", description="Meeting transcript",
    )
    canvas.accepted_checkpoint = checkpoint
    canvas.save(update_fields=["accepted_checkpoint"])
    set_active_canvas(thread.id, canvas)

    # Copy meeting attachments into fresh ChatAttachment rows before we build
    # the seed, so the hidden seed can tell the LLM how many supporting files
    # are attached.
    accepted, skipped = _copy_meeting_attachments_to_thread(meeting, thread, user)

    # Hidden seed user message that primes the LLM. The chat consumer's
    # `pending_initial_turn` flag triggers an auto-assistant turn on first
    # WS load.
    model_label = meeting.transcription_model or "uploaded text"
    seed_content = _build_seed_message(
        meeting,
        canvas_title,
        model_label,
        skill_name=(summarizer_skill.name if summarizer_skill else None),
        attachment_count=len(accepted),
    )
    ChatMessage.objects.create(
        thread=thread,
        role="user",
        content=seed_content,
        is_hidden_from_user=True,
    )

    # Visible user message carrying the attachments, mirroring the "+ → Add
    # images & files" flow: the user's opening turn arrives with the files.
    # attachment_ids in metadata is what the consumer's enrichment reads to feed
    # the files to the LLM (the message FK alone isn't enough) — so meeting
    # attachments get the same docx/pdf extraction + embedded-image handling as
    # chat attachments.
    if accepted or skipped:
        disclaimer = _build_attachments_disclaimer(len(accepted), skipped)
        disclaimer_msg = ChatMessage.objects.create(
            thread=thread,
            role="user",
            content=disclaimer,
            is_hidden_from_user=False,
            metadata={"attachment_ids": [str(a.id) for a in accepted]} if accepted else {},
        )
        if accepted:
            ChatAttachment = accepted[0].__class__
            ChatAttachment.objects.filter(id__in=[a.id for a in accepted]).update(
                message=disclaimer_msg,
            )

    logger.info(
        "create_minutes_thread: created thread %s for meeting %s "
        "(user=%s, attachments_accepted=%d, attachments_skipped=%d)",
        thread.id, meeting.uuid, user.id, len(accepted), len(skipped),
    )
    return thread, None
