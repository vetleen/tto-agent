"""Service that prepares a fresh chat thread for the "Create meeting minutes
with Wilfred" flow.

Mirrors ``agent_skills/views.py:476-544`` (skills_edit_in_chat). The chat
consumer's existing ``pending_initial_turn`` machinery handles the auto-fire
of the first assistant turn — there is no consumer change required.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_eligible_summarizer_skills(user):
    """Return all skills accessible to *user* that include ``save_meeting_minutes`` in tool_names."""
    from agent_skills.services import get_accessible_skills

    return sorted(
        [s for s in get_accessible_skills(user) if "save_meeting_minutes" in (s.tool_names or [])],
        key=lambda s: s.name,
    )


def resolve_summarizer_skill(user, meeting):
    """Determine which summarizer skill to use.

    Priority: per-meeting override > per-user default > system fallback.
    Each level validates access and eligibility before accepting.
    """
    from accounts.models import UserSettings
    from agent_skills.models import AgentSkill
    from agent_skills.services import get_skill_for_user

    def _is_eligible(skill):
        return skill and "save_meeting_minutes" in (skill.tool_names or [])

    # 1. Per-meeting override
    if meeting.summarizer_skill_id:
        skill = get_skill_for_user(user, str(meeting.summarizer_skill_id))
        if _is_eligible(skill):
            return skill

    # 2. Per-user default
    try:
        us = UserSettings.objects.get(user=user)
        skill_id = (us.preferences or {}).get("meetings", {}).get("summarizer_skill_id")
        if skill_id:
            skill = get_skill_for_user(user, skill_id)
            if _is_eligible(skill):
                return skill
    except UserSettings.DoesNotExist:
        pass

    # 3. System fallback
    return AgentSkill.objects.filter(
        slug="meeting-summarizer", level="system", is_active=True,
    ).first()


def _format_duration_minutes(seconds) -> str:
    if not seconds:
        return "unknown"
    minutes = max(1, int(seconds) // 60)
    return f"~{minutes} minutes"


def _build_seed_message(
    meeting,
    canvas_title: str,
    model_label: str,
    skill_name: str = "Meeting Summarizer",
    attachment_count: int = 0,
) -> str:
    transcript = meeting.transcript or ""
    n_chars = len(transcript)
    parts: list[str] = []
    parts.append(
        f"The user opened this thread to create meeting minutes (or a summary) "
        f"for **{meeting.name}**. A transcript of the meeting "
        f"({n_chars} characters, {_format_duration_minutes(meeting.duration_seconds)} "
        f'of audio, transcribed via `{model_label}`) is preloaded in the canvas titled "{canvas_title}".'
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
    parts.append(
        "Your job is to produce well-structured meeting minutes (or a summary) "
        f"and save them with `save_meeting_minutes` (the meeting_id is `{meeting.uuid}`). "
        "If important context is missing — attendees, meeting purpose, the boundary "
        "between decisions and action items — greet the user briefly and ask one "
        f"focused question before drafting. Use the attached {skill_name} skill "
        "to complete the task."
    )
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
    from django.core.files.base import ContentFile
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
            with ma.file.open("rb") as fh:
                data = fh.read()
            att = ChatAttachment.objects.create(
                thread=thread,
                message=None,
                uploaded_by=user,
                file=ContentFile(data, name=ma.original_filename),
                original_filename=(ma.original_filename or "")[:255],
                content_type=ct,
                size_bytes=len(data),
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


def create_minutes_thread(user, meeting, summarizer_skill=None):
    """Create a ChatThread pre-loaded with a summarizer skill.

    *summarizer_skill* overrides the cascade resolution when provided.
    Returns ``(thread, error_message)``: exactly one is non-None.
    """
    from chat.models import ChatCanvas, ChatMessage, ChatThread
    from chat.services import (
        CANVAS_MAX_CHARS,
        create_canvas_checkpoint,
        set_active_canvas,
    )

    if summarizer_skill is None:
        summarizer_skill = resolve_summarizer_skill(user, meeting)
    if summarizer_skill is None:
        logger.error(
            "create_minutes_thread: No eligible summarizer skill found "
            "(did the post_migrate seed run?)"
        )
        return None, "Meeting summarization is unavailable right now (skill missing)."

    if not (meeting.transcript or "").strip():
        return None, "This meeting has no transcript yet."

    thread = ChatThread.objects.create(
        created_by=user,
        skill=summarizer_skill,
        title=f"Minutes for {meeting.name}",
        metadata={
            "source_meeting_id": str(meeting.uuid),
            "pending_initial_turn": True,
        },
    )

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
        skill_name=summarizer_skill.name,
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
    if accepted or skipped:
        disclaimer = _build_attachments_disclaimer(len(accepted), skipped)
        disclaimer_msg = ChatMessage.objects.create(
            thread=thread,
            role="user",
            content=disclaimer,
            is_hidden_from_user=False,
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
