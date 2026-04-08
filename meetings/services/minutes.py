"""Service that prepares a fresh chat thread for the "Create meeting minutes
with Wilfred" flow.

Mirrors ``agent_skills/views.py:476-544`` (skills_edit_in_chat). The chat
consumer's existing ``pending_initial_turn`` machinery handles the auto-fire
of the first assistant turn — there is no consumer change required.
"""
from __future__ import annotations

import logging

from django.core.files.base import ContentFile

logger = logging.getLogger(__name__)


def _format_duration_minutes(seconds) -> str:
    if not seconds:
        return "unknown"
    minutes = max(1, int(seconds) // 60)
    return f"~{minutes} minutes"


def _build_seed_message(meeting, transcript_filename: str, model_label: str) -> str:
    transcript = meeting.transcript or ""
    n_chars = len(transcript)
    parts: list[str] = []
    parts.append(
        f"The user opened this thread to create meeting minutes for "
        f"**{meeting.name}**. The transcript is attached to this thread as "
        f"`{transcript_filename}` ({n_chars} characters, "
        f"{_format_duration_minutes(meeting.duration_seconds)} of audio, "
        f"transcribed via `{model_label}`)."
    )
    if meeting.agenda and meeting.agenda.strip():
        parts.append(f"The agenda was: {meeting.agenda.strip()}.")
    if meeting.participants and meeting.participants.strip():
        parts.append(f"Participants: {meeting.participants.strip()}.")
    linked_rooms = list(meeting.data_rooms.all().values_list("name", flat=True))
    if linked_rooms:
        parts.append(
            "Linked data rooms (search them with the standard document tools): "
            + ", ".join(linked_rooms)
            + "."
        )
    parts.append(
        "Your job is to produce well-structured meeting minutes and save them "
        f"with `save_meeting_minutes` (the meeting_id is `{meeting.uuid}`). "
        "If anything important is missing — attendees, meeting purpose, the "
        "boundary between decisions and action items — greet the user briefly "
        "and ask one focused question before drafting. If the transcript is "
        "rich enough to draft directly, do so: open a canvas titled "
        f"\"Meeting minutes — {meeting.name}\", write the minutes there, then "
        "call `save_meeting_minutes` with `kind=\"minutes\"`. Use the Meeting "
        "Summarizer playbook."
    )
    return " ".join(parts)


def create_minutes_thread(user, meeting):
    """Create a ChatThread pre-loaded with the Meeting Summarizer skill.

    Returns ``(thread, error_message)``: exactly one is non-None.
    """
    from agent_skills.models import AgentSkill
    from chat.models import ChatAttachment, ChatMessage, ChatThread, ChatThreadDataRoom

    summarizer = (
        AgentSkill.objects
        .filter(slug="meeting-summarizer", level="system", is_active=True)
        .first()
    )
    if summarizer is None:
        logger.error(
            "create_minutes_thread: Meeting Summarizer system skill not found "
            "(did the post_migrate seed run?)"
        )
        return None, "Meeting summarization is unavailable right now (skill missing)."

    if not (meeting.transcript or "").strip():
        return None, "This meeting has no transcript yet."

    thread = ChatThread.objects.create(
        created_by=user,
        skill=summarizer,
        title=f"Minutes for {meeting.name}",
        metadata={
            "source_meeting_id": str(meeting.uuid),
            "pending_initial_turn": True,
        },
    )

    # Auto-link any data rooms the meeting has — gives Wilfred RAG access via
    # the standard document tools during the minutes session.
    for dr in meeting.data_rooms.all():
        ChatThreadDataRoom.objects.get_or_create(thread=thread, data_room=dr)

    # Attach the transcript as a ChatAttachment (programmatic, message=None).
    payload = (meeting.transcript or "").encode("utf-8")
    transcript_filename = f"transcript-{meeting.slug}.txt"
    ChatAttachment.objects.create(
        thread=thread,
        message=None,
        uploaded_by=user,
        file=ContentFile(payload, name=transcript_filename),
        original_filename=transcript_filename,
        content_type="text/plain",
        size_bytes=len(payload),
    )

    # Hidden seed user message that primes the LLM. The chat consumer's
    # `pending_initial_turn` flag triggers an auto-assistant turn on first
    # WS load.
    model_label = meeting.transcription_model or "uploaded text"
    seed_content = _build_seed_message(meeting, transcript_filename, model_label)
    ChatMessage.objects.create(
        thread=thread,
        role="user",
        content=seed_content,
        is_hidden_from_user=True,
    )

    logger.info(
        "create_minutes_thread: created thread %s for meeting %s (user=%s)",
        thread.id, meeting.uuid, user.id,
    )
    return thread, None
