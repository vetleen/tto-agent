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


def _build_seed_message(meeting, canvas_title: str, model_label: str, skill_name: str = "Meeting Summarizer") -> str:
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
    parts.append(
        "Your job is to produce well-structured meeting minutes (or a summary) "
        f"and save them with `save_meeting_minutes` (the meeting_id is `{meeting.uuid}`). "
        "If important context is missing — attendees, meeting purpose, the boundary "
        "between decisions and action items — greet the user briefly and ask one "
        f"focused question before drafting. Use the attached {skill_name} skill "
        "to complete the task."
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

    # Hidden seed user message that primes the LLM. The chat consumer's
    # `pending_initial_turn` flag triggers an auto-assistant turn on first
    # WS load.
    model_label = meeting.transcription_model or "uploaded text"
    seed_content = _build_seed_message(meeting, canvas_title, model_label, skill_name=summarizer_skill.name)
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
