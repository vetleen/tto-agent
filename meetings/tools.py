"""Chat tools exposed by the meetings app.

The ``SaveMeetingMinutesTool`` is registered on import. ``MeetingsConfig.ready()``
imports this module so the LLM can call the tool on startup.
"""
from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from llm.tools import ContextAwareTool, ReasonBaseModel, get_tool_registry

logger = logging.getLogger(__name__)


class SaveMeetingMinutesInput(ReasonBaseModel):
    meeting_id: str = Field(
        description="UUID of the meeting to save minutes to.",
    )
    content_md: str = Field(
        description="Markdown content of the meeting minutes (the artifact body).",
    )
    kind: str = Field(
        default="minutes",
        description="One of: minutes, summary, notes.",
    )
    title: str = Field(
        default="",
        description="Optional title for the artifact (e.g. 'Meeting minutes — Acme call').",
    )


class SaveMeetingMinutesTool(ContextAwareTool):
    """Persist a Markdown artifact (minutes/summary/notes) to a meeting."""

    name: str = "save_meeting_minutes"
    description: str = (
        "Persist meeting minutes (or a summary/notes) back to the originating meeting "
        "as a Markdown artifact. Use this once the user is satisfied with the draft."
    )
    args_schema: type[BaseModel] = SaveMeetingMinutesInput
    section: str = "meetings"

    def _run(self, meeting_id: str, content_md: str, kind: str = "minutes", title: str = "", **kwargs) -> str:
        from django.contrib.auth import get_user_model

        from chat.models import ChatThread

        from .models import Meeting, MeetingArtifact

        if not content_md or not content_md.strip():
            return json.dumps({"status": "error", "message": "content_md is empty."})

        kind_normalized = (kind or "minutes").strip().lower()
        valid_kinds = {choice.value for choice in MeetingArtifact.Kind}
        if kind_normalized not in valid_kinds:
            return json.dumps({
                "status": "error",
                "message": f"Invalid kind '{kind}'. Must be one of: {sorted(valid_kinds)}.",
            })

        user_id = self.context.user_id if self.context else None
        thread_id = self.context.conversation_id if self.context else None
        if not user_id or not thread_id:
            return json.dumps({"status": "error", "message": "No context available."})

        User = get_user_model()
        try:
            user = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return json.dumps({"status": "error", "message": "User not found."})

        thread = ChatThread.objects.filter(pk=thread_id, created_by=user).first()
        # Prefer the meeting recorded in thread.metadata to prevent the LLM from
        # writing minutes to the wrong meeting. Fall back to the LLM-supplied id.
        source_meeting_id = (thread.metadata or {}).get("source_meeting_id") if thread else None
        effective_meeting_id = source_meeting_id or meeting_id
        if source_meeting_id and meeting_id and str(source_meeting_id) != str(meeting_id):
            logger.warning(
                "save_meeting_minutes: LLM-supplied meeting_id %s differs from "
                "thread.metadata.source_meeting_id %s; using metadata value",
                meeting_id,
                source_meeting_id,
            )

        try:
            meeting = Meeting.objects.get(uuid=effective_meeting_id)
        except (Meeting.DoesNotExist, ValueError):
            return json.dumps({"status": "error", "message": "Meeting not found."})

        if meeting.created_by_id != user.id:
            return json.dumps({"status": "error", "message": "You do not have access to this meeting."})

        artifact = MeetingArtifact.objects.create(
            meeting=meeting,
            kind=kind_normalized,
            title=title or f"Meeting {kind_normalized} — {meeting.name}",
            content_md=content_md,
            created_by=user,
            source_thread=thread,
        )

        # Touch the meeting so it floats to the top of the list view.
        meeting.save(update_fields=["updated_at"])

        logger.info(
            "save_meeting_minutes: created artifact %s (kind=%s) for meeting %s by user %s",
            artifact.id,
            kind_normalized,
            meeting.uuid,
            user.id,
        )
        return json.dumps({
            "status": "ok",
            "artifact_id": str(artifact.id),
            "meeting_id": str(meeting.uuid),
            "kind": kind_normalized,
        })


# Register on import
_registry = get_tool_registry()
_registry.register_tool(SaveMeetingMinutesTool())
