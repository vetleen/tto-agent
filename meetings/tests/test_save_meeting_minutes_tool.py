"""Tests for the SaveMeetingMinutesTool."""
from __future__ import annotations

import json

from django.contrib.auth import get_user_model
from django.test import TestCase

from chat.models import ChatThread
from llm.types.context import RunContext
from meetings.models import Meeting, MeetingArtifact
from meetings.tools import SaveMeetingMinutesTool

User = get_user_model()


class SaveMeetingMinutesToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="tool@example.com", password="pw")
        self.other = User.objects.create_user(email="tool-other@example.com", password="pw")
        self.meeting = Meeting.objects.create(
            name="M", slug="m-tool", created_by=self.user, transcript="content",
        )
        self.thread = ChatThread.objects.create(
            created_by=self.user,
            metadata={"source_meeting_id": str(self.meeting.uuid)},
        )

    def _run_tool(self, **overrides):
        ctx = RunContext.create(
            user_id=self.user.id,
            conversation_id=str(self.thread.id),
        )
        tool = SaveMeetingMinutesTool().set_context(ctx)
        kwargs = {
            "meeting_id": str(self.meeting.uuid),
            "content_md": "# Minutes\n\n- Decision A",
            "kind": "minutes",
        }
        kwargs.update(overrides)
        return json.loads(tool._run(**kwargs))

    def test_creates_artifact_for_owner(self):
        result = self._run_tool()
        self.assertEqual(result["status"], "ok")
        artifact = MeetingArtifact.objects.get(pk=result["artifact_id"])
        self.assertEqual(artifact.meeting, self.meeting)
        self.assertEqual(artifact.created_by, self.user)
        self.assertEqual(artifact.source_thread, self.thread)
        self.assertEqual(artifact.kind, MeetingArtifact.Kind.MINUTES)
        self.assertIn("Decision A", artifact.content_md)

    def test_rejects_empty_content(self):
        result = self._run_tool(content_md="   ")
        self.assertEqual(result["status"], "error")

    def test_rejects_invalid_kind(self):
        result = self._run_tool(kind="garbage")
        self.assertEqual(result["status"], "error")
        self.assertIn("Invalid kind", result["message"])

    def test_kind_summary_creates_summary_artifact(self):
        result = self._run_tool(kind="summary")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(MeetingArtifact.objects.get(pk=result["artifact_id"]).kind, "summary")

    def test_thread_metadata_overrides_llm_meeting_id(self):
        # The LLM passes a wrong meeting_id; the metadata should win.
        other_meeting = Meeting.objects.create(
            name="Wrong", slug="wrong-meeting-tool", created_by=self.user, transcript="x",
        )
        result = self._run_tool(meeting_id=str(other_meeting.uuid))
        self.assertEqual(result["status"], "ok")
        artifact = MeetingArtifact.objects.get(pk=result["artifact_id"])
        self.assertEqual(artifact.meeting, self.meeting)

    def test_rejects_non_owner(self):
        thread = ChatThread.objects.create(
            created_by=self.other,
            metadata={"source_meeting_id": str(self.meeting.uuid)},
        )
        ctx = RunContext.create(user_id=self.other.id, conversation_id=str(thread.id))
        tool = SaveMeetingMinutesTool().set_context(ctx)
        result = json.loads(tool._run(
            meeting_id=str(self.meeting.uuid),
            content_md="# m",
            kind="minutes",
        ))
        self.assertEqual(result["status"], "error")
        self.assertIn("access", result["message"].lower())

    def test_rejects_missing_context(self):
        tool = SaveMeetingMinutesTool()  # no .set_context()
        result = json.loads(tool._run(
            meeting_id=str(self.meeting.uuid),
            content_md="# m",
            kind="minutes",
        ))
        self.assertEqual(result["status"], "error")
