"""Integration tests for the 'Create meeting minutes with Wilfred' flow."""
from __future__ import annotations

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from agent_skills.models import AgentSkill
from chat.models import ChatAttachment, ChatMessage, ChatThread, ChatThreadDataRoom
from documents.models import DataRoom
from meetings.models import Meeting, MeetingDataRoom
from meetings.services.minutes import create_minutes_thread

User = get_user_model()


def _seed_meeting_summarizer():
    return AgentSkill.objects.create(
        slug="meeting-summarizer",
        name="Meeting Summarizer",
        description="Test seed.",
        instructions="Test instructions.",
        level="system",
        tool_names=["save_meeting_minutes"],
    )


class CreateMinutesThreadTests(TestCase):
    def setUp(self):
        AgentSkill.objects.filter(slug="meeting-summarizer").delete()
        self.skill = _seed_meeting_summarizer()
        self.user = User.objects.create_user(email="cmt@example.com", password="pw")
        self.meeting = Meeting.objects.create(
            name="Acme call",
            slug="acme-call-cmt",
            created_by=self.user,
            transcript="Speaker says hello. Speaker says goodbye.",
            transcription_model="openai/gpt-4o-mini-transcribe",
            agenda="Discuss licensing terms.",
            participants="Alice, Bob",
        )

    def test_creates_thread_with_skill_and_metadata(self):
        thread, err = create_minutes_thread(self.user, self.meeting)
        self.assertIsNone(err)
        self.assertIsNotNone(thread)
        self.assertEqual(thread.skill_id, self.skill.id)
        self.assertEqual(thread.metadata.get("source_meeting_id"), str(self.meeting.uuid))
        self.assertTrue(thread.metadata.get("pending_initial_turn"))

    def test_attaches_transcript_as_chat_attachment(self):
        thread, _ = create_minutes_thread(self.user, self.meeting)
        attachments = list(ChatAttachment.objects.filter(thread=thread))
        self.assertEqual(len(attachments), 1)
        att = attachments[0]
        self.assertIsNone(att.message_id)
        self.assertIn("transcript-", att.original_filename)
        self.assertEqual(att.size_bytes, len(self.meeting.transcript.encode("utf-8")))

    def test_creates_hidden_seed_message(self):
        thread, _ = create_minutes_thread(self.user, self.meeting)
        seeds = list(ChatMessage.objects.filter(thread=thread, is_hidden_from_user=True))
        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0].role, "user")
        self.assertIn("Acme call", seeds[0].content)
        self.assertIn("Discuss licensing terms", seeds[0].content)
        self.assertIn("Alice, Bob", seeds[0].content)

    def test_propagates_linked_data_rooms(self):
        room = DataRoom.objects.create(name="R", slug="r-cmt", created_by=self.user)
        MeetingDataRoom.objects.create(meeting=self.meeting, data_room=room)
        thread, _ = create_minutes_thread(self.user, self.meeting)
        self.assertTrue(ChatThreadDataRoom.objects.filter(thread=thread, data_room=room).exists())

    def test_refuses_meeting_without_transcript(self):
        m = Meeting.objects.create(name="Empty", slug="empty-m", created_by=self.user)
        thread, err = create_minutes_thread(self.user, m)
        self.assertIsNone(thread)
        self.assertIsNotNone(err)

    def test_returns_error_when_skill_missing(self):
        AgentSkill.objects.filter(slug="meeting-summarizer").delete()
        thread, err = create_minutes_thread(self.user, self.meeting)
        self.assertIsNone(thread)
        self.assertIsNotNone(err)


@override_settings(ALLOWED_HOSTS=["testserver"])
class MeetingCreateMinutesViewTests(TestCase):
    def setUp(self):
        AgentSkill.objects.filter(slug="meeting-summarizer").delete()
        _seed_meeting_summarizer()
        self.user = User.objects.create_user(email="mcv@example.com", password="pw")
        self.meeting = Meeting.objects.create(
            name="Call",
            slug="call-mcv",
            created_by=self.user,
            transcript="hi.",
        )
        self.client.force_login(self.user)

    def test_redirects_to_chat_with_thread_param(self):
        response = self.client.post(
            reverse("meeting_create_minutes_thread", args=[self.meeting.uuid])
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("/chat/", response["Location"])
        self.assertIn("thread=", response["Location"])
        self.assertEqual(ChatThread.objects.filter(created_by=self.user).count(), 1)

    def test_refuses_when_no_transcript(self):
        m = Meeting.objects.create(name="Empty", slug="empty-mcv", created_by=self.user)
        response = self.client.post(reverse("meeting_create_minutes_thread", args=[m.uuid]))
        self.assertEqual(response.status_code, 302)
        self.assertEqual(ChatThread.objects.filter(created_by=self.user).count(), 0)
