"""Integration tests for the 'Create meeting minutes with Wilfred' flow."""
from __future__ import annotations

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from agent_skills.models import AgentSkill
from chat.models import ChatCanvas, ChatMessage, ChatThread
from meetings.models import Meeting
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
            description="Quarterly check-in with Acme.",
        )

    def test_creates_thread_with_skill_and_metadata(self):
        thread, err = create_minutes_thread(self.user, self.meeting)
        self.assertIsNone(err)
        self.assertIsNotNone(thread)
        self.assertEqual(thread.skill_id, self.skill.id)
        self.assertEqual(thread.metadata.get("source_meeting_id"), str(self.meeting.uuid))
        self.assertTrue(thread.metadata.get("pending_initial_turn"))

    def test_preloads_transcript_into_active_canvas(self):
        thread, _ = create_minutes_thread(self.user, self.meeting)
        canvases = list(ChatCanvas.objects.filter(thread=thread))
        self.assertEqual(len(canvases), 1)
        canvas = canvases[0]
        self.assertEqual(canvas.content, self.meeting.transcript)
        self.assertIn("Acme call", canvas.title)
        self.assertIsNotNone(canvas.accepted_checkpoint_id)
        thread.refresh_from_db()
        self.assertEqual(thread.active_canvas_id, canvas.id)

    def test_creates_hidden_seed_message(self):
        thread, _ = create_minutes_thread(self.user, self.meeting)
        seeds = list(ChatMessage.objects.filter(thread=thread, is_hidden_from_user=True))
        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0].role, "user")
        content = seeds[0].content
        self.assertIn("Acme call", content)
        self.assertIn("Discuss licensing terms", content)
        self.assertIn("Alice, Bob", content)
        self.assertIn("Quarterly check-in with Acme.", content)
        self.assertIn("Meeting Summarizer skill", content)
        self.assertNotIn("playbook", content.lower())

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
