"""Integration tests for the 'Create meeting minutes with Wilfred' flow."""
from __future__ import annotations

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from accounts.models import UserSettings
from agent_skills.models import AgentSkill
from chat.models import ChatCanvas, ChatMessage, ChatThread
from meetings.models import Meeting
from meetings.services.minutes import (
    create_minutes_thread,
    get_eligible_summarizer_skills,
    resolve_summarizer_skill,
)

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


class GetEligibleSummarizerSkillsTests(TestCase):
    def setUp(self):
        AgentSkill.objects.filter(slug="meeting-summarizer").delete()
        self.system_skill = _seed_meeting_summarizer()
        self.user = User.objects.create_user(email="elig@example.com", password="pw")

    def test_returns_system_skill_with_save_meeting_minutes(self):
        skills = get_eligible_summarizer_skills(self.user)
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0].id, self.system_skill.id)

    def test_excludes_skills_without_save_meeting_minutes(self):
        AgentSkill.objects.create(
            slug="other-skill", name="Other", instructions="x",
            level="system", tool_names=["some_other_tool"],
        )
        skills = get_eligible_summarizer_skills(self.user)
        self.assertEqual(len(skills), 1)

    def test_includes_user_skill_with_save_meeting_minutes(self):
        user_skill = AgentSkill.objects.create(
            slug="my-summarizer", name="My Summarizer", instructions="x",
            level="user", created_by=self.user, tool_names=["save_meeting_minutes"],
        )
        skills = get_eligible_summarizer_skills(self.user)
        self.assertEqual(len(skills), 2)
        ids = {s.id for s in skills}
        self.assertIn(user_skill.id, ids)


class ResolveSummarizerSkillTests(TestCase):
    def setUp(self):
        AgentSkill.objects.filter(slug="meeting-summarizer").delete()
        self.system_skill = _seed_meeting_summarizer()
        self.user = User.objects.create_user(email="resolve@example.com", password="pw")
        self.meeting = Meeting.objects.create(
            name="Test", slug="test-resolve", created_by=self.user, transcript="hi",
        )

    def test_falls_back_to_system_default(self):
        skill = resolve_summarizer_skill(self.user, self.meeting)
        self.assertEqual(skill.id, self.system_skill.id)

    def test_per_user_default_overrides_system(self):
        user_skill = AgentSkill.objects.create(
            slug="user-sum", name="User Sum", instructions="x",
            level="user", created_by=self.user, tool_names=["save_meeting_minutes"],
        )
        us, _ = UserSettings.objects.get_or_create(user=self.user)
        us.preferences = {"meetings": {"summarizer_skill_id": str(user_skill.id)}}
        us.save()
        skill = resolve_summarizer_skill(self.user, self.meeting)
        self.assertEqual(skill.id, user_skill.id)

    def test_per_meeting_override_wins(self):
        user_skill = AgentSkill.objects.create(
            slug="mtg-sum", name="Mtg Sum", instructions="x",
            level="user", created_by=self.user, tool_names=["save_meeting_minutes"],
        )
        self.meeting.summarizer_skill = user_skill
        self.meeting.save()
        skill = resolve_summarizer_skill(self.user, self.meeting)
        self.assertEqual(skill.id, user_skill.id)

    def test_stale_per_meeting_override_falls_back(self):
        # Skill exists but no longer has save_meeting_minutes -> ineligible
        stale = AgentSkill.objects.create(
            slug="stale-sum", name="Stale", instructions="x",
            level="user", created_by=self.user, tool_names=["other_tool"],
        )
        self.meeting.summarizer_skill = stale
        self.meeting.save()
        skill = resolve_summarizer_skill(self.user, self.meeting)
        self.assertEqual(skill.id, self.system_skill.id)

    def test_stale_user_pref_falls_back(self):
        us, _ = UserSettings.objects.get_or_create(user=self.user)
        us.preferences = {"meetings": {"summarizer_skill_id": "00000000-0000-0000-0000-000000000000"}}
        us.save()
        skill = resolve_summarizer_skill(self.user, self.meeting)
        self.assertEqual(skill.id, self.system_skill.id)


class CreateMinutesThreadWithSkillTests(TestCase):
    def setUp(self):
        AgentSkill.objects.filter(slug="meeting-summarizer").delete()
        self.system_skill = _seed_meeting_summarizer()
        self.user = User.objects.create_user(email="cmts@example.com", password="pw")
        self.meeting = Meeting.objects.create(
            name="Call", slug="call-cmts", created_by=self.user,
            transcript="Hello world.",
        )

    def test_uses_custom_skill_when_provided(self):
        custom = AgentSkill.objects.create(
            slug="custom-sum", name="Custom Summarizer", instructions="x",
            level="user", created_by=self.user, tool_names=["save_meeting_minutes"],
        )
        thread, err = create_minutes_thread(self.user, self.meeting, summarizer_skill=custom)
        self.assertIsNone(err)
        self.assertEqual(thread.skill_id, custom.id)

    def test_seed_message_uses_custom_skill_name(self):
        custom = AgentSkill.objects.create(
            slug="board-min", name="Board Minutes Drafter", instructions="x",
            level="user", created_by=self.user, tool_names=["save_meeting_minutes"],
        )
        thread, _ = create_minutes_thread(self.user, self.meeting, summarizer_skill=custom)
        seed = ChatMessage.objects.filter(thread=thread, is_hidden_from_user=True).first()
        self.assertIn("Board Minutes Drafter skill", seed.content)
        self.assertNotIn("Meeting Summarizer skill", seed.content)


@override_settings(ALLOWED_HOSTS=["testserver"])
class MeetingCreateMinutesWithSkillViewTests(TestCase):
    def setUp(self):
        AgentSkill.objects.filter(slug="meeting-summarizer").delete()
        self.system_skill = _seed_meeting_summarizer()
        self.user = User.objects.create_user(email="mcvs@example.com", password="pw")
        self.meeting = Meeting.objects.create(
            name="Call", slug="call-mcvs", created_by=self.user, transcript="hi",
        )
        self.client.force_login(self.user)

    def test_posts_with_skill_id_saves_per_meeting_override(self):
        custom = AgentSkill.objects.create(
            slug="custom", name="Custom", instructions="x",
            level="user", created_by=self.user, tool_names=["save_meeting_minutes"],
        )
        response = self.client.post(
            reverse("meeting_create_minutes_thread", args=[self.meeting.uuid]),
            {"skill_id": str(custom.id)},
        )
        self.assertEqual(response.status_code, 302)
        self.meeting.refresh_from_db()
        self.assertEqual(self.meeting.summarizer_skill_id, custom.id)
        thread = ChatThread.objects.filter(created_by=self.user).first()
        self.assertEqual(thread.skill_id, custom.id)

    def test_posts_with_invalid_skill_id_uses_default(self):
        response = self.client.post(
            reverse("meeting_create_minutes_thread", args=[self.meeting.uuid]),
            {"skill_id": "00000000-0000-0000-0000-000000000000"},
        )
        self.assertEqual(response.status_code, 302)
        thread = ChatThread.objects.filter(created_by=self.user).first()
        self.assertEqual(thread.skill_id, self.system_skill.id)

    def test_posts_without_skill_id_uses_cascade(self):
        response = self.client.post(
            reverse("meeting_create_minutes_thread", args=[self.meeting.uuid]),
        )
        self.assertEqual(response.status_code, 302)
        thread = ChatThread.objects.filter(created_by=self.user).first()
        self.assertEqual(thread.skill_id, self.system_skill.id)

    def test_rejects_skill_without_save_meeting_minutes(self):
        ineligible = AgentSkill.objects.create(
            slug="ineligible", name="Ineligible", instructions="x",
            level="user", created_by=self.user, tool_names=["some_tool"],
        )
        response = self.client.post(
            reverse("meeting_create_minutes_thread", args=[self.meeting.uuid]),
            {"skill_id": str(ineligible.id)},
        )
        self.assertEqual(response.status_code, 302)
        # Should fall back to system default, not use the ineligible skill
        thread = ChatThread.objects.filter(created_by=self.user).first()
        self.assertEqual(thread.skill_id, self.system_skill.id)
