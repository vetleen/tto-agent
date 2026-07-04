"""Integration tests for the 'Create meeting minutes with Wilfred' flow."""
from __future__ import annotations

from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.core.files.storage import InMemoryStorage
from django.test import TestCase, override_settings
from django.urls import reverse

from agent_skills.models import AgentSkill
from chat.models import ChatAttachment, ChatCanvas, ChatMessage, ChatThread, ChatThreadSkill
from meetings.models import Meeting, MeetingAttachment
from meetings.services.minutes import (
    create_minutes_thread,
    get_eligible_summarizer_skills,
)

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

User = get_user_model()


def _thread_skill_ids(thread):
    """Attached skill ids for a thread, in attach order."""
    return list(
        ChatThreadSkill.objects.filter(thread=thread).values_list("skill_id", flat=True)
    )


def _seed_meeting_summarizer():
    return AgentSkill.objects.create(
        slug="meeting-summarizer",
        name="Meeting Summarizer",
        description="Test seed.",
        instructions="Test instructions.",
        level="system",
        tool_names=[],
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
        self.assertEqual(_thread_skill_ids(thread), [self.skill.id])
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

    def test_thread_title_truncated_for_long_meeting_name(self):
        self.meeting.name = "N" * 250
        self.meeting.save(update_fields=["name"])
        thread, err = create_minutes_thread(self.user, self.meeting)
        self.assertIsNone(err)
        self.assertLessEqual(len(thread.title), 255)

    def test_seed_notes_truncation_for_oversized_transcript(self):
        from chat.services import CANVAS_MAX_CHARS

        self.meeting.transcript = "word " * (CANVAS_MAX_CHARS // 2)  # well over the cap
        self.meeting.save(update_fields=["transcript"])
        thread, _ = create_minutes_thread(self.user, self.meeting)

        canvas = ChatCanvas.objects.get(thread=thread)
        self.assertEqual(len(canvas.content), CANVAS_MAX_CHARS)  # canvas is capped

        seed = ChatMessage.objects.get(thread=thread, is_hidden_from_user=True)
        # The seed must state truncation, not claim the full transcript is preloaded.
        self.assertIn("truncated", seed.content.lower())
        self.assertIn(f"{CANVAS_MAX_CHARS:,}", seed.content)

    def test_refuses_meeting_without_transcript(self):
        m = Meeting.objects.create(name="Empty", slug="empty-m", created_by=self.user)
        thread, err = create_minutes_thread(self.user, m)
        self.assertIsNone(thread)
        self.assertIsNotNone(err)

    def test_no_skill_creates_valid_thread_without_chatthreadskill(self):
        # "No skill" is now a valid outcome — a missing/disabled summarizer must
        # yield a working thread (drafted by the model itself), not an error.
        from chat.models import ChatThreadSkill

        AgentSkill.objects.filter(slug="meeting-summarizer").delete()
        thread, err = create_minutes_thread(self.user, self.meeting)
        self.assertIsNone(err)
        self.assertIsNotNone(thread)
        self.assertEqual(ChatThreadSkill.objects.filter(thread=thread).count(), 0)
        seed = ChatMessage.objects.get(thread=thread, is_hidden_from_user=True)
        self.assertNotIn("Use the attached", seed.content)


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

    def test_explicit_no_skill_persists_preference(self):
        from accounts.models import UserSettings

        self.client.post(
            reverse("meeting_create_minutes_thread", args=[self.meeting.uuid]),
            {"skill_id": "none", "skill_explicit": "1"},
        )
        prefs = UserSettings.objects.get(user=self.user).preferences
        self.assertIsNone(prefs["meetings"]["summarizer_skill_id"])

    def test_explicit_skill_choice_persists_id(self):
        from accounts.models import UserSettings
        from agent_skills.models import AgentSkill

        skill = AgentSkill.objects.get(slug="meeting-summarizer", level="system")
        self.client.post(
            reverse("meeting_create_minutes_thread", args=[self.meeting.uuid]),
            {"skill_id": str(skill.id), "skill_explicit": "1"},
        )
        prefs = UserSettings.objects.get(user=self.user).preferences
        self.assertEqual(prefs["meetings"]["summarizer_skill_id"], str(skill.id))

    def test_non_explicit_submit_does_not_persist(self):
        from accounts.models import UserSettings

        self.client.post(
            reverse("meeting_create_minutes_thread", args=[self.meeting.uuid]),
            {"skill_id": "none"},  # no skill_explicit flag
        )
        settings_row = UserSettings.objects.filter(user=self.user).first()
        prefs = (settings_row.preferences if settings_row else {}) or {}
        self.assertNotIn("summarizer_skill_id", prefs.get("meetings", {}))


class GetEligibleSummarizerSkillsTests(TestCase):
    def setUp(self):
        AgentSkill.objects.filter(slug="meeting-summarizer").delete()
        self.system_skill = _seed_meeting_summarizer()
        self.user = User.objects.create_user(email="elig@example.com", password="pw")

    def test_returns_system_skill(self):
        skills = get_eligible_summarizer_skills(self.user)
        ids = {s.id for s in skills}
        self.assertIn(self.system_skill.id, ids)

    def test_includes_all_accessible_skills(self):
        other = AgentSkill.objects.create(
            slug="other-skill", name="Other", instructions="x",
            level="system", tool_names=["some_other_tool"],
        )
        skills = get_eligible_summarizer_skills(self.user)
        ids = {s.id for s in skills}
        self.assertIn(other.id, ids)
        self.assertIn(self.system_skill.id, ids)

    def test_includes_user_skills(self):
        user_skill = AgentSkill.objects.create(
            slug="my-summarizer", name="My Summarizer", instructions="x",
            level="user", created_by=self.user, tool_names=[],
        )
        skills = get_eligible_summarizer_skills(self.user)
        ids = {s.id for s in skills}
        self.assertIn(user_skill.id, ids)

    def test_excludes_subagent_audience_skill(self):
        # A minutes thread is a MAIN thread — sub-agent specializations must not
        # be offered (nor attachable).
        sub = AgentSkill.objects.create(
            slug="sub-spec", name="Sub Spec", instructions="x",
            level="user", created_by=self.user, tool_names=[], audience="subagent",
        )
        ids = {s.id for s in get_eligible_summarizer_skills(self.user)}
        self.assertNotIn(sub.id, ids)


class DefaultSummarizerOrgGateTests(TestCase):
    def setUp(self):
        from accounts.models import Membership, Organization

        AgentSkill.objects.filter(slug="meeting-summarizer").delete()
        self.system_skill = _seed_meeting_summarizer()
        self.user = User.objects.create_user(email="orggate@example.com", password="pw")
        self.org = Organization.objects.create(name="Org", slug="org-gate")
        Membership.objects.create(user=self.user, org=self.org)

    def test_org_member_without_enablement_gets_no_default(self):
        from meetings.services.minutes import resolve_default_summarizer_skill

        # System seed skills are OFF by default for org members.
        self.assertIsNone(resolve_default_summarizer_skill(self.user))

    def test_org_member_with_enablement_gets_skill(self):
        from meetings.services.minutes import resolve_default_summarizer_skill

        self.org.preferences = {"skills": {"meeting-summarizer": {"enabled": True}}}
        self.org.save(update_fields=["preferences"])
        skill = resolve_default_summarizer_skill(self.user)
        self.assertEqual(skill.id, self.system_skill.id)


class ResolveDefaultSummarizerSkillTests(TestCase):
    def setUp(self):
        AgentSkill.objects.filter(slug="meeting-summarizer").delete()
        self.system_skill = _seed_meeting_summarizer()
        self.user = User.objects.create_user(email="resolve@example.com", password="pw")

    def test_returns_system_default_for_no_org_user(self):
        from meetings.services.minutes import resolve_default_summarizer_skill

        # No-org user: the gate leaves system skills accessible, so the default
        # is the meeting-summarizer.
        skill = resolve_default_summarizer_skill(self.user)
        self.assertEqual(skill.id, self.system_skill.id)

    def test_returns_none_when_missing(self):
        from meetings.services.minutes import resolve_default_summarizer_skill

        AgentSkill.objects.filter(slug="meeting-summarizer").delete()
        self.assertIsNone(resolve_default_summarizer_skill(self.user))

    def test_saved_preference_none_returns_no_skill(self):
        from meetings.services.minutes import (
            resolve_default_summarizer_skill,
            set_default_summarizer_preference,
        )

        set_default_summarizer_preference(self.user, None)  # explicit "no skill"
        self.assertIsNone(resolve_default_summarizer_skill(self.user))

    def test_saved_preference_specific_skill_resolves(self):
        from meetings.services.minutes import (
            resolve_default_summarizer_skill,
            set_default_summarizer_preference,
        )

        set_default_summarizer_preference(self.user, str(self.system_skill.id))
        skill = resolve_default_summarizer_skill(self.user)
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
            level="user", created_by=self.user, tool_names=[],
        )
        thread, err = create_minutes_thread(self.user, self.meeting, summarizer_skill=custom)
        self.assertIsNone(err)
        self.assertEqual(_thread_skill_ids(thread), [custom.id])

    def test_seed_message_uses_custom_skill_name(self):
        custom = AgentSkill.objects.create(
            slug="board-min", name="Board Minutes Drafter", instructions="x",
            level="user", created_by=self.user, tool_names=[],
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

    def test_posts_with_skill_id_uses_chosen_skill(self):
        custom = AgentSkill.objects.create(
            slug="custom", name="Custom", instructions="x",
            level="user", created_by=self.user, tool_names=[],
        )
        response = self.client.post(
            reverse("meeting_create_minutes_thread", args=[self.meeting.uuid]),
            {"skill_id": str(custom.id)},
        )
        self.assertEqual(response.status_code, 302)
        thread = ChatThread.objects.filter(created_by=self.user).first()
        self.assertEqual(_thread_skill_ids(thread), [custom.id])

    def test_posts_with_invalid_skill_id_uses_default(self):
        response = self.client.post(
            reverse("meeting_create_minutes_thread", args=[self.meeting.uuid]),
            {"skill_id": "00000000-0000-0000-0000-000000000000"},
        )
        self.assertEqual(response.status_code, 302)
        thread = ChatThread.objects.filter(created_by=self.user).first()
        self.assertEqual(_thread_skill_ids(thread), [self.system_skill.id])

    def test_posts_without_skill_id_uses_default(self):
        response = self.client.post(
            reverse("meeting_create_minutes_thread", args=[self.meeting.uuid]),
        )
        self.assertEqual(response.status_code, 302)
        thread = ChatThread.objects.filter(created_by=self.user).first()
        self.assertEqual(_thread_skill_ids(thread), [self.system_skill.id])


@override_settings(
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.InMemoryStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    },
)
class CreateMinutesThreadAttachmentTests(TestCase):
    def setUp(self):
        AgentSkill.objects.filter(slug="meeting-summarizer").delete()
        self.skill = _seed_meeting_summarizer()
        self.user = User.objects.create_user(email="att-mt@example.com", password="pw")
        self.meeting = Meeting.objects.create(
            name="Acme call",
            slug="acme-call-att",
            created_by=self.user,
            transcript="Speaker says hello. Speaker says goodbye.",
        )

    def _add_attachment(self, filename, data, content_type, size_bytes=None):
        return MeetingAttachment.objects.create(
            meeting=self.meeting,
            uploaded_by=self.user,
            file=ContentFile(data, name=filename),
            original_filename=filename,
            content_type=content_type,
            size_bytes=size_bytes if size_bytes is not None else len(data),
        )

    def test_no_attachments_no_extra_message(self):
        thread, err = create_minutes_thread(self.user, self.meeting)
        self.assertIsNone(err)
        msgs = list(ChatMessage.objects.filter(thread=thread).order_by("created_at"))
        self.assertEqual(len(msgs), 1)
        self.assertTrue(msgs[0].is_hidden_from_user)
        self.assertEqual(ChatAttachment.objects.filter(thread=thread).count(), 0)

    def test_supported_pdf_copied_and_linked(self):
        ma = self._add_attachment("slides.pdf", b"%PDF-1.4 fake body", "application/pdf")
        thread, err = create_minutes_thread(self.user, self.meeting)
        self.assertIsNone(err)
        atts = list(ChatAttachment.objects.filter(thread=thread))
        self.assertEqual(len(atts), 1)
        ca = atts[0]
        self.assertEqual(ca.original_filename, "slides.pdf")
        self.assertEqual(ca.content_type, "application/pdf")
        # Fresh storage path — chat copy is independent of the meeting file.
        self.assertNotEqual(ca.file.name, ma.file.name)
        self.assertTrue(ca.file.name.startswith("chat_attachments/"))
        # Linked to a visible user message.
        self.assertIsNotNone(ca.message_id)
        self.assertFalse(ca.message.is_hidden_from_user)
        self.assertEqual(ca.message.role, "user")
        # attachment_ids in metadata is what the consumer enriches from — without
        # it the file content never reaches the LLM.
        self.assertIn(str(ca.id), ca.message.metadata.get("attachment_ids", []))

    def test_pdf_attachment_embedded_images_persist_via_chat_path(self):
        """A meeting PDF with an embedded image gets the same persistent
        message-scoped Asset treatment as a chat attachment (meetings
        inherits the chat enrichment path)."""
        import io
        from unittest.mock import patch

        from PIL import Image

        from chat.models import Asset
        from chat.services import get_or_extract_attachment_text

        buf = io.BytesIO()
        Image.new("RGB", (120, 80), (10, 80, 160)).save(buf, format="PDF")
        self._add_attachment("deck.pdf", buf.getvalue(), "application/pdf")

        thread, err = create_minutes_thread(self.user, self.meeting)
        self.assertIsNone(err)
        att = ChatAttachment.objects.get(thread=thread)

        att.file.open("rb")
        try:
            data = att.file.read()
        finally:
            att.file.close()
        with patch("chat.services.describe_image", return_value="A blue rectangle"):
            text = get_or_extract_attachment_text(att, data, user=self.user)

        assets = list(Asset.objects.filter(message=att.message))
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].description, "A blue rectangle")
        self.assertIn(f"[[image:{assets[0].id}|", text)

    def test_supported_docx_with_octet_stream_normalized(self):
        self._add_attachment("notes.docx", b"PK fake docx bytes", "application/octet-stream")
        thread, _ = create_minutes_thread(self.user, self.meeting)
        atts = list(ChatAttachment.objects.filter(thread=thread))
        self.assertEqual(len(atts), 1)
        self.assertEqual(atts[0].content_type, _DOCX_MIME)

    def test_unsupported_type_skipped_in_disclaimer(self):
        self._add_attachment("archive.zip", b"PK zip bytes", "application/zip")
        thread, _ = create_minutes_thread(self.user, self.meeting)
        self.assertEqual(ChatAttachment.objects.filter(thread=thread).count(), 0)
        visible = ChatMessage.objects.filter(
            thread=thread, is_hidden_from_user=False,
        ).first()
        self.assertIsNotNone(visible)
        self.assertIn("archive.zip", visible.content)
        self.assertIn("unsupported file type", visible.content)

    def test_oversized_pdf_skipped(self):
        # Lie about size without allocating 31 MB in memory — validation uses
        # MeetingAttachment.size_bytes.
        self._add_attachment(
            "huge.pdf", b"%PDF-1.4 stub", "application/pdf",
            size_bytes=31 * 1024 * 1024,
        )
        thread, _ = create_minutes_thread(self.user, self.meeting)
        self.assertEqual(ChatAttachment.objects.filter(thread=thread).count(), 0)
        visible = ChatMessage.objects.filter(
            thread=thread, is_hidden_from_user=False,
        ).first()
        self.assertIsNotNone(visible)
        self.assertIn("huge.pdf", visible.content)
        self.assertIn("too large", visible.content)

    def test_mixed_accepted_and_skipped(self):
        self._add_attachment("slides.pdf", b"%PDF-1.4 body", "application/pdf")
        self._add_attachment("archive.zip", b"PK zip", "application/zip")
        thread, _ = create_minutes_thread(self.user, self.meeting)
        # One accepted, one skipped.
        self.assertEqual(ChatAttachment.objects.filter(thread=thread).count(), 1)
        visible = ChatMessage.objects.filter(
            thread=thread, is_hidden_from_user=False,
        ).first()
        self.assertIsNotNone(visible)
        self.assertIn("automatically included", visible.content)
        self.assertIn("archive.zip", visible.content)
        self.assertIn("unsupported file type", visible.content)
        # Hidden seed mentions the 1 accepted file.
        hidden = ChatMessage.objects.filter(
            thread=thread, is_hidden_from_user=True,
        ).first()
        self.assertIn("1 supporting file", hidden.content)

    def test_copy_failure_does_not_break_thread(self):
        self._add_attachment("slides.pdf", b"%PDF-1.4 body", "application/pdf")
        # Simulate storage.open raising — the helper should log and record the
        # file as skipped rather than propagating.
        with patch.object(InMemoryStorage, "open", side_effect=OSError("boom")):
            thread, err = create_minutes_thread(self.user, self.meeting)
        self.assertIsNone(err)
        self.assertIsNotNone(thread)
        self.assertEqual(ChatAttachment.objects.filter(thread=thread).count(), 0)
        visible = ChatMessage.objects.filter(
            thread=thread, is_hidden_from_user=False,
        ).first()
        self.assertIsNotNone(visible)
        self.assertIn("slides.pdf", visible.content)
        self.assertIn("copy failed", visible.content)

    def test_visible_message_ordered_after_hidden_seed(self):
        self._add_attachment("slides.pdf", b"%PDF-1.4 body", "application/pdf")
        thread, _ = create_minutes_thread(self.user, self.meeting)
        msgs = list(ChatMessage.objects.filter(thread=thread).order_by("created_at"))
        self.assertEqual(len(msgs), 2)
        self.assertTrue(msgs[0].is_hidden_from_user)
        self.assertFalse(msgs[1].is_hidden_from_user)

    def test_seed_mentions_attachment_count_only_when_accepted(self):
        thread_empty, _ = create_minutes_thread(self.user, self.meeting)
        seed_empty = ChatMessage.objects.filter(
            thread=thread_empty, is_hidden_from_user=True,
        ).first()
        self.assertNotIn("supporting file", seed_empty.content)

        self._add_attachment("a.pdf", b"%PDF-1.4", "application/pdf")
        self._add_attachment("b.pdf", b"%PDF-1.4", "application/pdf")
        thread_two, _ = create_minutes_thread(self.user, self.meeting)
        seed_two = ChatMessage.objects.filter(
            thread=thread_two, is_hidden_from_user=True,
        ).first()
        self.assertIn("2 supporting files", seed_two.content)
