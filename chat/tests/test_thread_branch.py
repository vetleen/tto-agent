"""Tests for "branch this chat" — forking a thread at a chosen message, and the
per-thread attachment-size cap added alongside it."""

import datetime
import json
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from agent_skills.models import AgentSkill
from chat.models import (
    ChatAttachment,
    ChatCanvas,
    ChatMessage,
    ChatThread,
    ChatThreadDataRoom,
    ChatThreadSkill,
    ThreadChunkUsage,
)
from documents.models import DataRoom

User = get_user_model()

_STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.InMemoryStorage"},
    "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
}


@override_settings(
    ALLOWED_HOSTS=["testserver"],
    LLM_ALLOWED_MODELS=["anthropic/claude-sonnet-4-5-20250929"],
    LLM_DEFAULT_MODEL="anthropic/claude-sonnet-4-5-20250929",
    STORAGES=_STORAGES,
)
class ThreadBranchTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="branch@example.com", password="pw")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.client.force_login(self.user)

        self.thread = ChatThread.objects.create(
            created_by=self.user, title="My chat", emoji="🌱", model="anthropic/x",
        )
        base = timezone.now() - datetime.timedelta(hours=1)

        def msg(role, content, minute, **kw):
            m = ChatMessage.objects.create(
                thread=self.thread, role=role, content=content, **kw
            )
            ts = base + datetime.timedelta(minutes=minute)
            ChatMessage.objects.filter(id=m.id).update(created_at=ts)
            m.refresh_from_db()
            return m

        self.m_q1 = msg("user", "Q1", 0)
        self.m_a1 = msg("assistant", "A1", 1)
        self.m_q2 = msg("user", "Q2", 2)
        self.m_tool = msg(
            "tool", "tool result", 3, is_hidden_from_user=True, tool_call_id="call_1"
        )
        self.m_a2 = msg("assistant", "A2", 4)  # assistant branch point
        self.m_q3 = msg("user", "Q3", 5)       # user branch point
        self.m_a3 = msg("assistant", "A3", 6)

    def _branch(self, message_id, *, expect=200):
        url = reverse("thread_branch", args=[self.thread.id])
        resp = self.client.post(
            url, data=json.dumps({"message_id": str(message_id)}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, expect)
        return resp

    def _new_thread(self, resp):
        return ChatThread.objects.get(id=resp.json()["thread_id"])

    # -- core copy semantics -------------------------------------------------

    def test_branch_on_assistant_copies_inclusive(self):
        new = self._new_thread(self._branch(self.m_a2.id))
        copied = list(new.messages.all())
        self.assertEqual(
            [(m.role, m.content) for m in copied],
            [("user", "Q1"), ("assistant", "A1"), ("user", "Q2"),
             ("tool", "tool result"), ("assistant", "A2")],
        )
        # Hidden/tool rows ride along intact.
        tool = new.messages.get(role="tool")
        self.assertTrue(tool.is_hidden_from_user)
        self.assertEqual(tool.tool_call_id, "call_1")
        # New rows, distinct ids, original timestamps preserved (ordering stable).
        self.assertNotEqual({m.id for m in copied}, {self.m_a2.id})
        self.assertEqual(new.messages.get(content="A2").created_at, self.m_a2.created_at)

    def test_branch_on_user_excludes_message_and_sets_draft(self):
        new = self._new_thread(self._branch(self.m_q3.id))
        self.assertEqual(
            [(m.role, m.content) for m in new.messages.all()],
            [("user", "Q1"), ("assistant", "A1"), ("user", "Q2"),
             ("tool", "tool result"), ("assistant", "A2")],
        )
        # The clicked user message is NOT copied; its text becomes the draft.
        self.assertFalse(new.messages.filter(content="Q3").exists())
        self.assertEqual(new.metadata.get("draft_input"), "Q3")

    def test_thread_fields_and_lineage(self):
        new = self._new_thread(self._branch(self.m_a2.id))
        self.assertEqual(new.title, "Branch: My chat")
        self.assertEqual(new.emoji, "🌱")
        self.assertEqual(new.model, "anthropic/x")
        self.assertEqual(new.metadata["branched_from"], {
            "thread_id": str(self.thread.id),
            "message_id": str(self.m_a2.id),
        })

    def test_canvas_and_summary_reset(self):
        ChatCanvas.objects.create(thread=self.thread, title="Doc")
        self.thread.summary = "an old summary"
        self.thread.summary_up_to_message_id = self.m_a1.id
        self.thread.save(update_fields=["summary", "summary_up_to_message_id"])

        new = self._new_thread(self._branch(self.m_a2.id))
        self.assertEqual(new.canvases.count(), 0)
        self.assertIsNone(new.active_canvas_id)
        self.assertEqual(new.summary, "")
        self.assertIsNone(new.summary_up_to_message_id)

    # -- related data --------------------------------------------------------

    def test_data_room_and_skill_links_copied(self):
        r1 = DataRoom.objects.create(name="R1", slug="r1", created_by=self.user)
        r2 = DataRoom.objects.create(name="R2", slug="r2", created_by=self.user)
        ChatThreadDataRoom.objects.create(thread=self.thread, data_room=r1)
        ChatThreadDataRoom.objects.create(thread=self.thread, data_room=r2)
        s1 = AgentSkill.objects.create(
            slug="s1", name="S1", instructions="i", description="d",
            level="user", created_by=self.user,
        )
        s2 = AgentSkill.objects.create(
            slug="s2", name="S2", instructions="i", description="d",
            level="user", created_by=self.user,
        )
        ChatThreadSkill.objects.create(thread=self.thread, skill=s1)
        ChatThreadSkill.objects.create(thread=self.thread, skill=s2)

        new = self._new_thread(self._branch(self.m_a2.id))
        self.assertEqual(
            set(new.thread_data_rooms.values_list("data_room_id", flat=True)),
            {r1.id, r2.id},
        )
        # Skill order preserved (id tie-break mirrors source attach order).
        self.assertEqual(
            list(new.thread_skills.order_by("attached_at", "id").values_list(
                "skill_id", flat=True)),
            [s1.id, s2.id],
        )

    def test_attachments_byte_copied_in_range_only(self):
        in_range = ChatAttachment.objects.create(
            thread=self.thread, message=self.m_q2, uploaded_by=self.user,
            file=SimpleUploadedFile("in.txt", b"in-range-bytes", content_type="text/plain"),
            original_filename="in.txt", content_type="text/plain", size_bytes=14,
            extracted_content="cached text",
        )
        # Out-of-range (m_a3 is after the cutoff) and an unlinked draft.
        ChatAttachment.objects.create(
            thread=self.thread, message=self.m_a3, uploaded_by=self.user,
            file=SimpleUploadedFile("out.txt", b"out", content_type="text/plain"),
            original_filename="out.txt", content_type="text/plain", size_bytes=3,
        )
        ChatAttachment.objects.create(
            thread=self.thread, message=None, uploaded_by=self.user,
            file=SimpleUploadedFile("draft.txt", b"draft", content_type="text/plain"),
            original_filename="draft.txt", content_type="text/plain", size_bytes=5,
        )

        new = self._new_thread(self._branch(self.m_a2.id))
        copies = list(ChatAttachment.objects.filter(thread=new))
        self.assertEqual(len(copies), 1)
        copy = copies[0]
        self.assertEqual(copy.original_filename, "in.txt")
        self.assertEqual(copy.size_bytes, 14)
        self.assertEqual(copy.extracted_content, "cached text")
        self.assertEqual(copy.message, new.messages.get(content="Q2"))
        # Distinct stored file, identical bytes.
        self.assertNotEqual(copy.file.name, in_range.file.name)
        with copy.file.open("rb") as fh:
            self.assertEqual(fh.read(), b"in-range-bytes")

    def test_chunk_usage_copied_up_to_cutoff(self):
        room = DataRoom.objects.create(name="R", slug="r", created_by=self.user)
        from documents.tests._helpers import make_document

        doc = make_document(room, self.user, chunks=["x"])
        base = timezone.now() - datetime.timedelta(hours=1)

        def usage(minute):
            u = ThreadChunkUsage.objects.create(thread=self.thread, document=doc)
            ts = base + datetime.timedelta(minutes=minute)
            ThreadChunkUsage.objects.filter(id=u.id).update(created_at=ts)

        usage(2)   # before assistant cutoff (minute 4)
        usage(6)   # after cutoff

        new = self._new_thread(self._branch(self.m_a2.id))
        copied = list(new.chunk_usages.all())
        self.assertEqual(len(copied), 1)
        self.assertEqual(copied[0].document_id, doc.id)

    # -- guards --------------------------------------------------------------

    def test_missing_message_id_returns_400(self):
        url = reverse("thread_branch", args=[self.thread.id])
        resp = self.client.post(url, data=json.dumps({}), content_type="application/json")
        self.assertEqual(resp.status_code, 400)

    def test_unknown_message_id_returns_404(self):
        import uuid

        self._branch(uuid.uuid4(), expect=404)

    def test_message_from_other_thread_returns_404(self):
        other = ChatThread.objects.create(created_by=self.user)
        foreign = ChatMessage.objects.create(thread=other, role="user", content="x")
        self._branch(foreign.id, expect=404)

    def test_other_users_thread_returns_404(self):
        intruder = User.objects.create_user(email="intruder@example.com", password="pw")
        intruder.email_verified = True
        intruder.save(update_fields=["email_verified"])
        self.client.force_login(intruder)
        self._branch(self.m_a2.id, expect=404)

    def test_draft_input_is_one_shot(self):
        """chat_home pops draft_input so a reload starts with a clean composer."""
        new = self._new_thread(self._branch(self.m_q3.id))
        resp = self.client.get(reverse("chat_home"), {"thread": str(new.id)})
        self.assertEqual(resp.context["draft_input"], "Q3")
        new.refresh_from_db()
        self.assertNotIn("draft_input", new.metadata)
        # Second load no longer seeds it.
        resp2 = self.client.get(reverse("chat_home"), {"thread": str(new.id)})
        self.assertEqual(resp2.context["draft_input"], "")


@override_settings(
    ALLOWED_HOSTS=["testserver"],
    LLM_ALLOWED_MODELS=["anthropic/claude-sonnet-4-5-20250929"],
    LLM_DEFAULT_MODEL="anthropic/claude-sonnet-4-5-20250929",
    STORAGES=_STORAGES,
)
class ThreadAttachmentCapTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="cap@example.com", password="pw")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.client.force_login(self.user)
        self.thread = ChatThread.objects.create(created_by=self.user)

    @patch("chat.services.MAX_THREAD_ATTACHMENT_BYTES", 50)
    def test_upload_over_thread_cap_rejected(self):
        url = reverse("chat_upload_attachments", args=[self.thread.id])
        f = SimpleUploadedFile("a.txt", b"x" * 100, content_type="text/plain")
        resp = self.client.post(url, {"files": f})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("attachment limit", resp.json()["error"])
        self.assertEqual(ChatAttachment.objects.filter(thread=self.thread).count(), 0)

    @patch("chat.services.MAX_THREAD_ATTACHMENT_BYTES", 1_000_000)
    def test_upload_under_thread_cap_ok(self):
        url = reverse("chat_upload_attachments", args=[self.thread.id])
        f = SimpleUploadedFile("a.txt", b"hello", content_type="text/plain")
        resp = self.client.post(url, {"files": f})
        self.assertEqual(resp.status_code, 200)

    @patch("chat.services.MAX_THREAD_ATTACHMENT_BYTES", 50)
    def test_reattach_over_thread_cap_rejected(self):
        msg = ChatMessage.objects.create(thread=self.thread, role="user", content="m")
        att = ChatAttachment.objects.create(
            thread=self.thread, message=msg, uploaded_by=self.user,
            file=SimpleUploadedFile("a.txt", b"x" * 100, content_type="text/plain"),
            original_filename="a.txt", content_type="text/plain", size_bytes=100,
        )
        url = reverse("chat_reattach_attachment", args=[self.thread.id, att.id])
        resp = self.client.post(url)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("attachment limit", resp.json()["error"])

    @patch("chat.services.MAX_THREAD_ATTACHMENT_BYTES", 50)
    def test_branch_is_exempt_from_cap(self):
        """A grandfathered over-cap thread is still branchable; files copy over."""
        msg = ChatMessage.objects.create(thread=self.thread, role="assistant", content="a")
        ChatAttachment.objects.create(
            thread=self.thread, message=msg, uploaded_by=self.user,
            file=SimpleUploadedFile("big.txt", b"x" * 5000, content_type="text/plain"),
            original_filename="big.txt", content_type="text/plain", size_bytes=5000,
        )
        url = reverse("thread_branch", args=[self.thread.id])
        resp = self.client.post(
            url, data=json.dumps({"message_id": str(msg.id)}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        new = ChatThread.objects.get(id=resp.json()["thread_id"])
        self.assertEqual(ChatAttachment.objects.filter(thread=new).count(), 1)
