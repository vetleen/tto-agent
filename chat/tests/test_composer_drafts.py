"""Composer drafts that survive navigation: unsent files restored on load, the
pill's "x" discarding them, and the sidebar label of a thread with no title."""

from datetime import timedelta

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from chat.models import ChatAttachment, ChatMessage, ChatThread

User = get_user_model()

_SETTINGS = dict(
    ALLOWED_HOSTS=["testserver"],
    LLM_ALLOWED_MODELS=["anthropic/claude-sonnet-4-5-20250929"],
    LLM_DEFAULT_MODEL="anthropic/claude-sonnet-4-5-20250929",
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.InMemoryStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    },
)


def _user(email):
    user = User.objects.create_user(email=email, password="testpass")
    user.email_verified = True
    user.save(update_fields=["email_verified"])
    return user


def _att(thread, user, name="a.txt", **kwargs):
    return ChatAttachment.objects.create(
        thread=thread,
        uploaded_by=user,
        file=SimpleUploadedFile(name, b"hello", content_type="text/plain"),
        original_filename=name,
        content_type="text/plain",
        size_bytes=5,
        **kwargs,
    )


def _age(obj, seconds):
    """Move a row's auto_now_add created_at into the past."""
    type(obj).objects.filter(pk=obj.pk).update(created_at=timezone.now() - timedelta(seconds=seconds))


@override_settings(**_SETTINGS)
class ChatHomePendingAttachmentsTests(TestCase):
    def setUp(self):
        self.user = _user("drafts@example.com")
        self.client.force_login(self.user)
        self.thread = ChatThread.objects.create(created_by=self.user, title="Drafts")

    def _pending(self):
        resp = self.client.get(reverse("chat_home") + f"?thread={self.thread.id}")
        self.assertEqual(resp.status_code, 200)
        return resp.context["pending_attachments"]

    def test_unsent_attachment_on_new_thread_is_restored(self):
        att = _att(self.thread, self.user, "brief.txt")
        pending = self._pending()
        self.assertEqual([p["id"] for p in pending], [str(att.id)])
        self.assertEqual(pending[0]["filename"], "brief.txt")
        self.assertIsNone(pending[0]["parent_id"])

    def test_rendered_as_json_script(self):
        _att(self.thread, self.user, "brief.txt")
        resp = self.client.get(reverse("chat_home") + f"?thread={self.thread.id}")
        self.assertContains(resp, 'id="pending-attachments-data"')
        self.assertContains(resp, "brief.txt")

    def test_sent_attachment_is_not_restored(self):
        msg = ChatMessage.objects.create(thread=self.thread, role="user", content="hi")
        _att(self.thread, self.user, message=msg)
        self.assertEqual(self._pending(), [])

    def test_leftover_from_before_the_last_message_is_not_restored(self):
        old = _att(self.thread, self.user, "removed-pill.txt")
        _age(old, 120)
        msg = ChatMessage.objects.create(thread=self.thread, role="user", content="hi")
        _age(msg, 60)
        fresh = _att(self.thread, self.user, "fresh.txt")
        self.assertEqual([p["id"] for p in self._pending()], [str(fresh.id)])

    def test_other_users_upload_is_not_restored(self):
        _att(self.thread, _user("other@example.com"))
        self.assertEqual(self._pending(), [])

    def test_email_children_follow_their_parent(self):
        email = _att(self.thread, self.user, "mail.eml")
        child = _att(self.thread, self.user, "inside.pdf", parent=email, email_ordinal=1)
        pending = self._pending()
        self.assertEqual([p["id"] for p in pending], [str(email.id), str(child.id)])
        self.assertEqual(pending[1]["parent_id"], str(email.id))

    def test_new_chat_page_has_none(self):
        resp = self.client.get(reverse("chat_home"))
        self.assertEqual(resp.context["pending_attachments"], [])


@override_settings(**_SETTINGS)
class DiscardAttachmentTests(TestCase):
    def setUp(self):
        self.user = _user("discard@example.com")
        self.client.force_login(self.user)
        self.thread = ChatThread.objects.create(created_by=self.user)

    def _url(self, att):
        return reverse("chat_discard_attachment", args=[att.id])

    def test_owner_discards_unsent_row(self):
        att = _att(self.thread, self.user)
        resp = self.client.post(self._url(att))
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(ChatAttachment.objects.filter(id=att.id).exists())

    def test_email_descendants_go_with_it(self):
        email = _att(self.thread, self.user, "mail.eml")
        nested = _att(self.thread, self.user, "fwd.eml", parent=email, email_ordinal=1)
        leaf = _att(self.thread, self.user, "deep.pdf", parent=nested, email_ordinal=1)
        keep = _att(self.thread, self.user, "other.txt")
        self.client.post(self._url(email))
        remaining = set(ChatAttachment.objects.values_list("id", flat=True))
        self.assertEqual(remaining, {keep.id})
        self.assertNotIn(leaf.id, remaining)

    def test_sent_row_is_kept(self):
        msg = ChatMessage.objects.create(thread=self.thread, role="user", content="hi")
        att = _att(self.thread, self.user, message=msg)
        resp = self.client.post(self._url(att))
        self.assertEqual(resp.status_code, 404)
        self.assertTrue(ChatAttachment.objects.filter(id=att.id).exists())

    def test_other_user_gets_404(self):
        att = _att(self.thread, self.user)
        self.client.force_login(_user("intruder@example.com"))
        resp = self.client.post(self._url(att))
        self.assertEqual(resp.status_code, 404)
        self.assertTrue(ChatAttachment.objects.filter(id=att.id).exists())

    def test_get_not_allowed(self):
        att = _att(self.thread, self.user)
        self.assertEqual(self.client.get(self._url(att)).status_code, 405)

    def test_csrf_required(self):
        att = _att(self.thread, self.user)
        client = Client(enforce_csrf_checks=True)
        client.force_login(self.user)
        self.assertEqual(client.post(self._url(att)).status_code, 403)
        self.assertTrue(ChatAttachment.objects.filter(id=att.id).exists())


@override_settings(**_SETTINGS)
class SidebarUntitledThreadTests(TestCase):
    def test_untitled_thread_reads_new_chat(self):
        user = _user("sidebar@example.com")
        self.client.force_login(user)
        thread = ChatThread.objects.create(created_by=user)
        resp = self.client.get(reverse("chat_home"))
        self.assertContains(resp, 'title="New chat"')
        self.assertNotContains(resp, f"Thread {thread.id}")
