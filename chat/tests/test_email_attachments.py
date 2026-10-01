"""Emails attached in chat (and copied from meetings) split into the email plus
one attachment per file it carries (chat/email_attachments.py)."""
import json
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.message import MIMEMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from chat.attachment_tools import AttachmentViewTool
from chat.models import ChatAttachment, ChatMessage, ChatThread
from llm.types.context import RunContext

User = get_user_model()

PDF_A = b"%PDF-1.4 first fake pdf"
PDF_B = b"%PDF-1.4 second fake pdf"
EML = "message/rfc822"

_STORAGE = override_settings(
    ALLOWED_HOSTS=["testserver"],
    LLM_ALLOWED_MODELS=["anthropic/claude-sonnet-4-5-20250929"],
    LLM_DEFAULT_MODEL="anthropic/claude-sonnet-4-5-20250929",
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.InMemoryStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    },
)


def _png():
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _eml(files=(), *, subject="Deal docs", extra_parts=()) -> bytes:
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = "a@example.com"
    msg["To"] = "b@example.com"
    msg.attach(MIMEText("Files attached.", "plain"))
    for name, data in files:
        part = MIMEApplication(data, Name=name)
        part.add_header("Content-Disposition", "attachment", filename=name)
        msg.attach(part)
    for part in extra_parts:
        msg.attach(part)
    return msg.as_bytes()


def _photo_part():
    photo = MIMEImage(_png(), "png")
    photo.add_header("Content-Disposition", "attachment", filename="photo.png")
    return photo


@_STORAGE
class ChatEmailUploadTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="mail@example.com", password="pw")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.client.force_login(self.user)
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.url = reverse("chat_upload_attachments", args=[self.thread.id])
        delay = patch("chat.tasks.process_chat_attachment.delay")
        self.delay = delay.start()
        self.addCleanup(delay.stop)

    def _upload(self, data, name="mail.eml", content_type=EML):
        with self.captureOnCommitCallbacks(execute=True):
            resp = self.client.post(self.url, {"files": SimpleUploadedFile(name, data, content_type=content_type)})
        self.assertEqual(resp.status_code, 200, resp.content)
        return resp.json()

    def test_email_splits_into_its_files(self):
        data = self._upload(_eml(
            [("a.pdf", PDF_A), ("b.pdf", PDF_B), ("sheet.xlsx", b"PK xlsx")],
            extra_parts=[_photo_part()],
        ))
        atts = data["attachments"]
        self.assertEqual([a["filename"] for a in atts], ["mail.eml", "a.pdf", "b.pdf"])
        email_id = atts[0]["id"]
        self.assertIsNone(atts[0]["parent_id"])
        self.assertEqual([a["parent_id"] for a in atts[1:]], [email_id, email_id])
        self.assertEqual([a["processing_state"] for a in atts], ["pending"] * 3)
        self.assertEqual(len(data["warnings"]), 1)
        self.assertIn("sheet.xlsx", data["warnings"][0])
        self.assertIn("spreadsheets can't be attached in chat", data["warnings"][0])

        email = ChatAttachment.objects.get(id=email_id)
        children = list(email.children.order_by("created_at"))
        self.assertEqual([c.email_ordinal for c in children], [1, 2])
        self.assertEqual([c.content_type for c in children], ["application/pdf"] * 2)
        with children[0].file.open("rb") as fh:
            self.assertEqual(fh.read(), PDF_A)
        self.assertEqual(self.delay.call_count, 3)

    def test_email_text_names_its_split_files(self):
        from chat.services import get_or_extract_attachment_text

        data = self._upload(_eml([("a.pdf", PDF_A), ("sheet.xlsx", b"PK xlsx")], extra_parts=[_photo_part()]))
        email = ChatAttachment.objects.get(id=data["attachments"][0]["id"])
        with email.file.open("rb") as fh:
            raw = fh.read()
        with patch("chat.services.resolve_vision_model", return_value="anthropic/claude-opus-4-8"), \
             patch("chat.services.describe_image", return_value="A red square"):
            text = get_or_extract_attachment_text(email, raw, user=self.user)

        self.assertIn("Files attached.", text)
        self.assertIn("a.pdf", text)
        self.assertIn("attached separately as #2", text)
        self.assertIn("sheet.xlsx", text)
        self.assertIn("not attached: spreadsheets", text)
        self.assertIn("photo.png", text)
        self.assertIn("A red square", text)
        self.assertIn("[[image:", text)

    def test_outlook_msg_reported_as_octet_stream_is_accepted(self):
        data = self._upload(b"not really an outlook file", name="note.msg", content_type="application/octet-stream")
        [att] = data["attachments"]
        self.assertEqual(att["content_type"], "application/vnd.ms-outlook")
        self.assertIn("could not be read", data["warnings"][0])

    def test_forwarded_email_splits_recursively(self):
        inner = MIMEMessage(MIMEMultipart())
        inner_msg = inner.get_payload(0)
        inner_msg["Subject"] = "Contract"
        inner_msg.attach(MIMEText("Inner body.", "plain"))
        pdf = MIMEApplication(PDF_A, Name="contract.pdf")
        pdf.add_header("Content-Disposition", "attachment", filename="contract.pdf")
        inner_msg.attach(pdf)
        atts = self._upload(_eml(extra_parts=[inner]))["attachments"]

        self.assertEqual([a["filename"] for a in atts], ["mail.eml", "Contract.eml", "contract.pdf"])
        self.assertEqual(atts[1]["parent_id"], atts[0]["id"])
        self.assertEqual(atts[2]["parent_id"], atts[1]["id"])

    def test_thread_byte_budget_limits_children(self):
        raw = _eml([("a.pdf", PDF_A)])
        with patch("chat.services.MAX_THREAD_ATTACHMENT_BYTES", len(raw) + 5):
            data = self._upload(raw)
        self.assertEqual(len(data["attachments"]), 1)
        self.assertIn("attachment limit", data["warnings"][0])

    def test_reattach_resplits_the_email(self):
        first = self._upload(_eml([("a.pdf", PDF_A)]))["attachments"]
        url = reverse("chat_reattach_attachment", args=[self.thread.id, first[0]["id"]])
        with self.captureOnCommitCallbacks(execute=True):
            resp = self.client.post(url)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertNotEqual(body["id"], first[0]["id"])
        self.assertEqual(body["processing_state"], "pending")
        self.assertEqual([c["filename"] for c in body["children"]], ["a.pdf"])
        self.assertEqual(body["children"][0]["parent_id"], body["id"])
        self.assertEqual(ChatAttachment.objects.get(id=body["id"]).extracted_content, "")

    def test_branch_keeps_children_with_their_email(self):
        atts = self._upload(_eml([("a.pdf", PDF_A)]))["attachments"]
        msg = ChatMessage.objects.create(thread=self.thread, role="user", content="see mail")
        ChatAttachment.objects.filter(thread=self.thread).update(message=msg)
        reply = ChatMessage.objects.create(thread=self.thread, role="assistant", content="ok")

        resp = self.client.post(
            reverse("thread_branch", args=[self.thread.id]),
            data=json.dumps({"message_id": str(reply.id)}), content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        new = ChatThread.objects.get(id=resp.json()["thread_id"])
        email = ChatAttachment.objects.get(thread=new, original_filename="mail.eml")
        child = ChatAttachment.objects.get(thread=new, original_filename="a.pdf")
        self.assertEqual(child.parent_id, email.id)
        self.assertEqual(child.email_ordinal, 1)
        self.assertNotEqual(str(email.id), atts[0]["id"])


@_STORAGE
class ChatEmailViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="mailview@example.com", password="pw")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.msg = ChatMessage.objects.create(thread=self.thread, role="user", content="here")

    def test_attachment_view_reads_email_text(self):
        from chat.email_attachments import split_email_attachment

        raw = _eml([("a.pdf", PDF_A)])
        email = ChatAttachment.objects.create(
            thread=self.thread, uploaded_by=self.user, message=self.msg,
            file=ContentFile(raw, name="mail.eml"), original_filename="mail.eml",
            content_type=EML, size_bytes=len(raw),
        )
        with patch("chat.tasks.process_chat_attachment.delay"):
            split_email_attachment(email, raw, byte_budget=10**9)

        ctx = RunContext.create(user_id=self.user.pk, conversation_id=str(self.thread.id))
        tool = AttachmentViewTool()
        tool.set_context(ctx)
        with patch("chat.services.resolve_vision_model", return_value=""):
            result = json.loads(tool.invoke({"attachment_number": 1}))
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["kind"], "email")
        self.assertEqual(result["representation"], "extracted")
        self.assertIn("attached separately as #2", json.dumps(result))


@_STORAGE
class MeetingEmailCopyTests(TestCase):
    def setUp(self):
        from agent_skills.models import AgentSkill
        from meetings.models import Meeting

        AgentSkill.objects.filter(slug="meeting-summarizer").delete()
        AgentSkill.objects.create(
            slug="meeting-summarizer", name="Meeting Summarizer", description="Test seed.",
            instructions="Test instructions.", level="system", tool_names=[],
        )
        self.user = User.objects.create_user(email="mt-mail@example.com", password="pw")
        self.meeting = Meeting.objects.create(
            name="Acme call", slug="acme-call-mail", created_by=self.user, transcript="Hello.",
        )

    def test_meeting_email_attachment_is_split(self):
        from meetings.models import MeetingAttachment
        from meetings.services.minutes import create_minutes_thread

        raw = _eml([("a.pdf", PDF_A)])
        MeetingAttachment.objects.create(
            meeting=self.meeting, uploaded_by=self.user, file=ContentFile(raw, name="mail.eml"),
            original_filename="mail.eml", content_type=EML, size_bytes=len(raw),
        )
        with patch("chat.tasks.process_chat_attachment.delay") as delay, \
             self.captureOnCommitCallbacks(execute=True):
            thread, err = create_minutes_thread(self.user, self.meeting)
        self.assertIsNone(err)
        email = ChatAttachment.objects.get(thread=thread, original_filename="mail.eml")
        child = ChatAttachment.objects.get(thread=thread, original_filename="a.pdf")
        self.assertEqual(child.parent_id, email.id)
        ids = email.message.metadata.get("attachment_ids", [])
        self.assertIn(str(email.id), ids)
        self.assertIn(str(child.id), ids)
        self.assertEqual(child.message_id, email.message_id)
        self.assertEqual(delay.call_count, 2)
