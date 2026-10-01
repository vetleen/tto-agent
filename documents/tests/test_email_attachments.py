"""Email attachments split into data-room documents of their own."""
import tempfile
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.message import MIMEMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.test import TestCase, override_settings

from core.files import sha256_of_bytes
from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentTag
from documents.services.email_attachments import (
    TAG_DEPTH,
    TAG_ORDINAL,
    TAG_PARENT_DOC,
    TAG_PARENT_VERSION,
    split_display_name,
)
from documents.services.email_split import _msg_attachment_bytes, parse_email

User = get_user_model()

PDF = b"%PDF-1.4 fake pdf bytes"
DOCX = b"PK fake docx"
PPTX = b"PK fake pptx"
XLSX = b"PK fake xlsx"


def _eml(attachments=(), *, subject="Deal docs", extra_parts=()):
    """Build an .eml; *attachments* = [(filename, bytes)]."""
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = "a@example.com"
    msg["To"] = "b@example.com"
    msg.attach(MIMEText("Please find the files attached.", "plain"))
    for name, data in attachments:
        part = MIMEApplication(data, Name=name)
        part.add_header("Content-Disposition", "attachment", filename=name)
        msg.attach(part)
    for part in extra_parts:
        msg.attach(part)
    return msg


def _tags(doc) -> dict:
    return dict(DataRoomDocumentTag.objects.filter(version__document=doc).values_list("key", "value"))


@override_settings(PGVECTOR_CONNECTION="")
class EmailAttachmentSplitTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="split@example.com", password="pw")
        self.room = DataRoom.objects.create(name="Room", slug="room", created_by=self.user)
        self._tmp = tempfile.TemporaryDirectory()
        self._media = self.settings(MEDIA_ROOT=self._tmp.name)
        self._media.enable()
        patcher = patch("guardrails.tasks.scan_document_version.delay")
        patcher.start()
        self.addCleanup(patcher.stop)
        dispatch = patch("documents.services.dispatch.safe_dispatch")
        self.mock_dispatch = dispatch.start()
        self.addCleanup(dispatch.stop)

    def tearDown(self):
        self._media.disable()
        self._tmp.cleanup()

    def _upload_email(self, msg, filename="mail.eml"):
        doc = DataRoomDocument(
            data_room=self.room, uploaded_by=self.user, original_filename=filename,
            mime_type="message/rfc822", status=DataRoomDocument.Status.UPLOADED,
        )
        doc.original_file.save(filename, ContentFile(msg.as_bytes()), save=True)
        return doc

    def _process(self, doc):
        from documents.services.process_document import process_document

        process_document(doc.id)
        doc.refresh_from_db()
        return doc

    def _text(self, doc) -> str:
        return "\n".join(doc.current_version.chunks.order_by("chunk_index").values_list("text", flat=True))

    def _children(self, parent):
        return DataRoomDocument.objects.filter(
            versions__tags__key=TAG_PARENT_DOC, versions__tags__value=str(parent.id),
        ).order_by("id")

    def test_supported_attachments_become_documents(self):
        parent = self._process(self._upload_email(_eml([
            ("report.pdf", PDF), ("memo.docx", DOCX), ("deck.pptx", PPTX), ("numbers.xlsx", XLSX),
        ])))

        children = list(self._children(parent))
        self.assertEqual(
            [c.original_filename for c in children], ["report.pdf", "memo.docx", "deck.pptx", "numbers.xlsx"],
        )
        for ordinal, child in enumerate(children, start=1):
            self.assertEqual(child.name, f"mail.eml › {child.original_filename}")
            self.assertEqual(child.uploaded_by, self.user)
            self.assertEqual(child.data_room, self.room)
            self.assertEqual(child.status, DataRoomDocument.Status.UPLOADED)
            self.assertIsNotNone(child.current_version.queued_at)
            with child.original_file.open("rb") as f:
                self.assertEqual(sha256_of_bytes(f.read()), child.content_sha256)
            tags = _tags(child)
            self.assertEqual(tags["source"], "email_attachment")
            self.assertEqual(tags[TAG_PARENT_VERSION], str(parent.current_version_id))
            self.assertEqual(tags[TAG_ORDINAL], str(ordinal))
            self.assertEqual(tags[TAG_DEPTH], "1")
        self.assertEqual(children[0].mime_type, "application/pdf")

        text = self._text(parent)
        self.assertIn(f'added to this data room as #{children[0].doc_index} "mail.eml › report.pdf"', text)
        self.assertNotIn("## Attachment:", text)
        self.mock_dispatch.assert_any_call("email_attachments")

        outcomes = parent.current_version.processing_metadata["email_attachments"]
        self.assertEqual([o["outcome"] for o in outcomes], ["split"] * 4)
        self.assertEqual(outcomes[0]["doc_id"], children[0].id)

    def test_duplicate_points_at_existing_document(self):
        existing = DataRoomDocument.objects.create(
            data_room=self.room, uploaded_by=self.user, original_filename="report.pdf",
            content_sha256=sha256_of_bytes(PDF), status=DataRoomDocument.Status.READY,
        )
        parent = self._process(self._upload_email(_eml([("copy.pdf", PDF)])))

        self.assertFalse(self._children(parent).exists())
        self.assertIn(f'identical to #{existing.doc_index} "report.pdf", already in this data room', self._text(parent))
        outcome = parent.current_version.processing_metadata["email_attachments"][0]
        self.assertEqual(outcome["outcome"], "duplicate")
        self.assertEqual(outcome["doc_id"], existing.id)

    def test_unsupported_and_oversized_are_skipped_with_reason(self):
        with self.settings(DOCUMENT_UPLOAD_MAX_SIZE_BYTES=len(PDF) - 1):
            parent = self._process(self._upload_email(_eml([("setup.exe", b"MZ"), ("big.pdf", PDF)])))

        self.assertFalse(self._children(parent).exists())
        text = self._text(parent)
        self.assertIn("setup.exe", text)
        self.assertIn("not added: unsupported file type", text)
        self.assertIn("not added: file is too large", text)
        outcomes = parent.current_version.processing_metadata["email_attachments"]
        self.assertEqual([o["outcome"] for o in outcomes], ["skipped", "skipped"])
        self.assertTrue(all(o["reason"] for o in outcomes))

    def test_audio_without_transcription_is_skipped(self):
        from unittest.mock import Mock

        with patch("core.preferences.get_preferences", return_value=Mock(allowed_transcription_models=[])):
            parent = self._process(self._upload_email(_eml([("call.m4a", b"audio")])))
        self.assertFalse(self._children(parent).exists())
        self.assertIn("audio transcription is not enabled", self._text(parent))

    @override_settings(DOCUMENT_EMAIL_MAX_SPLIT_ATTACHMENTS=1)
    def test_per_email_cap(self):
        parent = self._process(self._upload_email(_eml([("a.pdf", PDF), ("b.docx", DOCX)])))
        self.assertEqual(self._children(parent).count(), 1)
        self.assertIn("more than 1 attachments", self._text(parent))

    @override_settings(DOCUMENT_MAX_IN_FLIGHT_PER_USER=1)
    def test_in_flight_cap(self):
        # The email itself is in flight while it is processed.
        parent = self._process(self._upload_email(_eml([("a.pdf", PDF)])))
        self.assertFalse(self._children(parent).exists())
        self.assertIn("too many files are processing", self._text(parent))

    def test_rerun_does_not_duplicate_children(self):
        from documents.services.process_document import process_document_version

        parent = self._process(self._upload_email(_eml([("a.pdf", PDF), ("b.docx", DOCX)])))
        process_document_version(parent.current_version_id)
        self.assertEqual(self._children(parent).count(), 2)

    def test_image_attachment_is_described_inline_and_inline_image_skipped(self):
        from chat.tests.test_attachment_view import _png_bytes

        photo = MIMEImage(_png_bytes(), "png")
        photo.add_header("Content-Disposition", "attachment", filename="photo.png")
        logo = MIMEImage(_png_bytes() + b"x", "png")
        logo.add_header("Content-Disposition", "inline", filename="logo.png")
        logo.add_header("Content-ID", "<logo@x>")
        with patch("chat.services.describe_image", return_value="A photo") as describe, \
             patch("core.preferences.resolve_org_feature_model", return_value="anthropic/claude-opus-4-8"):
            parent = self._process(self._upload_email(_eml(extra_parts=[photo, logo])))

        self.assertFalse(self._children(parent).exists())
        self.assertEqual(describe.call_count, 1)
        text = self._text(parent)
        self.assertIn("photo.png", text)
        self.assertIn("[[image:", text)
        self.assertNotIn("logo.png", text)

    def test_forwarded_email_splits_recursively(self):
        inner = _eml([("contract.pdf", PDF)], subject="Contract")
        outer = _eml(subject="Fwd: Contract", extra_parts=[MIMEMessage(inner)])
        parent = self._process(self._upload_email(outer))

        [child] = self._children(parent)
        self.assertEqual(child.original_filename, "Contract.eml")
        self.assertEqual(_tags(child)[TAG_DEPTH], "1")

        child = self._process(child)
        [grandchild] = self._children(child)
        self.assertEqual(grandchild.original_filename, "contract.pdf")
        self.assertEqual(grandchild.name, "mail.eml › Contract.eml › contract.pdf")
        self.assertEqual(_tags(grandchild)[TAG_DEPTH], "2")

    def test_forward_chain_depth_is_capped(self):
        from documents.services.chunking import MAX_EMAIL_NESTING_DEPTH

        inner = _eml([("contract.pdf", PDF)], subject="Contract")
        doc = self._upload_email(_eml(extra_parts=[MIMEMessage(inner)]))
        from documents.services.process_document import ensure_initial_version

        v0 = ensure_initial_version(doc)
        DataRoomDocumentTag.objects.create(version=v0, key=TAG_DEPTH, value=str(MAX_EMAIL_NESTING_DEPTH - 1))
        parent = self._process(doc)
        self.assertFalse(self._children(parent).exists())
        self.assertIn("too many levels of forwarded emails", self._text(parent))


class SplitHelpersTests(TestCase):
    def test_display_name_shortens_parent_not_filename(self):
        self.assertEqual(split_display_name("mail.eml", "a.pdf"), "mail.eml › a.pdf")
        long_parent = "A very long email subject line that goes on and on and on.eml"
        name = split_display_name(long_parent, "quarterly-report.pdf")
        self.assertLessEqual(len(name), 75)
        self.assertTrue(name.endswith("… › quarterly-report.pdf"))
        self.assertEqual(split_display_name("x" * 60, "y" * 70 + ".pdf"), ("y" * 70 + ".pdf")[:75])

    def test_parse_eml_skips_inline_images_but_keeps_inline_files(self):
        pdf = MIMEApplication(PDF, Name="inline.pdf")
        pdf.add_header("Content-Disposition", "inline", filename="inline.pdf")
        pdf.add_header("Content-ID", "<pdf@x>")
        logo = MIMEImage(b"\x89PNG fake", "png")
        logo.add_header("Content-Disposition", "inline", filename="logo.png")
        logo.add_header("Content-ID", "<logo@x>")
        parsed = parse_email(_eml(extra_parts=[pdf, logo]).as_bytes(), "eml")
        self.assertEqual([(p.ordinal, p.filename) for p in parsed.parts], [(1, "inline.pdf")])
        self.assertEqual(parsed.parts[0].data, PDF)

    def test_msg_embedded_message_is_exported(self):
        class _Inner:
            subject = "Board minutes"

            def exportBytes(self):
                return b"msg-bytes"

        class _Att:
            longFilename = None
            shortFilename = None
            data = _Inner()

        self.assertEqual(_msg_attachment_bytes(_Att()), (b"msg-bytes", "Board minutes.msg"))

    def test_msg_unexportable_embedded_message_has_no_data(self):
        class _Inner:
            def exportBytes(self):
                raise RuntimeError("boom")

        class _Att:
            longFilename = "Fwd.msg"
            shortFilename = None
            data = _Inner()

        self.assertEqual(_msg_attachment_bytes(_Att()), (None, "Fwd.msg"))
