"""Tests for [[file:uuid]] download tokens (file-kind Asset references).

Mirrors test_image_assets.py but for the file-download path: a version-owned,
blob-less Asset with ``kind="file"`` whose serve view forces a download of the
document's LATEST native file.
"""
import json
import tempfile
import uuid as uuidlib

from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from chat.models import Asset
from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentVersion

User = get_user_model()
_MEDIA = tempfile.mkdtemp()
READY = DataRoomDocument.Status.READY
UPLOADED = DataRoomDocumentVersion.Origin.UPLOADED


def _make_doc(room, owner, *, filename="report.pdf", mime="application/pdf",
              parser_type="", original_file_bytes=None, doc_index=1):
    doc = DataRoomDocument.objects.create(
        data_room=room, uploaded_by=owner, original_filename=filename,
        mime_type=mime, doc_index=doc_index, status=READY,
    )
    if original_file_bytes is not None:
        doc.original_file.save(filename, ContentFile(original_file_bytes), save=True)
    return doc


def _make_version(doc, *, index=0, native_bytes=None, native_filename="",
                  mime="application/pdf", parser_type="", make_current=True):
    v = DataRoomDocumentVersion.objects.create(
        document=doc, version_index=index, status=READY, origin=UPLOADED,
        mime_type=mime, native_filename=native_filename, parser_type=parser_type,
        is_searchable=(index == 0),
    )
    if native_bytes is not None:
        v.native_blob.save(native_filename or "f.bin", ContentFile(native_bytes), save=True)
    if make_current:
        doc.current_version = v
        doc.active_searchable_version = v
        doc.save(update_fields=["current_version", "active_searchable_version"])
    return v


@override_settings(MEDIA_ROOT=_MEDIA)
class FileReferenceAssetTests(TestCase):
    """The file-kind reference Asset and its idempotent token helper."""

    def setUp(self):
        self.owner = User.objects.create_user(email="f1@test.com", password="pw")
        self.room = DataRoom.objects.create(name="R", slug="r-f1", created_by=self.owner)
        self.doc = _make_doc(self.room, self.owner)
        self.version = _make_version(self.doc, native_bytes=b"PDF", native_filename="report.pdf")

    def test_file_ref_is_blobless_version_owned_kind_file(self):
        from chat.assets import get_or_create_version_file_token

        get_or_create_version_file_token(version_id=self.version.id, mime="application/pdf")
        asset = Asset.objects.get(version=self.version, kind=Asset.KIND_FILE)
        self.assertFalse(bool(asset.blob))
        self.assertEqual(asset.version_id, self.version.id)
        self.assertEqual(asset.kind, "file")

    def test_token_is_idempotent(self):
        from chat.assets import get_or_create_version_file_token

        t1 = get_or_create_version_file_token(version_id=self.version.id)
        t2 = get_or_create_version_file_token(version_id=self.version.id)
        self.assertEqual(t1, t2)
        self.assertEqual(Asset.objects.filter(version=self.version, kind=Asset.KIND_FILE).count(), 1)

    def test_image_and_file_refs_for_same_version_are_distinct(self):
        """The dedup-collision regression: one version yields TWO distinct rows."""
        from chat.assets import (
            get_or_create_version_file_token,
            get_or_create_version_image_token,
        )

        img_tok = get_or_create_version_image_token(version_id=self.version.id)
        file_tok = get_or_create_version_file_token(version_id=self.version.id)
        self.assertNotEqual(img_tok, file_tok)
        self.assertTrue(img_tok.startswith("[[image:"))
        self.assertTrue(file_tok.startswith("[[file:"))
        self.assertEqual(Asset.objects.filter(version=self.version, blob="").count(), 2)
        self.assertEqual(
            Asset.objects.filter(version=self.version, kind=Asset.KIND_IMAGE).count(), 1
        )
        self.assertEqual(
            Asset.objects.filter(version=self.version, kind=Asset.KIND_FILE).count(), 1
        )


@override_settings(MEDIA_ROOT=_MEDIA)
class ServeFileAssetTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="f2@test.com", password="pw")
        self.other = User.objects.create_user(email="f2b@test.com", password="pw")
        self.room = DataRoom.objects.create(name="R", slug="r-f2", created_by=self.owner)
        self.doc = _make_doc(self.room, self.owner)
        self.v0 = _make_version(self.doc, native_bytes=b"PDF-v0", native_filename="report.pdf")
        self.asset = Asset.objects.create(
            version=self.v0, kind=Asset.KIND_FILE, content_type="application/pdf"
        )

    def _url(self):
        return reverse("chat_file_asset", args=[self.asset.id])

    def test_owner_downloads_as_attachment(self):
        self.client.force_login(self.owner)
        resp = self.client.get(self._url())
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["Content-Type"], "application/octet-stream")
        self.assertIn("attachment", resp["Content-Disposition"])
        self.assertIn('filename="report.pdf"', resp["Content-Disposition"])
        self.assertEqual(resp["X-Content-Type-Options"], "nosniff")
        self.assertEqual(b"".join(resp.streaming_content), b"PDF-v0")

    def test_non_owner_gets_404(self):
        self.client.force_login(self.other)
        self.assertEqual(self.client.get(self._url()).status_code, 404)

    def test_anonymous_redirected_to_login(self):
        self.assertEqual(self.client.get(self._url()).status_code, 302)

    def test_resolves_latest_version(self):
        """A token anchored on v0 serves the newest uploaded native file (symlink)."""
        _make_version(self.doc, index=1, native_bytes=b"PDF-v1", native_filename="final.pdf")
        self.client.force_login(self.owner)
        resp = self.client.get(self._url())
        self.assertEqual(b"".join(resp.streaming_content), b"PDF-v1")
        self.assertIn('filename="final.pdf"', resp["Content-Disposition"])

    def test_original_file_fallback(self):
        """A doc whose bytes live on original_file (no native_blob) still resolves."""
        doc = _make_doc(self.room, self.owner, filename="legacy.pdf",
                        original_file_bytes=b"LEGACY", doc_index=2)
        # Version carries no native bytes — bytes are only on the document.
        v = _make_version(doc, native_bytes=None)
        asset = Asset.objects.create(version=v, kind=Asset.KIND_FILE)
        self.client.force_login(self.owner)
        resp = self.client.get(reverse("chat_file_asset", args=[asset.id]))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(b"".join(resp.streaming_content), b"LEGACY")

    def test_no_native_file_gives_404(self):
        """A canvas/markdown-only doc has nothing to download."""
        doc = _make_doc(self.room, self.owner, filename="draft", doc_index=3)
        v = _make_version(doc, native_bytes=None)  # no native_blob, no original_file
        asset = Asset.objects.create(version=v, kind=Asset.KIND_FILE)
        self.client.force_login(self.owner)
        self.assertEqual(
            self.client.get(reverse("chat_file_asset", args=[asset.id])).status_code, 404
        )


@override_settings(MEDIA_ROOT=_MEDIA)
class FileAssetMetaTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="f3@test.com", password="pw")
        self.other = User.objects.create_user(email="f3b@test.com", password="pw")
        self.room = DataRoom.objects.create(name="R", slug="r-f3", created_by=self.owner)
        self.doc = _make_doc(self.room, self.owner)
        self.v0 = _make_version(self.doc, native_bytes=b"PDFBYTES", native_filename="report.pdf")
        self.asset = Asset.objects.create(
            version=self.v0, kind=Asset.KIND_FILE, content_type="application/pdf"
        )

    def _url(self):
        return reverse("chat_file_asset_meta", args=[self.asset.id])

    def test_owner_gets_metadata(self):
        self.client.force_login(self.owner)
        resp = self.client.get(self._url())
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["available"])
        self.assertEqual(data["name"], "report.pdf")
        self.assertEqual(data["size_bytes"], len(b"PDFBYTES"))
        self.assertEqual(data["mime"], "application/pdf")

    def test_non_owner_unavailable(self):
        self.client.force_login(self.other)
        resp = self.client.get(self._url())
        self.assertEqual(resp.status_code, 404)
        self.assertFalse(resp.json()["available"])


@override_settings(MEDIA_ROOT=_MEDIA)
class EmbedFileTokensTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="f4@test.com", password="pw")
        self.other = User.objects.create_user(email="f4b@test.com", password="pw")
        self.room = DataRoom.objects.create(name="R", slug="r-f4", created_by=self.owner)
        self.doc = _make_doc(self.room, self.owner)
        self.v0 = _make_version(self.doc, native_bytes=b"PDF", native_filename="report.pdf")
        self.asset = Asset.objects.create(
            version=self.v0, kind=Asset.KIND_FILE, content_type="application/pdf"
        )

    def test_accessible_token_becomes_link(self):
        from chat.views import _embed_file_tokens

        out = _embed_file_tokens(
            f"A [[file:{self.asset.id}|My Report]] B", "http://host", self.owner
        )
        self.assertIn(f'href="http://host/chat/file/{self.asset.id}/"', out)
        self.assertIn(">My Report</a>", out)
        self.assertNotIn("[[file:", out)

    def test_empty_label_falls_back_to_filename(self):
        from chat.views import _embed_file_tokens

        out = _embed_file_tokens(f"[[file:{self.asset.id}|]]", "http://host", self.owner)
        self.assertIn(">report.pdf</a>", out)

    def test_inaccessible_shows_placeholder(self):
        from chat.views import _embed_file_tokens

        out = _embed_file_tokens(f"[[file:{self.asset.id}|x]]", "http://host", self.other)
        self.assertNotIn("<a ", out)
        self.assertIn("no longer be accessed", out)

    def test_missing_token_shows_placeholder(self):
        from chat.views import _embed_file_tokens

        out = _embed_file_tokens(f"[[file:{uuidlib.uuid4()}|gone]]", "http://host", self.owner)
        self.assertNotIn("<a ", out)
        self.assertIn("no longer be accessed", out)


class ContentDispositionTests(SimpleTestCase):
    """Header-safety of the attachment Content-Disposition builder."""

    def test_unicode_filename_gets_rfc5987(self):
        from chat.views import _content_disposition

        disp = _content_disposition("rapport-café.pdf")
        self.assertIn("filename*=UTF-8''", disp)
        self.assertIn("attachment", disp)

    def test_strips_header_injection_chars(self):
        from chat.views import _content_disposition

        disp = _content_disposition('a"b\r\nSet-Cookie: x.pdf')
        self.assertNotIn("\r", disp)
        self.assertNotIn("\n", disp)
        self.assertNotIn('"b', disp)  # the embedded quote was stripped


@override_settings(MEDIA_ROOT=_MEDIA)
class FileTokenGatingTests(TestCase):
    """file_token_for_document gating + the document tools surfacing the handle."""

    def setUp(self):
        from llm.types.context import RunContext

        self.user = User.objects.create_user(email="f5@test.com", password="pw")
        self.room = DataRoom.objects.create(name="R", slug="r-f5", created_by=self.user)
        self.ctx = RunContext.create(user_id=self.user.pk, data_room_ids=[self.room.pk])

    def test_native_doc_gets_token_markdown_only_does_not(self):
        from chat.assets import file_token_for_document
        from documents.tests._helpers import make_document

        native = _make_doc(self.room, self.user, doc_index=1)
        _make_version(native, native_bytes=b"PDF", native_filename="report.pdf")
        self.assertTrue((file_token_for_document(native) or "").startswith("[[file:"))

        markdown_only = make_document(self.room, self.user, original_filename="note.md",
                                      chunks=["x"])
        self.assertIsNone(file_token_for_document(markdown_only))

    def test_list_tool_surfaces_file_handles(self):
        from chat.tools import ListDocumentsTool
        from documents.tests._helpers import make_document

        # 1: native, non-image -> file only
        native = _make_doc(self.room, self.user, filename="report.pdf", doc_index=1)
        _make_version(native, native_bytes=b"PDF", native_filename="report.pdf")
        # 2: image-as-document -> BOTH image and file
        image_doc = _make_doc(self.room, self.user, filename="chart.png",
                              mime="image/png", doc_index=2)
        _make_version(image_doc, native_bytes=b"PNG", native_filename="chart.png",
                      mime="image/png", parser_type="image")
        # 3: markdown-only -> neither
        make_document(self.room, self.user, original_filename="note.md",
                      chunks=["x"], doc_index=3)

        tool = ListDocumentsTool()
        tool.set_context(self.ctx)
        rows = {r["doc_index"]: r for r in json.loads(tool.invoke({}))["documents"]}

        self.assertIn("file", rows[1])
        self.assertNotIn("image", rows[1])
        self.assertIn("file", rows[2])
        self.assertIn("image", rows[2])
        self.assertNotIn("file", rows[3])
        self.assertNotIn("image", rows[3])
