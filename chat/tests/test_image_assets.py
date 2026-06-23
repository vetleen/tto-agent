"""Tests for the Asset model and its access-checked serve view."""

import tempfile

from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.db import IntegrityError, transaction
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from chat.models import ChatCanvas, ChatMessage, ChatThread, Asset

User = get_user_model()

_MEDIA = tempfile.mkdtemp()


def _make_asset(*, canvas=None, version=None, message=None, content_type="image/png"):
    return Asset.objects.create(
        canvas=canvas,
        version=version,
        message=message,
        blob=ContentFile(b"\x89PNG fake-image-bytes", name="x.png"),
        content_type=content_type,
        size_bytes=21,
    )


@override_settings(MEDIA_ROOT=_MEDIA)
class AssetModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="ia@test.com", password="pw")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.canvas = ChatCanvas.objects.create(thread=self.thread, title="C", content="")

    def test_single_owner_is_allowed(self):
        asset = _make_asset(canvas=self.canvas)
        self.assertIsNotNone(asset.pk)
        self.assertEqual(self.canvas.assets.count(), 1)

    def test_zero_owners_rejected(self):
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                _make_asset()

    def test_two_owners_rejected(self):
        msg = ChatMessage.objects.create(thread=self.thread, role="user", content="hi")
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                _make_asset(canvas=self.canvas, message=msg)


@override_settings(MEDIA_ROOT=_MEDIA)
class ServeAssetTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(email="own@test.com", password="pw")
        self.other = User.objects.create_user(email="oth@test.com", password="pw")
        self.thread = ChatThread.objects.create(created_by=self.owner)
        self.canvas = ChatCanvas.objects.create(thread=self.thread, title="C", content="")
        self.asset = _make_asset(canvas=self.canvas)

    def _url(self):
        return reverse("chat_image_asset", args=[self.asset.id])

    def test_owner_can_fetch_inline(self):
        self.client.force_login(self.owner)
        resp = self.client.get(self._url())
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["X-Content-Type-Options"], "nosniff")
        self.assertIn("inline", resp["Content-Disposition"])
        # The download filename carries an extension derived from the mime so a
        # "Save image as" lands a usable file rather than an extension-less blob.
        self.assertIn(f'filename="{self.asset.id}.png"', resp["Content-Disposition"])
        self.assertEqual(resp["Content-Type"], "image/png")

    def test_jpeg_filename_uses_jpg_extension(self):
        asset = _make_asset(canvas=self.canvas, content_type="image/jpeg")
        self.client.force_login(self.owner)
        resp = self.client.get(reverse("chat_image_asset", args=[asset.id]))
        self.assertEqual(resp.status_code, 200)
        self.assertIn(f'filename="{asset.id}.jpg"', resp["Content-Disposition"])

    def test_non_owner_gets_404(self):
        self.client.force_login(self.other)
        resp = self.client.get(self._url())
        self.assertEqual(resp.status_code, 404)

    def test_anonymous_redirected_to_login(self):
        resp = self.client.get(self._url())
        self.assertEqual(resp.status_code, 302)

    def test_non_image_forced_to_download(self):
        # A non-displayable content type is streamed as an attachment.
        asset = _make_asset(canvas=self.canvas, content_type="image/x-emf")
        self.client.force_login(self.owner)
        resp = self.client.get(reverse("chat_image_asset", args=[asset.id]))
        self.assertEqual(resp.status_code, 200)
        self.assertIn("attachment", resp["Content-Disposition"])
        # Unknown mime -> no extension to append, so the bare id is used.
        self.assertIn(f'filename="{asset.id}"', resp["Content-Disposition"])
        self.assertEqual(resp["Content-Type"], "application/octet-stream")


@override_settings(MEDIA_ROOT=_MEDIA)
class EmbedImageTokensTests(TestCase):
    """canvas export resolves [[image:uuid]] tokens to <img> data-URLs (ACL-gated)."""

    def setUp(self):
        self.owner = User.objects.create_user(email="emb@test.com", password="pw")
        self.other = User.objects.create_user(email="emb2@test.com", password="pw")
        self.thread = ChatThread.objects.create(created_by=self.owner)
        self.canvas = ChatCanvas.objects.create(thread=self.thread, title="C", content="")
        self.asset = _make_asset(canvas=self.canvas)

    def test_owner_token_becomes_img(self):
        from chat.views import _embed_image_tokens

        content = f"Before\n\n[[image:{self.asset.id}|Image 1: a chart]]\n\nAfter"
        out, _ = _embed_image_tokens(content, self.owner)
        self.assertIn('<img src="data:image/png;base64,', out)
        self.assertNotIn("[[image:", out)

    def test_inaccessible_token_dropped(self):
        from chat.views import _embed_image_tokens

        content = f"X [[image:{self.asset.id}|Image 1: a chart]] Y"
        out, _ = _embed_image_tokens(content, self.other)
        self.assertNotIn("<img", out)
        self.assertNotIn("[[image:", out)


@override_settings(MEDIA_ROOT=_MEDIA)
class CanvasImportAssetsTests(TestCase):
    """Docx import attaches embedded images to the canvas as Assets + tokens."""

    def setUp(self):
        self.user = User.objects.create_user(email="cimp@test.com", password="pw")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.canvas = ChatCanvas.objects.create(thread=self.thread, title="Deck", content="")

    def test_import_stores_canvas_assets_and_tokens(self):
        from unittest.mock import patch

        from django.core.files.uploadedfile import SimpleUploadedFile

        from chat.services import import_docx_to_canvas
        from chat.tests.test_attachments import _docx_with_image

        f = SimpleUploadedFile("deck.docx", _docx_with_image())
        with patch("chat.services.describe_image", return_value="a revenue chart"):
            _title, content, _truncated = import_docx_to_canvas(f, self.user, canvas=self.canvas)

        assets = list(Asset.objects.filter(canvas=self.canvas))
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].description, "a revenue chart")
        self.assertIn(f"[[image:{assets[0].id}|", content)


@override_settings(MEDIA_ROOT=_MEDIA)
class ReferenceAssetTests(TestCase):
    """A version-owned Asset with no blob serves the data-room image bytes
    (the bytes stay on the document's original_file — no copy)."""

    def setUp(self):
        from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentVersion

        self.owner = User.objects.create_user(email="ref@test.com", password="pw")
        self.other = User.objects.create_user(email="ref2@test.com", password="pw")
        self.room = DataRoom.objects.create(name="R", slug="r-ref", created_by=self.owner)
        self.doc = DataRoomDocument.objects.create(
            data_room=self.room, uploaded_by=self.owner,
            original_filename="chart.png", mime_type="image/png",
            doc_index=1, status=DataRoomDocument.Status.READY,
        )
        self.doc.original_file.save("chart.png", ContentFile(b"\x89PNG real-bytes"), save=True)
        self.version = DataRoomDocumentVersion.objects.create(
            document=self.doc, parser_type="image", mime_type="image/png",
        )
        self.doc.current_version = self.version
        self.doc.save(update_fields=["current_version"])
        # Reference asset: version-owned, NO blob.
        self.asset = Asset.objects.create(version=self.version, content_type="image/png")

    def test_reference_asset_has_no_blob(self):
        self.assertFalse(bool(self.asset.blob))

    def test_owner_fetches_original_file_bytes(self):
        self.client.force_login(self.owner)
        resp = self.client.get(reverse("chat_image_asset", args=[self.asset.id]))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["Content-Type"], "image/png")
        self.assertIn("inline", resp["Content-Disposition"])

    def test_non_owner_gets_404(self):
        self.client.force_login(self.other)
        resp = self.client.get(reverse("chat_image_asset", args=[self.asset.id]))
        self.assertEqual(resp.status_code, 404)

    def test_embed_resolves_reference_to_dataurl(self):
        from chat.views import _embed_image_tokens

        out, _ = _embed_image_tokens(f"A [[image:{self.asset.id}|chart]] B", self.owner)
        self.assertIn('<img src="data:image/png;base64,', out)
        self.assertNotIn("[[image:", out)

    def test_embed_inaccessible_shows_placeholder(self):
        from chat.views import _embed_image_tokens

        out, _ = _embed_image_tokens(f"A [[image:{self.asset.id}|chart]] B", self.other)
        self.assertNotIn("<img", out)
        self.assertIn("no longer be accessed", out)

    def test_embed_missing_token_shows_placeholder(self):
        import uuid as _uuid

        from chat.views import _embed_image_tokens

        out, _ = _embed_image_tokens(f"X [[image:{_uuid.uuid4()}|gone]] Y", self.owner)
        self.assertIn("no longer be accessed", out)


class SplitImageLabelSizeTests(SimpleTestCase):
    """The optional trailing ``|NN%`` display-width segment of a token interior.

    Mirrors the JS rule in templates/chat/chat.html (renderMarkdown) — keep both
    in sync.
    """

    def _split(self, label):
        from chat.views import _split_image_label_size

        return _split_image_label_size(label)

    def test_size_suffix_extracted(self):
        self.assertEqual(self._split("a chart|50%"), ("a chart", 50))

    def test_pipe_inside_caption_preserved(self):
        # Only the FINAL |NN% is the size; an earlier | stays in the caption.
        self.assertEqual(self._split("a|b|30%"), ("a|b", 30))

    def test_no_size_returns_none(self):
        self.assertEqual(self._split("just a caption"), ("just a caption", None))

    def test_percent_not_in_suffix_form_is_not_a_size(self):
        # A bare percent without the leading | is part of the caption.
        self.assertEqual(self._split("50% done"), ("50% done", None))

    def test_clamps_below_minimum(self):
        self.assertEqual(self._split("cap|5%"), ("cap", 10))
        self.assertEqual(self._split("cap|0%"), ("cap", 10))

    def test_clamps_above_maximum(self):
        self.assertEqual(self._split("cap|250%"), ("cap", 100))

    def test_empty_caption_with_size(self):
        self.assertEqual(self._split("|40%"), ("", 40))

    def test_empty_string(self):
        self.assertEqual(self._split(""), ("", None))


@override_settings(MEDIA_ROOT=_MEDIA)
class CanvasExportImageSizingTests(TestCase):
    """End-to-end docx export honours the ``|NN%`` display-width suffix by
    resizing the matching inline shape (height scaled to keep the aspect ratio)."""

    def setUp(self):
        from chat.tests.test_attachments import _tiny_png

        self.png = _tiny_png()
        self.owner = User.objects.create_user(email="sz@test.com", password="pw")
        self.thread = ChatThread.objects.create(created_by=self.owner)
        self.canvas = ChatCanvas.objects.create(thread=self.thread, title="Sized", content="")
        self.thread.active_canvas = self.canvas
        self.thread.save(update_fields=["active_canvas"])
        self.asset = Asset.objects.create(
            canvas=self.canvas,
            blob=ContentFile(self.png, name="a.png"),
            content_type="image/png",
            size_bytes=len(self.png),
        )
        # An asset owned by someone else — inaccessible to self.owner.
        self.other = User.objects.create_user(email="sz2@test.com", password="pw")
        other_thread = ChatThread.objects.create(created_by=self.other)
        other_canvas = ChatCanvas.objects.create(thread=other_thread, title="O", content="")
        self.foreign_asset = Asset.objects.create(
            canvas=other_canvas,
            blob=ContentFile(self.png, name="b.png"),
            content_type="image/png",
            size_bytes=len(self.png),
        )
        self.client.force_login(self.owner)

    def _export(self, content):
        import json

        url = f"/chat/threads/{self.thread.id}/canvas/export/"
        return self.client.post(url, json.dumps({"content": content}), content_type="application/json")

    def _open(self, response):
        import io

        from docx import Document as DocxDocument

        return DocxDocument(io.BytesIO(response.getvalue()))

    def _text_width(self, doc):
        section = doc.sections[0]
        return section.page_width - section.left_margin - section.right_margin

    def test_sized_token_resizes_inline_shape(self):
        resp = self._export(f"# T\n\n[[image:{self.asset.id}|a chart|50%]]\n")
        self.assertEqual(resp.status_code, 200)
        doc = self._open(resp)
        self.assertEqual(len(doc.inline_shapes), 1)
        self.assertEqual(doc.inline_shapes[0].width, int(self._text_width(doc) * 50 / 100))

    def test_unsized_token_keeps_default_width(self):
        resp = self._export(f"# T\n\n[[image:{self.asset.id}|a chart]]\n")
        self.assertEqual(resp.status_code, 200)
        doc = self._open(resp)
        self.assertEqual(len(doc.inline_shapes), 1)
        # No |NN% -> html2docx's native sizing, not our half-page width.
        self.assertNotEqual(doc.inline_shapes[0].width, int(self._text_width(doc) * 50 / 100))

    def test_sized_token_after_inaccessible_token_sizes_correct_shape(self):
        # The inaccessible token becomes a text placeholder (no inline shape), so
        # the following sized token's percentage must still land on its own shape.
        content = (
            f"[[image:{self.foreign_asset.id}|secret]]\n\n"
            f"[[image:{self.asset.id}|chart|40%]]\n"
        )
        resp = self._export(content)
        self.assertEqual(resp.status_code, 200)
        doc = self._open(resp)
        self.assertEqual(len(doc.inline_shapes), 1)
        self.assertEqual(doc.inline_shapes[0].width, int(self._text_width(doc) * 40 / 100))

    def test_mermaid_image_before_sized_token_sizes_token_shape(self):
        # A pre-rendered <img> (the client's mermaid render) reserves a None slot
        # so the token's percentage lands on the token's shape, not the mermaid's.
        import base64

        b64 = base64.b64encode(self.png).decode("ascii")
        mermaid = f'<img src="data:image/png;base64,{b64}" alt="diagram" />'
        content = f"{mermaid}\n\n[[image:{self.asset.id}|chart|30%]]\n"
        resp = self._export(content)
        self.assertEqual(resp.status_code, 200)
        doc = self._open(resp)
        self.assertEqual(len(doc.inline_shapes), 2)
        target = int(self._text_width(doc) * 30 / 100)
        # Shape 0 = mermaid (untouched); shape 1 = token (resized to 30%).
        self.assertEqual(doc.inline_shapes[1].width, target)
        self.assertNotEqual(doc.inline_shapes[0].width, target)
