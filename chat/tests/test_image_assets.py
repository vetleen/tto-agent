"""Tests for the ImageAsset model and its access-checked serve view."""

import tempfile

from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.db import IntegrityError, transaction
from django.test import TestCase, override_settings
from django.urls import reverse

from chat.models import ChatCanvas, ChatMessage, ChatThread, ImageAsset

User = get_user_model()

_MEDIA = tempfile.mkdtemp()


def _make_asset(*, canvas=None, version=None, message=None, content_type="image/png"):
    return ImageAsset.objects.create(
        canvas=canvas,
        version=version,
        message=message,
        blob=ContentFile(b"\x89PNG fake-image-bytes", name="x.png"),
        content_type=content_type,
        size_bytes=21,
    )


@override_settings(MEDIA_ROOT=_MEDIA)
class ImageAssetModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="ia@test.com", password="pw")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.canvas = ChatCanvas.objects.create(thread=self.thread, title="C", content="")

    def test_single_owner_is_allowed(self):
        asset = _make_asset(canvas=self.canvas)
        self.assertIsNotNone(asset.pk)
        self.assertEqual(self.canvas.image_assets.count(), 1)

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
class ServeImageAssetTests(TestCase):
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
        self.assertEqual(resp["Content-Type"], "image/png")

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
        self.assertEqual(resp["Content-Type"], "application/octet-stream")
