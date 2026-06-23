"""Tests for the user profile picture: validate/resize service and upload views."""
import io
import shutil
import tempfile
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from PIL import Image

from accounts.avatars import (
    AVATAR_MAX_EDGE,
    InvalidProfilePicture,
    process_profile_picture,
)

User = get_user_model()

_MEDIA = tempfile.mkdtemp(prefix="avatar-test-")


def tearDownModule():
    shutil.rmtree(_MEDIA, ignore_errors=True)


def _image_bytes(fmt="PNG", size=(10, 10), color=(10, 20, 30), mode="RGB"):
    img = Image.new(mode, size, color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _upload(name, data, content_type):
    return SimpleUploadedFile(name, data, content_type=content_type)


class ProcessProfilePictureTests(TestCase):
    """Unit tests for the pure validate/resize helper."""

    def test_resizes_large_image_preserving_aspect(self):
        data = _image_bytes(size=(600, 400))
        ext, original, resized = process_profile_picture(_upload("a.png", data, "image/png"))
        self.assertEqual(ext, "png")
        with Image.open(io.BytesIO(resized.read())) as img:
            self.assertLessEqual(max(img.size), AVATAR_MAX_EDGE)
            # Aspect ratio preserved: 600x400 -> 256x171-ish.
            self.assertEqual(img.size, (AVATAR_MAX_EDGE, round(AVATAR_MAX_EDGE * 400 / 600)))
        # Original keeps full resolution.
        with Image.open(io.BytesIO(original.read())) as img:
            self.assertEqual(img.size, (600, 400))

    def test_small_image_not_upscaled(self):
        data = _image_bytes(size=(40, 40))
        _ext, _original, resized = process_profile_picture(_upload("a.png", data, "image/png"))
        with Image.open(io.BytesIO(resized.read())) as img:
            self.assertEqual(img.size, (40, 40))

    def test_jpeg_passthrough(self):
        data = _image_bytes(fmt="JPEG", size=(300, 300))
        ext, _original, resized = process_profile_picture(_upload("a.jpg", data, "image/jpeg"))
        self.assertEqual(ext, "jpg")
        with Image.open(io.BytesIO(resized.read())) as img:
            self.assertEqual(img.format, "JPEG")

    def test_rgba_png_kept_as_png(self):
        # RGBA PNGs (transparency) must not crash and stay PNG.
        data = _image_bytes(fmt="PNG", size=(300, 300), color=(10, 20, 30, 128), mode="RGBA")
        ext, _original, resized = process_profile_picture(_upload("a.png", data, "image/png"))
        self.assertEqual(ext, "png")
        with Image.open(io.BytesIO(resized.read())) as img:
            self.assertEqual(img.format, "PNG")

    def test_rejects_non_image(self):
        with self.assertRaises(InvalidProfilePicture):
            process_profile_picture(_upload("a.png", b"not an image at all", "image/png"))

    def test_rejects_disallowed_format(self):
        data = _image_bytes(fmt="GIF", size=(20, 20), mode="P")
        with self.assertRaises(InvalidProfilePicture):
            process_profile_picture(_upload("a.gif", data, "image/gif"))

    @override_settings(PROFILE_PICTURE_MAX_BYTES=50)
    def test_rejects_oversize(self):
        data = _image_bytes(size=(300, 300))
        self.assertGreater(len(data), 50)
        with self.assertRaises(InvalidProfilePicture):
            process_profile_picture(_upload("a.png", data, "image/png"))

    def test_rejects_too_many_pixels(self):
        # Patch the cap low so we don't have to allocate a 50-megapixel image.
        data = _image_bytes(size=(50, 50))
        with patch("accounts.avatars._MAX_IMAGE_PIXELS", 100):
            with self.assertRaises(InvalidProfilePicture) as cm:
                process_profile_picture(_upload("a.png", data, "image/png"))
        msg = str(cm.exception)
        self.assertIn("pixel", msg.lower())
        # The message names the image's actual dimensions, not just a megapixel cap.
        self.assertIn("50", msg)


@override_settings(ALLOWED_HOSTS=["testserver"], MEDIA_ROOT=_MEDIA)
class ProfilePictureUploadViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="u@example.com", password="pw123456")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.url = reverse("accounts:profile_picture_update")
        self.delete_url = reverse("accounts:profile_picture_delete")
        self.client.login(email="u@example.com", password="pw123456")

    def _post_image(self, size=(400, 300), fmt="PNG", name="a.png", content_type="image/png"):
        data = _image_bytes(fmt=fmt, size=size)
        return self.client.post(self.url, {"picture": _upload(name, data, content_type)})

    def test_requires_login(self):
        self.client.logout()
        r = self._post_image()
        self.assertEqual(r.status_code, 302)

    def test_rejects_get(self):
        r = self.client.get(self.url)
        self.assertEqual(r.status_code, 405)

    def test_upload_sets_both_fields(self):
        r = self._post_image()
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["ok"])
        self.assertTrue(body["url"])
        self.user.refresh_from_db()
        self.assertTrue(self.user.profile_picture)
        self.assertTrue(self.user.profile_picture_original)
        # The display file is the resized one.
        with self.user.profile_picture.open("rb") as fh:
            with Image.open(fh) as img:
                self.assertLessEqual(max(img.size), AVATAR_MAX_EDGE)
        with self.user.profile_picture_original.open("rb") as fh:
            with Image.open(fh) as img:
                self.assertEqual(img.size, (400, 300))

    def test_jpeg_upload_succeeds(self):
        r = self._post_image(size=(800, 600), fmt="JPEG", name="photo.jpg", content_type="image/jpeg")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])
        self.user.refresh_from_db()
        self.assertTrue(self.user.profile_picture)
        with self.user.profile_picture.open("rb") as fh:
            with Image.open(fh) as img:
                self.assertEqual(img.format, "JPEG")

    def test_missing_file_returns_400(self):
        r = self.client.post(self.url, {})
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", r.json())

    def test_dimension_rejection_reason_is_surfaced(self):
        # A valid image that exceeds the pixel cap must report the real reason,
        # not a generic "use a JPEG/PNG/WebP" message that implies a format problem.
        with patch("accounts.avatars._MAX_IMAGE_PIXELS", 100):
            r = self._post_image(size=(200, 200))
        self.assertEqual(r.status_code, 400)
        self.assertIn("pixel", r.json()["error"].lower())

    def test_invalid_image_returns_400(self):
        r = self.client.post(self.url, {"picture": _upload("a.png", b"garbage", "image/png")})
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", r.json())
        self.user.refresh_from_db()
        self.assertFalse(self.user.profile_picture)

    def test_replace_removes_previous_file(self):
        self._post_image(name="first.png")
        self.user.refresh_from_db()
        first = self.user.profile_picture.name
        first_storage = self.user.profile_picture.storage
        self.assertTrue(first_storage.exists(first))

        self._post_image(name="second.png", size=(200, 200))
        self.user.refresh_from_db()
        second = self.user.profile_picture.name
        # New file present; if the path changed, the old one is gone.
        self.assertTrue(first_storage.exists(second))
        if second != first:
            self.assertFalse(first_storage.exists(first))

    def test_delete_clears_fields(self):
        self._post_image()
        self.user.refresh_from_db()
        name = self.user.profile_picture.name
        storage = self.user.profile_picture.storage

        r = self.client.post(self.delete_url)
        self.assertEqual(r.status_code, 200)
        self.user.refresh_from_db()
        self.assertFalse(self.user.profile_picture)
        self.assertFalse(self.user.profile_picture_original)
        self.assertFalse(storage.exists(name))

    def test_delete_without_picture_is_ok(self):
        r = self.client.post(self.delete_url)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])
