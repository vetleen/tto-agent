"""Tests for the org logo upload/delete endpoints and image processing."""
import io
import tempfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from accounts.avatars import InvalidLogo, process_org_logo
from accounts.models import Membership, Organization, User


def _png(w, h, fmt="PNG"):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA" if fmt == "PNG" else "RGB", (w, h), (10, 120, 200, 255)[: 4 if fmt == "PNG" else 3]).save(
        buf, format=fmt
    )
    return buf.getvalue()


def _upload(data, name="logo.png", content_type="image/png"):
    return SimpleUploadedFile(name, data, content_type=content_type)


class ProcessOrgLogoTests(SimpleTestCase):
    def _decode(self, content_file):
        from PIL import Image

        return Image.open(io.BytesIO(content_file.read()))

    def test_scales_within_bounding_box(self):
        ext, processed = process_org_logo(io.BytesIO(_png(2000, 1000)))
        self.assertEqual(ext, "png")
        img = self._decode(processed)
        self.assertLessEqual(img.width, 1000)
        self.assertLessEqual(img.height, 400)
        # Aspect (2:1) preserved.
        self.assertAlmostEqual(img.width / img.height, 2.0, places=1)

    def test_small_image_not_upscaled(self):
        _, processed = process_org_logo(io.BytesIO(_png(120, 60)))
        img = self._decode(processed)
        self.assertEqual((img.width, img.height), (120, 60))

    def test_jpeg_normalised_to_png(self):
        ext, processed = process_org_logo(io.BytesIO(_png(300, 200, fmt="JPEG")))
        self.assertEqual(ext, "png")
        self.assertEqual(self._decode(processed).format, "PNG")

    def test_non_image_rejected(self):
        with self.assertRaises(InvalidLogo):
            process_org_logo(io.BytesIO(b"definitely not an image"))


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class LogoUploadEndpointTests(TestCase):
    def setUp(self):
        self.org = Organization.objects.create(name="Acme", slug="acme")
        self.admin = User.objects.create_user(email="admin@test.com", password="pass")
        Membership.objects.create(user=self.admin, org=self.org, role=Membership.Role.ADMIN)
        self.upload_url = reverse("accounts:org_logo_upload")
        self.delete_url = reverse("accounts:org_logo_delete")

    def _login_admin(self):
        self.client.login(email="admin@test.com", password="pass")

    def test_upload_valid_logo(self):
        self._login_admin()
        resp = self.client.post(self.upload_url, {"logo": _upload(_png(400, 200))})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])
        self.org.refresh_from_db()
        self.assertTrue(self.org.logo)

    def test_upload_non_image_rejected(self):
        self._login_admin()
        resp = self.client.post(self.upload_url, {"logo": _upload(b"junk", name="x.png")})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.json())

    def test_upload_requires_admin(self):
        member = User.objects.create_user(email="member@test.com", password="pass")
        Membership.objects.create(user=member, org=self.org, role=Membership.Role.MEMBER)
        self.client.login(email="member@test.com", password="pass")
        resp = self.client.post(self.upload_url, {"logo": _upload(_png(400, 200))})
        self.assertEqual(resp.status_code, 403)

    @override_settings(PROFILE_PICTURE_MAX_BYTES=10)
    def test_upload_oversize_rejected(self):
        self._login_admin()
        resp = self.client.post(self.upload_url, {"logo": _upload(_png(400, 200))})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("large", resp.json()["error"].lower())

    def test_delete_removes_logo(self):
        self._login_admin()
        self.client.post(self.upload_url, {"logo": _upload(_png(400, 200))})
        self.org.refresh_from_db()
        self.assertTrue(self.org.logo)
        resp = self.client.post(self.delete_url)
        self.assertEqual(resp.status_code, 200)
        self.org.refresh_from_db()
        self.assertFalse(self.org.logo)
