"""Tests for the org brand-font upload/delete settings endpoints."""
import io
import tempfile
from pathlib import Path

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from accounts.models import FontAsset, Membership, Organization, User
from core import fonts as core_fonts

FONTS_DIR = Path(core_fonts.__file__).resolve().parent / "assets" / "fonts"


def _carlito_bytes():
    return (FONTS_DIR / "Carlito" / "Carlito-Regular.ttf").read_bytes()


def _restricted_font_bytes():
    from fontTools.ttLib import TTFont

    font = TTFont(io.BytesIO(_carlito_bytes()))
    font["OS/2"].fsType = 0x0002  # Restricted-License embedding
    buf = io.BytesIO()
    font.save(buf)
    return buf.getvalue()


def _upload(data, name="Carlito-Regular.ttf", content_type="font/ttf"):
    return SimpleUploadedFile(name, data, content_type=content_type)


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class FontUploadEndpointTests(TestCase):
    def setUp(self):
        self.org = Organization.objects.create(name="Acme", slug="acme")
        self.admin = User.objects.create_user(email="admin@test.com", password="pass")
        Membership.objects.create(user=self.admin, org=self.org, role=Membership.Role.ADMIN)
        self.upload_url = reverse("accounts:org_fonts_upload")
        self.delete_url = reverse("accounts:org_fonts_delete")

    def _login_admin(self):
        self.client.login(email="admin@test.com", password="pass")

    def test_upload_valid_font(self):
        self._login_admin()
        resp = self.client.post(self.upload_url, {"file": _upload(_carlito_bytes())})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["fonts"][0]["family"], "Carlito")
        self.assertEqual(
            FontAsset.objects.filter(organization=self.org, source="upload").count(), 1
        )

    def test_upload_non_font_rejected(self):
        self._login_admin()
        resp = self.client.post(self.upload_url, {"file": _upload(b"not a font", name="x.ttf")})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.json())

    def test_upload_restricted_font_rejected(self):
        self._login_admin()
        resp = self.client.post(self.upload_url, {"file": _upload(_restricted_font_bytes())})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("embedding", resp.json()["error"].lower())

    @override_settings(FONT_UPLOAD_MAX_SIZE_BYTES=1000)
    def test_upload_oversize_rejected(self):
        self._login_admin()
        resp = self.client.post(self.upload_url, {"file": _upload(_carlito_bytes())})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("large", resp.json()["error"].lower())

    def test_upload_requires_admin(self):
        member = User.objects.create_user(email="member@test.com", password="pass")
        Membership.objects.create(user=member, org=self.org, role=Membership.Role.MEMBER)
        self.client.login(email="member@test.com", password="pass")
        resp = self.client.post(self.upload_url, {"file": _upload(_carlito_bytes())})
        self.assertEqual(resp.status_code, 403)

    def test_delete_removes_family(self):
        self._login_admin()
        self.client.post(self.upload_url, {"file": _upload(_carlito_bytes())})
        self.assertEqual(FontAsset.objects.filter(organization=self.org).count(), 1)
        resp = self.client.post(
            self.delete_url,
            data='{"family_norm": "carlito"}',
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["fonts"], [])
        self.assertEqual(FontAsset.objects.filter(organization=self.org).count(), 0)
