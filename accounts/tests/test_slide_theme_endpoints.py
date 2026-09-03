"""HTTP endpoints for v2 slide-theme CRUD + per-theme logo (org + user scope)."""
from __future__ import annotations

import json
import tempfile
import uuid
from io import BytesIO

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase
from django.urls import reverse
from PIL import Image

from accounts.models import Membership, Organization


def _png() -> bytes:
    b = BytesIO()
    Image.new("RGB", (8, 8), (10, 120, 60)).save(b, "PNG")
    return b.getvalue()


class SlideThemeEndpointTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(email=f"u+{uuid.uuid4().hex[:6]}@ex.com", password="x")
        self.client = Client()
        self.client.force_login(self.user)

    def _post(self, name, payload):
        return self.client.post(reverse(f"accounts:{name}"), data=json.dumps(payload), content_type="application/json")

    def _list(self):
        return self.client.get(reverse("accounts:slide_themes_list")).json()

    def test_user_theme_crud_and_default(self):
        r = self._post("slide_theme_save", {"scope": "user", "theme": {"label": "Mine"}})
        self.assertEqual(r.status_code, 200)
        tid = r.json()["theme"]["id"]

        lst = self._list()
        by_id = {t["id"]: t for t in lst["themes"]}
        self.assertIn("forest", by_id)
        self.assertEqual(by_id[tid]["scope"], "user")

        self.assertEqual(self._post("slide_theme_set_default", {"scope": "user", "id": tid}).status_code, 200)
        self.assertEqual(self._list()["user_default"], tid)

        # update preserves the id
        r = self._post("slide_theme_save", {"scope": "user", "theme": {"id": tid, "label": "Renamed"}})
        self.assertEqual(r.json()["theme"]["id"], tid)

        self.assertEqual(self._post("slide_theme_delete", {"scope": "user", "id": tid}).status_code, 200)
        self.assertNotIn(tid, {t["id"] for t in self._list()["themes"]})

    def test_user_theme_logo_upload_serve_delete(self):
        tid = self._post("slide_theme_save", {"scope": "user", "theme": {"label": "L"}}).json()["theme"]["id"]
        with tempfile.TemporaryDirectory() as d, self.settings(MEDIA_ROOT=d):
            up = self.client.post(reverse("accounts:slide_theme_logo_upload"), {
                "scope": "user", "theme_id": tid,
                "logo": SimpleUploadedFile("l.png", _png(), "image/png")})
            self.assertEqual(up.status_code, 200)
            self.assertEqual(up.json()["logo_ext"], "png")
            self.assertTrue({t["id"]: t for t in self._list()["themes"]}[tid]["has_logo"])

            serve = self.client.get(reverse("accounts:slide_theme_logo_serve"), {"scope": "user", "id": tid})
            self.assertEqual(serve.status_code, 200)
            self.assertEqual(serve["Content-Type"], "image/png")

            self.assertEqual(self._post("slide_theme_logo_delete", {"scope": "user", "id": tid}).status_code, 200)
            self.assertFalse({t["id"]: t for t in self._list()["themes"]}[tid]["has_logo"])

    def test_logo_upload_requires_saved_theme(self):
        with tempfile.TemporaryDirectory() as d, self.settings(MEDIA_ROOT=d):
            up = self.client.post(reverse("accounts:slide_theme_logo_upload"), {
                "scope": "user", "theme_id": "tnope",
                "logo": SimpleUploadedFile("l.png", _png(), "image/png")})
        self.assertEqual(up.status_code, 400)

    def test_org_scope_requires_admin(self):
        org = Organization.objects.create(name="Acme", slug=f"acme-{uuid.uuid4().hex[:6]}")
        Membership.objects.create(user=self.user, org=org, role=Membership.Role.MEMBER)
        r = self._post("slide_theme_save", {"scope": "org", "theme": {"label": "OrgT"}})
        self.assertEqual(r.status_code, 403)

    def test_org_admin_saves_and_lists_org_scope(self):
        org = Organization.objects.create(name="Acme", slug=f"acme-{uuid.uuid4().hex[:6]}")
        Membership.objects.create(user=self.user, org=org, role=Membership.Role.ADMIN)
        r = self._post("slide_theme_save", {"scope": "org", "theme": {"label": "OrgT"}})
        self.assertEqual(r.status_code, 200)
        tid = r.json()["theme"]["id"]
        by_id = {t["id"]: t for t in self._list()["themes"]}
        self.assertEqual(by_id[tid]["scope"], "org")
        self.assertEqual(self._post("slide_theme_set_default", {"scope": "org", "id": tid}).status_code, 200)
        self.assertEqual(self._list()["org_default"], tid)
        self.assertTrue(self._list()["is_org_admin"])
