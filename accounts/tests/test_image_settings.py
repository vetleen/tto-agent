"""Tests for the image-generation settings endpoints (org + user)."""

from __future__ import annotations

import json

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from accounts.models import Membership, Organization, UserSettings

User = get_user_model()


def _verified(email):
    u = User.objects.create_user(email=email, password="test-pass-123")
    u.email_verified = True
    u.save(update_fields=["email_verified"])
    return u


@override_settings(
    ALLOWED_HOSTS=["testserver"],
    IMAGE_ALLOWED_MODELS=["gemini/gemini-2.5-flash-image", "gemini/gemini-3-pro-image"],
)
class OrgAllowedImageModelsUpdateTests(TestCase):
    def setUp(self):
        self.password = "test-pass-123"
        self.admin_user = _verified("imgadmin@example.com")
        self.member_user = _verified("imgmember@example.com")
        self.org = Organization.objects.create(name="ImgOrg", slug="imgorg")
        Membership.objects.create(user=self.admin_user, org=self.org, role=Membership.Role.ADMIN)
        Membership.objects.create(user=self.member_user, org=self.org, role=Membership.Role.MEMBER)
        self.url = reverse("accounts:org_allowed_image_models_update")

    def test_admin_sets_allowed(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"allowed_image_models": ["gemini/gemini-2.5-flash-image"]}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.org.refresh_from_db()
        self.assertEqual(
            self.org.preferences["allowed_image_models"], ["gemini/gemini-2.5-flash-image"]
        )

    def test_reject_model_not_in_system(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"allowed_image_models": ["not-a-model"]}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_requires_admin(self):
        self.client.login(email=self.member_user.email, password=self.password)
        response = self.client.post(
            self.url, json.dumps({"allowed_image_models": []}), content_type="application/json"
        )
        self.assertEqual(response.status_code, 403)

    def test_requires_login(self):
        response = self.client.post(
            self.url, json.dumps({"allowed_image_models": []}), content_type="application/json"
        )
        self.assertEqual(response.status_code, 302)

    def test_empty_list_disables(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url, json.dumps({"allowed_image_models": []}), content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        self.org.refresh_from_db()
        self.assertEqual(self.org.preferences["allowed_image_models"], [])


@override_settings(
    ALLOWED_HOSTS=["testserver"],
    IMAGE_ALLOWED_MODELS=["gemini/gemini-2.5-flash-image", "gemini/gemini-3-pro-image"],
)
class OrgImageModelUpdateTests(TestCase):
    def setUp(self):
        self.password = "test-pass-123"
        self.admin_user = _verified("imgdef-admin@example.com")
        self.org = Organization.objects.create(name="ImgDefOrg", slug="imgdeforg")
        Membership.objects.create(user=self.admin_user, org=self.org, role=Membership.Role.ADMIN)
        self.url = reverse("accounts:org_image_model_update")

    def test_admin_sets_default(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"model": "gemini/gemini-2.5-flash-image"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.org.refresh_from_db()
        self.assertEqual(
            self.org.preferences["image_models"]["default"], "gemini/gemini-2.5-flash-image"
        )

    def test_reject_model_not_allowed(self):
        # Narrow the org allow-list, then try to default to an excluded model.
        self.org.preferences = {"allowed_image_models": ["gemini/gemini-2.5-flash-image"]}
        self.org.save(update_fields=["preferences"])
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"model": "gemini/gemini-3-pro-image"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_admin_clears_default(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url, json.dumps({"model": ""}), content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        self.org.refresh_from_db()
        self.assertIsNone(self.org.preferences["image_models"]["default"])


@override_settings(
    ALLOWED_HOSTS=["testserver"],
    IMAGE_ALLOWED_MODELS=["gemini/gemini-2.5-flash-image"],
)
class UserImageModelUpdateTests(TestCase):
    def setUp(self):
        self.password = "test-pass-123"
        self.user = _verified("imguser@example.com")
        self.url = reverse("accounts:preferences_image_model_update")

    def test_set_allowed_model(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"model": "gemini/gemini-2.5-flash-image"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        settings = UserSettings.objects.get(user=self.user)
        self.assertEqual(
            settings.preferences["image_models"]["default"], "gemini/gemini-2.5-flash-image"
        )

    def test_reject_not_allowed(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(
            self.url, json.dumps({"model": "gemini/gemini-3-pro-image"}), content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
