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
    IMAGE_ALLOWED_MODELS=["gemini/gemini-3.1-flash-lite-image", "gemini/gemini-3.1-flash-image"],
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
            json.dumps({"allowed_image_models": ["gemini/gemini-3.1-flash-lite-image"]}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.org.refresh_from_db()
        self.assertEqual(
            self.org.preferences["allowed_image_models"], ["gemini/gemini-3.1-flash-lite-image"]
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

    def test_retired_model_stored_as_replacement(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"allowed_image_models": [
                "gemini/gemini-2.5-flash-image", "gemini/gemini-3.1-flash-lite-image",
            ]}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.org.refresh_from_db()
        self.assertEqual(
            self.org.preferences["allowed_image_models"], ["gemini/gemini-3.1-flash-lite-image"]
        )


@override_settings(
    ALLOWED_HOSTS=["testserver"],
    IMAGE_ALLOWED_MODELS=["gemini/gemini-3.1-flash-lite-image", "gemini/gemini-3.1-flash-image"],
)
class OrgSettingsImageSectionTests(TestCase):
    def setUp(self):
        self.admin_user = _verified("imgpage-admin@example.com")
        self.org = Organization.objects.create(
            name="ImgPageOrg",
            slug="imgpageorg",
            preferences={
                "allowed_image_models": ["gemini/gemini-2.5-flash-image"],
                "image_models": {"default": "gemini/gemini-2.5-flash-image"},
            },
        )
        Membership.objects.create(user=self.admin_user, org=self.org, role=Membership.Role.ADMIN)
        self.client.login(email=self.admin_user.email, password="test-pass-123")

    def test_context_canonicalizes_retired_prefs_and_populates_default_select(self):
        response = self.client.get(reverse("accounts:org_settings"))
        self.assertEqual(response.status_code, 200)
        lite = "gemini/gemini-3.1-flash-lite-image"
        self.assertEqual(response.context["org_allowed_image"], [lite])
        self.assertEqual(response.context["effective_image_allowed"], [lite])
        self.assertEqual(response.context["org_image_default"], lite)
        # Default dropdown renders the display name, selected.
        self.assertContains(
            response,
            f'<option value="{lite}" selected>Gemini 3.1 Flash Lite Image (Nano Banana 2 Lite)</option>',
            html=True,
        )


@override_settings(
    ALLOWED_HOSTS=["testserver"],
    IMAGE_ALLOWED_MODELS=["gemini/gemini-3.1-flash-lite-image", "gemini/gemini-3.1-flash-image"],
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
            json.dumps({"model": "gemini/gemini-3.1-flash-lite-image"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.org.refresh_from_db()
        self.assertEqual(
            self.org.preferences["image_models"]["default"], "gemini/gemini-3.1-flash-lite-image"
        )

    def test_reject_model_not_allowed(self):
        # Narrow the org allow-list, then try to default to an excluded model.
        self.org.preferences = {"allowed_image_models": ["gemini/gemini-3.1-flash-lite-image"]}
        self.org.save(update_fields=["preferences"])
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"model": "gemini/gemini-3.1-flash-image"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_retired_default_stored_as_replacement(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"model": "gemini/gemini-2.5-flash-image"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.org.refresh_from_db()
        self.assertEqual(
            self.org.preferences["image_models"]["default"], "gemini/gemini-3.1-flash-lite-image"
        )

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
    IMAGE_ALLOWED_MODELS=["gemini/gemini-3.1-flash-lite-image"],
)
class UserImageModelUpdateTests(TestCase):
    def setUp(self):
        self.password = "test-pass-123"
        self.user = _verified("imguser@example.com")
        self.url = reverse("accounts:preferences_image_model_update")

    def test_rejects_persistent_image_model(self):
        settings_obj, _ = UserSettings.objects.get_or_create(user=self.user)
        settings_obj.preferences = {
            "image_models": {"default": "gemini/gemini-3.1-flash-image"}
        }
        settings_obj.save()
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"model": "gemini/gemini-3.1-flash-lite-image"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 403)
        settings_obj.refresh_from_db()
        self.assertEqual(
            settings_obj.preferences["image_models"]["default"],
            "gemini/gemini-3.1-flash-image",
        )

    def test_requires_login(self):
        response = self.client.post(
            self.url,
            json.dumps({"model": "gemini/gemini-3.1-flash-lite-image"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 302)
