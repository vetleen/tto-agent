"""Tests for the image-generation preference cascade + tool gating."""

from __future__ import annotations

from unittest.mock import patch

from django.test import TestCase, override_settings

from accounts.models import Membership, Organization

_ALLOWED_LLM = ["openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5.4-nano"]
_TOOLS = {"document_search": None, "chat_generate_image": None}


def _create_user(email="img@example.com"):
    from django.contrib.auth import get_user_model

    return get_user_model().objects.create_user(email=email, password="testpass123")


@override_settings(
    LLM_DEFAULT_MODEL="openai/gpt-5.4",
    LLM_DEFAULT_MID_MODEL="openai/gpt-5.4-mini",
    LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5.4-nano",
    IMAGE_ALLOWED_MODELS=["gemini/gemini-2.5-flash-image"],
    IMAGE_DEFAULT_MODEL="gemini/gemini-2.5-flash-image",
)
class ImagePreferenceCascadeTests(TestCase):
    def setUp(self):
        self._p1 = patch("llm.service.policies.get_allowed_models", return_value=list(_ALLOWED_LLM))
        self._p2 = patch("llm.tools.registry.get_tool_registry")
        self._p1.start()
        reg = self._p2.start()
        reg.return_value.list_tools.return_value = dict(_TOOLS)
        self.addCleanup(self._p1.stop)
        self.addCleanup(self._p2.stop)

    def _prefs(self, user):
        from core.preferences import get_preferences

        return get_preferences(user)

    def test_enabled_no_org(self):
        prefs = self._prefs(_create_user())
        self.assertEqual(prefs.image_model, "gemini/gemini-2.5-flash-image")
        self.assertEqual(prefs.allowed_image_models, ["gemini/gemini-2.5-flash-image"])
        self.assertIn("chat_generate_image", prefs.allowed_tools)

    def test_org_empty_list_disables(self):
        user = _create_user("img2@example.com")
        org = Organization.objects.create(
            name="O", slug="o", preferences={"allowed_image_models": []}
        )
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)
        prefs = self._prefs(user)
        self.assertEqual(prefs.image_model, "")
        self.assertNotIn("chat_generate_image", prefs.allowed_tools)

    def test_user_override(self):
        # Two allowed models so the user's pick is meaningfully different.
        with override_settings(
            IMAGE_ALLOWED_MODELS=["gemini/gemini-2.5-flash-image", "gemini/gemini-3-pro-image"]
        ):
            user = _create_user("img3@example.com")
            from accounts.services import update_user_preferences

            update_user_preferences(
                user, lambda p: p.__setitem__("image_models", {"default": "gemini/gemini-3-pro-image"})
            )
            prefs = self._prefs(user)
            self.assertEqual(prefs.image_model, "gemini/gemini-3-pro-image")


@override_settings(
    LLM_DEFAULT_MODEL="openai/gpt-5.4",
    IMAGE_ALLOWED_MODELS=[],
    IMAGE_DEFAULT_MODEL="",
)
class ImageDisabledBySystemTests(TestCase):
    @patch("llm.service.policies.get_allowed_models", return_value=list(_ALLOWED_LLM))
    @patch("llm.tools.registry.get_tool_registry")
    def test_withheld_when_system_unset(self, mock_reg, _mock_allowed):
        mock_reg.return_value.list_tools.return_value = dict(_TOOLS)
        from core.preferences import get_preferences

        prefs = get_preferences(_create_user("img4@example.com"))
        self.assertEqual(prefs.image_model, "")
        self.assertEqual(prefs.allowed_image_models, [])
        self.assertNotIn("chat_generate_image", prefs.allowed_tools)
