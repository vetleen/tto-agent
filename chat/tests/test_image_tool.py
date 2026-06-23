"""Tests for the chat_generate_image tool and thread-owned ImageAsset.

The image-generation service is faked; no Gemini SDK or network is touched.
"""

from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase

from chat.image_assets import image_token, store_thread_image
from chat.image_tools import ChatGenerateImageTool
from chat.models import ChatThread, ImageAsset
from llm.service.image_generation_service import ImageGenerationError, ImageGenerationResult
from llm.types.context import RunContext

User = get_user_model()


class _FakeService:
    def __init__(self, result=None, exc=None):
        self.calls = []
        self._result = result
        self._exc = exc

    def generate(self, prompt, model_id, context=None, input_images=None, aspect_ratio=None):
        self.calls.append({"prompt": prompt, "input_images": list(input_images or [])})
        if self._exc:
            raise self._exc
        return self._result or ImageGenerationResult(
            img_bytes=b"PNGBYTES",
            media_type="image/png",
            model=model_id,
            width=1024,
            height=1024,
            cost_usd=Decimal("0.039"),
            is_edit=bool(input_images),
        )


_ENABLED_PREFS = SimpleNamespace(
    image_model="gemini/gemini-2.5-flash-image",
    allowed_image_models=["gemini/gemini-2.5-flash-image"],
)


def _patch_prefs(prefs=_ENABLED_PREFS):
    return patch("core.preferences.get_preferences", return_value=prefs)


def _patch_service(fake):
    return patch(
        "llm.service.image_generation_service.get_image_generation_service",
        return_value=fake,
    )


class ChatGenerateImageToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="gen@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.ctx = RunContext.create(user_id=self.user.pk, conversation_id=str(self.thread.id))

    def _invoke(self, args):
        tool = ChatGenerateImageTool()
        tool.set_context(self.ctx)
        return json.loads(tool.invoke(args))

    def test_happy_path_creates_thread_asset(self):
        fake = _FakeService()
        with _patch_prefs(), _patch_service(fake):
            result = self._invoke({"prompt": "a friendly robot"})

        self.assertEqual(result["status"], "ok")
        self.assertIn("[[image:", result["token"])
        self.assertFalse(result["is_edit"])
        self.assertEqual(ImageAsset.objects.filter(thread=self.thread).count(), 1)
        asset = ImageAsset.objects.get(thread=self.thread)
        self.assertTrue(asset.blob)
        self.assertEqual(asset.created_by, self.user)
        # The image is surfaced to the model for this turn.
        self.assertEqual(len(self.ctx.pending_image_assets), 1)
        self.assertEqual(self.ctx.pending_image_assets[0]["media_type"], "image/png")

    def test_disabled_returns_error(self):
        disabled = SimpleNamespace(image_model="", allowed_image_models=[])
        with _patch_prefs(disabled):
            result = self._invoke({"prompt": "x"})
        self.assertEqual(result["status"], "error")
        self.assertIn("not enabled", result["message"].lower())
        self.assertEqual(ImageAsset.objects.filter(thread=self.thread).count(), 0)

    def test_provider_error_returns_error(self):
        fake = _FakeService(exc=ImageGenerationError("blocked by safety filters"))
        with _patch_prefs(), _patch_service(fake):
            result = self._invoke({"prompt": "x"})
        self.assertEqual(result["status"], "error")
        self.assertIn("blocked", result["message"].lower())
        self.assertEqual(ImageAsset.objects.filter(thread=self.thread).count(), 0)

    def test_input_images_access_control(self):
        # One asset the user owns, one owned by someone else.
        mine = store_thread_image(
            self.thread, img_bytes=b"mine", content_type="image/png", created_by=self.user
        )
        other_user = User.objects.create_user(email="other@test.com", password="pass")
        other_thread = ChatThread.objects.create(created_by=other_user)
        theirs = store_thread_image(
            other_thread, img_bytes=b"theirs", content_type="image/png", created_by=other_user
        )

        fake = _FakeService()
        with _patch_prefs(), _patch_service(fake):
            result = self._invoke(
                {
                    "prompt": "restyle these",
                    "input_images": [image_token(mine.id, ""), image_token(theirs.id, "")],
                }
            )

        self.assertEqual(result["status"], "ok")
        # Only the accessible image is forwarded to the model.
        self.assertEqual(len(fake.calls[0]["input_images"]), 1)
        self.assertTrue(result["is_edit"])


class ImageAssetConstraintTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="c@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def test_thread_owner_allowed(self):
        asset = ImageAsset.objects.create(thread=self.thread, content_type="image/png")
        self.assertIsNotNone(asset.pk)

    def test_zero_owner_rejected(self):
        from django.db import IntegrityError, transaction

        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                ImageAsset.objects.create(content_type="image/png")
