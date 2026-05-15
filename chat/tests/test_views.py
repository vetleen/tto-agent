"""Tests for chat views."""

import json
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from chat.models import ChatCanvas, ChatThread
from documents.models import DataRoom, DataRoomDocument
from llm.models import LLMCallLog

User = get_user_model()


@override_settings(
    ALLOWED_HOSTS=["testserver"],
    LLM_ALLOWED_MODELS=["anthropic/claude-sonnet-4-5-20250929", "openai/gpt-5-mini"],
    LLM_DEFAULT_MODEL="anthropic/claude-sonnet-4-5-20250929",
)
class ChatHomeModelChoicesTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.client.force_login(self.user)

    def test_context_includes_model_choices_json(self):
        response = self.client.get(reverse("chat_home"))
        self.assertEqual(response.status_code, 200)
        self.assertIn("model_choices_json", response.context)
        choices = json.loads(response.context["model_choices_json"])
        self.assertIsInstance(choices, list)
        self.assertTrue(len(choices) > 0)
        # Each choice has the required keys
        for c in choices:
            self.assertIn("id", c)
            self.assertIn("display_name", c)
            self.assertIn("supports_thinking", c)

    def test_context_includes_default_model(self):
        response = self.client.get(reverse("chat_home"))
        self.assertIn("default_model", response.context)
        self.assertTrue(len(response.context["default_model"]) > 0)

    def test_context_includes_default_model_display(self):
        response = self.client.get(reverse("chat_home"))
        self.assertIn("default_model_display", response.context)
        self.assertTrue(len(response.context["default_model_display"]) > 0)

    def test_model_selector_rendered_in_html(self):
        response = self.client.get(reverse("chat_home"))
        self.assertContains(response, 'id="model-selector-btn"')
        self.assertContains(response, 'id="model-selector-dropdown"')
        self.assertContains(response, 'name="thinking-level"')


@override_settings(
    ALLOWED_HOSTS=["testserver"],
    LLM_ALLOWED_MODELS=["anthropic/claude-sonnet-4-5-20250929"],
    LLM_DEFAULT_MODEL="anthropic/claude-sonnet-4-5-20250929",
)
class ThreadCostTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="cost@example.com", password="testpass")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.client.force_login(self.user)

    def test_no_thread_returns_zero_cost(self):
        response = self.client.get(reverse("chat_home"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["thread_cost_usd"], 0.0)

    def test_thread_with_call_logs_returns_correct_cost(self):
        thread = ChatThread.objects.create(created_by=self.user)
        LLMCallLog.objects.create(
            model="test-model",
            prompt=[{"role": "user", "content": "Hi"}],
            raw_output="Hello!",
            status=LLMCallLog.Status.SUCCESS,
            conversation_id=str(thread.id),
            cost_usd=Decimal("0.00123456"),
        )
        LLMCallLog.objects.create(
            model="test-model",
            prompt=[{"role": "user", "content": "Bye"}],
            raw_output="Goodbye!",
            status=LLMCallLog.Status.SUCCESS,
            conversation_id=str(thread.id),
            cost_usd=Decimal("0.00200000"),
        )
        response = self.client.get(reverse("chat_home") + f"?thread={thread.id}")
        self.assertEqual(response.status_code, 200)
        self.assertAlmostEqual(response.context["thread_cost_usd"], 0.00323456, places=6)

    def test_thread_with_no_logs_returns_zero_cost(self):
        thread = ChatThread.objects.create(created_by=self.user)
        response = self.client.get(reverse("chat_home") + f"?thread={thread.id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["thread_cost_usd"], 0.0)


class CanvasSaveToDataRoomTests(TestCase):
    """Test that canvas_save_to_data_room requires owner access."""

    def setUp(self):
        self.owner = User.objects.create_user(email="owner@example.com", password="pass")
        self.owner.email_verified = True
        self.owner.save(update_fields=["email_verified"])
        self.other = User.objects.create_user(email="other@example.com", password="pass")
        self.other.email_verified = True
        self.other.save(update_fields=["email_verified"])
        self.data_room = DataRoom.objects.create(
            name="Owner Room", slug="owner-canvas", created_by=self.owner,
        )

    def test_owner_can_save_canvas_to_own_room(self):
        self.client.force_login(self.owner)
        thread = ChatThread.objects.create(created_by=self.owner)
        canvas = ChatCanvas.objects.create(
            thread=thread, title="Doc", content="# Hello",
        )
        thread.active_canvas_id = canvas.pk
        thread.save(update_fields=["active_canvas_id"])
        url = reverse("canvas_save_to_data_room", kwargs={"thread_id": thread.id})
        response = self.client.post(
            url,
            json.dumps({"data_room_id": self.data_room.pk}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(DataRoomDocument.objects.filter(data_room=self.data_room).exists())

    def test_non_owner_cannot_save_canvas(self):
        self.client.force_login(self.other)
        thread = ChatThread.objects.create(created_by=self.other)
        canvas = ChatCanvas.objects.create(
            thread=thread, title="Doc", content="# Hello",
        )
        thread.active_canvas_id = canvas.pk
        thread.save(update_fields=["active_canvas_id"])
        url = reverse("canvas_save_to_data_room", kwargs={"thread_id": thread.id})
        response = self.client.post(
            url,
            json.dumps({"data_room_id": self.data_room.pk}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 403)
        self.assertFalse(DataRoomDocument.objects.filter(data_room=self.data_room).exists())


@override_settings(
    ALLOWED_HOSTS=["testserver"],
    LLM_ALLOWED_MODELS=["openai/gpt-5-mini"],
    LLM_DEFAULT_MODEL="openai/gpt-5-mini",
)
class CanvasImportValidationTests(TestCase):
    """canvas_import must validate file type and size."""

    def setUp(self):
        self.user = User.objects.create_user(email="imp@example.com", password="pass")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.client.force_login(self.user)

    def test_rejects_non_docx_file(self):
        from django.core.files.uploadedfile import SimpleUploadedFile

        f = SimpleUploadedFile("evil.txt", b"hello", content_type="text/plain")
        url = reverse("canvas_import", kwargs={"thread_id": self.thread.id})
        response = self.client.post(url, {"file": f})
        self.assertEqual(response.status_code, 400)
        self.assertIn("docx", response.json()["error"].lower())

    def test_rejects_oversized_file(self):
        from django.core.files.uploadedfile import SimpleUploadedFile

        large = b"x" * (10 * 1024 * 1024 + 1)
        f = SimpleUploadedFile(
            "big.docx", large,
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        url = reverse("canvas_import", kwargs={"thread_id": self.thread.id})
        response = self.client.post(url, {"file": f})
        self.assertEqual(response.status_code, 400)
        self.assertIn("too large", response.json()["error"].lower())
