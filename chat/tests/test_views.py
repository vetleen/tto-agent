"""Tests for chat views."""

import json
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from chat.models import ChatCanvas, ChatMessage, ChatThread
from documents.models import DataRoom, DataRoomDocument
from llm.models import LLMCallLog

User = get_user_model()


@override_settings(ALLOWED_HOSTS=["testserver"])
class ChatHomeIntermediateMessagesTests(TestCase):
    """Hidden tool-loop assistant messages with narration/thinking render as
    collapsed blocks on reload; empty ones stay hidden."""

    def setUp(self):
        self.user = User.objects.create_user(email="inter@example.com", password="testpass")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.client.force_login(self.user)
        self.thread = ChatThread.objects.create(created_by=self.user)

    def _get_messages(self):
        response = self.client.get(
            reverse("chat_home"), {"thread": str(self.thread.id)},
        )
        self.assertEqual(response.status_code, 200)
        return response, list(response.context["messages"])

    def test_narration_message_included_and_flagged(self):
        ChatMessage.objects.create(
            thread=self.thread, role="user", content="Search for X",
        )
        narration = ChatMessage.objects.create(
            thread=self.thread, role="assistant",
            content="Let me search the documents...",
            metadata={"tool_calls": [{"id": "c1", "name": "web_search", "arguments": {}}]},
            is_hidden_from_user=True,
        )
        ChatMessage.objects.create(
            thread=self.thread, role="assistant", content="Final answer.",
        )

        response, messages = self._get_messages()
        included_pks = [m.pk for m in messages]
        self.assertIn(narration.pk, included_pks)
        narration_ctx = next(m for m in messages if m.pk == narration.pk)
        self.assertTrue(narration_ctx.is_intermediate)
        self.assertContains(response, "Thought further")

    def test_empty_tool_loop_message_excluded(self):
        ChatMessage.objects.create(
            thread=self.thread, role="user", content="Search for X",
        )
        empty_hidden = ChatMessage.objects.create(
            thread=self.thread, role="assistant", content="",
            metadata={"tool_calls": [{"id": "c1", "name": "web_search", "arguments": {}}]},
            is_hidden_from_user=True,
        )

        _, messages = self._get_messages()
        self.assertNotIn(empty_hidden.pk, [m.pk for m in messages])

    def test_hidden_tool_and_user_messages_stay_hidden(self):
        ChatMessage.objects.create(
            thread=self.thread, role="tool", content="{\"results\": []}",
            tool_call_id="c1", is_hidden_from_user=True,
        )
        ChatMessage.objects.create(
            thread=self.thread, role="user",
            content="[Sub-agent result: abc12345]\nFindings.",
            metadata={"source": "subagent"}, is_hidden_from_user=True,
        )

        _, messages = self._get_messages()
        self.assertEqual(messages, [])

    def test_thinking_metadata_renders_on_visible_message(self):
        ChatMessage.objects.create(
            thread=self.thread, role="user", content="Question",
        )
        ChatMessage.objects.create(
            thread=self.thread, role="assistant", content="Answer.",
            metadata={"thinking": "Deliberating carefully."},
        )

        response, _ = self._get_messages()
        self.assertContains(response, "Deliberating carefully.")
        self.assertContains(response, "data-server-thinking")


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

    def test_csp_header_enforced_with_nonce(self):
        """The page carries a strict, nonce-based Content-Security-Policy and its
        inline scripts carry the matching nonce."""
        response = self.client.get(reverse("chat_home"))
        csp = response.headers.get("Content-Security-Policy", "")
        self.assertIn("script-src", csp)
        self.assertIn("'self'", csp)
        self.assertIn("object-src 'none'", csp)
        self.assertIn("base-uri 'self'", csp)
        # script-src must not fall back to unsafe-inline (that would defeat the policy)
        script_src = next(d for d in csp.split(";") if d.strip().startswith("script-src"))
        self.assertNotIn("unsafe-inline", script_src)
        self.assertIn("'nonce-", script_src)
        # Inline scripts in the rendered page carry a nonce attribute.
        self.assertContains(response, 'nonce="')


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
