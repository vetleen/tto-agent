"""Tests for chat views."""

import json
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from accounts.models import Membership, Organization
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
    """Test that canvas_save_to_data_room requires write (owner) access."""

    def setUp(self):
        self.owner = User.objects.create_user(email="owner@example.com", password="pass")
        self.owner.email_verified = True
        self.owner.save(update_fields=["email_verified"])
        self.colleague = User.objects.create_user(email="colleague@example.com", password="pass")
        self.colleague.email_verified = True
        self.colleague.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="TestOrg", slug="testorg-canvas")
        Membership.objects.create(user=self.owner, org=self.org)
        Membership.objects.create(user=self.colleague, org=self.org)
        self.data_room = DataRoom.objects.create(
            name="Shared", slug="shared-canvas", created_by=self.owner, is_shared=True,
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

    def test_shared_member_cannot_save_canvas_to_shared_room(self):
        """Non-owner org members should not be able to write to shared rooms."""
        self.client.force_login(self.colleague)
        thread = ChatThread.objects.create(created_by=self.colleague)
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
class SharedDataRoomVisibilityTests(TestCase):
    """Shared data rooms must appear in the chat attach dropdown."""

    def setUp(self):
        self.owner = User.objects.create_user(email="owner@dr.com", password="pass")
        self.member = User.objects.create_user(email="member@dr.com", password="pass")
        self.member.email_verified = True
        self.member.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="Org", slug="org-vis")
        Membership.objects.create(user=self.owner, org=self.org)
        Membership.objects.create(user=self.member, org=self.org)
        self.shared_room = DataRoom.objects.create(
            name="Shared Room", slug="shared-vis", created_by=self.owner, is_shared=True,
        )
        self.private_room = DataRoom.objects.create(
            name="Private Room", slug="private-vis", created_by=self.owner, is_shared=False,
        )
        self.own_room = DataRoom.objects.create(
            name="My Room", slug="my-vis", created_by=self.member,
        )

    def test_data_rooms_for_user_includes_shared(self):
        self.client.force_login(self.member)
        response = self.client.get(reverse("chat_data_rooms_api"))
        rooms = response.json()["data_rooms"]
        room_names = {r["name"] for r in rooms}
        self.assertIn("My Room", room_names)
        self.assertIn("Shared Room", room_names)
        self.assertNotIn("Private Room", room_names)

    def test_chat_home_context_includes_shared(self):
        self.client.force_login(self.member)
        response = self.client.get(reverse("chat_home"))
        room_names = {r["name"] for r in response.context["data_rooms"]}
        self.assertIn("My Room", room_names)
        self.assertIn("Shared Room", room_names)
        self.assertNotIn("Private Room", room_names)
