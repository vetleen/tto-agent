from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from llm_chat.models import ChatThread


User = get_user_model()


@override_settings(
    CHANNEL_LAYERS={
        "default": {
            "BACKEND": "channels.layers.InMemoryChannelLayer",
        }
    }
)
class ChatViewTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")
        self.client.force_login(self.user)

    def test_chat_view_requires_login(self):
        self.client.logout()
        response = self.client.get(reverse("chat"))
        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login", response["Location"])

    def test_chat_view_get_renders_template(self):
        response = self.client.get(reverse("chat"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "llm_chat/chat.html")
        self.assertIn("threads", response.context)
        self.assertIn("messages", response.context)

    @mock.patch("llm_chat.views.get_channel_layer")
    def test_chat_view_post_triggers_group_send(self, mock_get_layer):
        layer = mock.Mock()
        # async_to_sync expects an async callable; AsyncMock satisfies that.
        layer.group_send = mock.AsyncMock()
        mock_get_layer.return_value = layer

        response = self.client.post(
            reverse("chat"),
            data={"message": "Hello from test"},
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )

        # When no thread exists, a new one is created and we get 201
        # When a thread exists, we get 204 (or 200 for non-AJAX)
        self.assertIn(response.status_code, [200, 201, 204])
        # Ensure group_send was awaited with expected event data
        self.assertTrue(layer.group_send.await_count >= 1)
        args, kwargs = layer.group_send.await_args
        self.assertIn("thread_", args[0])
        event = args[1]
        self.assertEqual(event["type"], "chat.start_stream")
        self.assertEqual(event["content"], "Hello from test")

