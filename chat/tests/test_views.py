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

    def test_all_messages_shown_when_no_user_turns(self):
        """A thread with no visible user messages forms zero turns, so the whole
        thread loads (turn-based paging has no flat message cap). Chronological
        order, newest last, and no "Show earlier messages" control."""
        for i in range(120):
            ChatMessage.objects.create(
                thread=self.thread, role="assistant", content=f"msg-{i:03d}",
            )

        response, messages = self._get_messages()
        contents = [m.content for m in messages]
        self.assertEqual(len(contents), 120)
        self.assertEqual(contents[0], "msg-000")
        self.assertEqual(contents[-1], "msg-119")
        self.assertEqual(contents, sorted(contents))
        self.assertFalse(response.context["history_has_more"])


@override_settings(ALLOWED_HOSTS=["testserver"])
class ChatHistoryPaginationTests(TestCase):
    """Turn-based history paging: newest 20 turns initially, "Show earlier
    messages" prepends the previous 20. A turn = a visible user message plus its
    hidden-assistant narration and final answer (which ride along, never split)."""

    PAGE = 20  # keep in sync with load_thread_message_page(turns=...)

    def setUp(self):
        from datetime import timedelta

        from django.utils import timezone

        self.user = User.objects.create_user(email="page@example.com", password="testpass")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.client.force_login(self.user)
        self.thread = ChatThread.objects.create(created_by=self.user)
        # Deterministic, strictly-increasing created_at (auto_now_add otherwise
        # stamps "now" for every row, scrambling turn boundaries).
        self._clock = timezone.now() - timedelta(days=3)
        self._tick = timedelta(seconds=1)

    def _msg(self, role, content, **kw):
        """Create one message at the next tick and pin its created_at."""
        m = ChatMessage.objects.create(
            thread=self.thread, role=role, content=content, **kw
        )
        self._clock = self._clock + self._tick
        ChatMessage.objects.filter(pk=m.pk).update(created_at=self._clock)
        m.created_at = self._clock
        return m

    def _turn(self, n, *, narration=1):
        """Emit one turn: visible user msg → `narration` hidden-assistant narration
        blocks (with content, so they survive as "Thought further") → final answer."""
        u = self._msg("user", f"user-{n}")
        for k in range(narration):
            self._msg(
                "assistant", f"narr-{n}-{k}", is_hidden_from_user=True,
                metadata={"tool_calls": [{"id": "c", "name": "web_search", "arguments": {}}]},
            )
        a = self._msg("assistant", f"answer-{n}")
        return u, a

    def _home(self):
        resp = self.client.get(reverse("chat_home"), {"thread": str(self.thread.id)})
        self.assertEqual(resp.status_code, 200)
        return resp

    def _older(self, before, thread=None):
        params = {"before": before} if before else {}
        return self.client.get(
            reverse("chat_load_older_messages",
                    kwargs={"thread_id": (thread or self.thread).id}),
            params,
        )

    def test_initial_page_is_newest_20_turns(self):
        for n in range(1, 26):  # turns 1 (oldest) .. 25 (newest)
            self._turn(n)
        resp = self._home()
        self.assertTrue(resp.context["history_has_more"])
        self.assertTrue(resp.context["history_cursor"])
        contents = {m.content for m in resp.context["messages"]}
        # Newest 20 turns are 6..25; turn 5 and older fall off.
        self.assertIn("user-25", contents)
        self.assertIn("user-6", contents)
        self.assertNotIn("user-5", contents)
        self.assertNotIn("answer-5", contents)

    def test_narration_rides_with_its_turn_and_never_splits(self):
        for n in range(1, 22):  # 21 turns => paginates
            self._turn(n, narration=2)
        resp = self._home()
        contents = {m.content for m in resp.context["messages"]}
        # Oldest turn on the initial page (turn 2) keeps ALL of its messages.
        for c in ("user-2", "narr-2-0", "narr-2-1", "answer-2"):
            self.assertIn(c, contents)
        # The older page (turn 1) must not contain any of turn 2's messages.
        older = self._older(resp.context["history_cursor"]).json()
        self.assertIn("user-1", older["html"])
        self.assertIn("narr-1-1", older["html"])
        self.assertNotIn("user-2", older["html"])
        self.assertNotIn("narr-2-0", older["html"])

    def test_show_more_walks_back_in_pages_of_20(self):
        for n in range(1, 46):  # 45 turns
            self._turn(n)
        resp = self._home()
        c1 = resp.context["history_cursor"]
        self.assertTrue(resp.context["history_has_more"])

        page2 = self._older(c1).json()
        self.assertEqual(set(page2.keys()), {"html", "cursor", "has_more", "compressed_above"})
        self.assertTrue(page2["has_more"])
        self.assertIn("user-25", page2["html"])   # turns 6..25
        self.assertIn("user-6", page2["html"])
        self.assertNotIn("user-26", page2["html"])
        self.assertNotIn("user-5", page2["html"])

        page3 = self._older(page2["cursor"]).json()
        self.assertFalse(page3["has_more"])
        self.assertEqual(page3["cursor"], "")
        self.assertIn("user-1", page3["html"])     # turns 1..5
        self.assertIn("user-5", page3["html"])
        self.assertNotIn("user-6", page3["html"])

    def test_leading_non_user_messages_land_on_the_final_page(self):
        # A visible assistant message before any user turn (e.g. a seeded opener).
        self._msg("assistant", "lead-a")
        for n in range(1, 22):  # 21 turns => paginates
            self._turn(n)
        resp = self._home()
        older = self._older(resp.context["history_cursor"]).json()
        self.assertFalse(older["has_more"])
        self.assertIn("lead-a", older["html"])
        self.assertIn("user-1", older["html"])

    def test_single_giant_turn_loads_atomically(self):
        self._msg("user", "user-1")
        for k in range(50):
            self._msg("assistant", f"narr-1-{k}", is_hidden_from_user=True,
                      metadata={"tool_calls": [{"id": "c", "name": "x", "arguments": {}}]})
        self._msg("assistant", "answer-1")
        resp = self._home()
        self.assertFalse(resp.context["history_has_more"])
        contents = {m.content for m in resp.context["messages"]}
        self.assertIn("user-1", contents)
        self.assertIn("answer-1", contents)
        self.assertIn("narr-1-49", contents)
        self.assertEqual(len(resp.context["messages"]), 52)

    def test_few_turns_show_no_control(self):
        for n in range(1, 16):  # 15 turns
            self._turn(n)
        resp = self._home()
        self.assertFalse(resp.context["history_has_more"])
        self.assertEqual(resp.context["history_cursor"], "")
        self.assertNotContains(resp, 'id="show-more-btn"')

    def test_hidden_user_message_does_not_start_a_turn(self):
        for n in range(1, 21):  # exactly 20 visible turns
            self._turn(n)
        # Sprinkle hidden user messages (e.g. sub-agent result injections) — they
        # must not count as turn starts, so paging stays at 20 turns (no overflow).
        for n in range(1, 6):
            self._msg("user", f"[sub-agent result {n}]", is_hidden_from_user=True)
        resp = self._home()
        self.assertFalse(resp.context["history_has_more"])
        contents = {m.content for m in resp.context["messages"]}
        self.assertIn("user-1", contents)
        self.assertIn("user-20", contents)

    def test_compression_divider_moves_from_top_to_inline_across_pages(self):
        for n in range(1, 26):  # 25 turns
            self._turn(n)
        # Summary boundary in an OLD turn (turn 3), below the newest-20 window.
        boundary = ChatMessage.objects.filter(
            thread=self.thread, content="answer-3"
        ).first()
        self.thread.summary_up_to_message_id = boundary.pk
        self.thread.save(update_fields=["summary_up_to_message_id"])

        resp = self._home()
        # Newest page is entirely after the boundary → top divider, no inline one.
        self.assertTrue(resp.context["history_compressed_above"])

        # The older page that contains turn 3 carries the boundary inline instead.
        older = self._older(resp.context["history_cursor"]).json()
        self.assertFalse(older["compressed_above"])
        self.assertIn("data-compression-divider", older["html"])

    def test_endpoint_requires_ownership(self):
        other = User.objects.create_user(email="other@example.com", password="x")
        other_thread = ChatThread.objects.create(created_by=other)
        resp = self._older("", thread=other_thread)
        self.assertEqual(resp.status_code, 404)

    def test_endpoint_rejects_bad_cursor(self):
        resp = self._older("not-a-date")
        self.assertEqual(resp.status_code, 400)


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

    def test_attach_accept_offers_photo_picker(self):
        # The attachment file input must advertise image types + image/* so iOS
        # Safari offers the photo library instead of the camera/video flow.
        response = self.client.get(reverse("chat_home"))
        accept = response.context["attach_accept"]
        self.assertIn("image/*", accept)
        self.assertIn(".png", accept)
        self.assertIn(".docx", accept)
        # Chat consumes images/PDF/docx/text only — no audio.
        self.assertNotIn(".mp3", accept)

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
