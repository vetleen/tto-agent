"""Tests for chat views."""

import json
from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from chat.models import ChatCanvas, ChatMessage, ChatThread
from documents.models import DataRoom, DataRoomDocument
from llm.models import LLMCallLog

User = get_user_model()


@override_settings(ALLOWED_HOSTS=["testserver"])
class ChatHomeIntermediateMessagesTests(TestCase):
    """Hidden tool-loop assistant messages fold into their turn's activity
    timeline on reload; empty ones stay hidden."""

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

    def test_narration_folds_into_final_answer_activity(self):
        ChatMessage.objects.create(
            thread=self.thread, role="user", content="Search for X",
        )
        narration = ChatMessage.objects.create(
            thread=self.thread, role="assistant",
            content="Let me search the documents...",
            metadata={"tool_calls": [{"id": "c1", "name": "web_search", "arguments": {}}]},
            is_hidden_from_user=True,
        )
        final = ChatMessage.objects.create(
            thread=self.thread, role="assistant", content="Final answer.",
        )

        response, messages = self._get_messages()
        self.assertNotIn(narration.pk, [m.pk for m in messages])
        final_ctx = next(m for m in messages if m.pk == final.pk)
        self.assertEqual(
            [r["kind"] for r in final_ctx.activity["rows"]], ["narration", "tools"],
        )
        self.assertEqual(final_ctx.activity["rows"][0]["snippet"], "“Let me search the documents...”")
        self.assertFalse(final_ctx.activity["collapsed"])
        self.assertContains(response, "“Let me search the documents...”")
        self.assertNotContains(response, "Thought further")
        self.assertContains(response, "wmsg__answer")

    def test_empty_tool_loop_message_excluded(self):
        ChatMessage.objects.create(
            thread=self.thread, role="user", content="Search for X",
        )
        empty_hidden = ChatMessage.objects.create(
            thread=self.thread, role="assistant", content="", metadata={},
            is_hidden_from_user=True,
        )

        _, messages = self._get_messages()
        self.assertNotIn(empty_hidden.pk, [m.pk for m in messages])

    def test_tool_only_round_renders_single_tool_group(self):
        """A round with tool calls but no thinking/narration is kept, and a lone
        call's group header is the tool's own completion label."""
        ChatMessage.objects.create(
            thread=self.thread, role="user", content="Search for X",
        )
        tool_round = ChatMessage.objects.create(
            thread=self.thread, role="assistant", content="",
            metadata={"tool_calls": [
                {"id": "c1", "name": "web_search", "arguments": {"reason": "find sources"}},
            ]},
            is_hidden_from_user=True,
        )
        ChatMessage.objects.create(
            thread=self.thread, role="tool", content='{"results": []}',
            tool_call_id="c1", is_hidden_from_user=True,
        )

        response, messages = self._get_messages()
        # No final answer: the round renders as a timeline-only activity bubble.
        self.assertNotIn(tool_round.pk, [m.pk for m in messages])
        ctx = next(m for m in messages if m.role == "activity")
        self.assertEqual(
            ctx.activity["rows"],
            [{"kind": "tools", "failed": 0, "rows": [
                {"label": "Searched the web", "is_error": False, "reason": "find sources"},
            ]}],
        )
        self.assertContains(response, '<span class="tool-group-label">Searched the web</span>', html=False)
        self.assertContains(response, "(find sources)")

    def test_multi_tool_round_counts_and_flags_failures(self):
        """A batch says "Used n tools · k failed"; an unknown tool falls back to
        "Done"; a result with status=error is marked failed."""
        ChatMessage.objects.create(
            thread=self.thread, role="user", content="Do things",
        )
        ChatMessage.objects.create(
            thread=self.thread, role="assistant", content="",
            metadata={
                "thinking": "Plan the calls.",
                "tool_calls": [
                    {"id": "a", "name": "web_search", "arguments": {}},
                    {"id": "b", "name": "retired_tool_xyz", "arguments": {}},
                    {"id": "c", "name": "web_search", "arguments": {}},
                ],
            },
            is_hidden_from_user=True,
        )
        ChatMessage.objects.create(
            thread=self.thread, role="tool", content='{"results": []}',
            tool_call_id="a", is_hidden_from_user=True,
        )
        ChatMessage.objects.create(
            thread=self.thread, role="tool", content="plain text result",
            tool_call_id="b", is_hidden_from_user=True,
        )
        ChatMessage.objects.create(
            thread=self.thread, role="tool",
            content='{"status": "error", "message": "boom"}',
            tool_call_id="c", is_hidden_from_user=True,
        )

        response, messages = self._get_messages()
        ctx = next(m for m in messages if m.role == "activity")
        tools_row = ctx.activity["rows"][-1]
        self.assertEqual(
            [(r["label"], r["is_error"]) for r in tools_row["rows"]],
            [("Searched the web", False), ("Done", False), ("Searched the web", True)],
        )
        self.assertContains(response, "Used 3 tools · 1 failed")
        # Thinking renders before the tool group (live-stream order).
        body = response.content.decode()
        self.assertLess(body.index("Plan the calls."), body.index("Used 3 tools"))

    def test_tool_group_label_uses_end_label_for_result(self):
        """A dynamic completion label (end_label_for_result) survives reload."""
        from llm.tools.registry import get_tool_registry

        tool = get_tool_registry().get_tool("web_search")
        ChatMessage.objects.create(
            thread=self.thread, role="user", content="Search",
        )
        ChatMessage.objects.create(
            thread=self.thread, role="assistant", content="",
            metadata={"tool_calls": [{"id": "d1", "name": "web_search", "arguments": {}}]},
            is_hidden_from_user=True,
        )
        ChatMessage.objects.create(
            thread=self.thread, role="tool", content='{"results": [1, 2, 3]}',
            tool_call_id="d1", is_hidden_from_user=True,
        )
        with patch.object(type(tool), "end_label_for_result", return_value="Found 3 results"):
            _, messages = self._get_messages()
        ctx = next(m for m in messages if m.role == "activity")
        self.assertEqual(ctx.activity["rows"][0]["rows"][0]["label"], "Found 3 results")

    def test_subagent_panel_data_in_page(self):
        from chat.models import SubAgentRun

        SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="Dig into X",
            reason="Check the licence terms", status=SubAgentRun.Status.PENDING,
        )
        response, _ = self._get_messages()
        panel = response.context["subagent_panel"]
        self.assertEqual(len(panel["runs"]), 1)
        self.assertEqual(panel["runs"][0]["reason"], "Check the licence terms")
        self.assertEqual(panel["runs"][0]["state"], "queued")
        self.assertContains(response, 'id="subagent-panel-data"')
        self.assertEqual(response.context["active_subagent_count"], 1)
        self.assertEqual(response.context["waiting_subagent_count"], 1)

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
class TurnActivityFoldTests(TestCase):
    """A turn renders as one bubble: its tool-loop rounds fold into an activity
    timeline on the final answer; past TIMELINE_COLLAPSE_AFTER rows the timeline is
    one collapsed history block with adjacent tool batches merged."""

    def setUp(self):
        from datetime import timedelta

        from django.utils import timezone

        self.user = User.objects.create_user(email="fold@example.com", password="testpass")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.client.force_login(self.user)
        self.thread = ChatThread.objects.create(created_by=self.user)
        self._clock = timezone.now() - timedelta(days=1)
        self._call = 0

    def _msg(self, role, content="", seconds=1, **kw):
        from datetime import timedelta

        m = ChatMessage.objects.create(thread=self.thread, role=role, content=content, **kw)
        self._clock += timedelta(seconds=seconds)
        ChatMessage.objects.filter(pk=m.pk).update(created_at=self._clock)
        return m

    def _round(self, *, thinking="", narration="", tools=1, seconds=1):
        calls = []
        for _ in range(tools):
            self._call += 1
            calls.append({"id": f"c{self._call}", "name": "web_search", "arguments": {}})
        meta = {"tool_calls": calls}
        if thinking:
            meta["thinking"] = thinking
        m = self._msg("assistant", narration, seconds=seconds, metadata=meta, is_hidden_from_user=True)
        for c in calls:
            self._msg("tool", '{"results": []}', seconds=0, tool_call_id=c["id"], is_hidden_from_user=True)
        return m

    def _messages(self):
        response = self.client.get(reverse("chat_home"), {"thread": str(self.thread.id)})
        self.assertEqual(response.status_code, 200)
        return response, list(response.context["messages"])

    def test_short_turn_stays_flat_in_order(self):
        self._msg("user", "Q")
        self._round(thinking="t1", narration="Looking it up.")
        final = self._msg("assistant", "Answer.", metadata={"thinking": "t-final"})
        response, messages = self._messages()
        self.assertEqual([m.role for m in messages], ["user", "assistant"])
        act = next(m for m in messages if m.pk == final.pk).activity
        self.assertFalse(act["collapsed"])
        self.assertEqual(
            [r["kind"] for r in act["rows"]], ["thinking", "narration", "tools", "thinking"],
        )
        self.assertNotContains(response, '<div class="activity-history collapsed">')
        # One bubble per turn: the timeline precedes the answer inside it.
        body = response.content.decode()
        self.assertLess(body.index("t-final"), body.index("Answer."))

    def test_long_turn_collapses_merges_tool_batches_and_counts(self):
        self._msg("user", "Q")
        self._round(thinking="t1", tools=2, seconds=10)
        self._round(tools=1, seconds=10)       # tools right after tools → merged
        self._round(tools=1, seconds=10)       # merged again
        self._round(thinking="t2", tools=1, seconds=10)
        self._round(narration="Checking the PDF.", tools=1, seconds=10)
        final = self._msg("assistant", "Answer.", seconds=14, metadata={"thinking": "t3"})
        response, messages = self._messages()
        act = next(m for m in messages if m.pk == final.pk).activity
        self.assertTrue(act["collapsed"])
        self.assertEqual(
            [(r["kind"], len(r.get("rows", []))) for r in act["rows"]],
            [("thinking", 0), ("tools", 4), ("thinking", 0), ("tools", 1),
             ("narration", 0), ("tools", 1), ("thinking", 0)],
        )
        self.assertEqual((act["thoughts"], act["tools"]), (3, 6))
        self.assertEqual(act["summary"], "Worked for 1m 4s · 3 thoughts · 6 tools")
        self.assertContains(response, "activity-history collapsed")
        self.assertContains(response, "Worked for 1m 4s · 3 thoughts · 6 tools")
        self.assertContains(response, "Used 4 tools")

    def test_turn_without_user_anchor_omits_duration(self):
        """A seeded continuation (no visible user message) has no start to measure from."""
        self._msg("user", "Q")
        self._msg("assistant", "First answer.")
        self._msg("user", "[Sub-agent result]", is_hidden_from_user=True)
        for _ in range(6):
            self._round(thinking="t")
        final = self._msg("assistant", "Follow-up.")
        _, messages = self._messages()
        act = next(m for m in messages if m.pk == final.pk).activity
        self.assertEqual(act["summary"], "6 thoughts · 6 tools")

    def test_rounds_without_final_answer_become_activity_item(self):
        self._msg("user", "Q")
        self._round(thinking="t1")
        _, messages = self._messages()
        self.assertEqual([m.role for m in messages], ["user", "activity"])
        self.assertEqual([r["kind"] for r in messages[1].activity["rows"]], ["thinking", "tools"])

    def test_redacted_final_answer_gets_no_activity(self):
        self._msg("user", "Q")
        self._round(thinking="t1")
        redacted = self._msg("assistant", "x", is_redacted=True)
        _, messages = self._messages()
        self.assertEqual([m.role for m in messages], ["user", "activity", "assistant"])
        self.assertFalse(getattr(messages[2], "activity", None))
        self.assertEqual(messages[2].pk, redacted.pk)

    def test_answer_without_rounds_or_thinking_has_no_timeline(self):
        self._msg("user", "Q")
        final = self._msg("assistant", "Plain answer.")
        response, messages = self._messages()
        self.assertIsNone(next(m for m in messages if m.pk == final.pk).activity)
        self.assertContains(response, 'class="text-sm text-body wmsg__answer markdown-content')
        self.assertNotContains(response, 'data-server-thinking>')

    def test_narration_snippet_truncates(self):
        from chat.views import NARRATION_SNIPPET_CHARS, _narration_snippet

        snippet = _narration_snippet("word  \n" * 40)
        self.assertTrue(snippet.startswith("“word word"))
        self.assertTrue(snippet.endswith("…”"))
        self.assertLessEqual(len(snippet), NARRATION_SNIPPET_CHARS + 3)

    def test_format_duration(self):
        from chat.views import _format_duration

        self.assertEqual(_format_duration(45), "45s")
        self.assertEqual(_format_duration(134), "2m 14s")
        self.assertEqual(_format_duration(3780), "1h 3m")


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
        blocks (with content, so they survive as narration rows) → final answer."""
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
        # Oldest turn on the initial page (turn 2) keeps ALL of its messages.
        for c in ("user-2", "narr-2-0", "narr-2-1", "answer-2"):
            self.assertContains(resp, c)
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
        # The 50 rounds fold into the answer's activity: one user msg + one bubble.
        messages = resp.context["messages"]
        self.assertEqual([m.content for m in messages], ["user-1", "answer-1"])
        self.assertEqual(messages[1].activity["tools"], 50)
        self.assertContains(resp, "narr-1-49")

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
            self.assertIn("thinking_levels", c)
            self.assertIn("default_thinking_level", c)

    def test_context_includes_default_model(self):
        response = self.client.get(reverse("chat_home"))
        self.assertIn("default_model", response.context)
        self.assertTrue(len(response.context["default_model"]) > 0)

    @patch("core.preferences.get_preferences")
    def test_new_chat_uses_org_chat_feature_override(self, mock_preferences):
        from core.preferences import ResolvedPreferences

        mock_preferences.return_value = ResolvedPreferences(
            top_model="openai/gpt-5.6-sol",
            mid_model="openai/gpt-6-luna",
            cheap_model="openai/gpt-5.4-nano",
            allowed_models=["openai/gpt-5.6-sol", "openai/gpt-6-luna"],
            feature_models={"chat": "openai/gpt-6-luna"},
        )

        response = self.client.get(reverse("chat_home"))

        self.assertEqual(
            response.context["default_model"], "openai/gpt-6-luna"
        )

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

    def test_model_choices_carry_rating_not_blended_price(self):
        # The "$" rating comes from a backend-only blended price; the picker
        # gets the rating and the real list price (tooltip), never the blend.
        response = self.client.get(reverse("chat_home"))
        choices = json.loads(response.context["model_choices_json"])
        self.assertTrue(choices)
        for m in choices:
            with self.subTest(model=m["id"]):
                self.assertIn("price_level", m)
                self.assertFalse(any("blend" in k for k in m))

    @patch("core.preferences.get_preferences")
    def test_model_guide_covers_only_enabled_models(self, mock_preferences):
        from core.preferences import ResolvedPreferences

        enabled = ["anthropic/claude-opus-5-5", "openai/gpt-6-luna"]
        mock_preferences.return_value = ResolvedPreferences(
            top_model="anthropic/claude-opus-5-5",
            mid_model="openai/gpt-6-luna",
            cheap_model="openai/gpt-6-luna",
            allowed_models=enabled,
        )

        response = self.client.get(reverse("chat_home"))

        guide = response.context["model_guide"]
        self.assertEqual([m["id"] for m in guide["models"]], enabled)
        for bench in guide["benchmarks"]:
            self.assertLessEqual({r["model_id"] for r in bench["rows"]}, set(enabled))
        self.assertContains(response, 'id="model-guide-data"')
        self.assertContains(response, 'id="model-guide-btn"')
        self.assertContains(response, "js/model-guide.js")
        self.assertContains(response, 'id="reasoning-toggle"')
        self.assertContains(response, 'id="reasoning-level-picker"')
        self.assertContains(response, 'id="reasoning-level-select"')
        self.assertContains(response, 'style="min-width: 8rem"')

    def test_reasoning_preference_is_scoped_to_the_current_thread(self):
        response = self.client.get(reverse("chat_home"))
        self.assertContains(response, "chat_reasoning_levels_by_thread")
        self.assertContains(response, "chat_enabled_reasoning_levels_by_thread")
        self.assertNotContains(response, "chat_reasoning_levels_by_model")
        self.assertNotContains(response, "chat_enabled_reasoning_levels_by_model")

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
        with patch("documents.tasks.process_document_version_task.delay"):
            response = self.client.post(
                url,
                json.dumps({"data_room_id": self.data_room.pk}),
                content_type="application/json",
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["verdict"], "queued")  # scanned async, polled
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

    def test_rejects_unsupported_file(self):
        # Word, PDF and text are importable (CANVAS_IMPORT_KINDS); anything
        # else is refused before a canvas is created.
        from django.core.files.uploadedfile import SimpleUploadedFile

        f = SimpleUploadedFile("evil.exe", b"MZ\x90\x00", content_type="application/x-msdownload")
        url = reverse("canvas_import", kwargs={"thread_id": self.thread.id})
        response = self.client.post(url, {"file": f})
        self.assertEqual(response.status_code, 400)
        self.assertIn("docx", response.json()["error"].lower())
        self.assertFalse(ChatCanvas.objects.filter(thread=self.thread).exists())

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
