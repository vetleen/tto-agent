"""Tests for the Loops page + create/edit/pause/restart views."""

import json

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from chat.models import ChatThread, Loop
from documents.models import DataRoom

User = get_user_model()


class LoopViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            email="loops@example.com", password="pass123",
        )
        self.client.force_login(self.user)

    def _create(self, **overrides):
        payload = {
            "prompt": "Summarize new docs.",
            "history_mode": "fresh",
            "cadence_kind": "interval",
            "interval_value": 6, "interval_unit": "hours",
            "first_run_mode": "now",
        }
        payload.update(overrides)
        return self.client.post(
            reverse("loop_create"), data=json.dumps(payload),
            content_type="application/json",
        )

    def test_loops_page_renders(self):
        resp = self.client.get(reverse("loops_list"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Loops")

    def test_create_makes_loop_and_thread(self):
        resp = self._create()
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        loop = Loop.objects.get(id=data["loop_id"])
        self.assertEqual(loop.created_by, self.user)
        self.assertEqual(loop.interval_seconds, 6 * 3600)
        self.assertEqual(loop.history_mode, "fresh")
        self.assertTrue(ChatThread.objects.filter(id=data["thread_id"]).exists())
        # No run cap by default: the loop runs until paused.
        self.assertIsNone(data["max_runs"])
        self.assertIsNone(loop.max_runs)

    def test_create_with_max_runs_sets_cap(self):
        resp = self._create(max_runs=25)
        data = resp.json()
        self.assertEqual(data["max_runs"], 25)
        self.assertEqual(Loop.objects.get(id=data["loop_id"]).max_runs, 25)

    def test_create_zero_max_runs_is_unlimited(self):
        resp = self._create(max_runs=0)
        data = resp.json()
        self.assertIsNone(data["max_runs"])
        self.assertIsNone(Loop.objects.get(id=data["loop_id"]).max_runs)

    def test_create_requires_prompt(self):
        resp = self._create(prompt="   ")
        self.assertEqual(resp.status_code, 400)

    def test_create_links_data_room(self):
        dr = DataRoom.objects.create(created_by=self.user, name="Room A")
        resp = self._create(data_room_ids=[dr.pk])
        loop = Loop.objects.get(id=resp.json()["loop_id"])
        self.assertEqual(
            list(loop.thread.data_rooms.values_list("pk", flat=True)), [dr.pk],
        )

    def test_clock_cadence_create(self):
        resp = self._create(
            cadence_kind="clock", clock_frequency="weekdays", clock_time="09:00",
            first_run_mode="scheduled",
        )
        loop = Loop.objects.get(id=resp.json()["loop_id"])
        self.assertEqual(loop.cadence_kind, "clock")
        self.assertEqual(loop.clock_frequency, "weekdays")
        self.assertEqual(loop.clock_time.strftime("%H:%M"), "09:00")

    def test_edit_updates_prompt(self):
        loop_id = self._create().json()["loop_id"]
        resp = self.client.post(
            reverse("loop_edit", args=[loop_id]),
            data=json.dumps({
                "prompt": "New prompt.", "history_mode": "conversational",
                "cadence_kind": "interval", "interval_value": 2,
                "interval_unit": "hours", "first_run_mode": "now",
            }),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        loop = Loop.objects.get(id=loop_id)
        self.assertEqual(loop.prompt, "New prompt.")
        self.assertEqual(loop.history_mode, "conversational")
        self.assertEqual(loop.interval_seconds, 2 * 3600)

    def test_edit_keep_preserves_next_run(self):
        loop_id = self._create().json()["loop_id"]
        loop = Loop.objects.get(id=loop_id)
        original_next = loop.next_run
        resp = self.client.post(
            reverse("loop_edit", args=[loop_id]),
            data=json.dumps({
                "prompt": "Tweaked.", "history_mode": "fresh",
                "cadence_kind": "interval", "interval_value": 6,
                "interval_unit": "hours", "first_run_mode": "keep",
            }),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        loop.refresh_from_db()
        self.assertEqual(loop.prompt, "Tweaked.")
        self.assertEqual(loop.next_run, original_next)

    def test_edit_restart_reactivates_paused_loop(self):
        loop_id = self._create().json()["loop_id"]
        loop = Loop.objects.get(id=loop_id)
        loop.status = Loop.Status.PAUSED
        loop.runs_completed = 5
        loop.save(update_fields=["status", "runs_completed"])

        before = timezone.now()
        resp = self.client.post(
            reverse("loop_edit", args=[loop_id]),
            data=json.dumps({
                "prompt": "Restarted.", "history_mode": "fresh",
                "cadence_kind": "interval", "interval_value": 6,
                "interval_unit": "hours", "first_run_mode": "keep",
                "restart": True,
            }),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        loop.refresh_from_db()
        self.assertEqual(loop.status, Loop.Status.ACTIVE)
        self.assertEqual(loop.runs_completed, 0)  # run count starts over
        self.assertEqual(loop.prompt, "Restarted.")
        self.assertGreaterEqual(loop.next_run, before)  # keep + restart → now
        self.assertFalse(loop.running)

    def test_pause_and_restart(self):
        loop_id = self._create().json()["loop_id"]
        self.client.post(reverse("loop_pause", args=[loop_id]))
        loop = Loop.objects.get(id=loop_id)
        self.assertEqual(loop.status, Loop.Status.PAUSED)
        # Pausing archives the backing thread.
        self.assertTrue(loop.thread.is_archived)
        loop.runs_completed = 4
        loop.save(update_fields=["runs_completed"])

        # The user-facing revive action restarts: re-activate and reset the count.
        before = timezone.now()
        self.client.post(reverse("loop_restart", args=[loop_id]))
        loop.refresh_from_db()
        self.assertEqual(loop.status, Loop.Status.ACTIVE)
        self.assertEqual(loop.runs_completed, 0)
        self.assertGreaterEqual(loop.next_run, before)
        self.assertFalse(loop.running)
        # Restarting unarchives it again.
        loop.thread.refresh_from_db()
        self.assertFalse(loop.thread.is_archived)

    def test_opening_paused_loop_thread_keeps_it_archived(self):
        loop_id = self._create().json()["loop_id"]
        self.client.post(reverse("loop_pause", args=[loop_id]))
        loop = Loop.objects.get(id=loop_id)

        # Merely opening the thread to read results must not unarchive/revive it.
        self.client.get(reverse("chat_home") + f"?thread={loop.thread_id}")
        loop.refresh_from_db()
        self.assertEqual(loop.status, Loop.Status.PAUSED)
        self.assertTrue(loop.thread.is_archived)

    def test_open_loop_thread_clears_unread_and_nav_badge(self):
        loop_id = self._create().json()["loop_id"]
        loop = Loop.objects.get(id=loop_id)
        loop.last_result_at = timezone.now()
        loop.save(update_fields=["last_result_at"])
        self.assertTrue(loop.is_unread)

        # Nav badge count is 1 before the user opens the thread.
        resp = self.client.get(reverse("loops_list"))
        self.assertEqual(resp.context["loops_unread_count"], 1)

        # Opening the thread sets last_seen_at and clears the unread state.
        self.client.get(reverse("chat_home") + f"?thread={loop.thread_id}")
        loop.refresh_from_db()
        self.assertFalse(loop.is_unread)
        resp2 = self.client.get(reverse("loops_list"))
        self.assertEqual(resp2.context["loops_unread_count"], 0)

    def test_edit_deeplink_opens_edit(self):
        loop_id = self._create().json()["loop_id"]
        resp = self.client.get(reverse("loops_list") + f"?edit={loop_id}")
        self.assertEqual(resp.context["open_edit_id"], loop_id)
        # An unknown id is ignored (no modal).
        resp2 = self.client.get(
            reverse("loops_list") + "?edit=00000000-0000-0000-0000-000000000000"
        )
        self.assertEqual(resp2.context["open_edit_id"], "")

    def test_chat_home_exposes_thread_loop_id(self):
        loop_id = self._create().json()["loop_id"]
        loop = Loop.objects.get(id=loop_id)
        resp = self.client.get(reverse("chat_home") + f"?thread={loop.thread_id}")
        self.assertEqual(resp.context["thread_loop_id"], str(loop.id))
        # A plain (non-loop) thread exposes no loop id.
        plain = ChatThread.objects.create(created_by=self.user)
        resp2 = self.client.get(reverse("chat_home") + f"?thread={plain.id}")
        self.assertIsNone(resp2.context["thread_loop_id"])

    def test_create_with_model_sets_thread_model(self):
        from core.preferences import get_preferences

        model = get_preferences(self.user).allowed_models[0]
        loop_id = self._create(model=model).json()["loop_id"]
        loop = Loop.objects.get(id=loop_id)
        self.assertEqual(loop.thread.model, model)

    def test_create_with_invalid_model_ignored(self):
        loop_id = self._create(model="vendor/not-allowed").json()["loop_id"]
        loop = Loop.objects.get(id=loop_id)
        self.assertEqual(loop.thread.model, "")

    def test_edit_changes_thread_model(self):
        from core.preferences import get_preferences

        model = get_preferences(self.user).allowed_models[0]
        loop_id = self._create().json()["loop_id"]
        self.client.post(
            reverse("loop_edit", args=[loop_id]),
            data=json.dumps({
                "prompt": "x", "history_mode": "fresh", "cadence_kind": "interval",
                "interval_value": 6, "interval_unit": "hours", "first_run_mode": "keep",
                "model": model,
            }),
            content_type="application/json",
        )
        self.assertEqual(Loop.objects.get(id=loop_id).thread.model, model)

    def test_chat_home_default_model_follows_thread(self):
        from core.preferences import get_preferences

        model = get_preferences(self.user).allowed_models[0]
        thread = ChatThread.objects.create(created_by=self.user, model=model)
        resp = self.client.get(reverse("chat_home") + f"?thread={thread.id}")
        self.assertEqual(resp.context["default_model"], model)
        # A new chat (no thread) defaults to the preferred chat model.
        resp2 = self.client.get(reverse("chat_home"))
        self.assertEqual(resp2.context["default_model"], resp2.context["preferred_chat_model"])

    def test_cannot_touch_other_users_loop(self):
        other = User.objects.create_user(email="other@example.com", password="x")
        thread = ChatThread.objects.create(created_by=other)
        loop = Loop.objects.create(
            thread=thread, created_by=other, prompt="x",
            cadence_kind="interval", interval_seconds=3600, next_run=timezone.now(),
        )
        resp = self.client.post(reverse("loop_pause", args=[loop.id]))
        self.assertEqual(resp.status_code, 404)
