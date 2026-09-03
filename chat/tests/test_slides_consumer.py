"""Lightweight regression tests for the slide-deck consumer + skill wiring."""

from __future__ import annotations

import json
import uuid
from unittest import mock

from asgiref.sync import async_to_sync
from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TransactionTestCase


class SlidesUpdatedToolsMembershipTests(SimpleTestCase):
    def test_deck_mutating_tools_are_listed(self):
        """Any deck-mutating tool MUST be in SLIDES_UPDATED_TOOLS or its deck never
        refreshes in the UI (mirror of the CANVAS_UPDATED_TOOLS assertion)."""
        from chat.consumers import SLIDES_UPDATED_TOOLS

        for name in ("slides_create_deck", "slide_canvas_write", "slide_canvas_edit", "slides_add_slide"):
            self.assertIn(name, SLIDES_UPDATED_TOOLS)


class SkillWiringTests(SimpleTestCase):
    def test_skill_tools_registered_and_audience_compatible(self):
        from agent_skills.seed_skills.slide_deck_collaborator import SLIDE_DECK_COLLABORATOR
        from llm.tools import get_tool_registry

        reg = get_tool_registry()
        for name in SLIDE_DECK_COLLABORATOR["tool_names"]:
            tool = reg.get_tool(name)
            self.assertIsNotNone(tool, f"{name} not registered")
            self.assertEqual(tool.section, "skills", name)
            self.assertIn(tool.audience, ("main", "shared"), name)

    def test_skill_in_system_seed_list(self):
        from agent_skills.seed_skills import SYSTEM_SKILLS

        slugs = {s["slug"] for s in SYSTEM_SKILLS}
        self.assertIn("slide_deck_collaborator", slugs)


class SetThemeHandlerTests(TransactionTestCase):
    """Exercise the ``chat.slides_set_theme`` handler end-to-end (DB writes +
    the outgoing ``slidedeck.updated`` event), with the Celery dispatch mocked."""

    def setUp(self):
        from chat.models import ChatThread, SlideSet

        User = get_user_model()
        self.user = User.objects.create_user(
            email=f"th+{uuid.uuid4().hex[:6]}@ex.com", password="x"
        )
        self.thread = ChatThread.objects.create(created_by=self.user, title="t")
        self.deck = SlideSet.objects.create(
            thread=self.thread, title="Deck",
            content={"version": 1, "size": {"w": 960, "h": 540}, "slides": [
                {"id": "s1", "name": "A", "elements": []},
                {"id": "s2", "name": "B", "elements": []},
            ]},
        )

    def _run_handler(self, theme_name):
        """Instantiate the consumer, call the handler, return the sent events."""
        from chat.consumers import ChatConsumer

        consumer = ChatConsumer()
        consumer.user = self.user
        sent = []

        async def _capture(text_data=None, **_kw):
            sent.append(json.loads(text_data))

        consumer.send = _capture
        with mock.patch("chat.tasks.render_deck_task.delay") as m_delay:
            m_delay.return_value = mock.Mock(id="task-123")
            async_to_sync(consumer._handle_slides_set_theme)({
                "thread_id": str(self.thread.pk),
                "deck_id": str(self.deck.pk),
                "theme": theme_name,
            })
        return sent, m_delay

    def test_applies_preset_and_emits_update(self):
        from chat.slides import theme as theme_mod

        sent, m_delay = self._run_handler("slate")
        self.deck.refresh_from_db()
        # The deck's theme override now carries the preset's full palette + provenance.
        th = self.deck.content["theme"]
        self.assertEqual(th["colors"]["dk1"], "#1E293B")  # slate body colour
        self.assertEqual(th["_theme_id"], "slate")
        # A render run was dispatched (all slides re-render on a theme change).
        m_delay.assert_called_once()
        # The client is told every slide changed so the filmstrip shimmers.
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["event_type"], "slidedeck.updated")
        self.assertEqual(sorted(sent[0]["changed_slide_ids"]), ["s1", "s2"])

    def test_checkpoint_is_recorded(self):
        from chat.models import SlideSetCheckpoint

        before = SlideSetCheckpoint.objects.filter(slide_set=self.deck).count()
        self._run_handler("ocean")
        after = SlideSetCheckpoint.objects.filter(slide_set=self.deck).count()
        self.assertEqual(after, before + 1)

    def test_unknown_theme_is_a_noop(self):
        sent, m_delay = self._run_handler("not-a-real-theme")
        self.deck.refresh_from_db()
        # No override applied, no render dispatched, no event sent.
        self.assertNotIn("theme", self.deck.content)
        m_delay.assert_not_called()
        self.assertEqual(sent, [])

    def test_foreign_deck_is_rejected(self):
        # A deck the requesting user doesn't own must not be mutated.
        other = get_user_model().objects.create_user(
            email=f"o+{uuid.uuid4().hex[:6]}@ex.com", password="x"
        )
        from chat.consumers import ChatConsumer

        consumer = ChatConsumer()
        consumer.user = other
        sent = []

        async def _capture(text_data=None, **_kw):
            sent.append(json.loads(text_data))

        consumer.send = _capture
        with mock.patch("chat.tasks.render_deck_task.delay") as m_delay:
            async_to_sync(consumer._handle_slides_set_theme)({
                "thread_id": str(self.thread.pk),
                "deck_id": str(self.deck.pk),
                "theme": "slate",
            })
        self.deck.refresh_from_db()
        self.assertNotIn("theme", self.deck.content)
        m_delay.assert_not_called()
        self.assertEqual(sent, [])


class SlidesOpenOwnershipTests(TransactionTestCase):
    """`chat.slides_open` must not leak another user's active deck for a guessed
    thread_id (the deck_id-omitted path used to skip the ownership check)."""

    def setUp(self):
        from chat.models import ChatThread, SlideSet

        User = get_user_model()
        self.owner = User.objects.create_user(email=f"own+{uuid.uuid4().hex[:6]}@ex.com", password="x")
        self.thread = ChatThread.objects.create(created_by=self.owner, title="t")
        self.deck = SlideSet.objects.create(
            thread=self.thread, title="Secret Deck", is_active=True,
            content={"version": 1, "size": {"w": 960, "h": 540},
                     "slides": [{"id": "s1", "name": "Confidential", "elements": []}]},
        )

    def _open(self, user):
        from chat.consumers import ChatConsumer

        consumer = ChatConsumer()
        consumer.user = user
        sent = []

        async def _capture(text_data=None, **_kw):
            sent.append(json.loads(text_data))

        consumer.send = _capture
        # deck_id omitted => the active-deck path that used to be unscoped.
        async_to_sync(consumer._handle_slides_open)({"thread_id": str(self.thread.pk)})
        return sent

    def test_owner_sees_active_deck(self):
        sent = self._open(self.owner)
        self.assertEqual(sent[0]["event_type"], "slidedeck.state")
        self.assertEqual(sent[0]["title"], "Secret Deck")

    def test_foreign_user_gets_closed_not_deck(self):
        other = get_user_model().objects.create_user(email=f"att+{uuid.uuid4().hex[:6]}@ex.com", password="x")
        sent = self._open(other)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["event_type"], "slidedeck.closed")
        # No deck data may leak to a non-owner.
        self.assertNotIn("title", sent[0])
        self.assertNotIn("slides", sent[0])


class ServiceRobustnessTests(TransactionTestCase):
    """Edge cases in chat.slides.service that would otherwise raise IntegrityError."""

    def setUp(self):
        from chat.models import ChatThread
        User = get_user_model()
        self.user = User.objects.create_user(email=f"sv+{uuid.uuid4().hex[:6]}@ex.com", password="x")
        self.thread = ChatThread.objects.create(created_by=self.user, title="t")

    def test_restore_disambiguates_a_taken_title(self):
        # Soft-delete 'Foo', create a new live 'Foo', then restore the old one:
        # the restore must not violate the (thread,title) unique constraint.
        from chat.models import SlideSet
        from chat.slides import service

        c = {"version": 1, "size": {"w": 960, "h": 540}, "slides": []}
        old = SlideSet.objects.create(thread=self.thread, title="Foo", content=c)
        service.soft_delete_deck(str(self.thread.pk), old)
        SlideSet.objects.create(thread=self.thread, title="Foo", content=c)  # new live 'Foo'

        service.restore_deck(str(self.thread.pk), old)  # must not raise
        old.refresh_from_db()
        self.assertIsNone(old.deleted_at)
        self.assertNotEqual(old.title, "Foo")           # disambiguated
        self.assertTrue(old.title.startswith("Foo ("))


class UserCustomThemeHandlerTests(TransactionTestCase):
    """Save / delete / apply a user's custom slide theme via the consumer."""

    def setUp(self):
        from chat.models import ChatThread, SlideSet

        User = get_user_model()
        self.user = User.objects.create_user(
            email=f"uc+{uuid.uuid4().hex[:6]}@ex.com", password="x"
        )
        self.thread = ChatThread.objects.create(created_by=self.user, title="t")
        self.deck = SlideSet.objects.create(
            thread=self.thread, title="Deck",
            content={"version": 1, "size": {"w": 960, "h": 540},
                     "slides": [{"id": "s1", "name": "A", "elements": []}]},
        )

    def _consumer(self):
        from chat.consumers import ChatConsumer

        consumer = ChatConsumer()
        consumer.user = self.user
        sent = []

        async def _capture(text_data=None, **_kw):
            sent.append(json.loads(text_data))

        consumer.send = _capture
        return consumer, sent

    def _valid_theme(self):
        return {"label": "Acme brand", "base": "slate",
                "colors": {"accent1": "#123456", "dk2": "#223344", "lt1": "#FFFFFF"}}

    def test_save_persists_and_returns_swatch(self):
        from chat.slides import theme as theme_mod

        consumer, sent = self._consumer()
        async_to_sync(consumer._handle_slides_save_theme)({"theme": self._valid_theme()})

        saved = theme_mod.user_slide_themes(self.user)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["label"], "Acme brand")
        self.assertTrue(saved[0]["id"].startswith("c"))
        # The client is handed the refreshed swatch list.
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["event_type"], "slidedeck.user_themes")
        self.assertEqual(sent[0]["themes"][0]["label"], "Acme brand")

    def test_save_invalid_reports_error_and_saves_nothing(self):
        from chat.slides import theme as theme_mod

        consumer, sent = self._consumer()
        async_to_sync(consumer._handle_slides_save_theme)(
            {"theme": {"label": "", "base": "slate", "colors": {}}}
        )
        self.assertEqual(theme_mod.user_slide_themes(self.user), [])
        self.assertEqual(sent[0]["event_type"], "slidedeck.user_themes")
        self.assertTrue(sent[0].get("error"))

    def test_apply_custom_theme_to_deck(self):
        from chat.slides import theme as theme_mod

        consumer, _ = self._consumer()
        async_to_sync(consumer._handle_slides_save_theme)({"theme": self._valid_theme()})
        tid = theme_mod.user_slide_themes(self.user)[0]["id"]

        consumer2, sent = self._consumer()
        with mock.patch("chat.tasks.render_deck_task.delay") as m_delay:
            m_delay.return_value = mock.Mock(id="t")
            async_to_sync(consumer2._handle_slides_set_theme)({
                "thread_id": str(self.thread.pk),
                "deck_id": str(self.deck.pk),
                "theme": tid,
            })
        self.deck.refresh_from_db()
        self.assertEqual(self.deck.content["theme"]["colors"]["accent1"], "#123456")
        m_delay.assert_called_once()
        self.assertEqual(sent[0]["event_type"], "slidedeck.updated")

    def test_delete_removes_theme(self):
        from chat.slides import theme as theme_mod

        consumer, _ = self._consumer()
        async_to_sync(consumer._handle_slides_save_theme)({"theme": self._valid_theme()})
        tid = theme_mod.user_slide_themes(self.user)[0]["id"]

        consumer2, sent = self._consumer()
        async_to_sync(consumer2._handle_slides_delete_theme)({"theme_id": tid})
        self.assertEqual(theme_mod.user_slide_themes(self.user), [])
        self.assertEqual(sent[0]["event_type"], "slidedeck.user_themes")
        self.assertEqual(sent[0]["themes"], [])

    def test_cap_enforced(self):
        from chat.slides import theme as theme_mod

        consumer, _ = self._consumer()
        for i in range(theme_mod.MAX_USER_SLIDE_THEMES + 3):
            t = self._valid_theme()
            t["label"] = f"T{i}"
            async_to_sync(consumer._handle_slides_save_theme)({"theme": t})
        self.assertLessEqual(
            len(theme_mod.user_slide_themes(self.user)), theme_mod.MAX_USER_SLIDE_THEMES
        )
