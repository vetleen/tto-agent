"""Automatic model choice: categories, cache boundaries, skills, floor, router."""

import asyncio
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase, override_settings
from django.utils import timezone

from chat import auto_model
from chat.auto_model import (
    AUTO_CATEGORIES,
    RouteDecision,
    apply_floor,
    build_question,
    build_route_facts,
    build_router_input,
    is_cache_warm,
    resolve_category,
    route_turn,
    skill_category,
)
from agent_skills.models import AgentSkill
from chat.auto_model_prompt import CATEGORY_DEFINITIONS
from chat.models import ChatAttachment, ChatCanvas, ChatMessage, ChatThread, ChatThreadDataRoom, ChatThreadSkill
from documents.models import DataRoom, DataRoomDocument
from core.preferences import FEATURE_DEFAULTS

OPUS = "anthropic/claude-opus-5-5"
SONNET = "anthropic/claude-sonnet-5-5"
LUNA = "openai/gpt-6-luna"
SOL = "openai/gpt-6-sol"
ARGON = "gemini/gemini-4-argon"
HAIKU = "anthropic/claude-haiku-5-5"

TTLS = {"anthropic": 3600, "openai": 1800, "google_genai": 300}


def _prefs(**feature_models):
    models = {f"auto_{c.key}": SONNET for c in AUTO_CATEGORIES}
    models.update({f"auto_{k}": v for k, v in feature_models.items()})
    return SimpleNamespace(
        feature_models=models,
        feature_reasoning={},
        medium_model=SONNET,
        allowed_models=[OPUS, SONNET, LUNA, SOL, ARGON, HAIKU],
    )


def _run(coro):
    return asyncio.run(coro)


class CategoryTableTests(SimpleTestCase):
    def test_every_category_has_a_feature_and_a_router_definition(self):
        keys = [c.key for c in AUTO_CATEGORIES]
        self.assertEqual(sorted(keys), sorted(CATEGORY_DEFINITIONS))
        for c in AUTO_CATEGORIES:
            self.assertIn(c.feature_key, FEATURE_DEFAULTS)
            self.assertTrue(FEATURE_DEFAULTS[c.feature_key].reasoning_control)

    def test_rank_follows_the_default_slot(self):
        slot_rank = {"cheap": 0, "medium": 1, "primary": 2}
        for c in AUTO_CATEGORIES:
            self.assertEqual(c.rank, slot_rank[FEATURE_DEFAULTS[c.feature_key].default_slot], c.key)

    def test_question_offers_every_category(self):
        q = build_question()
        self.assertEqual([c.value for c in q.choices], [c.key for c in AUTO_CATEGORIES])

    def test_resolve_category_uses_pinned_level_only_for_its_model(self):
        prefs = _prefs(writing=OPUS)
        prefs.feature_reasoning = {"auto_writing": {"model": OPUS, "level": "high"}}
        self.assertEqual(resolve_category(prefs, "writing"), (OPUS, "high"))
        prefs.feature_reasoning = {"auto_writing": {"model": SONNET, "level": "high"}}
        self.assertEqual(resolve_category(prefs, "writing"), (OPUS, None))
        # A skill's level wins when the model offers it, else the org pin.
        self.assertEqual(resolve_category(prefs, "writing", "max"), (OPUS, "max"))
        self.assertEqual(resolve_category(prefs, "writing", "none"), (OPUS, None))

    def test_unknown_category_resolves_to_default(self):
        prefs = _prefs(medium=LUNA)
        self.assertEqual(resolve_category(prefs, "nope")[0], LUNA)


@override_settings(AUTO_MODEL_CACHE_TTL_SECONDS=TTLS)
class CacheWarmthTests(SimpleTestCase):
    def setUp(self):
        self.now = timezone.now()

    def _state(self, model, minutes_ago, **extra):
        return {
            "enabled": True, "category": "writing", "model": model,
            "last_turn_at": (self.now - timedelta(minutes=minutes_ago)).isoformat(),
            "summary_up_to": None, **extra,
        }

    def _warm(self, state, model, summary_up_to=None):
        return is_cache_warm(
            state, thread_model=model, summary_up_to=summary_up_to,
            allowed_models=_prefs().allowed_models, now=self.now,
        )

    def test_ttl_per_provider(self):
        self.assertTrue(self._warm(self._state(OPUS, 59), OPUS))
        self.assertFalse(self._warm(self._state(OPUS, 61), OPUS))
        self.assertTrue(self._warm(self._state(SOL, 29), SOL))
        self.assertFalse(self._warm(self._state(SOL, 31), SOL))
        self.assertTrue(self._warm(self._state(ARGON, 4), ARGON))
        self.assertFalse(self._warm(self._state(ARGON, 6), ARGON))

    def test_summarise_since_last_turn_is_a_boundary(self):
        state = self._state(OPUS, 1)
        self.assertFalse(self._warm(state, OPUS, summary_up_to="0b1c"))
        state["summary_up_to"] = "0b1c"
        self.assertTrue(self._warm(state, OPUS, summary_up_to="0b1c"))

    def test_cheap_category_always_reroutes(self):
        self.assertFalse(self._warm(self._state(HAIKU, 1, category="cheap"), HAIKU))

    def test_model_change_or_disallowed_model_is_cold(self):
        self.assertFalse(self._warm(self._state(OPUS, 1), SONNET))
        state = self._state("anthropic/retired", 1)
        self.assertFalse(self._warm(state, "anthropic/retired"))

    def test_no_state_is_cold(self):
        self.assertFalse(self._warm(None, OPUS))
        self.assertFalse(self._warm({"model": OPUS}, OPUS))


class FloorAndSkillTests(SimpleTestCase):
    def test_floor(self):
        self.assertEqual(apply_floor("medium", "writing"), ("writing", True))
        self.assertEqual(apply_floor("cheap", "medium"), ("medium", True))
        self.assertEqual(apply_floor("complex_analysis", "writing"), ("complex_analysis", False))
        self.assertEqual(apply_floor("writing", "medium"), ("writing", False))
        # A cheap thread (or none yet) has no floor.
        self.assertEqual(apply_floor("cheap", "cheap"), ("cheap", False))
        self.assertEqual(apply_floor("cheap", None), ("cheap", False))

    def test_skill_highest_rank_wins(self):
        prefs = _prefs()
        skills = [
            {"preferred_model_category": "light_reading"},
            {"preferred_model_category": "slide_design", "preferred_reasoning_level": "high"},
            {"preferred_model_category": ""},
        ]
        self.assertEqual(skill_category(skills, prefs), ("slide_design", "high"))

    def test_skill_tie_goes_to_cheapest_model(self):
        prefs = _prefs(writing=OPUS, complex_analysis=LUNA)
        skills = [{"preferred_model_category": "writing"}, {"preferred_model_category": "complex_analysis"}]
        self.assertEqual(skill_category(skills, prefs), ("complex_analysis", None))

    def test_no_skill_category(self):
        self.assertIsNone(skill_category([{"preferred_model_category": "bogus"}, {}], _prefs()))
        self.assertIsNone(skill_category([], _prefs()))


@override_settings(AUTO_MODEL_CACHE_TTL_SECONDS=TTLS, AUTO_MODEL_ROUTE_TIMEOUT_SECONDS=1)
class RouteTurnTests(SimpleTestCase):
    def _route(self, *, state=None, thread_model=SONNET, skills=(), prefs=None):
        return _run(route_turn(
            thread_id="t", org_id=None, prefs=prefs or _prefs(writing=OPUS, medium=LUNA),
            state=state, thread_model=thread_model, summary_up_to=None,
            skills=list(skills), facts={},
        ))

    @patch.object(auto_model, "decide_category", new_callable=AsyncMock, return_value="writing")
    def test_router_pick(self, mock_decide):
        decision = self._route()
        self.assertEqual(decision, RouteDecision(OPUS, None, "writing", "router", "writing", False))
        mock_decide.assert_awaited_once()

    @patch.object(auto_model, "decide_category", new_callable=AsyncMock, return_value="medium")
    def test_router_downgrade_is_clamped(self, _decide):
        state = {"enabled": True, "category": "writing", "model": OPUS}
        decision = self._route(state=state, thread_model=OPUS)
        self.assertEqual((decision.category, decision.raw_category, decision.floor_applied), ("writing", "medium", True))
        self.assertEqual(decision.model, OPUS)

    @patch.object(auto_model, "decide_category", new_callable=AsyncMock, return_value="writing")
    def test_cached_turn_keeps_model_and_level_without_deciding(self, mock_decide):
        state = {
            "enabled": True, "category": "writing", "model": OPUS, "thinking_level": "high",
            "last_turn_at": timezone.now().isoformat(), "summary_up_to": None,
        }
        decision = self._route(state=state, thread_model=OPUS)
        self.assertEqual((decision.model, decision.thinking_level, decision.source), (OPUS, "high", "cached"))
        mock_decide.assert_not_awaited()

    @patch.object(auto_model, "decide_category", new_callable=AsyncMock, return_value="writing")
    def test_skill_wins_without_deciding(self, mock_decide):
        decision = self._route(skills=[{"preferred_model_category": "slide_design", "preferred_reasoning_level": "max"}])
        self.assertEqual((decision.category, decision.source, decision.thinking_level), ("slide_design", "skill", "max"))
        mock_decide.assert_not_awaited()

    def test_failures_fall_back_to_medium(self):
        async def slow(*a, **k):
            await asyncio.sleep(5)

        for side_effect in (RuntimeError("boom"), auto_model.RouterNoAnswer("refusal"), slow):
            with self.subTest(side_effect=side_effect), \
                    patch.object(auto_model, "decide_category", side_effect=side_effect):
                with self.assertLogs("chat.auto_model", "WARNING"):
                    decision = self._route()
                self.assertEqual((decision.model, decision.category, decision.source), (LUNA, "medium", "fallback"))

    @patch.object(auto_model, "decide_category", side_effect=RuntimeError("boom"))
    def test_fallback_respects_floor(self, _decide):
        state = {"enabled": True, "category": "writing", "model": SONNET}
        with self.assertLogs("chat.auto_model", "WARNING"):
            decision = self._route(state=state)
        self.assertEqual(decision.category, "writing")

    @patch.object(auto_model, "decide_category", new_callable=AsyncMock, return_value="cheap")
    def test_manual_state_has_no_floor(self, _decide):
        # The thread was on a manual model: no category, so nothing to clamp to.
        state = {"enabled": False, "category": "writing", "model": OPUS}
        decision = self._route(state=state, thread_model=OPUS)
        self.assertEqual(decision.category, "cheap")


class RouterInputTests(TestCase):
    def setUp(self):
        from chat.models import ChatMessage, ChatThread

        user = get_user_model().objects.create_user(email="router@example.com", password="x")
        self.thread = ChatThread.objects.create(created_by=user)
        self.msg = lambda role, content, **kw: ChatMessage.objects.create(
            thread=self.thread, role=role, content=content, **kw
        )

    def _data(self, **kw):
        messages = build_router_input(
            self.thread.id, current_category=kw.get("current"),
            decision_message_id=kw.get("decided_by"), facts={"data_rooms": []},
        )
        self.assertEqual(messages[0].role, "developer")
        return messages[1].content[0].data

    def test_shape(self):
        self.msg("user", "hi")
        self.msg("user", "first task")
        for i in range(5):
            self.msg("user", f"u{i}")
        self.msg("user", "hidden notice", is_hidden_from_user=True)
        self.msg("assistant", "A" * 5000)
        self.msg("assistant", "ui note", metadata={"ui_only": True})
        self.msg("user", "thanks")

        data = self._data(current="writing")
        self.assertEqual(data["latest_user_message"], "thanks")
        self.assertEqual(data["earlier_user_messages"], ["u2", "u3", "u4"])
        self.assertEqual(data["first_user_messages"], ["hi", "first task"])
        self.assertEqual(data["last_assistant_message"], "A" * 5000)  # never clipped
        self.assertEqual(data["current_category"], "writing")
        self.assertNotIn("current_category_decision_point", data)
        self.assertEqual(data["facts"], {"data_rooms": []})

    def test_first_message_not_repeated(self):
        self.msg("user", "only message")
        data = self._data()
        self.assertEqual(data["latest_user_message"], "only message")
        self.assertNotIn("first_user_messages", data)
        self.assertNotIn("earlier_user_messages", data)
        self.assertNotIn("last_assistant_message", data)

    def test_first_messages_skip_those_already_included(self):
        for text in ("one", "two", "three"):
            self.msg("user", text)
        data = self._data()
        self.assertEqual(data["earlier_user_messages"], ["one", "two"])
        self.assertNotIn("first_user_messages", data)

    def test_decision_point(self):
        decided = self.msg("user", "draft the licence agreement")
        self.msg("user", "make it shorter")
        data = self._data(current="writing", decided_by=decided.pk)
        self.assertEqual(data["current_category_decision_point"], "draft the licence agreement")
        # Without a current category there is nothing to explain.
        self.assertNotIn("current_category_decision_point", self._data(decided_by=decided.pk))


class RouteFactsTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(email="facts@example.com", password="x")
        self.thread = ChatThread.objects.create(created_by=self.user)

    @staticmethod
    def _at(obj, field, when):
        type(obj).objects.filter(pk=obj.pk).update(**{field: when})

    def _attachment(self, name, content_type, **kw):
        return ChatAttachment.objects.create(
            thread=self.thread, uploaded_by=self.user, file=name, original_filename=name,
            content_type=content_type, size_bytes=10, **kw,
        )

    def test_facts(self):
        t0 = timezone.now() - timedelta(hours=1)
        first = ChatMessage.objects.create(thread=self.thread, role="user", content="hi")
        self._at(first, "created_at", t0)
        second = ChatMessage.objects.create(thread=self.thread, role="user", content="read this")
        self._at(second, "created_at", t0 + timedelta(minutes=10))

        self._attachment("patent_claims.pdf", "application/pdf", message=second, page_count=2)
        self._attachment(
            "a-very-long-file-name-for-the-board-meeting.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            message=first, extracted_content="x" * 4000,
        )
        pending = self._attachment("photo.png", "image/png")  # this turn's, not linked yet
        self._attachment("draft.txt", "text/plain")  # unsent, not on this turn: left out

        room = DataRoom.objects.create(name="Project Alpha", created_by=self.user)
        for i, archived in enumerate((False, False, True)):
            DataRoomDocument.objects.create(
                data_room=room, uploaded_by=self.user, original_filename=f"d{i}.pdf", is_archived=archived,
            )
        link = ChatThreadDataRoom.objects.create(thread=self.thread, data_room=room)
        # Attached between the first and the second message: belongs to round 2.
        self._at(link, "attached_at", t0 + timedelta(minutes=5))

        canvas = ChatCanvas.objects.create(thread=self.thread, title="Memo", content="y" * 800)
        self._at(canvas, "created_at", t0 + timedelta(minutes=1))  # during round 1's turn
        ChatCanvas.objects.create(thread=self.thread, title="Gone", content="z", deleted_at=timezone.now())

        skill = AgentSkill.objects.create(
            slug="drafting", name="Drafting", instructions="x", level="user", created_by=self.user,
            preferred_model_category="writing",
        )
        ChatThreadSkill.objects.create(thread=self.thread, skill=skill)

        skills, facts = build_route_facts(self.thread.id, [str(pending.pk)])
        self.assertEqual(skills, [{
            "name": "Drafting", "preferred_model_category": "writing", "preferred_reasoning_level": "",
        }])
        self.assertEqual(facts, {
            "current_round": 2,
            "attachments": [
                "patent_claims.pdf (2 pages, added round #2)",
                "a-very-long-file-name-fo….docx (~1,000 tokens, added round #1)",
                "photo.png (image, added round #2)",
            ],
            "data_rooms": ["Project Alpha (2 files, added round #2)"],
            "canvases": ["Memo (~200 tokens, added round #1)"],
            "skills": ["Drafting (playbook, added round #2)"],
        })

    def test_empty_thread(self):
        skills, facts = build_route_facts(self.thread.id)
        self.assertEqual(skills, [])
        self.assertEqual(facts, {
            "current_round": 1, "attachments": [], "data_rooms": [], "canvases": [], "skills": [],
        })


class SpecializationModelTests(SimpleTestCase):
    def test_category_overrides_tier(self):
        from chat.subagent_service import specialization_model

        prefs = _prefs(light_reading=LUNA)
        skill = SimpleNamespace(preferred_model_category="light_reading", preferred_reasoning_level="low")
        self.assertEqual(specialization_model(skill, prefs), (LUNA, "low"))
        skill.preferred_reasoning_level = ""
        self.assertEqual(specialization_model(skill, prefs), (LUNA, None))

    def test_no_category_keeps_tier(self):
        from chat.subagent_service import specialization_model

        self.assertEqual(specialization_model(None, _prefs()), (None, None))
        skill = SimpleNamespace(preferred_model_category="", preferred_reasoning_level="")
        self.assertEqual(specialization_model(skill, _prefs()), (None, None))
