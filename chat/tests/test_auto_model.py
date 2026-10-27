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
    build_router_input,
    is_cache_warm,
    resolve_category,
    route_turn,
    skill_category,
)
from chat.auto_model_prompt import CATEGORY_DEFINITIONS
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
        mid_model=SONNET,
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
        prefs = _prefs(mid=LUNA)
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
        self.assertEqual(apply_floor("mid", "writing"), ("writing", True))
        self.assertEqual(apply_floor("cheap", "mid"), ("mid", True))
        self.assertEqual(apply_floor("complex_analysis", "writing"), ("complex_analysis", False))
        self.assertEqual(apply_floor("writing", "mid"), ("writing", False))
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
        prefs = _prefs(writing=OPUS, research=LUNA)
        skills = [{"preferred_model_category": "writing"}, {"preferred_model_category": "research"}]
        self.assertEqual(skill_category(skills, prefs), ("research", None))

    def test_no_skill_category(self):
        self.assertIsNone(skill_category([{"preferred_model_category": "bogus"}, {}], _prefs()))
        self.assertIsNone(skill_category([], _prefs()))


@override_settings(AUTO_MODEL_CACHE_TTL_SECONDS=TTLS, AUTO_MODEL_ROUTE_TIMEOUT_SECONDS=1)
class RouteTurnTests(SimpleTestCase):
    def _route(self, *, state=None, thread_model=SONNET, skills=(), prefs=None):
        return _run(route_turn(
            thread_id="t", org_id=None, prefs=prefs or _prefs(writing=OPUS, mid=LUNA),
            state=state, thread_model=thread_model, summary_up_to=None,
            skills=list(skills), facts={},
        ))

    @patch.object(auto_model, "decide_category", new_callable=AsyncMock, return_value="writing")
    def test_router_pick(self, mock_decide):
        decision = self._route()
        self.assertEqual(decision, RouteDecision(OPUS, None, "writing", "router", "writing", False))
        mock_decide.assert_awaited_once()

    @patch.object(auto_model, "decide_category", new_callable=AsyncMock, return_value="mid")
    def test_router_downgrade_is_clamped(self, _decide):
        state = {"enabled": True, "category": "writing", "model": OPUS}
        decision = self._route(state=state, thread_model=OPUS)
        self.assertEqual((decision.category, decision.raw_category, decision.floor_applied), ("writing", "mid", True))
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

    def test_failures_fall_back_to_mid(self):
        async def slow(*a, **k):
            await asyncio.sleep(5)

        for side_effect in (RuntimeError("boom"), auto_model.RouterNoAnswer("refusal"), slow):
            with self.subTest(side_effect=side_effect), \
                    patch.object(auto_model, "decide_category", side_effect=side_effect):
                with self.assertLogs("chat.auto_model", "WARNING"):
                    decision = self._route()
                self.assertEqual((decision.model, decision.category, decision.source), (LUNA, "mid", "fallback"))

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
        messages = build_router_input(self.thread.id, current_category=kw.get("current"), facts={"data_rooms": 1})
        self.assertEqual(messages[0].role, "developer")
        return messages[1].content[0].data

    def test_shape(self):
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
        self.assertEqual(data["first_user_message"], "first task")
        self.assertTrue(data["last_assistant_message_head"].endswith("…[truncated]"))
        self.assertLess(len(data["last_assistant_message_head"]), 1700)
        self.assertEqual(data["current_category"], "writing")
        self.assertEqual(data["facts"], {"data_rooms": 1})

    def test_first_message_not_repeated(self):
        self.msg("user", "only message")
        data = self._data()
        self.assertEqual(data["latest_user_message"], "only message")
        self.assertNotIn("first_user_message", data)
        self.assertNotIn("earlier_user_messages", data)
        self.assertNotIn("last_assistant_message_head", data)


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
