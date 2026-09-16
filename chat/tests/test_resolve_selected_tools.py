"""``_resolve_selected_tools`` strips BOTH of the agent's skill-management tools
when the user has disabled agent-driven skill attachment (a half-stripped pair
would let the agent detach but not attach, or vice versa)."""

from types import SimpleNamespace

from asgiref.sync import async_to_sync
from django.test import TestCase

from chat.consumers import ChatConsumer


def _consumer(allow_agent_attach_skills):
    c = ChatConsumer()
    c.resolved_prefs = SimpleNamespace(
        allowed_tools=["chat_skill_attach", "chat_skill_detach", "chat_task_update"],
        allowed_skills=[],
        allow_agent_attach_skills=allow_agent_attach_skills,
    )
    c.data_room_ids = []
    c.active_skill_ids = []
    return c


class ResolveSelectedToolsSkillGateTests(TestCase):
    def _resolve(self, allow):
        return async_to_sync(_consumer(allow)._resolve_selected_tools)(is_loop_turn=False)

    def test_disabled_strips_attach_and_detach(self):
        tools, schemas = self._resolve(False)
        self.assertNotIn("chat_skill_attach", tools)
        self.assertNotIn("chat_skill_detach", tools)
        self.assertIn("chat_task_update", tools)
        self.assertEqual({t.name for t in schemas}, {"chat_task_update"})

    def test_enabled_keeps_both(self):
        tools, schemas = self._resolve(True)
        self.assertIn("chat_skill_attach", tools)
        self.assertIn("chat_skill_detach", tools)
        self.assertEqual(
            {t.name for t in schemas},
            {"chat_skill_attach", "chat_skill_detach", "chat_task_update"},
        )
