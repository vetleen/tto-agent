"""Live tests for the decisions service against the real providers.

Run only when TEST_APIS=True and OPENAI_API_KEY is set: one call per pathway.
"""

import os
from unittest.mock import patch

from django.test import TestCase, override_settings

from llm.models import LLMCallLog
from llm.service.decisions_service import DecisionsService
from llm.tests.utils import require_test_apis
from llm.types.decisions import Choice, Level, Question

_INPUT = (
    "Invention disclosure: a researcher developed a catalyst that lowers the temperature "
    "needed for ammonia synthesis by 150 C. Preliminary results were presented on a poster "
    "at a conference last month."
)
_QUESTIONS = [
    Question(type="predicate", name="disclosed", instructions="Has the invention already been publicly disclosed?"),
    Question(type="choice", name="field", instructions="Which technology field does this belong to?",
             choices=[Choice(value="chemistry", description="Chemistry, materials, catalysis."),
                      Choice(value="software"), Choice(value="other")]),
    Question(type="score", name="potential", instructions="How strong is the commercial potential?",
             levels=[Level(label="Low"), Level(label="Medium"), Level(label="High")]),
]


@require_test_apis()
@override_settings(LLM_DEFAULT_CHEAP_MODEL="openai/gpt-6-luna")
class DecisionsLiveTests(TestCase):
    def _service(self):
        return DecisionsService()

    def _check(self, result):
        self.assertEqual([a.name for a in result.answers], ["disclosed", "field", "potential"])
        disclosed, field, potential = result.answers
        if disclosed.type != "refusal":
            self.assertGreaterEqual(disclosed.probability, 0.5)  # a poster is a disclosure
        if field.type != "refusal":
            self.assertEqual(field.choice, "chemistry")
            self.assertAlmostEqual(sum(p.probability for p in field.probabilities), 1.0, places=1)
        if potential.type != "refusal":
            self.assertEqual([p.value for p in potential.probabilities], [0, 1, 2])  # 0-based levels
            self.assertTrue(0 <= potential.score <= 2)
        self.assertGreater(result.usage.prompt_tokens or 0, 0)

    def test_quick_pathway(self):
        with patch.dict(os.environ, {"LLM_ALLOWED_MODELS": "openai/gpt-6-luna"}, clear=False):
            result = self._service().decide(_QUESTIONS, _INPUT, feature_key="live_test", org_id=None)
        self.assertEqual(result.pathway, "quick")
        self._check(result)
        self.assertGreater(result.usage.cost_usd, 0)
        row = LLMCallLog.objects.get(response_metadata__kind="decisions")
        self.assertEqual(row.response_metadata["pathway"], "quick")
        self.assertEqual(row.output_tokens, 0)

    def test_structured_pathway(self):
        with patch.dict(os.environ, {"LLM_ALLOWED_MODELS": "openai/gpt-6-luna"}, clear=False):
            service = self._service()
            service._quick_available = lambda prefs: (False, "test")  # force the structured pathway
            result = service.decide(_QUESTIONS, _INPUT, feature_key="live_test", org_id=None)
        self.assertEqual(result.pathway, "structured")
        self._check(result)
        row = LLMCallLog.objects.get(response_metadata__kind="decisions")
        self.assertEqual(row.response_metadata["pathway"], "structured")
        self.assertEqual(row.response_metadata["feature"], "live_test")
