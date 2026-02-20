"""
Live test: each chat dropdown model works with the chat request contract.

Run only when TEST_APIS=True and the relevant API keys are set:
    TEST_APIS=True python manage.py test llm_chat.tests.test_live_chat_setup -v 2

Uses the same contract as the chat UI: system instruction asking for valid JSON
with key "message", response_format json_object. Fails if any model doesn't
work (e.g. keyword requirements, unsupported format).
"""
import json
import os
import unittest

from django.test import TestCase

from llm_chat.constants import CHAT_KEY_MODELS
from llm_service.conf import get_allowed_models

RUN_LIVE = os.environ.get("TEST_APIS", "").strip().lower() in ("1", "true", "yes")
REQUIRES_LIVE = unittest.skipUnless(
    RUN_LIVE,
    "Live chat setup test disabled. Set TEST_APIS=True and API keys to run.",
)


def _live_chat_style_completion_test(model: str) -> None:
    """
    Run one completion with the same contract as the chat UI: system instruction
    asking for valid JSON with key "message", response_format json_object.
    """
    from llm_service.client import completion

    system_instruction = (
        "You must respond with valid JSON only. Use a single key \"message\" "
        "whose value is your reply text."
    )
    user_message = "Reply with exactly: {\"message\": \"OK\"}."
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_message},
    ]
    resp = completion(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    assert resp is not None
    assert getattr(resp, "choices", None) and len(resp.choices) > 0
    content = (getattr(resp.choices[0].message, "content", None) or "").strip()
    assert len(content) > 0, f"Model {model}: empty response"
    parsed = json.loads(content)
    assert isinstance(parsed, dict), f"Model {model}: response is not a JSON object: {content!r}"
    assert "message" in parsed, f"Model {model}: missing 'message' key in {list(parsed.keys())}"


@REQUIRES_LIVE
class LiveChatSetupCompatibilityTest(TestCase):
    """
    For each chat dropdown model (in LLM_ALLOWED_MODELS), run a chat-style request.
    Fails if any model doesn't work with the chat setup.
    """

    def tearDown(self):
        import llm_service.client as client_mod
        client_mod._client = None

    def test_each_chat_model_accepts_chat_style_request(self):
        model_ids = [m[0] for m in CHAT_KEY_MODELS]
        allowed = set(get_allowed_models())
        to_test = [m for m in model_ids if m in allowed]
        if not to_test:
            self.skipTest(
                "No chat key models are in LLM_ALLOWED_MODELS. "
                "Add at least one of: " + ", ".join(model_ids)
            )
        for model in to_test:
            with self.subTest(model=model):
                _live_chat_style_completion_test(model)
