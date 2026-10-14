"""Gemini thought-signed reply blocks: what is stored and what is replayed."""

from django.test import SimpleTestCase

from chat.assistant_replay import build_assistant_replay, replay_metadata_for

SIGNED = [
    {"type": "text", "text": "The ball costs 5 cents."},
    {"type": "text", "text": "", "extras": {"signature": "c2ln"}},
]
CONTENT = "The ball costs 5 cents."


class BuildAssistantReplayTests(SimpleTestCase):
    def test_stores_blocks_with_canonical_model(self):
        replay = build_assistant_replay("gemini-4-argon", SIGNED, CONTENT)
        self.assertEqual(replay, {"model": "gemini/gemini-4-argon", "blocks": SIGNED})

    def test_nothing_without_blocks_or_model(self):
        self.assertIsNone(build_assistant_replay("gemini/gemini-4-argon", None, CONTENT))
        self.assertIsNone(build_assistant_replay(None, SIGNED, CONTENT))

    def test_nothing_when_text_differs(self):
        self.assertIsNone(build_assistant_replay("gemini/gemini-4-argon", SIGNED, "Other"))


class ReplayMetadataForTests(SimpleTestCase):
    def _msg(self, **overrides):
        msg = {
            "role": "assistant",
            "content": CONTENT,
            "replay": {"model": "gemini/gemini-4-argon", "blocks": SIGNED},
        }
        msg.update(overrides)
        return msg

    def test_same_model_replays_blocks(self):
        self.assertEqual(
            replay_metadata_for(self._msg(), "gemini/gemini-4-argon"),
            {"content_blocks": SIGNED},
        )

    def test_other_model_does_not(self):
        self.assertIsNone(replay_metadata_for(self._msg(), "gemini/gemini-3.8-flash"))
        self.assertIsNone(replay_metadata_for(self._msg(), "anthropic/claude-opus-5-5"))

    def test_changed_content_does_not(self):
        self.assertIsNone(replay_metadata_for(self._msg(content="Edited"), "gemini/gemini-4-argon"))

    def test_only_assistant_messages(self):
        self.assertIsNone(replay_metadata_for(self._msg(role="user"), "gemini/gemini-4-argon"))
        self.assertIsNone(replay_metadata_for({"role": "assistant", "content": CONTENT}, "gemini/gemini-4-argon"))
