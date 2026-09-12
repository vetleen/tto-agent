"""Tests for the send-time native-asset request-ceiling enforcer + pruning."""

from django.test import SimpleTestCase, override_settings

from llm.core.native_limits import (
    enforce_native_request_limits,
    provider_native_b64_ceiling,
)
from llm.types.messages import Message


def _img(pathway, size, label="f"):
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "A" * size},
        "_wf_pathway": pathway,
        "_wf_b64len": size,
        "_wf_label": label,
    }


def _text(t):
    return {"type": "text", "text": t}


def _is_stub(block):
    return block.get("type") == "text" and "omitted to fit" in block.get("text", "")


def _has_markers(block):
    return any(k.startswith("_wf") for k in block)


@override_settings(
    NATIVE_ASSET_BUDGET_B64_BYTES=10_000,
    NATIVE_REQUEST_MAX_B64_BYTES_ANTHROPIC=1_000,
)
class NativeRequestLimitTests(SimpleTestCase):
    def test_no_native_blocks_returns_same_list(self):
        messages = [Message(role="user", content="hello")]
        out = enforce_native_request_limits(messages, "anthropic")
        self.assertIs(out, messages)

    def test_under_ceiling_keeps_block_but_strips_markers(self):
        messages = [Message(role="user", content=[_text("hi"), _img("attachment", 500)])]
        out = enforce_native_request_limits(messages, "anthropic")
        block = out[0].content[1]
        self.assertEqual(block["type"], "image")           # kept, not a stub
        self.assertFalse(_has_markers(block))               # markers stripped
        self.assertEqual(block["source"]["data"], "A" * 500)

    def test_over_ceiling_evicts_oldest_nonskill_first(self):
        messages = [
            Message(role="user", content=[_img("attachment", 600, "old")]),
            Message(role="user", content=[_img("dataroom", 600, "new")]),
        ]
        out = enforce_native_request_limits(messages, "anthropic")
        self.assertTrue(_is_stub(out[0].content[0]))        # oldest evicted
        self.assertEqual(out[1].content[0]["type"], "image")  # newest kept
        self.assertIn("old", out[0].content[0]["text"])

    def test_skill_survives_when_nonskill_can_be_evicted(self):
        messages = [
            Message(role="user", content=[_img("attachment", 600, "att")]),
            Message(role="user", content=[_img("skill", 600, "skill")]),
        ]
        out = enforce_native_request_limits(messages, "anthropic")
        self.assertTrue(_is_stub(out[0].content[0]))        # non-skill evicted
        self.assertEqual(out[1].content[0]["type"], "image")  # skill kept

    def test_skill_evicted_only_as_last_resort(self):
        messages = [
            Message(role="user", content=[_img("skill", 600, "a")]),
            Message(role="user", content=[_img("skill", 600, "b")]),
        ]
        out = enforce_native_request_limits(messages, "anthropic")
        # 1200 > 1000: evict the oldest skill block, keep the newest.
        self.assertTrue(_is_stub(out[0].content[0]))
        self.assertEqual(out[1].content[0]["type"], "image")

    def test_nonanthropic_bounded_only_by_pool(self):
        # pool is 10_000; two 600-byte blocks are well under it → no eviction.
        messages = [
            Message(role="user", content=[_img("attachment", 600)]),
            Message(role="user", content=[_img("dataroom", 600)]),
        ]
        out = enforce_native_request_limits(messages, "openai")
        self.assertEqual(out[0].content[0]["type"], "image")
        self.assertEqual(out[1].content[0]["type"], "image")

    def test_original_messages_not_mutated(self):
        block = _img("attachment", 600)
        messages = [
            Message(role="user", content=[block]),
            Message(role="user", content=[_img("dataroom", 600)]),
        ]
        enforce_native_request_limits(messages, "anthropic")
        # The source list/block still carries its markers (a copy was pruned).
        self.assertTrue(_has_markers(messages[0].content[0]))

    def test_untagged_native_block_size_derived_and_treated_nonskill(self):
        raw = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "A" * 1_200},
        }
        messages = [Message(role="user", content=[raw])]
        out = enforce_native_request_limits(messages, "anthropic")
        # 1200 > 1000 ceiling and it's the only block → evicted to a stub.
        self.assertTrue(_is_stub(out[0].content[0]))

    def test_ceiling_is_min_of_pool_and_provider(self):
        self.assertEqual(provider_native_b64_ceiling("anthropic"), 1_000)
        self.assertEqual(provider_native_b64_ceiling("openai"), 10_000)
