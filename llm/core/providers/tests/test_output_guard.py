"""Unit tests for the degenerate-output guard (``llm.core.output_guard``)."""

from __future__ import annotations

import random
from unittest import TestCase

from langchain_core.messages import AIMessageChunk

from llm.core.output_guard import OutputGuard

_WORDS = (
    "polymer intrinsic microporosity lithium sulfur cathode separator membrane "
    "polysulfide shuttle coating layer electrode battery amidoxime priority "
    "publication claim abstract family examiner search table feature novelty "
    "inventive step problem solution closest prior art disclosure applicant "
    "document classification keyword concept strategy union screening report "
    "canvas template orchestrator sub-agent brief evidence citation legal status"
).split()


def _prose(chars: int, seed: int = 1) -> str:
    rng = random.Random(seed)
    out: list[str] = []
    n = 0
    while n < chars:
        sentence = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(6, 18)))
        sentence = sentence.capitalize() + rng.choice([". ", ", ", "; ", ".\n"])
        out.append(sentence)
        n += len(sentence)
    return "".join(out)[:chars]


def _feed_in_pieces(guard: OutputGuard, text: str, piece: int = 37, **kw) -> str | None:
    reason = None
    for i in range(0, len(text), piece):
        reason = guard.feed(text[i:i + piece], **kw)
        if reason:
            break
    return reason


class OutputGuardTests(TestCase):
    def test_varied_prose_never_flags(self):
        guard = OutputGuard()
        self.assertIsNone(_feed_in_pieces(guard, _prose(60_000)))
        self.assertEqual(guard.total_chars, 60_000)

    def test_pipe_loop_flags_as_repetition_quickly(self):
        guard = OutputGuard()
        _feed_in_pieces(guard, _prose(700))
        reason = _feed_in_pieces(guard, " |" * 20_000, piece=20)
        self.assertEqual(reason, OutputGuard.REPETITION)
        # Caught at the first window check, not after tens of thousands of chars.
        self.assertLessEqual(guard.total_chars, OutputGuard.MIN_CHARS + OutputGuard.STEP_CHARS + 700)

    def test_repeated_sentence_flags(self):
        guard = OutputGuard()
        loop = "The quick brown fox jumps over the lazy dog near the river. " * 500
        self.assertEqual(_feed_in_pieces(guard, loop), OutputGuard.REPETITION)

    def test_repeated_paragraph_flags(self):
        guard = OutputGuard()
        paragraph = _prose(400, seed=7)
        self.assertEqual(_feed_in_pieces(guard, paragraph * 120), OutputGuard.REPETITION)

    def test_realistic_markdown_table_does_not_flag(self):
        rng = random.Random(3)
        rows = [
            "| %s | %d | %.2f | H01M%d/%d | %s |"
            % (
                rng.choice(["WO2022255672A1", "US20200306745A1", "CN120089903A"]),
                2010 + rng.randint(0, 15), rng.random() * 100,
                rng.randint(4, 50), rng.randint(1, 500),
                " ".join(rng.choice(_WORDS) for _ in range(4)),
            )
            for _ in range(300)
        ]
        guard = OutputGuard()
        self.assertIsNone(_feed_in_pieces(guard, "\n".join(rows)))

    def test_thinking_counts_for_repetition_but_not_the_ceiling(self):
        guard = OutputGuard()
        reason = _feed_in_pieces(guard, " |" * 10_000, counts_toward_ceiling=False)
        self.assertEqual(reason, OutputGuard.REPETITION)
        self.assertEqual(guard.ceiling_chars, 0)

    def test_hard_ceiling_flags_output_ceiling(self):
        guard = OutputGuard()
        reason = _feed_in_pieces(guard, _prose(OutputGuard.HARD_CEILING_CHARS + 1_000), piece=5_000)
        self.assertEqual(reason, OutputGuard.OUTPUT_CEILING)

    def test_reason_is_sticky(self):
        guard = OutputGuard()
        _feed_in_pieces(guard, " |" * 10_000)
        self.assertEqual(guard.feed(_prose(5_000)), OutputGuard.REPETITION)
        self.assertEqual(guard.reason, OutputGuard.REPETITION)

    def test_observe_feeds_tool_call_chunk_args(self):
        guard = OutputGuard()
        head = AIMessageChunk(
            content="",
            tool_call_chunks=[{"name": "canvas_edit", "args": '{"edits": [', "id": "c1", "index": 0}],
        )
        self.assertIsNone(guard.observe(head, []))
        reason = None
        for _ in range(60):
            frag = AIMessageChunk(
                content="",
                tool_call_chunks=[{"name": None, "args": " |" * 200, "id": None, "index": 0}],
            )
            reason = guard.observe(frag, [])
            if reason:
                break
        self.assertEqual(reason, OutputGuard.REPETITION)

    def test_observe_feeds_token_and_thinking_events(self):
        guard = OutputGuard()
        chunk = AIMessageChunk(content="")
        guard.observe(chunk, [("thinking", {"text": "plan"}), ("token", {"text": "answer"})])
        self.assertEqual(guard.total_chars, len("plan") + len("answer"))
        self.assertEqual(guard.ceiling_chars, len("answer"))
        self.assertEqual(guard.text(), "plananswer")

    def test_observe_tolerates_missing_args(self):
        guard = OutputGuard()
        chunk = AIMessageChunk(
            content="", tool_call_chunks=[{"name": "t", "args": None, "id": "c1", "index": 0}],
        )
        self.assertIsNone(guard.observe(chunk, []))
        self.assertEqual(guard.total_chars, 0)
