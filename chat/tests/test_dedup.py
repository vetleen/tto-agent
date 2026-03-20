"""Tests for chat.dedup — tool result deduplication."""

import json

from django.test import SimpleTestCase

from chat.dedup import (
    _CONTENT_PLACEHOLDER,
    _DESC_PLACEHOLDER,
    _extract_context_doc_indices,
    _parse_read_coverage,
    _parse_search_coverage,
    _redact_read_content,
    _redact_search_content,
    deduplicate_tool_results,
)
from llm.types.messages import Message, ToolCall


# ---------------------------------------------------------------------------
# Helpers to build test messages
# ---------------------------------------------------------------------------

def _assistant_msg(tool_calls: list[tuple[str, str]]) -> Message:
    """Build an assistant message with tool_calls.  Each tuple is (id, name)."""
    return Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id=tc_id, name=name, arguments={}) for tc_id, name in tool_calls],
    )


def _tool_msg(tool_call_id: str, content: str) -> Message:
    """Build a tool-result message."""
    return Message(role="tool", content=content, tool_call_id=tool_call_id)


def _user_msg(content: str = "hello") -> Message:
    return Message(role="user", content=content)


# ---------------------------------------------------------------------------
# Realistic tool output builders
# ---------------------------------------------------------------------------

def _search_result(
    result_num: int,
    doc_index: int,
    filename: str = "doc.pdf",
    chunk_start: int = 0,
    chunk_end: int | None = None,
    total_chunks: int = 10,
    description: str = "A document",
    doc_type: str = "Report",
    content: str = "Some chunk text here.",
) -> str:
    """Build a single search-result block matching SearchDocumentsTool format."""
    chunk_end = chunk_end if chunk_end is not None else chunk_start
    if chunk_start == chunk_end:
        chunk_label = f"Chunk #{chunk_start} of {total_chunks}"
    else:
        chunk_label = f"Chunks #{chunk_start}\u2013#{chunk_end} of {total_chunks}"

    lines = [
        f"## {result_num}.",
        f'**Document:** "{filename}" [doc #{doc_index}]',
        f"**Type:** {doc_type}",
        f"**Description:** {description}",
        f"**Data room:** Test Room",
        f"**{chunk_label}:**",
        content,
        "",
    ]
    return "\n".join(lines)


def _full_search_output(*blocks: str) -> str:
    return "# Search Results\n\n" + "\n".join(blocks)


def _read_output(documents: list[dict]) -> str:
    return json.dumps({"documents": documents})


def _read_doc_entry(
    doc_index: int,
    total_chunks: int = 10,
    content: str = "Full document text",
    chunk_range: str | None = None,
) -> dict:
    entry = {
        "doc_index": doc_index,
        "filename": "doc.pdf",
        "data_room_id": 1,
        "total_chunks": total_chunks,
        "content": content,
    }
    if chunk_range is not None:
        entry["chunk_range"] = chunk_range
        parts = chunk_range.split("-")
        entry["chunks_returned"] = int(parts[1]) - int(parts[0]) + 1
    return entry


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------

class ParseSearchCoverageTests(SimpleTestCase):

    def test_single_chunk(self):
        content = _full_search_output(
            _search_result(1, doc_index=3, chunk_start=4, total_chunks=10)
        )
        cov = _parse_search_coverage(content)
        self.assertEqual(cov, {3: {4}})

    def test_chunk_range(self):
        content = _full_search_output(
            _search_result(1, doc_index=5, chunk_start=2, chunk_end=5, total_chunks=20)
        )
        cov = _parse_search_coverage(content)
        self.assertEqual(cov, {5: {2, 3, 4, 5}})

    def test_multiple_blocks(self):
        content = _full_search_output(
            _search_result(1, doc_index=1, chunk_start=0, total_chunks=5),
            _search_result(2, doc_index=2, chunk_start=3, chunk_end=4, total_chunks=8),
        )
        cov = _parse_search_coverage(content)
        self.assertEqual(cov, {1: {0}, 2: {3, 4}})

    def test_same_doc_multiple_blocks(self):
        content = _full_search_output(
            _search_result(1, doc_index=1, chunk_start=0, total_chunks=5),
            _search_result(2, doc_index=1, chunk_start=3, chunk_end=4, total_chunks=5),
        )
        cov = _parse_search_coverage(content)
        self.assertEqual(cov, {1: {0, 3, 4}})

    def test_no_chunk_label(self):
        # A block without chunk info should be skipped
        content = '## 1.\n**Document:** "doc.pdf" [doc #1]\nSome text\n'
        cov = _parse_search_coverage(content)
        self.assertEqual(cov, {})

    def test_no_results(self):
        cov = _parse_search_coverage("# Search Results\n\nNo results found.")
        self.assertEqual(cov, {})


class ParseReadCoverageTests(SimpleTestCase):

    def test_full_document(self):
        content = _read_output([_read_doc_entry(3, total_chunks=10)])
        cov = _parse_read_coverage(content)
        self.assertEqual(cov, {3: set(range(10))})

    def test_chunk_range(self):
        content = _read_output([_read_doc_entry(5, total_chunks=20, chunk_range="5-10")])
        cov = _parse_read_coverage(content)
        self.assertEqual(cov, {5: {5, 6, 7, 8, 9, 10}})

    def test_multiple_documents(self):
        content = _read_output([
            _read_doc_entry(1, total_chunks=5),
            _read_doc_entry(2, total_chunks=3, chunk_range="1-2"),
        ])
        cov = _parse_read_coverage(content)
        self.assertEqual(cov, {1: set(range(5)), 2: {1, 2}})

    def test_error_entry_skipped(self):
        content = _read_output([{"doc_index": 1, "error": "Not found"}])
        cov = _parse_read_coverage(content)
        self.assertEqual(cov, {})

    def test_invalid_json(self):
        cov = _parse_read_coverage("not json at all")
        self.assertEqual(cov, {})

    def test_error_entry_with_content_not_skipped(self):
        """An entry with both error and content (e.g. truncation warning) should still parse."""
        content = _read_output([{
            "doc_index": 1,
            "total_chunks": 3,
            "content": "some text\n\n[... truncated ...]",
            "error": "Output size limit reached",
        }])
        cov = _parse_read_coverage(content)
        self.assertEqual(cov, {1: {0, 1, 2}})


# ---------------------------------------------------------------------------
# Coverage & redaction logic tests
# ---------------------------------------------------------------------------

class DeduplicateToolResultsTests(SimpleTestCase):
    """End-to-end tests for deduplicate_tool_results()."""

    def test_search_then_read_same_doc(self):
        """search→read for same doc: search content should be redacted."""
        search_content = _full_search_output(
            _search_result(1, doc_index=1, chunk_start=0, chunk_end=2, total_chunks=10)
        )
        read_content = _read_output([_read_doc_entry(1, total_chunks=10)])

        messages = [
            _user_msg(),
            _assistant_msg([("call_1", "search_documents")]),
            _tool_msg("call_1", search_content),
            _assistant_msg([("call_2", "read_document")]),
            _tool_msg("call_2", read_content),
        ]

        result = deduplicate_tool_results(messages)

        # Search result (older, index 2) should be redacted
        self.assertIn(_CONTENT_PLACEHOLDER, result[2].content)
        # Read result (newer, index 4) should be intact
        self.assertEqual(result[4].content, read_content)

    def test_read_then_search_same_doc(self):
        """read→search for same doc: read content should be redacted."""
        read_content = _read_output([_read_doc_entry(1, total_chunks=10)])
        search_content = _full_search_output(
            _search_result(1, doc_index=1, chunk_start=0, chunk_end=9, total_chunks=10)
        )

        messages = [
            _user_msg(),
            _assistant_msg([("call_1", "read_document")]),
            _tool_msg("call_1", read_content),
            _assistant_msg([("call_2", "search_documents")]),
            _tool_msg("call_2", search_content),
        ]

        result = deduplicate_tool_results(messages)

        # Read result (older, index 2) should be redacted
        read_data = json.loads(result[2].content)
        self.assertEqual(read_data["documents"][0]["content"], _CONTENT_PLACEHOLDER)
        # Search result (newer, index 4) should be intact
        self.assertEqual(result[4].content, search_content)

    def test_different_docs_no_redaction(self):
        """Different doc_indices: no redaction."""
        search_content = _full_search_output(
            _search_result(1, doc_index=1, chunk_start=0, total_chunks=5)
        )
        read_content = _read_output([_read_doc_entry(2, total_chunks=5)])

        messages = [
            _user_msg(),
            _assistant_msg([("call_1", "search_documents")]),
            _tool_msg("call_1", search_content),
            _assistant_msg([("call_2", "read_document")]),
            _tool_msg("call_2", read_content),
        ]

        result = deduplicate_tool_results(messages)

        # Nothing should change
        self.assertEqual(result[2].content, search_content)
        self.assertEqual(result[4].content, read_content)

    def test_partial_overlap_no_redaction(self):
        """Older result has chunks not covered by newer: no redaction."""
        search_content = _full_search_output(
            _search_result(1, doc_index=1, chunk_start=0, chunk_end=5, total_chunks=10)
        )
        # Read only covers chunks 0-2
        read_content = _read_output([_read_doc_entry(1, total_chunks=10, chunk_range="0-2")])

        messages = [
            _user_msg(),
            _assistant_msg([("call_1", "search_documents")]),
            _tool_msg("call_1", search_content),
            _assistant_msg([("call_2", "read_document")]),
            _tool_msg("call_2", read_content),
        ]

        result = deduplicate_tool_results(messages)

        # Search has chunks 0-5, read only has 0-2 → not fully covered
        self.assertEqual(result[2].content, search_content)

    def test_union_of_newer_results_covers_older(self):
        """Multiple newer results whose union covers the older result."""
        search_content = _full_search_output(
            _search_result(1, doc_index=1, chunk_start=0, chunk_end=5, total_chunks=10)
        )
        read1 = _read_output([_read_doc_entry(1, total_chunks=10, chunk_range="0-3")])
        read2 = _read_output([_read_doc_entry(1, total_chunks=10, chunk_range="4-5")])

        messages = [
            _user_msg(),
            _assistant_msg([("call_1", "search_documents")]),
            _tool_msg("call_1", search_content),
            _assistant_msg([("call_2", "read_document")]),
            _tool_msg("call_2", read1),
            _assistant_msg([("call_3", "read_document")]),
            _tool_msg("call_3", read2),
        ]

        result = deduplicate_tool_results(messages)

        # Search (chunks 0-5) is fully covered by read1 (0-3) + read2 (4-5)
        self.assertIn(_CONTENT_PLACEHOLDER, result[2].content)
        # Both reads should be intact
        self.assertEqual(result[4].content, read1)
        self.assertEqual(result[6].content, read2)

    def test_two_search_results_newer_covers_older(self):
        """Two search results for same doc, newer fully covers older."""
        search1 = _full_search_output(
            _search_result(1, doc_index=1, chunk_start=2, chunk_end=3, total_chunks=10)
        )
        search2 = _full_search_output(
            _search_result(1, doc_index=1, chunk_start=0, chunk_end=5, total_chunks=10)
        )

        messages = [
            _user_msg(),
            _assistant_msg([("call_1", "search_documents")]),
            _tool_msg("call_1", search1),
            _assistant_msg([("call_2", "search_documents")]),
            _tool_msg("call_2", search2),
        ]

        result = deduplicate_tool_results(messages)

        # Older search (chunks 2-3) is subset of newer (0-5)
        self.assertIn(_CONTENT_PLACEHOLDER, result[2].content)
        self.assertEqual(result[4].content, search2)


# ---------------------------------------------------------------------------
# Description dedup tests
# ---------------------------------------------------------------------------

class DescriptionDedupTests(SimpleTestCase):

    def _dynamic_context_with_docs(self, doc_indices: list[int]) -> str:
        lines = ["# Retrieved Documents", "The attached data rooms contain 5 documents total."]
        for idx in doc_indices:
            lines.append(f'{idx}. [{idx}] "doc{idx}.pdf" (Report) (~1,000 tokens) \u2014 A description of doc {idx}')
        return "\n".join(lines)

    def test_description_in_context_redacted(self):
        ctx = self._dynamic_context_with_docs([1, 2])
        search_content = _full_search_output(
            _search_result(1, doc_index=1, chunk_start=0, total_chunks=5, description="A description of doc 1"),
        )

        messages = [
            _user_msg(),
            _assistant_msg([("call_1", "search_documents")]),
            _tool_msg("call_1", search_content),
        ]

        result = deduplicate_tool_results(messages, dynamic_context=ctx)

        self.assertIn(_DESC_PLACEHOLDER, result[2].content)
        # Chunk content should still be there (no chunk redaction)
        self.assertIn("Some chunk text here.", result[2].content)

    def test_description_not_in_context_kept(self):
        ctx = self._dynamic_context_with_docs([2])  # doc 1 not in context
        search_content = _full_search_output(
            _search_result(1, doc_index=1, chunk_start=0, total_chunks=5, description="Keep this"),
        )

        messages = [
            _user_msg(),
            _assistant_msg([("call_1", "search_documents")]),
            _tool_msg("call_1", search_content),
        ]

        result = deduplicate_tool_results(messages, dynamic_context=ctx)

        self.assertIn("Keep this", result[2].content)
        self.assertNotIn(_DESC_PLACEHOLDER, result[2].content)

    def test_description_redacted_by_doc_index_not_text(self):
        """Even if description text differs, redact if doc_index matches context."""
        ctx = self._dynamic_context_with_docs([1])
        search_content = _full_search_output(
            _search_result(1, doc_index=1, chunk_start=0, total_chunks=5,
                           description="Different description text"),
        )

        messages = [
            _user_msg(),
            _assistant_msg([("call_1", "search_documents")]),
            _tool_msg("call_1", search_content),
        ]

        result = deduplicate_tool_results(messages, dynamic_context=ctx)

        self.assertIn(_DESC_PLACEHOLDER, result[2].content)


# ---------------------------------------------------------------------------
# Output quality tests
# ---------------------------------------------------------------------------

class OutputQualityTests(SimpleTestCase):

    def test_redacted_search_preserves_metadata(self):
        search_content = _full_search_output(
            _search_result(1, doc_index=1, chunk_start=0, chunk_end=2, total_chunks=10,
                           filename="patent.pdf", doc_type="Patent",
                           description="A patent doc")
        )
        read_content = _read_output([_read_doc_entry(1, total_chunks=10)])

        messages = [
            _user_msg(),
            _assistant_msg([("call_1", "search_documents")]),
            _tool_msg("call_1", search_content),
            _assistant_msg([("call_2", "read_document")]),
            _tool_msg("call_2", read_content),
        ]

        result = deduplicate_tool_results(messages)
        redacted = result[2].content

        # Metadata preserved
        self.assertIn('**Document:** "patent.pdf" [doc #1]', redacted)
        self.assertIn("**Type:** Patent", redacted)
        self.assertIn("**Description:** A patent doc", redacted)
        self.assertIn("**Data room:** Test Room", redacted)
        self.assertIn("Chunks #0\u2013#2 of 10", redacted)
        # Content replaced
        self.assertIn(_CONTENT_PLACEHOLDER, redacted)
        self.assertNotIn("Some chunk text here.", redacted)

    def test_redacted_read_preserves_valid_json(self):
        read_content = _read_output([
            _read_doc_entry(1, total_chunks=10, content="secret text"),
        ])
        search_content = _full_search_output(
            _search_result(1, doc_index=1, chunk_start=0, chunk_end=9, total_chunks=10)
        )

        messages = [
            _user_msg(),
            _assistant_msg([("call_1", "read_document")]),
            _tool_msg("call_1", read_content),
            _assistant_msg([("call_2", "search_documents")]),
            _tool_msg("call_2", search_content),
        ]

        result = deduplicate_tool_results(messages)
        redacted_data = json.loads(result[2].content)

        self.assertEqual(redacted_data["documents"][0]["doc_index"], 1)
        self.assertEqual(redacted_data["documents"][0]["filename"], "doc.pdf")
        self.assertEqual(redacted_data["documents"][0]["total_chunks"], 10)
        self.assertEqual(redacted_data["documents"][0]["content"], _CONTENT_PLACEHOLDER)

    def test_non_doc_tool_results_untouched(self):
        messages = [
            _user_msg(),
            _assistant_msg([("call_1", "update_tasks")]),
            _tool_msg("call_1", '{"status": "ok"}'),
        ]

        result = deduplicate_tool_results(messages)

        self.assertEqual(result[2].content, '{"status": "ok"}')

    def test_plain_messages_pass_through(self):
        messages = [
            _user_msg("hi"),
            Message(role="assistant", content="hello"),
            _user_msg("bye"),
        ]

        result = deduplicate_tool_results(messages)

        self.assertEqual(len(result), 3)
        for orig, res in zip(messages, result):
            self.assertEqual(orig.content, res.content)

    def test_original_messages_not_mutated(self):
        search_content = _full_search_output(
            _search_result(1, doc_index=1, chunk_start=0, chunk_end=2, total_chunks=10)
        )
        read_content = _read_output([_read_doc_entry(1, total_chunks=10)])

        messages = [
            _user_msg(),
            _assistant_msg([("call_1", "search_documents")]),
            _tool_msg("call_1", search_content),
            _assistant_msg([("call_2", "read_document")]),
            _tool_msg("call_2", read_content),
        ]

        # Save original content
        original_search = messages[2].content
        original_read = messages[4].content

        deduplicate_tool_results(messages)

        # Originals untouched
        self.assertEqual(messages[2].content, original_search)
        self.assertEqual(messages[4].content, original_read)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class EdgeCaseTests(SimpleTestCase):

    def test_empty_message_list(self):
        result = deduplicate_tool_results([])
        self.assertEqual(result, [])

    def test_error_json_tool_result(self):
        """read_document returning an error should not crash."""
        content = json.dumps({"error": "No data rooms attached", "documents": []})
        messages = [
            _user_msg(),
            _assistant_msg([("call_1", "read_document")]),
            _tool_msg("call_1", content),
        ]

        result = deduplicate_tool_results(messages)
        self.assertEqual(result[2].content, content)

    def test_multimodal_user_message(self):
        """User messages with list content should pass through fine."""
        messages = [
            Message(role="user", content=[{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "data:..."}}]),
            Message(role="assistant", content="hello"),
        ]

        result = deduplicate_tool_results(messages)
        self.assertEqual(len(result), 2)

    def test_tool_result_without_matching_assistant(self):
        """Tool result with no matching assistant tool_call should be skipped."""
        messages = [
            _user_msg(),
            _tool_msg("orphan_id", _read_output([_read_doc_entry(1)])),
        ]

        result = deduplicate_tool_results(messages)
        self.assertEqual(result[1].content, messages[1].content)


# ---------------------------------------------------------------------------
# Context doc index extraction tests
# ---------------------------------------------------------------------------

class ExtractContextDocIndicesTests(SimpleTestCase):

    def test_extracts_indices(self):
        ctx = (
            "# Retrieved Documents\n"
            "The attached data rooms contain 3 documents total.\n"
            '1. [1] "doc1.pdf" (Report) (~500 tokens) \u2014 First doc\n'
            '2. [2] "doc2.pdf" (~1,000 tokens) \u2014 Second doc\n'
            '3. [3] "doc3.pdf" (Patent) (~2,000 tokens)\n'
        )
        indices = _extract_context_doc_indices(ctx)
        # doc 3 has no description (no " — "), so should not be included
        self.assertEqual(indices, {1, 2})

    def test_empty_context(self):
        self.assertEqual(_extract_context_doc_indices(""), set())

    def test_no_retrieved_documents(self):
        ctx = "# Some other section\nNo docs here."
        self.assertEqual(_extract_context_doc_indices(ctx), set())
