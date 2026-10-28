"""Several data rooms on one thread: doc_index restarts at 1 in every room.

The document tools must never guess between same-numbered documents in
different rooms, and everything the model reads must name the room id the
tools take.
"""

import json
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase

from chat.prompts import build_semi_static_prompt, build_system_prompt, multi_room_index_note
from chat.subagent_prompts import build_subagent_system_prompt
from chat.tools import (
    GetDocumentStatusTool,
    ListDocumentsTool,
    ReadDocumentTool,
    SearchDocumentsTool,
    _resolve_document,
)
from documents.models import DataRoom
from documents.tests._helpers import make_document
from llm.types.context import RunContext

User = get_user_model()


class _TwoRooms(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="rooms@test.com", password="pass")
        self.acme = DataRoom.objects.create(name="Acme", slug="acme-mr", created_by=self.user)
        self.beta = DataRoom.objects.create(name="Beta", slug="beta-mr", created_by=self.user)
        self.acme_doc = make_document(
            self.acme, self.user, original_filename="term_sheet.txt", chunks=["ACME royalty 3%"],
        )
        self.beta_doc = make_document(
            self.beta, self.user, original_filename="term_sheet.txt", chunks=["BETA royalty 5%"],
        )
        self.beta_only = make_document(
            self.beta, self.user, original_filename="business_plan.txt", chunks=["BETA plan"],
        )
        # Both rooms number their first document #1.
        self.assertEqual(self.acme_doc.doc_index, 1)
        self.assertEqual(self.beta_doc.doc_index, 1)
        self.assertEqual(self.beta_only.doc_index, 2)

    def _ctx(self, rooms=None):
        rooms = rooms if rooms is not None else [self.acme, self.beta]
        return RunContext.create(user_id=self.user.pk, data_room_ids=[r.pk for r in rooms])

    def _run(self, tool_cls, args, ctx=None):
        tool = tool_cls()
        tool.set_context(ctx or self._ctx())
        return json.loads(tool.invoke(args))


class ResolveDocumentTests(_TwoRooms):

    def test_shared_index_without_room_is_ambiguous(self):
        doc, err = _resolve_document(self._ctx(), 1)
        self.assertIsNone(doc)
        self.assertIn("ambiguous", err)
        self.assertIn(f"data_room_id {self.acme.pk}", err)
        self.assertIn(f"data_room_id {self.beta.pk}", err)

    def test_room_id_picks_the_right_document(self):
        doc, err = _resolve_document(self._ctx(), 1, self.acme.pk)
        self.assertIsNone(err)
        self.assertEqual(doc.pk, self.acme_doc.pk)
        doc, _ = _resolve_document(self._ctx(), 1, self.beta.pk)
        self.assertEqual(doc.pk, self.beta_doc.pk)

    def test_unique_index_needs_no_room(self):
        doc, err = _resolve_document(self._ctx(), 2)
        self.assertIsNone(err)
        self.assertEqual(doc.pk, self.beta_only.pk)

    def test_unattached_room_id_is_an_error_not_ignored(self):
        """Luna guessed data_room_id=2; the old code silently searched every room."""
        doc, err = _resolve_document(self._ctx(), 1, 999999)
        self.assertIsNone(doc)
        self.assertIn("not attached", err)
        self.assertIn(f'"Acme" (data_room_id {self.acme.pk})', err)

    def test_single_room_thread_unchanged(self):
        doc, err = _resolve_document(self._ctx([self.acme]), 1)
        self.assertIsNone(err)
        self.assertEqual(doc.pk, self.acme_doc.pk)

    def test_index_tool_reports_ambiguity(self):
        result = self._run(GetDocumentStatusTool, {"doc_index": 1})
        self.assertIn("ambiguous", result["error"])


class ReadDocumentMultiRoomTests(_TwoRooms):

    def test_ambiguous_index_is_an_error_entry(self):
        result = self._run(ReadDocumentTool, {"doc_indices": [1]})
        entry = result["documents"][0]
        self.assertIn("ambiguous", entry["error"])
        self.assertNotIn("content", entry)

    def test_room_id_reads_that_room(self):
        result = self._run(ReadDocumentTool, {"doc_indices": [1], "data_room_id": self.acme.pk})
        entry = result["documents"][0]
        self.assertIn("ACME", entry["content"])
        self.assertEqual(entry["data_room_id"], self.acme.pk)

    def test_unattached_room_id_is_an_error(self):
        result = self._run(ReadDocumentTool, {"doc_indices": [1], "data_room_id": 999999})
        self.assertIn("not attached", result["error"])
        self.assertEqual(result["documents"], [])


class ListDocumentsMultiRoomTests(_TwoRooms):

    def test_rows_carry_their_room_and_are_grouped(self):
        result = self._run(ListDocumentsTool, {})
        rows = [(r["data_room"], r["data_room_id"], r["doc_index"]) for r in result["documents"]]
        self.assertEqual(rows, [
            ("Acme", self.acme.pk, 1),
            ("Beta", self.beta.pk, 1),
            ("Beta", self.beta.pk, 2),
        ])

    def test_unattached_room_id_is_an_error(self):
        result = self._run(ListDocumentsTool, {"data_room_id": 999999})
        self.assertIn("not attached", result["error"])


class SearchDocumentsMultiRoomTests(_TwoRooms):

    def _search(self, rooms):
        from langchain_core.documents import Document

        chunk = self.beta_doc.active_searchable_version.chunks.get()
        hit = Document(page_content=chunk.text, metadata={
            "chunk_id": chunk.pk, "document_id": self.beta_doc.pk,
            "doc_index": 1, "data_room_id": self.beta.pk, "chunk_index": 0,
        })
        window = {
            "document_id": self.beta_doc.pk, "chunks_included": [0],
            "context_text": chunk.text, "page_start": None, "page_end": None,
        }
        with patch("documents.services.retrieval.similarity_search_chunks", return_value=[hit]), \
                patch("documents.services.retrieval.get_merged_context_windows", return_value=[window]):
            tool = SearchDocumentsTool()
            tool.set_context(self._ctx(rooms))
            return tool.invoke({"query": "royalty"})

    def test_result_tag_names_the_room(self):
        out = self._search([self.acme, self.beta])
        self.assertIn(f"[doc #1, data_room_id {self.beta.pk}]", out)
        self.assertIn(f'"Beta"** (data_room_id {self.beta.pk})', out)

    def test_single_room_tag_unchanged(self):
        out = self._search([self.beta])
        self.assertIn("[doc #1]", out)


class PromptRoomIdTests(TestCase):
    ROOMS = [
        {"id": 7, "name": "Acme", "description": "", "document_count": 3},
        {"id": 8, "name": "Beta", "description": "Spinout", "document_count": 1},
    ]

    def test_main_prompt_lists_room_ids_and_index_note(self):
        prompt = build_semi_static_prompt(data_rooms=self.ROOMS)
        self.assertIn('- "Acme" (data_room_id 7, 3 documents)', prompt)
        self.assertIn('- **"Beta"** (data_room_id 8, 1 document): Spinout', prompt)
        self.assertIn("Document numbers restart at 1 in each data room", prompt)

    def test_single_room_has_no_index_note(self):
        prompt = build_semi_static_prompt(data_rooms=self.ROOMS[:1])
        self.assertIn("data_room_id 7", prompt)
        self.assertEqual(multi_room_index_note(self.ROOMS[:1]), "")
        self.assertNotIn("restart at 1", prompt)

    def test_subagent_prompt_lists_room_ids_and_index_note(self):
        prompt = build_subagent_system_prompt(
            data_rooms=[{"id": r["id"], "name": r["name"], "description": r["description"]} for r in self.ROOMS],
        )
        self.assertIn('- "Acme" (data_room_id 7)', prompt)
        self.assertIn("restart at 1", prompt)

    def test_retrieved_documents_name_the_room_id(self):
        prompt = build_system_prompt(
            data_rooms=self.ROOMS,
            doc_context={"total_doc_count": 4, "documents": [
                {"doc_index": 1, "filename": "term_sheet.txt", "data_room_name": "Acme", "data_room_id": 7},
                {"doc_index": 1, "filename": "term_sheet.txt", "data_room_name": "Beta", "data_room_id": 8},
            ]},
        )
        self.assertIn('[1] "term_sheet.txt" (data room: Acme, data_room_id 7)', prompt)
        self.assertIn('[1] "term_sheet.txt" (data room: Beta, data_room_id 8)', prompt)
