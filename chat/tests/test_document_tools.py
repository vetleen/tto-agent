"""Tests for the versioned document-management agent tools."""
from __future__ import annotations

import json

from django.contrib.auth import get_user_model
from django.test import TestCase

from chat.tools import (
    ArchiveDocumentTool,
    GetDocumentStatusTool,
    ListDocumentsTool,
    ListVersionsTool,
    RenameDocumentTool,
    RestoreVersionTool,
)
from documents.models import DataRoom
from documents.tests._helpers import make_document, make_version
from llm.types.context import RunContext

User = get_user_model()


class DocumentToolsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="tools@x.io", password="p")
        self.room = DataRoom.objects.create(name="R", slug="r", created_by=self.user)
        self.ctx = RunContext.create(user_id=self.user.pk, data_room_ids=[self.room.pk])

    def _invoke(self, tool, args):
        tool.set_context(self.ctx)
        return json.loads(tool.invoke(args))

    def test_list_documents_paginates(self):
        for i in range(3):
            make_document(self.room, self.user, original_filename=f"d{i}.md", chunks=["x"])
        res = self._invoke(ListDocumentsTool(), {"limit": 2, "offset": 0})
        self.assertEqual(res["count"], 3)
        self.assertEqual(len(res["documents"]), 2)
        self.assertIn("Showing documents 1", res["header"])
        row = res["documents"][0]
        for key in ("doc_index", "name", "origin", "status", "versions", "total_chunks"):
            self.assertIn(key, row)

    def test_archive_document(self):
        doc = make_document(self.room, self.user, chunks=["x"])
        res = self._invoke(ArchiveDocumentTool(), {"doc_index": doc.doc_index})
        self.assertEqual(res["status"], "ok")
        self.assertTrue(res["archived"])
        doc.refresh_from_db()
        self.assertTrue(doc.is_archived)

    def test_rename_sets_name_not_original_filename(self):
        doc = make_document(self.room, self.user, original_filename="orig.md", chunks=["x"])
        res = self._invoke(RenameDocumentTool(), {"doc_index": doc.doc_index, "name": "Nice Name"})
        self.assertEqual(res["status"], "ok")
        doc.refresh_from_db()
        self.assertEqual(doc.name, "Nice Name")
        self.assertEqual(doc.original_filename, "orig.md")  # provenance preserved

    def test_list_and_restore_versions(self):
        doc = make_document(self.room, self.user, chunks=["v0 content"])
        make_version(doc, version_index=1, chunks=["v1 content"])  # v1 now current+active

        listed = self._invoke(ListVersionsTool(), {"doc_index": doc.doc_index})
        self.assertEqual(len(listed["versions"]), 2)

        restored = self._invoke(
            RestoreVersionTool(), {"doc_index": doc.doc_index, "version_index": 0}
        )
        self.assertEqual(restored["restored_version"], 0)
        doc.refresh_from_db()
        v0 = doc.versions.get(version_index=0)
        self.assertEqual(doc.active_searchable_version_id, v0.id)
        self.assertEqual(doc.current_version_id, v0.id)

    def test_restore_rejects_unknown_version(self):
        doc = make_document(self.room, self.user, chunks=["x"])
        res = self._invoke(RestoreVersionTool(), {"doc_index": doc.doc_index, "version_index": 99})
        self.assertIn("error", res)

    def test_get_document_status_ready(self):
        doc = make_document(self.room, self.user, chunks=["x"])
        res = self._invoke(GetDocumentStatusTool(), {"doc_index": doc.doc_index})
        self.assertEqual(res["state"], "ready")
        self.assertEqual(res["active_version"], 0)

    def test_tools_reject_unknown_doc_index(self):
        res = self._invoke(GetDocumentStatusTool(), {"doc_index": 999})
        self.assertIn("error", res)

    def test_tools_enforce_room_access(self):
        other = User.objects.create_user(email="other@x.io", password="p")
        other_room = DataRoom.objects.create(name="O", slug="o", created_by=other)
        doc = make_document(other_room, other, chunks=["secret"])
        # Our context only has access to self.room, not other_room.
        ctx = RunContext.create(user_id=self.user.pk, data_room_ids=[other_room.pk])
        tool = GetDocumentStatusTool()
        tool.set_context(ctx)
        res = json.loads(tool.invoke({"doc_index": doc.doc_index}))
        self.assertIn("error", res)
