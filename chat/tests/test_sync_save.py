"""Tests for synchronous scan-at-save wiring in the chat tools, the origin gate that
locks quarantined uploads, the button endpoint, and the document_open_to_canvas
broadcast regression. The scan itself (scan_version_synchronously) is mocked to return
controlled verdicts — the scan logic is covered in documents.tests.test_sync_scan.
"""
from __future__ import annotations

import json
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase

from chat.models import ChatCanvas, ChatThread
from chat.tools import CanvasSaveToDocumentTool, EditDocumentTool, OpenDocumentToCanvasTool
from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentVersion
from documents.tests._helpers import make_document, make_version
from llm.types.context import RunContext

User = get_user_model()
Origin = DataRoomDocumentVersion.Origin
Status = DataRoomDocument.Status

_SYNC_SCAN = "documents.services.sync_scan.scan_version_synchronously"


def _verdict(status, **kw):
    from documents.services.sync_scan import Verdict
    defaults = dict(
        status=status,
        is_quarantined=(status == "blocked"),
        is_partially_quarantined=(status == "warn"),
        reasons=(
            ["Contains GDPR Article 9 (special category) personal data."] if status == "blocked"
            else (["Reviewer: prompt_injection (confidence: 0.95)"] if status == "warn" else [])
        ),
        reviewer_reasoning=("Reviewer: prompt_injection (confidence: 0.95)" if status == "warn" else None),
        version_index=1,
        became_active=(status in ("clean", "warn")),
    )
    defaults.update(kw)
    return Verdict(**defaults)


class CanvasSaveRetryPolicyTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="cs@x.io", password="p")
        self.room = DataRoom.objects.create(name="R", slug="r", created_by=self.user)
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.canvas = ChatCanvas.objects.create(thread=self.thread, title="C", content="canvas body text")
        self.doc = make_document(self.room, self.user, chunks=["v0 content"])
        self.ctx = RunContext.create(
            user_id=self.user.pk, conversation_id=str(self.thread.id), data_room_ids=[self.room.pk],
        )

    def _overwrite(self, verdict):
        tool = CanvasSaveToDocumentTool()
        tool.set_context(self.ctx)
        with patch(_SYNC_SCAN, return_value=verdict):
            return json.loads(tool.invoke({"mode": "overwrite", "doc_index": self.doc.doc_index, "canvas_name": "C"}))

    def _attempts(self):
        return ChatCanvas.objects.get(pk=self.canvas.pk).dr_save_attempts

    def test_blocked_attempts_discard_and_count_then_defer(self):
        # Attempts 1 & 2: blocked, version discarded, counter climbs.
        for expected in (1, 2):
            res = self._overwrite(_verdict("blocked"))
            self.assertEqual(res["verdict"], "blocked")
            self.assertEqual(self._attempts(), expected)
            self.assertEqual(self.doc.versions.count(), 1)  # rejected version discarded

        # Attempt 3: deferred — keep the quarantined draft, reset counter, warn the user.
        res = self._overwrite(_verdict("blocked"))
        self.assertEqual(res["verdict"], "deferred")
        self.assertEqual(self._attempts(), 0)
        self.assertEqual(self.doc.versions.count(), 2)  # draft kept

    def test_clean_save_resets_counter_and_keeps_version(self):
        self._overwrite(_verdict("blocked"))
        self.assertEqual(self._attempts(), 1)
        res = self._overwrite(_verdict("clean"))
        self.assertEqual(res["verdict"], "clean")
        self.assertEqual(self._attempts(), 0)
        self.assertEqual(self.doc.versions.count(), 2)

    def test_warn_is_success_and_does_not_consume_budget(self):
        res = self._overwrite(_verdict("warn"))
        self.assertEqual(res["verdict"], "warn")
        self.assertEqual(self._attempts(), 0)
        self.assertEqual(self.doc.versions.count(), 2)

    def test_new_mode_blocked_discards_whole_doc(self):
        tool = CanvasSaveToDocumentTool()
        tool.set_context(self.ctx)
        before = DataRoomDocument.objects.filter(data_room=self.room).count()
        with patch(_SYNC_SCAN, return_value=_verdict("blocked")):
            res = json.loads(tool.invoke({"mode": "new", "new_name": "Fresh", "canvas_name": "C"}))
        self.assertEqual(res["verdict"], "blocked")
        self.assertEqual(DataRoomDocument.objects.filter(data_room=self.room).count(), before)


class EditDocumentSyncTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="ed@x.io", password="p")
        self.room = DataRoom.objects.create(name="R", slug="r", created_by=self.user)
        self.ctx = RunContext.create(user_id=self.user.pk, data_room_ids=[self.room.pk])
        self.doc = make_document(self.room, self.user, chunks=["original text here"])

    def test_blocked_edit_is_discarded(self):
        tool = EditDocumentTool()
        tool.set_context(self.ctx)
        with patch(_SYNC_SCAN, return_value=_verdict("blocked")):
            res = json.loads(tool.invoke({
                "doc_index": self.doc.doc_index, "mode": "edit",
                "edits": [{"old_text": "original", "new_text": "changed"}],
            }))
        self.assertEqual(res["verdict"], "blocked")
        self.assertEqual(self.doc.versions.count(), 1)  # rejected version rolled back

    def test_clean_edit_is_kept(self):
        tool = EditDocumentTool()
        tool.set_context(self.ctx)
        with patch(_SYNC_SCAN, return_value=_verdict("clean")):
            res = json.loads(tool.invoke({
                "doc_index": self.doc.doc_index, "mode": "edit",
                "edits": [{"old_text": "original", "new_text": "changed"}],
            }))
        self.assertEqual(res["verdict"], "clean")
        self.assertEqual(self.doc.versions.count(), 2)


class OriginGateTests(TestCase):
    """Quarantined uploads are locked from the editing tools; agent drafts stay editable."""

    def setUp(self):
        self.user = User.objects.create_user(email="og@x.io", password="p")
        self.room = DataRoom.objects.create(name="R", slug="r", created_by=self.user)
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.ctx = RunContext.create(
            user_id=self.user.pk, conversation_id=str(self.thread.id), data_room_ids=[self.room.pk],
        )

    def _quarantined_doc(self, origin):
        doc = DataRoomDocument.objects.create(
            data_room=self.room, uploaded_by=self.user,
            original_filename="q.md", status=Status.READY, is_quarantined=True,
        )
        make_version(doc, origin=origin, is_quarantined=True, status=Status.READY,
                     searchable=False, chunks=["sensitive text"])
        return doc

    def test_quarantined_upload_is_locked_for_open_to_canvas(self):
        doc = self._quarantined_doc(Origin.UPLOADED)
        tool = OpenDocumentToCanvasTool()
        tool.set_context(self.ctx)
        res = json.loads(tool.invoke({"doc_index": doc.doc_index}))
        self.assertIn("locked", res.get("error", ""))

    def test_quarantined_upload_is_locked_for_edit(self):
        doc = self._quarantined_doc(Origin.UPLOADED)
        tool = EditDocumentTool()
        tool.set_context(self.ctx)
        res = json.loads(tool.invoke({
            "doc_index": doc.doc_index, "mode": "rewrite", "content": "new clean text",
        }))
        self.assertIn("locked", res.get("error", ""))

    def test_quarantined_agent_draft_is_openable(self):
        doc = self._quarantined_doc(Origin.CANVAS_EXPORT)
        tool = OpenDocumentToCanvasTool()
        tool.set_context(self.ctx)
        res = json.loads(tool.invoke({"doc_index": doc.doc_index}))
        self.assertNotIn("locked", res.get("error", ""))
        self.assertEqual(res.get("status"), "ok")


class SaveToDataRoomEndpointTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="btn@x.io", password="p")
        self.client.login(email="btn@x.io", password="p")
        self.room = DataRoom.objects.create(name="R", slug="r", created_by=self.user)
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.canvas = ChatCanvas.objects.create(thread=self.thread, title="C", content="button body text")

    def _post(self, verdict):
        url = f"/chat/api/threads/{self.thread.id}/canvas/{self.canvas.id}/save-to-data-room/"
        with patch(_SYNC_SCAN, return_value=verdict):
            resp = self.client.post(url, data=json.dumps({"data_room_id": self.room.pk}),
                                    content_type="application/json")
        return resp

    def test_clean_save_is_ok(self):
        resp = self._post(_verdict("clean"))
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["verdict"], "clean")

    def test_blocked_save_kept_as_quarantined_draft_with_reason(self):
        before = DataRoomDocument.objects.filter(data_room=self.room).count()
        resp = self._post(_verdict("blocked"))
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["ok"])
        self.assertEqual(data["verdict"], "blocked")
        self.assertIn("Article 9", data["reason"])
        self.assertTrue(data["saved"])
        # The draft persists (not discarded for the button path).
        self.assertEqual(DataRoomDocument.objects.filter(data_room=self.room).count(), before + 1)


class CanvasBroadcastRegressionTests(TestCase):
    def test_document_open_to_canvas_is_broadcast(self):
        # Guards the regression where the tool ran but its canvas never reached the UI.
        from chat.consumers import CANVAS_UPDATED_TOOLS
        self.assertIn("document_open_to_canvas", CANVAS_UPDATED_TOOLS)
