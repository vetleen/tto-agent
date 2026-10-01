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
from django.urls import reverse

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
        reviewer_reasoning=(
            "Reviewer: prompt_injection (confidence: 0.95)" if status == "warn"
            else ("Row 3 states a named patient's diagnosis." if status == "blocked" else None)
        ),
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
            # The reviewer's user-facing finding rides along so the agent can remediate.
            self.assertEqual(res["reviewer_finding"], "Row 3 states a named patient's diagnosis.")
            self.assertEqual(self._attempts(), expected)
            self.assertEqual(self.doc.versions.count(), 1)  # rejected version discarded

        # Attempt 3: deferred — keep the quarantined draft, reset counter, warn the user.
        res = self._overwrite(_verdict("blocked"))
        self.assertEqual(res["verdict"], "deferred")
        self.assertEqual(res["reviewer_finding"], "Row 3 states a named patient's diagnosis.")
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

    # The button hands the save to the async pipeline and polls the verdict: the inline
    # scan took 25–45s, past Heroku's 30s router timeout (WILFRED-8R).
    @patch("documents.tasks.process_document_version_task.delay")
    def _post(self, mock_delay):
        url = f"/chat/api/threads/{self.thread.id}/canvas/{self.canvas.id}/save-to-data-room/"
        with patch(_SYNC_SCAN) as scan:
            resp = self.client.post(url, data=json.dumps({"data_room_id": self.room.pk}),
                                    content_type="application/json")
        scan.assert_not_called()  # nothing scans inline on the web dyno
        return resp, mock_delay

    def _verdict_url(self, doc, version):
        return reverse("document_version_verdict", kwargs={
            "data_room_id": self.room.uuid, "document_id": doc.id, "version_id": version.id,
        })

    def test_save_is_queued_with_verdict_url(self):
        before = DataRoomDocument.objects.filter(data_room=self.room).count()
        resp, mock_delay = self._post()
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertTrue(data["saved"])
        self.assertEqual(data["verdict"], "queued")
        self.assertEqual(data["data_room_name"], "R")
        self.assertEqual(DataRoomDocument.objects.filter(data_room=self.room).count(), before + 1)
        doc = DataRoomDocument.objects.get(pk=data["document_id"])
        version = DataRoomDocumentVersion.objects.get(pk=data["version_id"])
        self.assertEqual(version.document_id, doc.id)
        self.assertEqual(version.origin, Origin.CANVAS_EXPORT)
        # Joined the dispatch gate like an upload (mark_version_queued + safe_dispatch).
        self.assertIsNotNone(version.queued_at)
        mock_delay.assert_called_once_with(version.id)
        self.assertEqual(data["verdict_url"], self._verdict_url(doc, version))

    def test_verdict_url_reports_pending_then_clean(self):
        data = self._post()[0].json()
        url = data["verdict_url"]
        self.assertTrue(self.client.get(url).json()["pending"])

        DataRoomDocumentVersion.objects.filter(pk=data["version_id"]).update(status=Status.READY)
        polled = self.client.get(url).json()
        self.assertFalse(polled["pending"])
        self.assertTrue(polled["ok"])
        self.assertEqual(polled["verdict"], "clean")

    def test_verdict_url_reports_blocked_draft_with_reason(self):
        data = self._post()[0].json()
        DataRoomDocumentVersion.objects.filter(pk=data["version_id"]).update(
            status=Status.READY, is_quarantined=True,
            quarantine_reason="Contains GDPR Article 9 (special category) personal data.",
        )
        polled = self.client.get(data["verdict_url"]).json()
        self.assertFalse(polled["pending"])
        self.assertFalse(polled["ok"])
        self.assertEqual(polled["verdict"], "blocked")
        self.assertIn("Article 9", polled["reason"])
        # The draft persists (not discarded for the button path).
        self.assertTrue(DataRoomDocument.objects.filter(pk=data["document_id"]).exists())


class CanvasBroadcastRegressionTests(TestCase):
    def test_document_open_to_canvas_is_broadcast(self):
        # Guards the regression where the tool ran but its canvas never reached the UI.
        from chat.consumers import CANVAS_UPDATED_TOOLS
        self.assertIn("document_open_to_canvas", CANVAS_UPDATED_TOOLS)

    def test_new_canvas_ingress_tools_are_broadcast(self):
        # Every canvas-mutating tool must be here or its canvas never refreshes.
        from chat.consumers import CANVAS_UPDATED_TOOLS
        self.assertIn("canvas_paste_user_text", CANVAS_UPDATED_TOOLS)
        self.assertIn("chat_attachment_open_to_canvas", CANVAS_UPDATED_TOOLS)


class CanvasTruncationTests(TestCase):
    """A canvas cut to CANVAS_MAX_CHARS says so, remembers it, and can't silently
    replace a document with the partial copy (confirm first; warn on new saves)."""

    def setUp(self):
        from chat.services import CANVAS_MAX_CHARS

        self.cap = CANVAS_MAX_CHARS
        self.user = User.objects.create_user(email="trunc@x.io", password="p")
        self.room = DataRoom.objects.create(name="R", slug="r", created_by=self.user)
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.ctx = RunContext.create(
            user_id=self.user.pk, conversation_id=str(self.thread.id), data_room_ids=[self.room.pk],
        )
        self.big_doc = make_document(self.room, self.user, chunks=["x" * (self.cap + 5_000)])
        self.small_doc = make_document(self.room, self.user, chunks=["short body"])

    def _tool(self, cls):
        tool = cls()
        tool.set_context(self.ctx)
        return tool

    def _open(self, doc, canvas_name="Big"):
        return json.loads(self._tool(OpenDocumentToCanvasTool).invoke(
            {"doc_index": doc.doc_index, "canvas_name": canvas_name},
        ))

    def _save(self, verdict="clean", **args):
        with patch(_SYNC_SCAN, return_value=_verdict(verdict)) as scan:
            res = json.loads(self._tool(CanvasSaveToDocumentTool).invoke({"canvas_name": "Big", **args}))
        return res, scan

    # -- document_open_to_canvas --------------------------------------------------

    def test_open_oversized_document_reports_and_records_truncation(self):
        res = self._open(self.big_doc)
        self.assertEqual(res["status"], "ok")
        self.assertTrue(res["truncated"])
        self.assertEqual(res["original_chars"], self.cap + 5_000)
        self.assertEqual(res["kept_chars"], self.cap)
        self.assertIn("TRUNCATED", res["truncation_note"])
        self.assertIn("document_read", res["truncation_note"])
        canvas = ChatCanvas.objects.get(thread=self.thread, title="Big")
        self.assertEqual(len(canvas.content), self.cap)
        self.assertEqual(canvas.truncated_from_chars, self.cap + 5_000)
        self.assertEqual(canvas.source_document_id, self.big_doc.pk)

    def test_open_small_document_has_no_truncation(self):
        res = self._open(self.small_doc, canvas_name="Small")
        self.assertNotIn("truncated", res)
        canvas = ChatCanvas.objects.get(thread=self.thread, title="Small")
        self.assertIsNone(canvas.truncated_from_chars)
        self.assertEqual(canvas.source_document_id, self.small_doc.pk)

    def test_reopening_a_small_document_clears_the_marker(self):
        self._open(self.big_doc)
        self._open(self.small_doc)  # same canvas title "Big"
        canvas = ChatCanvas.objects.get(thread=self.thread, title="Big")
        self.assertIsNone(canvas.truncated_from_chars)
        self.assertEqual(canvas.source_document_id, self.small_doc.pk)

    # -- canvas_save_to_document ----------------------------------------------------

    def test_overwrite_source_with_truncated_canvas_needs_confirmation(self):
        self._open(self.big_doc)
        prior_current = DataRoomDocument.objects.get(pk=self.big_doc.pk).current_version_id
        versions_before = self.big_doc.versions.count()

        res, scan = self._save(mode="overwrite", doc_index=self.big_doc.doc_index)

        self.assertEqual(res["status"], "confirmation_required")
        self.assertEqual(res["reason"], "truncated_canvas")
        self.assertEqual(res["original_chars"], self.cap + 5_000)
        self.assertIn("TRUNCATED copy of", res["message"])
        self.assertIn("mode='new'", res["message"])
        self.assertIn("acknowledge_truncation=true", res["message"])
        scan.assert_not_called()
        self.assertEqual(self.big_doc.versions.count(), versions_before)
        self.assertEqual(DataRoomDocument.objects.get(pk=self.big_doc.pk).current_version_id, prior_current)
        self.assertEqual(ChatCanvas.objects.get(thread=self.thread, title="Big").dr_save_attempts, 0)

    def test_overwrite_other_document_with_truncated_canvas_needs_confirmation(self):
        self._open(self.big_doc)
        res, _scan = self._save(mode="overwrite", doc_index=self.small_doc.doc_index)
        self.assertEqual(res["status"], "confirmation_required")
        self.assertIn("would be replaced", res["message"])
        self.assertEqual(self.small_doc.versions.count(), 1)

    def test_acknowledged_overwrite_saves_with_warning(self):
        self._open(self.big_doc)
        res, scan = self._save(
            mode="overwrite", doc_index=self.big_doc.doc_index, acknowledge_truncation=True,
        )
        self.assertEqual(res["verdict"], "clean")
        scan.assert_called_once()
        self.assertEqual(self.big_doc.versions.count(), 2)
        self.assertIn("rolled back", res["truncation_warning"])

    def test_new_mode_saves_truncated_canvas_with_partial_copy_warning(self):
        self._open(self.big_doc)
        res, _scan = self._save(mode="new", new_name="Partial")
        self.assertEqual(res["verdict"], "clean")
        self.assertIn("PARTIAL copy", res["truncation_warning"])

    def test_complete_canvas_saves_without_truncation_keys(self):
        self._open(self.small_doc, canvas_name="Big")
        res, _scan = self._save(mode="overwrite", doc_index=self.small_doc.doc_index)
        self.assertEqual(res["verdict"], "clean")
        self.assertNotIn("truncation_warning", res)

    def test_confirmation_result_has_a_not_saved_label(self):
        label = CanvasSaveToDocumentTool().end_label_for_result({"status": "confirmation_required"})
        self.assertEqual(label, "Not saved: canvas is truncated")

    # -- UI save button -------------------------------------------------------------

    @patch("documents.tasks.process_document_version_task.delay")
    def test_button_save_reports_partial_copy(self, _mock_delay):
        self.client.login(email="trunc@x.io", password="p")
        canvas = ChatCanvas.objects.create(
            thread=self.thread, title="Btn", content="y" * self.cap, truncated_from_chars=self.cap + 10,
        )
        url = f"/chat/api/threads/{self.thread.id}/canvas/{canvas.id}/save-to-data-room/"
        data = self.client.post(url, data=json.dumps({"data_room_id": self.room.pk}),
                                content_type="application/json").json()
        self.assertTrue(data["saved"])
        self.assertTrue(data["truncated"])
        self.assertEqual(data["original_chars"], self.cap + 10)
        self.assertEqual(data["kept_chars"], self.cap)
