"""Tests for the document dispatch gate: slot-gated, arrival-order dispatch.

Mirrors chat/tests/test_subagent_dispatch.py. The real pipeline never runs —
``documents.tasks.process_document_version_task`` is mocked (the dispatcher
imports it lazily, so patching the module attribute works).
"""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.utils import timezone

from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentVersion
from documents.services.dispatch import (
    DocumentPipelineTask,
    dispatch_pending_document_versions,
    get_queue_depth,
    mark_version_queued,
    safe_dispatch,
)
from documents.services.pii_scan import SCAN_DISPATCH_RETRY_MESSAGE
from documents.tests._helpers import make_version

User = get_user_model()
Status = DataRoomDocument.Status


class _GateTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="gate@example.com", password="pw")
        self.data_room = DataRoom.objects.create(
            name="Gate", slug="gate", created_by=self.user,
        )

    def _version(
        self, status=Status.UPLOADED, *,
        queued_minutes_ago=None, dispatched_minutes_ago=None,
        processing_error=None, filename="doc.txt",
    ):
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename=filename, status=status,
        )
        version = make_version(doc, status=status, make_active=False, searchable=False)
        updates = {}
        if queued_minutes_ago is not None:
            updates["queued_at"] = timezone.now() - timedelta(minutes=queued_minutes_ago)
        if dispatched_minutes_ago is not None:
            updates["dispatched_at"] = timezone.now() - timedelta(minutes=dispatched_minutes_ago)
        if processing_error is not None:
            updates["processing_error"] = processing_error
        if updates:
            DataRoomDocumentVersion.objects.filter(pk=version.pk).update(**updates)
            version.refresh_from_db()
        return version


class DispatchPendingDocumentVersionsTests(_GateTestCase):
    @patch("documents.services.dispatch.DOCUMENT_WORKER_SLOTS", 2)
    @patch("documents.tasks.process_document_version_task")
    def test_dispatches_oldest_queued_first_up_to_free_slots(self, mock_task):
        mock_task.delay.return_value = MagicMock(id="celery-1")
        oldest = self._version(queued_minutes_ago=3)
        middle = self._version(queued_minutes_ago=2)
        newest = self._version(queued_minutes_ago=1)

        dispatched = dispatch_pending_document_versions()

        self.assertEqual(dispatched, [oldest.id, middle.id])
        for version in (oldest, middle):
            version.refresh_from_db()
            self.assertEqual(version.status, Status.UPLOADED)
            self.assertIsNotNone(version.dispatched_at)
        newest.refresh_from_db()
        self.assertIsNone(newest.dispatched_at)
        self.assertEqual(mock_task.delay.call_count, 2)
        mock_task.delay.assert_any_call(oldest.id)

    @patch("documents.services.dispatch.DOCUMENT_WORKER_SLOTS", 2)
    @patch("documents.tasks.process_document_version_task")
    def test_dispatched_nonterminal_rows_hold_slots(self, mock_task):
        # One dispatched-but-not-started, one mid-pipeline: both hold slots.
        self._version(Status.UPLOADED, queued_minutes_ago=5, dispatched_minutes_ago=4)
        self._version(Status.SCANNING, queued_minutes_ago=9, dispatched_minutes_ago=8)
        waiting = self._version(queued_minutes_ago=1)

        self.assertEqual(dispatch_pending_document_versions(), [])

        mock_task.delay.assert_not_called()
        waiting.refresh_from_db()
        self.assertIsNone(waiting.dispatched_at)

    @patch("documents.services.dispatch.DOCUMENT_WORKER_SLOTS", 1)
    @patch("documents.tasks.process_document_version_task")
    def test_terminal_rows_do_not_hold_slots(self, mock_task):
        mock_task.delay.return_value = MagicMock(id="celery-1")
        self._version(Status.READY, queued_minutes_ago=9, dispatched_minutes_ago=8)
        self._version(Status.FAILED, queued_minutes_ago=9, dispatched_minutes_ago=8)
        self._version(Status.SCAN_FAILED, queued_minutes_ago=9, dispatched_minutes_ago=8)
        waiting = self._version(queued_minutes_ago=1)

        self.assertEqual(dispatch_pending_document_versions(), [waiting.id])

    @patch("documents.services.dispatch.DOCUMENT_WORKER_SLOTS", 1)
    @patch("documents.tasks.process_document_version_task")
    def test_scan_retry_marker_rows_hold_no_slot(self, mock_task):
        # A version parked on the transient scan-retry marker is not occupying
        # the worker; its sweeper re-dispatch rides outside this gate.
        mock_task.delay.return_value = MagicMock(id="celery-1")
        self._version(
            Status.SCAN_FAILED, queued_minutes_ago=9, dispatched_minutes_ago=8,
            processing_error=SCAN_DISPATCH_RETRY_MESSAGE,
        )
        waiting = self._version(queued_minutes_ago=1)

        self.assertEqual(dispatch_pending_document_versions(), [waiting.id])

    @patch("documents.services.dispatch.DOCUMENT_WORKER_SLOTS", 4)
    @patch("documents.tasks.process_document_version_task")
    def test_sync_path_versions_are_invisible(self, mock_task):
        # queued_at NULL = a synchronous canvas/agent save mid-run on the web
        # dyno. The dispatcher must never publish it into Celery.
        sync_version = self._version(Status.UPLOADED)  # no queued_at

        self.assertEqual(dispatch_pending_document_versions(), [])

        mock_task.delay.assert_not_called()
        sync_version.refresh_from_db()
        self.assertIsNone(sync_version.dispatched_at)

    @patch("documents.services.dispatch.DOCUMENT_WORKER_SLOTS", 2)
    @patch("documents.tasks.process_document_version_task")
    def test_stale_reset_processing_row_is_redispatchable(self, mock_task):
        # The sweeper clears dispatched_at on a silent row; it rejoins the line
        # in PROCESSING status and must be dispatchable again.
        mock_task.delay.return_value = MagicMock(id="celery-1")
        version = self._version(Status.PROCESSING, queued_minutes_ago=30)

        self.assertEqual(dispatch_pending_document_versions(), [version.id])

    @patch("documents.tasks.process_document_version_task")
    def test_publish_failure_reverts_claim(self, mock_task):
        mock_task.delay.side_effect = ConnectionError("broker down")
        waiting = self._version(queued_minutes_ago=1)

        self.assertEqual(dispatch_pending_document_versions(), [])

        waiting.refresh_from_db()
        self.assertEqual(waiting.status, Status.UPLOADED)
        self.assertIsNotNone(waiting.queued_at)  # still in line
        self.assertIsNone(waiting.dispatched_at)

    @patch("documents.tasks.process_document_version_task")
    def test_publish_failure_does_not_revert_terminal_transition(self, mock_task):
        waiting = self._version(queued_minutes_ago=1)

        def finish_then_fail(version_id):
            DataRoomDocumentVersion.objects.filter(pk=version_id).update(
                status=Status.READY,
            )
            raise ConnectionError("broker down")

        mock_task.delay.side_effect = finish_then_fail

        self.assertEqual(dispatch_pending_document_versions(), [])

        waiting.refresh_from_db()
        self.assertEqual(waiting.status, Status.READY)

    @patch("documents.tasks.process_document_version_task")
    def test_noop_when_nothing_waiting(self, mock_task):
        self._version(Status.PROCESSING, queued_minutes_ago=5, dispatched_minutes_ago=4)

        self.assertEqual(dispatch_pending_document_versions(), [])

        mock_task.delay.assert_not_called()

    def test_mark_version_queued_is_idempotent(self):
        version = self._version()
        self.assertTrue(mark_version_queued(version.id))
        version.refresh_from_db()
        first = version.queued_at
        self.assertIsNotNone(first)

        self.assertFalse(mark_version_queued(version.id))
        version.refresh_from_db()
        self.assertEqual(version.queued_at, first)

    def test_get_queue_depth_counts_slots_and_waiting(self):
        self._version(Status.PROCESSING, queued_minutes_ago=5, dispatched_minutes_ago=4)
        self._version(Status.SCANNING, queued_minutes_ago=5, dispatched_minutes_ago=4)
        self._version(queued_minutes_ago=1)
        self._version(Status.READY, queued_minutes_ago=9, dispatched_minutes_ago=8)
        self._version(Status.UPLOADED)  # sync-path: neither slot nor waiting

        depth = get_queue_depth()

        self.assertEqual(depth["running"], 2)
        self.assertEqual(depth["waiting"], 1)

    @patch("documents.services.dispatch.dispatch_pending_document_versions")
    def test_safe_dispatch_swallows_dispatcher_errors(self, mock_dispatch):
        mock_dispatch.side_effect = RuntimeError("db gone")
        safe_dispatch("test")  # must not raise


class DocumentPipelineTaskHookTests(_GateTestCase):
    def test_all_pipeline_tasks_share_the_base(self):
        from documents.tasks import (
            finalize_document_metadata,
            process_document_task,
            process_document_version_task,
        )
        from guardrails.tasks import scan_document_version

        for task in (
            process_document_task,
            process_document_version_task,
            finalize_document_metadata,
            scan_document_version,
        ):
            self.assertIsInstance(task, DocumentPipelineTask)

    @patch("documents.services.dispatch.safe_dispatch")
    def test_after_return_dispatches(self, mock_safe):
        from documents.tasks import process_document_version_task

        process_document_version_task.after_return(
            "SUCCESS", None, "task-id", [1], {}, None,
        )

        mock_safe.assert_called_once()

    @patch("documents.services.dispatch.dispatch_pending_document_versions")
    def test_after_return_swallows_dispatcher_error(self, mock_dispatch):
        from documents.tasks import process_document_version_task

        mock_dispatch.side_effect = RuntimeError("boom")
        process_document_version_task.after_return(
            "FAILURE", None, "task-id", [1], {}, None,
        )  # must not raise

    @patch("documents.services.dispatch.DOCUMENT_WORKER_SLOTS", 1)
    @patch("documents.tasks.process_document_version_task.delay")
    def test_terminal_transition_then_after_return_frees_slot(self, mock_delay):
        # End-to-end: a slot-holder finishing lets after_return dispatch the
        # next waiting version. Only .delay is mocked — the task object (and its
        # real after_return hook) stays intact.
        mock_delay.return_value = MagicMock(id="celery-next")
        holder = self._version(
            Status.PROCESSING, queued_minutes_ago=9, dispatched_minutes_ago=8,
        )
        waiting = self._version(queued_minutes_ago=1)

        self.assertEqual(dispatch_pending_document_versions(), [])  # slot held

        DataRoomDocumentVersion.objects.filter(pk=holder.pk).update(status=Status.FAILED)
        from documents.tasks import process_document_version_task as real_task
        real_task.after_return("SUCCESS", None, "task-id", [holder.id], {}, None)

        waiting.refresh_from_db()
        self.assertIsNotNone(waiting.dispatched_at)
        mock_delay.assert_called_once_with(waiting.id)


class CreateVersionQueueTests(_GateTestCase):
    def _doc(self):
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="base.md", status=Status.READY,
        )
        make_version(doc, status=Status.READY)
        return doc

    @patch("documents.tasks.process_document_version_task")
    def test_enqueue_true_queues_and_dispatches(self, mock_task):
        from documents.services.versioning import create_version

        mock_task.delay.return_value = MagicMock(id="celery-1")
        doc = self._doc()

        version = create_version(
            doc, content="hello", origin=DataRoomDocumentVersion.Origin.CANVAS_EXPORT,
        )

        version.refresh_from_db()
        self.assertIsNotNone(version.queued_at)
        self.assertIsNotNone(version.dispatched_at)
        mock_task.delay.assert_called_once_with(version.id)

    @patch("documents.services.dispatch.DOCUMENT_WORKER_SLOTS", 1)
    @patch("documents.tasks.process_document_version_task")
    def test_enqueue_true_waits_when_slots_full(self, mock_task):
        from documents.services.versioning import create_version

        self._version(Status.PROCESSING, queued_minutes_ago=9, dispatched_minutes_ago=8)
        doc = self._doc()

        version = create_version(
            doc, content="hello", origin=DataRoomDocumentVersion.Origin.CANVAS_EXPORT,
        )

        version.refresh_from_db()
        self.assertIsNotNone(version.queued_at)
        self.assertIsNone(version.dispatched_at)
        mock_task.delay.assert_not_called()

    @patch("documents.services.dispatch.dispatch_pending_document_versions")
    def test_dispatcher_error_leaves_version_queued(self, mock_dispatch):
        from documents.services.versioning import create_version

        mock_dispatch.side_effect = RuntimeError("db hiccup")
        doc = self._doc()

        version = create_version(
            doc, content="hello", origin=DataRoomDocumentVersion.Origin.CANVAS_EXPORT,
        )

        version.refresh_from_db()
        self.assertEqual(version.status, Status.UPLOADED)
        self.assertIsNotNone(version.queued_at)
        self.assertIsNone(version.dispatched_at)

    def test_enqueue_false_never_joins_the_queue(self):
        from documents.services.versioning import create_version

        doc = self._doc()

        version = create_version(
            doc, content="hello",
            origin=DataRoomDocumentVersion.Origin.CANVAS_EXPORT,
            enqueue=False,
        )

        version.refresh_from_db()
        self.assertIsNone(version.queued_at)
        self.assertIsNone(version.dispatched_at)
