"""Tests for documents.tasks.requeue_stale_documents.

The sweeper operates on *versions* (the processing unit). The real pipeline is
never run — ``process_document_version_task.delay`` is mocked.
"""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.utils import timezone

from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentVersion
from documents.services.pii_scan import SCAN_FAILED_MESSAGE
from documents.tasks import MAX_REQUEUES, requeue_stale_documents
from documents.tests._helpers import make_version

User = get_user_model()


class RequeueStaleDocumentsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="sweep@example.com", password="pw")
        self.data_room = DataRoom.objects.create(
            name="Sweep", slug="sweep", created_by=self.user,
        )

    def _make(self, status, minutes_old=0, requeue_count=0, processed_at=None):
        """Create a fresh-upload document whose working (v0) version has *status*."""
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="doc.txt",
            status=status,
        )
        version = make_version(doc, status=status, make_active=False, searchable=False)
        # updated_at is auto_now — backdate via queryset update so the staleness
        # windows fire; set per-version requeue_count / processed_at too.
        DataRoomDocumentVersion.objects.filter(pk=version.pk).update(
            updated_at=timezone.now() - timedelta(minutes=minutes_old),
            requeue_count=requeue_count,
            processed_at=processed_at,
        )
        doc.refresh_from_db()
        return doc

    def _version(self, doc):
        return DataRoomDocumentVersion.objects.get(pk=doc.current_version_id)

    @patch("documents.tasks.process_document_version_task.delay")
    def test_stale_uploaded_version_requeued(self, mock_delay):
        doc = self._make(DataRoomDocument.Status.UPLOADED, minutes_old=20)

        handled = requeue_stale_documents()

        self.assertEqual(handled, 1)
        mock_delay.assert_called_once_with(doc.current_version_id)
        self.assertEqual(self._version(doc).requeue_count, 1)

    @patch("documents.tasks.process_document_version_task.delay")
    def test_stale_processing_version_requeued(self, mock_delay):
        doc = self._make(DataRoomDocument.Status.PROCESSING, minutes_old=20)

        handled = requeue_stale_documents()

        self.assertEqual(handled, 1)
        mock_delay.assert_called_once_with(doc.current_version_id)
        self.assertEqual(self._version(doc).requeue_count, 1)

    @patch("documents.tasks.process_document_version_task.delay")
    def test_requeue_leaves_updated_at_stale(self, mock_delay):
        """The requeue must NOT refresh the version's updated_at — the stale
        guard in process_document_version would otherwise skip it."""
        doc = self._make(DataRoomDocument.Status.PROCESSING, minutes_old=20)
        before = self._version(doc).updated_at

        requeue_stale_documents()

        self.assertEqual(self._version(doc).updated_at, before)

    @patch("documents.tasks.process_document_version_task.delay")
    def test_fresh_versions_untouched(self, mock_delay):
        uploaded = self._make(DataRoomDocument.Status.UPLOADED, minutes_old=5)
        processing = self._make(DataRoomDocument.Status.PROCESSING, minutes_old=5)

        handled = requeue_stale_documents()

        self.assertEqual(handled, 0)
        mock_delay.assert_not_called()
        self.assertEqual(self._version(uploaded).requeue_count, 0)
        self.assertEqual(self._version(uploaded).status, DataRoomDocument.Status.UPLOADED)
        self.assertEqual(self._version(processing).status, DataRoomDocument.Status.PROCESSING)

    @patch("documents.tasks.process_document_version_task.delay")
    def test_terminal_states_untouched(self, mock_delay):
        ready = self._make(DataRoomDocument.Status.READY, minutes_old=120)
        failed = self._make(DataRoomDocument.Status.FAILED, minutes_old=120)
        scan_failed = self._make(DataRoomDocument.Status.SCAN_FAILED, minutes_old=120)

        handled = requeue_stale_documents()

        self.assertEqual(handled, 0)
        mock_delay.assert_not_called()
        for doc, status in (
            (ready, DataRoomDocument.Status.READY),
            (failed, DataRoomDocument.Status.FAILED),
            (scan_failed, DataRoomDocument.Status.SCAN_FAILED),
        ):
            self.assertEqual(self._version(doc).status, status)

    @patch("documents.tasks.process_document_version_task.delay")
    def test_requeue_cap_marks_failed(self, mock_delay):
        doc = self._make(
            DataRoomDocument.Status.PROCESSING, minutes_old=20, requeue_count=MAX_REQUEUES,
        )

        handled = requeue_stale_documents()

        self.assertEqual(handled, 1)
        mock_delay.assert_not_called()
        self.assertEqual(self._version(doc).status, DataRoomDocument.Status.FAILED)
        # Mirrored onto the (fresh-upload) document too.
        doc.refresh_from_db()
        self.assertEqual(doc.status, DataRoomDocument.Status.FAILED)

    @patch("documents.tasks.process_document_version_task.delay")
    def test_stale_scanning_marked_scan_failed(self, mock_delay):
        doc = self._make(
            DataRoomDocument.Status.SCANNING,
            minutes_old=5,  # updated_at fresh (description gen refreshes it)
            processed_at=timezone.now() - timedelta(minutes=90),
        )

        handled = requeue_stale_documents()

        self.assertEqual(handled, 1)
        mock_delay.assert_not_called()
        self.assertEqual(self._version(doc).status, DataRoomDocument.Status.SCAN_FAILED)
        self.assertEqual(self._version(doc).processing_error, SCAN_FAILED_MESSAGE)

    @patch("documents.tasks.process_document_version_task.delay")
    def test_scanning_without_processed_at_falls_back_to_updated_at(self, mock_delay):
        doc = self._make(
            DataRoomDocument.Status.SCANNING, minutes_old=90, processed_at=None,
        )

        handled = requeue_stale_documents()

        self.assertEqual(handled, 1)
        self.assertEqual(self._version(doc).status, DataRoomDocument.Status.SCAN_FAILED)

    @patch("documents.tasks.process_document_version_task.delay")
    def test_fresh_scanning_untouched(self, mock_delay):
        doc = self._make(
            DataRoomDocument.Status.SCANNING,
            minutes_old=5,
            processed_at=timezone.now() - timedelta(minutes=30),
        )

        handled = requeue_stale_documents()

        self.assertEqual(handled, 0)
        self.assertEqual(self._version(doc).status, DataRoomDocument.Status.SCANNING)

    def test_swallows_transient_db_error(self):
        from django.db.utils import OperationalError

        with patch.object(
            DataRoomDocumentVersion.objects,
            "filter",
            side_effect=OperationalError("the database system is starting up"),
        ):
            result = requeue_stale_documents()

        self.assertEqual(result, 0)

    def test_propagates_non_transient_db_error(self):
        from django.db.utils import ProgrammingError

        with patch.object(
            DataRoomDocumentVersion.objects,
            "filter",
            side_effect=ProgrammingError("column does not exist"),
        ):
            with self.assertRaises(ProgrammingError):
                requeue_stale_documents()
