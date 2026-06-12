"""Tests for documents.tasks.requeue_stale_documents.

The real pipeline is never run — ``process_document_task.delay`` is mocked
(running it would also trip the ambient PII scan gate locally).
"""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.utils import timezone

from documents.models import DataRoom, DataRoomDocument
from documents.services.pii_scan import SCAN_FAILED_MESSAGE
from documents.tasks import MAX_REQUEUES, requeue_stale_documents

User = get_user_model()


class RequeueStaleDocumentsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="sweep@example.com", password="pw")
        self.data_room = DataRoom.objects.create(
            name="Sweep", slug="sweep", created_by=self.user,
        )

    def _make_doc(self, status, minutes_old=0, requeue_count=0, processed_at=None):
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="doc.txt",
            status=status,
        )
        # updated_at is auto_now — backdate via queryset update.
        DataRoomDocument.objects.filter(pk=doc.pk).update(
            updated_at=timezone.now() - timedelta(minutes=minutes_old),
            requeue_count=requeue_count,
            processed_at=processed_at,
        )
        doc.refresh_from_db()
        return doc

    @patch("documents.tasks.process_document_task.delay")
    def test_stale_uploaded_doc_requeued(self, mock_delay):
        doc = self._make_doc(DataRoomDocument.Status.UPLOADED, minutes_old=20)

        handled = requeue_stale_documents()

        self.assertEqual(handled, 1)
        mock_delay.assert_called_once_with(doc.pk)
        doc.refresh_from_db()
        self.assertEqual(doc.status, DataRoomDocument.Status.UPLOADED)
        self.assertEqual(doc.requeue_count, 1)

    @patch("documents.tasks.process_document_task.delay")
    def test_stale_processing_doc_requeued(self, mock_delay):
        doc = self._make_doc(DataRoomDocument.Status.PROCESSING, minutes_old=20)

        handled = requeue_stale_documents()

        self.assertEqual(handled, 1)
        mock_delay.assert_called_once_with(doc.pk)
        doc.refresh_from_db()
        self.assertEqual(doc.requeue_count, 1)

    @patch("documents.tasks.process_document_task.delay")
    def test_requeue_leaves_updated_at_stale(self, mock_delay):
        """The requeue must NOT refresh updated_at — process_document's
        stale-PROCESSING guard would otherwise see a fresh timestamp and
        skip the very document we just re-enqueued."""
        doc = self._make_doc(DataRoomDocument.Status.PROCESSING, minutes_old=20)
        before = doc.updated_at

        requeue_stale_documents()

        doc.refresh_from_db()
        self.assertEqual(doc.updated_at, before)

    @patch("documents.tasks.process_document_task.delay")
    def test_fresh_docs_untouched(self, mock_delay):
        uploaded = self._make_doc(DataRoomDocument.Status.UPLOADED, minutes_old=5)
        processing = self._make_doc(DataRoomDocument.Status.PROCESSING, minutes_old=5)

        handled = requeue_stale_documents()

        self.assertEqual(handled, 0)
        mock_delay.assert_not_called()
        uploaded.refresh_from_db()
        processing.refresh_from_db()
        self.assertEqual(uploaded.status, DataRoomDocument.Status.UPLOADED)
        self.assertEqual(uploaded.requeue_count, 0)
        self.assertEqual(processing.status, DataRoomDocument.Status.PROCESSING)

    @patch("documents.tasks.process_document_task.delay")
    def test_terminal_states_untouched(self, mock_delay):
        ready = self._make_doc(DataRoomDocument.Status.READY, minutes_old=120)
        failed = self._make_doc(DataRoomDocument.Status.FAILED, minutes_old=120)
        scan_failed = self._make_doc(DataRoomDocument.Status.SCAN_FAILED, minutes_old=120)

        handled = requeue_stale_documents()

        self.assertEqual(handled, 0)
        mock_delay.assert_not_called()
        for doc, status in (
            (ready, DataRoomDocument.Status.READY),
            (failed, DataRoomDocument.Status.FAILED),
            (scan_failed, DataRoomDocument.Status.SCAN_FAILED),
        ):
            doc.refresh_from_db()
            self.assertEqual(doc.status, status)

    @patch("documents.tasks.process_document_task.delay")
    def test_requeue_cap_marks_failed(self, mock_delay):
        doc = self._make_doc(
            DataRoomDocument.Status.PROCESSING,
            minutes_old=20,
            requeue_count=MAX_REQUEUES,
        )

        handled = requeue_stale_documents()

        self.assertEqual(handled, 1)
        mock_delay.assert_not_called()
        doc.refresh_from_db()
        self.assertEqual(doc.status, DataRoomDocument.Status.FAILED)
        self.assertIn("interrupted repeatedly", doc.processing_error)

    @patch("documents.tasks.process_document_task.delay")
    def test_stale_scanning_marked_scan_failed(self, mock_delay):
        doc = self._make_doc(
            DataRoomDocument.Status.SCANNING,
            minutes_old=5,  # updated_at fresh (description gen refreshes it)
            processed_at=timezone.now() - timedelta(minutes=90),
        )

        handled = requeue_stale_documents()

        self.assertEqual(handled, 1)
        mock_delay.assert_not_called()
        doc.refresh_from_db()
        self.assertEqual(doc.status, DataRoomDocument.Status.SCAN_FAILED)
        self.assertEqual(doc.processing_error, SCAN_FAILED_MESSAGE)

    @patch("documents.tasks.process_document_task.delay")
    def test_scanning_without_processed_at_falls_back_to_updated_at(self, mock_delay):
        doc = self._make_doc(
            DataRoomDocument.Status.SCANNING, minutes_old=90, processed_at=None,
        )

        handled = requeue_stale_documents()

        self.assertEqual(handled, 1)
        doc.refresh_from_db()
        self.assertEqual(doc.status, DataRoomDocument.Status.SCAN_FAILED)

    @patch("documents.tasks.process_document_task.delay")
    def test_fresh_scanning_untouched(self, mock_delay):
        doc = self._make_doc(
            DataRoomDocument.Status.SCANNING,
            minutes_old=5,
            processed_at=timezone.now() - timedelta(minutes=30),
        )

        handled = requeue_stale_documents()

        self.assertEqual(handled, 0)
        doc.refresh_from_db()
        self.assertEqual(doc.status, DataRoomDocument.Status.SCANNING)

    def test_swallows_transient_db_error(self):
        from django.db.utils import OperationalError

        with patch.object(
            DataRoomDocument.objects,
            "filter",
            side_effect=OperationalError("the database system is starting up"),
        ):
            result = requeue_stale_documents()

        self.assertEqual(result, 0)

    def test_propagates_non_transient_db_error(self):
        from django.db.utils import ProgrammingError

        with patch.object(
            DataRoomDocument.objects,
            "filter",
            side_effect=ProgrammingError("column does not exist"),
        ):
            with self.assertRaises(ProgrammingError):
                requeue_stale_documents()
