"""Tests for documents.tasks.requeue_stale_documents.

The sweeper operates on *versions* (the processing unit). The real pipeline is
never run — ``process_document_version_task.delay`` is mocked.
"""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.utils import timezone

from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentVersion
from documents.services.pii_scan import SCAN_DISPATCH_RETRY_MESSAGE, SCAN_FAILED_MESSAGE
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


class ScanDispatchRetryRecoveryTests(TestCase):
    """The sweeper auto-recovers versions a broker blip left SCAN_FAILED with the
    transient SCAN_DISPATCH_RETRY_MESSAGE marker (B-robust)."""

    def setUp(self):
        self.user = User.objects.create_user(email="scanretry@example.com", password="pw")
        self.data_room = DataRoom.objects.create(
            name="ScanRetry", slug="scanretry", created_by=self.user,
        )

    def _scan_failed(self, *, marker, requeue_count=0):
        """A fresh-upload document whose (v0) version is SCAN_FAILED with *marker*."""
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="doc.txt",
            status=DataRoomDocument.Status.SCAN_FAILED,
        )
        version = make_version(
            doc, status=DataRoomDocument.Status.SCAN_FAILED, make_active=False, searchable=False,
        )
        DataRoomDocumentVersion.objects.filter(pk=version.pk).update(
            processing_error=marker, requeue_count=requeue_count,
        )
        doc.refresh_from_db()
        return doc

    def _version(self, doc):
        return DataRoomDocumentVersion.objects.get(pk=doc.current_version_id)

    @patch("guardrails.tasks.scan_document_version.delay")
    def test_retryable_marker_redispatched(self, mock_delay):
        doc = self._scan_failed(marker=SCAN_DISPATCH_RETRY_MESSAGE, requeue_count=0)

        requeue_stale_documents()

        version = self._version(doc)
        self.assertEqual(version.status, DataRoomDocument.Status.SCANNING)
        self.assertIsNone(version.processing_error)
        self.assertEqual(version.requeue_count, 1)
        mock_delay.assert_called_once_with(doc.current_version_id)
        # Mirrored onto the fresh-upload document.
        doc.refresh_from_db()
        self.assertEqual(doc.status, DataRoomDocument.Status.SCANNING)

    @patch("guardrails.tasks.scan_document_version.delay")
    def test_exhausted_marker_goes_terminal(self, mock_delay):
        doc = self._scan_failed(marker=SCAN_DISPATCH_RETRY_MESSAGE, requeue_count=MAX_REQUEUES)

        requeue_stale_documents()

        version = self._version(doc)
        self.assertEqual(version.status, DataRoomDocument.Status.SCAN_FAILED)
        self.assertEqual(version.processing_error, SCAN_FAILED_MESSAGE)  # terminal now
        mock_delay.assert_not_called()

    @patch("guardrails.tasks.scan_document_version.delay")
    def test_genuine_scan_failed_untouched(self, mock_delay):
        # A real scan failure (terminal message, not the retry marker) is left alone.
        doc = self._scan_failed(marker=SCAN_FAILED_MESSAGE, requeue_count=0)

        requeue_stale_documents()

        version = self._version(doc)
        self.assertEqual(version.status, DataRoomDocument.Status.SCAN_FAILED)
        self.assertEqual(version.processing_error, SCAN_FAILED_MESSAGE)
        self.assertEqual(version.requeue_count, 0)
        mock_delay.assert_not_called()

    @patch(
        "guardrails.tasks.scan_document_version.delay",
        side_effect=RuntimeError("broker still down"),
    )
    def test_broker_still_down_reverts_to_retry_marker(self, mock_delay):
        doc = self._scan_failed(marker=SCAN_DISPATCH_RETRY_MESSAGE, requeue_count=0)

        requeue_stale_documents()

        version = self._version(doc)
        # Reverted to SCAN_FAILED + the retry marker so the next tick retries; the
        # attempt is still counted (bounds a sustained outage).
        self.assertEqual(version.status, DataRoomDocument.Status.SCAN_FAILED)
        self.assertEqual(version.processing_error, SCAN_DISPATCH_RETRY_MESSAGE)
        self.assertEqual(version.requeue_count, 1)
        mock_delay.assert_called_once_with(doc.current_version_id)


@override_settings(
    CACHES={"default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache", "LOCATION": "kick-test",
    }}
)
class FinalizeRecoveryKickTests(TestCase):
    """A successful finalize opportunistically kicks the sweeper (Option 2), throttled
    to ~once/45s so a draining batch fires it roughly once rather than per-document."""

    @patch("documents.tasks.requeue_stale_documents.delay")
    @patch("documents.tasks.finalize_version")  # no-op the real finalize body
    def test_finalize_kick_fires_once_then_throttles(self, _mock_final, mock_requeue):
        from django.core.cache import cache
        from documents.tasks import finalize_document_metadata

        cache.clear()
        finalize_document_metadata(1)  # throttle key free → kicks recovery
        finalize_document_metadata(2)  # key held → skipped
        self.assertEqual(mock_requeue.call_count, 1)

    @patch("documents.tasks.requeue_stale_documents.delay", side_effect=RuntimeError("broker down"))
    @patch("documents.tasks.finalize_version")
    def test_finalize_survives_kick_dispatch_failure(self, _mock_final, _mock_requeue):
        from django.core.cache import cache
        from documents.tasks import finalize_document_metadata

        cache.clear()
        # A broker blip on the recovery kick must not propagate out of finalize.
        finalize_document_metadata(1)
