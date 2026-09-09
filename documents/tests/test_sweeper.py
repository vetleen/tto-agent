"""Tests for documents.tasks.requeue_stale_documents.

The sweeper operates on *versions* (the processing unit). With the dispatch
gate, stale recovery means clearing the ``dispatched_at`` claim so the version
rejoins the queue (never a direct ``.delay``); never-queued orphans (stranded
synchronous saves) are given a ``queued_at`` instead. The end-of-sweep
dispatcher kick is isolated via a ``safe_dispatch`` patch so these tests assert
queue-state transitions, not dispatch behaviour (test_dispatch.py covers that).
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


class _SweeperTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="sweep@example.com", password="pw")
        self.data_room = DataRoom.objects.create(
            name="Sweep", slug="sweep", created_by=self.user,
        )
        # Isolate the end-of-sweep dispatcher kick; asserted explicitly in
        # SweeperDispatchKickTests.
        patcher = patch("documents.tasks.safe_dispatch")
        self.mock_safe_dispatch = patcher.start()
        self.addCleanup(patcher.stop)

    def _make(
        self, status, minutes_old=0, requeue_count=0, processed_at=None,
        queued_minutes_ago=None, dispatched_minutes_ago=None,
    ):
        """Create a fresh-upload document whose working (v0) version has *status*."""
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="doc.txt",
            status=status,
        )
        version = make_version(doc, status=status, make_active=False, searchable=False)
        # updated_at is auto_now — backdate via queryset update so the staleness
        # windows fire; set the queue/claim timestamps the same way.
        now = timezone.now()
        DataRoomDocumentVersion.objects.filter(pk=version.pk).update(
            updated_at=now - timedelta(minutes=minutes_old),
            requeue_count=requeue_count,
            processed_at=processed_at,
            queued_at=(
                now - timedelta(minutes=queued_minutes_ago)
                if queued_minutes_ago is not None else None
            ),
            dispatched_at=(
                now - timedelta(minutes=dispatched_minutes_ago)
                if dispatched_minutes_ago is not None else None
            ),
        )
        doc.refresh_from_db()
        return doc

    def _version(self, doc):
        return DataRoomDocumentVersion.objects.get(pk=doc.current_version_id)


class RequeueStaleDocumentsTests(_SweeperTestCase):
    @patch("documents.tasks.process_document_version_task.delay")
    def test_stale_dispatched_uploaded_returned_to_queue(self, mock_delay):
        # Dispatched 20 min ago, never started: claim cleared, budget spent,
        # NO direct .delay — re-entry goes through the gate.
        doc = self._make(
            DataRoomDocument.Status.UPLOADED,
            minutes_old=20, queued_minutes_ago=25, dispatched_minutes_ago=20,
        )

        handled = requeue_stale_documents()

        self.assertEqual(handled, 1)
        mock_delay.assert_not_called()
        version = self._version(doc)
        self.assertEqual(version.requeue_count, 1)
        self.assertIsNone(version.dispatched_at)
        self.assertIsNotNone(version.queued_at)  # keeps its place in line

    @patch("documents.tasks.process_document_version_task.delay")
    def test_stale_dispatched_processing_returned_to_queue(self, mock_delay):
        doc = self._make(
            DataRoomDocument.Status.PROCESSING,
            minutes_old=20, queued_minutes_ago=25, dispatched_minutes_ago=20,
        )

        handled = requeue_stale_documents()

        self.assertEqual(handled, 1)
        mock_delay.assert_not_called()
        version = self._version(doc)
        self.assertEqual(version.requeue_count, 1)
        self.assertIsNone(version.dispatched_at)

    @patch("documents.tasks.process_document_version_task.delay")
    def test_requeue_leaves_updated_at_stale(self, mock_delay):
        """The claim reset must NOT refresh the version's updated_at — the stale
        guard in process_document_version would otherwise skip the re-run."""
        doc = self._make(
            DataRoomDocument.Status.PROCESSING,
            minutes_old=20, queued_minutes_ago=25, dispatched_minutes_ago=20,
        )
        before = self._version(doc).updated_at

        requeue_stale_documents()

        self.assertEqual(self._version(doc).updated_at, before)

    @patch("documents.tasks.process_document_version_task.delay")
    def test_waiting_uploaded_version_is_never_stale(self, mock_delay):
        """THE critical gate interaction: a version waiting in line (queued, no
        claim) may sit in UPLOADED far past the stale window — it is inert, not
        stranded, and must not be touched or penalized."""
        doc = self._make(
            DataRoomDocument.Status.UPLOADED, minutes_old=120, queued_minutes_ago=120,
        )

        handled = requeue_stale_documents()

        self.assertEqual(handled, 0)
        mock_delay.assert_not_called()
        version = self._version(doc)
        self.assertEqual(version.requeue_count, 0)
        self.assertIsNone(version.dispatched_at)
        self.assertEqual(version.status, DataRoomDocument.Status.UPLOADED)

    @patch("documents.tasks.process_document_version_task.delay")
    def test_waiting_processing_version_is_never_stale(self, mock_delay):
        # A stale-reset row waiting for re-dispatch, however long.
        doc = self._make(
            DataRoomDocument.Status.PROCESSING,
            minutes_old=120, queued_minutes_ago=120, requeue_count=1,
        )

        handled = requeue_stale_documents()

        self.assertEqual(handled, 0)
        self.assertEqual(self._version(doc).requeue_count, 1)

    @patch("documents.tasks.process_document_version_task.delay")
    def test_fresh_dispatched_versions_untouched(self, mock_delay):
        uploaded = self._make(
            DataRoomDocument.Status.UPLOADED,
            minutes_old=5, queued_minutes_ago=6, dispatched_minutes_ago=5,
        )
        processing = self._make(
            DataRoomDocument.Status.PROCESSING,
            minutes_old=5, queued_minutes_ago=6, dispatched_minutes_ago=5,
        )

        handled = requeue_stale_documents()

        self.assertEqual(handled, 0)
        mock_delay.assert_not_called()
        self.assertEqual(self._version(uploaded).requeue_count, 0)
        self.assertIsNotNone(self._version(uploaded).dispatched_at)
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
            DataRoomDocument.Status.PROCESSING,
            minutes_old=20, queued_minutes_ago=25, dispatched_minutes_ago=20,
            requeue_count=MAX_REQUEUES,
        )

        handled = requeue_stale_documents()

        self.assertEqual(handled, 1)
        mock_delay.assert_not_called()
        self.assertEqual(self._version(doc).status, DataRoomDocument.Status.FAILED)
        # Mirrored onto the (fresh-upload) document too.
        doc.refresh_from_db()
        self.assertEqual(doc.status, DataRoomDocument.Status.FAILED)


class SweeperOrphanRecoveryTests(_SweeperTestCase):
    """Never-queued orphans: a synchronous save (enqueue=False) killed mid-run
    by a web-dyno restart joins the dispatch queue instead of being re-delayed
    straight into Celery."""

    @patch("documents.tasks.process_document_version_task.delay")
    def test_stranded_sync_version_joins_queue(self, mock_delay):
        doc = self._make(DataRoomDocument.Status.PROCESSING, minutes_old=20)

        handled = requeue_stale_documents()

        self.assertEqual(handled, 1)
        mock_delay.assert_not_called()
        version = self._version(doc)
        self.assertIsNotNone(version.queued_at)
        self.assertIsNone(version.dispatched_at)
        self.assertEqual(version.requeue_count, 1)

    @patch("documents.tasks.process_document_version_task.delay")
    def test_fresh_sync_version_untouched(self, mock_delay):
        # A sync save currently running on the web dyno (fresh updated_at).
        doc = self._make(DataRoomDocument.Status.PROCESSING, minutes_old=5)

        handled = requeue_stale_documents()

        self.assertEqual(handled, 0)
        self.assertIsNone(self._version(doc).queued_at)

    @patch("documents.tasks.process_document_version_task.delay")
    def test_orphan_cap_marks_failed(self, mock_delay):
        doc = self._make(
            DataRoomDocument.Status.UPLOADED, minutes_old=20, requeue_count=MAX_REQUEUES,
        )

        handled = requeue_stale_documents()

        self.assertEqual(handled, 1)
        self.assertEqual(self._version(doc).status, DataRoomDocument.Status.FAILED)


class SweeperScanningTests(_SweeperTestCase):
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


class SweeperDispatchKickTests(_SweeperTestCase):
    def test_sweep_kicks_dispatcher_even_when_idle(self):
        handled = requeue_stale_documents()

        self.assertEqual(handled, 0)
        self.mock_safe_dispatch.assert_called_once_with("sweeper")

    def test_db_blip_skips_the_dispatch_kick(self):
        from django.db.utils import OperationalError

        with patch.object(
            DataRoomDocumentVersion.objects,
            "filter",
            side_effect=OperationalError("the database system is starting up"),
        ):
            result = requeue_stale_documents()

        self.assertEqual(result, 0)
        self.mock_safe_dispatch.assert_not_called()


class SweeperDbErrorTests(_SweeperTestCase):
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


class ScanDispatchRetryRecoveryTests(_SweeperTestCase):
    """The sweeper auto-recovers versions a broker blip left SCAN_FAILED with the
    transient SCAN_DISPATCH_RETRY_MESSAGE marker (B-robust). These ride OUTSIDE
    the dispatch gate — scan work is network-bound and re-entry must not depend
    on a free document slot."""

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
    def test_broker_still_down_reverts_without_spending_budget(self, mock_delay):
        doc = self._scan_failed(marker=SCAN_DISPATCH_RETRY_MESSAGE, requeue_count=0)

        requeue_stale_documents()

        version = self._version(doc)
        # Reverted to SCAN_FAILED + the retry marker so the next tick retries — but a
        # dispatch that failed on a busy broker must NOT burn a retry attempt, or a
        # choppy broker at recovery time could terminally fail a never-scanned doc.
        self.assertEqual(version.status, DataRoomDocument.Status.SCAN_FAILED)
        self.assertEqual(version.processing_error, SCAN_DISPATCH_RETRY_MESSAGE)
        self.assertEqual(version.requeue_count, 0)  # budget intact
        mock_delay.assert_called_once_with(doc.current_version_id)

    @patch("guardrails.tasks.scan_document_version.delay", side_effect=RuntimeError("down"))
    def test_repeated_broker_failures_never_exhaust(self, _mock_delay):
        # Three failed recovery attempts in a row must leave the version retryable
        # (still scan_failed + marker, requeue_count 0), never terminally failed.
        doc = self._scan_failed(marker=SCAN_DISPATCH_RETRY_MESSAGE, requeue_count=0)

        for _ in range(3):
            requeue_stale_documents()

        version = self._version(doc)
        self.assertEqual(version.status, DataRoomDocument.Status.SCAN_FAILED)
        self.assertEqual(version.processing_error, SCAN_DISPATCH_RETRY_MESSAGE)
        self.assertEqual(version.requeue_count, 0)


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
