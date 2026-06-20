"""Tests for the nightly version-prune policy (documents.tasks.prune_document_versions)."""
from __future__ import annotations

from datetime import timedelta

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.utils import timezone

from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentVersion
from documents.tasks import MAX_VERSIONS_PER_DOCUMENT, prune_document_versions

User = get_user_model()


class PruneVersionsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="prune@x.io", password="p")
        self.room = DataRoom.objects.create(name="R", slug="r", created_by=self.user)
        self.doc = DataRoomDocument.objects.create(
            data_room=self.room, uploaded_by=self.user, original_filename="d.md",
        )

    def _v(self, idx):
        return DataRoomDocumentVersion.objects.create(document=self.doc, version_index=idx)

    def _remaining(self):
        return set(
            DataRoomDocumentVersion.objects.filter(document=self.doc)
            .values_list("version_index", flat=True)
        )

    def test_keeps_original_current_active_and_one_per_day(self):
        vs = [self._v(i) for i in range(6)]  # v0..v5, same calendar day (test runtime)
        DataRoomDocument.objects.filter(pk=self.doc.pk).update(
            current_version=vs[5], active_searchable_version=vs[5],
        )
        prune_document_versions()
        remaining = self._remaining()
        # Always protected: v0 (original) + v5 (current+active). Among the same-day
        # rest (v1..v4) only the newest (v4) survives.
        self.assertIn(0, remaining)
        self.assertIn(5, remaining)
        self.assertIn(4, remaining)
        for dropped in (1, 2, 3):
            self.assertNotIn(dropped, remaining)

    def test_never_prunes_the_original(self):
        vs = [self._v(i) for i in range(15)]
        DataRoomDocument.objects.filter(pk=self.doc.pk).update(
            current_version=vs[14], active_searchable_version=vs[14],
        )
        prune_document_versions()
        self.assertIn(0, self._remaining())

    def test_caps_total_when_spread_across_days(self):
        base = timezone.now()
        vs = [self._v(i) for i in range(15)]
        # Distinct calendar days so the 1-per-day rule keeps them all; the hard cap
        # then trims the oldest non-protected down to MAX_VERSIONS_PER_DOCUMENT.
        for i, v in enumerate(vs):
            DataRoomDocumentVersion.objects.filter(pk=v.pk).update(
                created_at=base - timedelta(days=i)
            )
        DataRoomDocument.objects.filter(pk=self.doc.pk).update(
            current_version=vs[0], active_searchable_version=vs[0],
        )
        prune_document_versions()
        total = DataRoomDocumentVersion.objects.filter(document=self.doc).count()
        self.assertLessEqual(total, MAX_VERSIONS_PER_DOCUMENT)
        self.assertIn(0, self._remaining())  # original always survives

    def test_single_version_is_left_alone(self):
        v0 = self._v(0)
        DataRoomDocument.objects.filter(pk=self.doc.pk).update(
            current_version=v0, active_searchable_version=v0,
        )
        pruned = prune_document_versions()
        self.assertEqual(pruned, 0)
        self.assertEqual(self._remaining(), {0})

    def test_versions_fetched_in_batches_not_per_document(self):
        """Versions are read in one grouped query per batch, not one per document.

        Regression for WILFRED-64 (N+1): the old code issued a SELECT against the
        version table for every document. Give several documents one version each
        (so no pruning/deletes muddy the query log) and assert the version-table
        reads don't scale with the document count.
        """
        from django.db import connection
        from django.test.utils import CaptureQueriesContext

        for i in range(6):
            d = DataRoomDocument.objects.create(
                data_room=self.room, uploaded_by=self.user, original_filename=f"d{i}.md",
            )
            DataRoomDocumentVersion.objects.create(document=d, version_index=0)

        with CaptureQueriesContext(connection) as ctx:
            prune_document_versions()

        version_selects = [
            q["sql"] for q in ctx.captured_queries
            if 'FROM "documents_dataroomdocumentversion"' in q["sql"]
            and q["sql"].lstrip().upper().startswith("SELECT")
        ]
        # Batched: a single grouped SELECT covers all 7 documents (6 here + the
        # setUp doc). An N+1 would issue one SELECT per document.
        self.assertLessEqual(len(version_selects), 2, version_selects)
