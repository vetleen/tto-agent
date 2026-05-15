"""Tests for documents retention signals."""
from __future__ import annotations

from datetime import timedelta

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.utils import timezone

from documents.models import DataRoom, DataRoomDocument

User = get_user_model()


class DataRoomRetainUntilTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="dr@example.com", password="testpass")

    def test_save_sets_retain_until(self):
        before = timezone.now()
        dr = DataRoom.objects.create(name="DR", slug="dr-test", created_by=self.user)
        self.assertIsNotNone(dr.retain_until)
        self.assertGreaterEqual(dr.retain_until, before + timedelta(days=364))

    def test_document_save_extends_dataroom_retain(self):
        dr = DataRoom.objects.create(name="DR", slug="dr-doc", created_by=self.user)
        DataRoom.objects.filter(pk=dr.pk).update(
            retain_until=timezone.now() - timedelta(days=1),
        )

        DataRoomDocument.objects.create(
            data_room=dr,
            uploaded_by=self.user,
            original_filename="test.txt",
            status=DataRoomDocument.Status.UPLOADED,
        )
        dr.refresh_from_db()

        self.assertGreater(dr.retain_until, timezone.now() + timedelta(days=363))
