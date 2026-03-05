from django.contrib.admin.sites import site as admin_site
from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentChunk

User = get_user_model()


class AdminRegistrationTests(TestCase):
    """Verify all documents models are registered with the admin."""

    def test_data_room_admin_registered(self):
        self.assertIn(DataRoom, admin_site._registry)

    def test_document_admin_registered(self):
        self.assertIn(DataRoomDocument, admin_site._registry)

    def test_chunk_admin_registered(self):
        self.assertIn(DataRoomDocumentChunk, admin_site._registry)


class AdminChangelistSmokeTests(TestCase):
    """Smoke-test that admin changelist pages render without error."""

    def setUp(self):
        self.superuser = User.objects.create_superuser(
            email="admin@example.com",
            password="adminpass",
        )
        self.client.force_login(self.superuser)

    def test_data_room_changelist_loads(self):
        response = self.client.get(reverse("admin:documents_dataroom_changelist"))
        self.assertEqual(response.status_code, 200)

    def test_document_changelist_loads(self):
        response = self.client.get(reverse("admin:documents_dataroomdocument_changelist"))
        self.assertEqual(response.status_code, 200)

    def test_chunk_changelist_loads(self):
        response = self.client.get(reverse("admin:documents_dataroomdocumentchunk_changelist"))
        self.assertEqual(response.status_code, 200)
