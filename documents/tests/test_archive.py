"""Tests for data room and document archive/restore functionality."""

from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentChunk

User = get_user_model()


@override_settings(ALLOWED_HOSTS=["testserver"])
class DataRoomArchiveViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.data_room = DataRoom.objects.create(name="Test", slug="test", created_by=self.user)
        self.other = User.objects.create_user(email="other@example.com", password="testpass")

    def test_archive_data_room_sets_is_archived(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("data_room_archive", kwargs={"data_room_id": self.data_room.uuid}),
        )
        self.assertRedirects(response, reverse("data_room_list"))
        self.data_room.refresh_from_db()
        self.assertTrue(self.data_room.is_archived)

    def test_restore_data_room_clears_is_archived(self):
        self.data_room.is_archived = True
        self.data_room.save(update_fields=["is_archived"])
        self.client.force_login(self.user)
        self.client.post(
            reverse("data_room_archive", kwargs={"data_room_id": self.data_room.uuid}),
        )
        self.data_room.refresh_from_db()
        self.assertFalse(self.data_room.is_archived)

    def test_archive_data_room_other_user_blocked(self):
        self.client.force_login(self.other)
        self.client.post(
            reverse("data_room_archive", kwargs={"data_room_id": self.data_room.uuid}),
        )
        self.data_room.refresh_from_db()
        self.assertFalse(self.data_room.is_archived)

    def test_archive_data_room_requires_post(self):
        self.client.force_login(self.user)
        response = self.client.get(
            reverse("data_room_archive", kwargs={"data_room_id": self.data_room.uuid}),
        )
        self.assertEqual(response.status_code, 405)

    def test_data_room_list_separates_active_and_archived(self):
        archived = DataRoom.objects.create(
            name="Archived", slug="archived", created_by=self.user, is_archived=True,
        )
        self.client.force_login(self.user)
        response = self.client.get(reverse("data_room_list"))
        self.assertEqual(response.status_code, 200)
        self.assertIn(self.data_room, response.context["data_rooms"])
        self.assertNotIn(archived, response.context["data_rooms"])
        self.assertIn(archived, response.context["archived_data_rooms"])
        self.assertNotIn(self.data_room, response.context["archived_data_rooms"])

    def test_data_room_list_shows_archive_button(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse("data_room_list"))
        self.assertContains(response, "Archive")

    def test_data_room_list_shows_restore_for_archived(self):
        DataRoom.objects.create(
            name="Archived", slug="archived", created_by=self.user, is_archived=True,
        )
        self.client.force_login(self.user)
        response = self.client.get(reverse("data_room_list"))
        self.assertContains(response, "Restore")
        self.assertContains(response, "Archived data rooms")


@override_settings(ALLOWED_HOSTS=["testserver"])
class DocumentArchiveViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.data_room = DataRoom.objects.create(name="Test", slug="test", created_by=self.user)
        self.other = User.objects.create_user(email="other@example.com", password="testpass")

    def _make_doc(self, **kwargs):
        defaults = dict(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="doc.txt",
            status=DataRoomDocument.Status.READY,
        )
        defaults.update(kwargs)
        return DataRoomDocument.objects.create(**defaults)

    def test_archive_document_sets_is_archived(self):
        doc = self._make_doc()
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("document_archive", kwargs={"data_room_id": self.data_room.uuid, "document_id": doc.id}),
        )
        self.assertRedirects(
            response,
            reverse("data_room_documents", kwargs={"data_room_id": self.data_room.uuid}),
        )
        doc.refresh_from_db()
        self.assertTrue(doc.is_archived)

    def test_restore_document_clears_is_archived(self):
        doc = self._make_doc(is_archived=True)
        self.client.force_login(self.user)
        self.client.post(
            reverse("document_archive", kwargs={"data_room_id": self.data_room.uuid, "document_id": doc.id}),
        )
        doc.refresh_from_db()
        self.assertFalse(doc.is_archived)

    def test_archive_document_other_user_blocked(self):
        doc = self._make_doc()
        self.client.force_login(self.other)
        self.client.post(
            reverse("document_archive", kwargs={"data_room_id": self.data_room.uuid, "document_id": doc.id}),
        )
        doc.refresh_from_db()
        self.assertFalse(doc.is_archived)

    def test_archive_document_requires_post(self):
        doc = self._make_doc()
        self.client.force_login(self.user)
        response = self.client.get(
            reverse("document_archive", kwargs={"data_room_id": self.data_room.uuid, "document_id": doc.id}),
        )
        self.assertEqual(response.status_code, 405)

    def test_data_room_documents_separates_active_and_archived(self):
        active_doc = self._make_doc(original_filename="active.txt")
        archived_doc = self._make_doc(original_filename="archived.txt", is_archived=True)
        self.client.force_login(self.user)
        response = self.client.get(
            reverse("data_room_documents", kwargs={"data_room_id": self.data_room.uuid}),
        )
        self.assertEqual(response.status_code, 200)
        active_filenames = [d.original_filename for d in response.context["documents"]]
        archived_filenames = [d.original_filename for d in response.context["archived_documents"]]
        self.assertIn("active.txt", active_filenames)
        self.assertNotIn("archived.txt", active_filenames)
        self.assertIn("archived.txt", archived_filenames)
        self.assertNotIn("active.txt", archived_filenames)

    def test_data_room_documents_shows_archive_button(self):
        self._make_doc()
        self.client.force_login(self.user)
        response = self.client.get(
            reverse("data_room_documents", kwargs={"data_room_id": self.data_room.uuid}),
        )
        self.assertContains(response, "Archive")

    def test_data_room_documents_shows_restore_for_archived(self):
        self._make_doc(is_archived=True)
        self.client.force_login(self.user)
        response = self.client.get(
            reverse("data_room_documents", kwargs={"data_room_id": self.data_room.uuid}),
        )
        self.assertContains(response, "Restore")
        self.assertContains(response, "Archived documents")


class ArchivedDocumentsExcludedFromRAGTests(TestCase):
    """Verify archived documents are excluded from retrieval and tool access."""

    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="Test", slug="test", created_by=self.user)
        self.active_doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="active.txt",
            status=DataRoomDocument.Status.READY,
        )
        self.archived_doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="archived.txt",
            status=DataRoomDocument.Status.READY,
            is_archived=True,
        )
        # Create chunks for both docs
        DataRoomDocumentChunk.objects.create(
            document=self.active_doc, chunk_index=0, text="Active content", token_count=2,
        )
        DataRoomDocumentChunk.objects.create(
            document=self.archived_doc, chunk_index=0, text="Archived content", token_count=2,
        )

    def test_get_chunks_by_data_room_excludes_archived(self):
        from documents.services.retrieval import get_chunks_by_data_room

        chunks = get_chunks_by_data_room(self.data_room.pk)
        doc_ids = {c["document_id"] for c in chunks}
        self.assertIn(self.active_doc.pk, doc_ids)
        self.assertNotIn(self.archived_doc.pk, doc_ids)

    def test_read_document_tool_excludes_archived(self):
        from chat.tools import ReadDocumentTool
        from llm.types.context import RunContext

        tool = ReadDocumentTool()
        ctx = RunContext.create(user_id=self.user.pk, data_room_ids=[self.data_room.pk])
        result = tool.run({"doc_indices": [self.archived_doc.doc_index]}, ctx)
        # Should get a "not found" error for the archived doc
        self.assertEqual(len(result["documents"]), 1)
        self.assertIn("error", result["documents"][0])

    def test_read_document_tool_allows_active(self):
        from chat.tools import ReadDocumentTool
        from llm.types.context import RunContext

        tool = ReadDocumentTool()
        ctx = RunContext.create(user_id=self.user.pk, data_room_ids=[self.data_room.pk])
        result = tool.run({"doc_indices": [self.active_doc.doc_index]}, ctx)
        self.assertEqual(len(result["documents"]), 1)
        self.assertIn("content", result["documents"][0])

    @patch("documents.services.retrieval.vs.similarity_search")
    def test_hybrid_search_excludes_archived_semantic_results(self, mock_vs):
        from documents.services.retrieval import hybrid_search_chunks

        # Simulate pgvector returning a chunk from the archived doc
        mock_semantic_doc = MagicMock()
        mock_semantic_doc.metadata = {
            "chunk_id": self.archived_doc.chunks.first().pk,
            "document_id": self.archived_doc.pk,
            "chunk_index": 0,
        }
        mock_semantic_doc.page_content = "Archived content"
        mock_vs.return_value = [mock_semantic_doc]

        results = hybrid_search_chunks(
            data_room_ids=[self.data_room.pk], query="test", k=10, fulltext_weight=0,
        )
        result_doc_ids = {r["document_id"] for r in results}
        self.assertNotIn(self.archived_doc.pk, result_doc_ids)
