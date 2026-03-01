import unittest
from io import StringIO

from django.contrib.auth import get_user_model
from django.db import connection
from django.test import TestCase
from django.core.management import call_command

from documents.models import Project, ProjectDocument, ProjectDocumentChunk

User = get_user_model()


class BackfillSearchVectorsCommandTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="cmd@example.com", password="testpass")
        self.project = Project.objects.create(name="CmdProject", slug="cmd-project", created_by=self.user)

    def test_nothing_to_do_when_no_ready_docs(self):
        """No READY documents at all → command exits early with success message."""
        out = StringIO()
        call_command("backfill_search_vectors", stdout=out)
        self.assertIn("Nothing to do", out.getvalue())

    def test_nothing_to_do_when_no_chunks_with_null_vector(self):
        """READY doc whose chunks all have non-null search_vector → nothing to backfill."""
        doc = ProjectDocument.objects.create(
            project=self.project,
            uploaded_by=self.user,
            original_filename="indexed.txt",
            status=ProjectDocument.Status.READY,
        )
        chunk = ProjectDocumentChunk.objects.create(
            document=doc, chunk_index=0, text="Already indexed", token_count=3
        )
        # Set search_vector to a non-null value using DB-agnostic raw SQL.
        # On Postgres the column type is tsvector; on SQLite it's stored as text.
        if connection.vendor == "postgresql":
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE documents_projectdocumentchunk SET search_vector = to_tsvector('english', 'indexed') WHERE id = %s",
                    [chunk.id],
                )
        else:
            # SQLite stores SearchVectorField as text; any non-empty string counts as non-null.
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE documents_projectdocumentchunk SET search_vector = 'indexed' WHERE id = %s",
                    [chunk.id],
                )

        out = StringIO()
        call_command("backfill_search_vectors", stdout=out)
        self.assertIn("Nothing to do", out.getvalue())

    @unittest.skipUnless(connection.vendor == "postgresql", "requires postgres")
    def test_backfill_updates_chunks_with_null_search_vector(self):
        """READY doc with null search_vector chunks are updated by the command."""
        doc = ProjectDocument.objects.create(
            project=self.project,
            uploaded_by=self.user,
            original_filename="unindexed.txt",
            status=ProjectDocument.Status.READY,
        )
        ProjectDocumentChunk.objects.create(document=doc, chunk_index=0, text="Chunk one", token_count=2)
        ProjectDocumentChunk.objects.create(document=doc, chunk_index=1, text="Chunk two", token_count=2)

        out = StringIO()
        call_command("backfill_search_vectors", stdout=out)

        output = out.getvalue()
        self.assertIn("2 chunk(s) updated", output)
        self.assertFalse(
            ProjectDocumentChunk.objects.filter(document=doc, search_vector__isnull=True).exists()
        )
