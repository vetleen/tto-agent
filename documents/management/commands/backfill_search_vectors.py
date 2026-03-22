"""
Backfill search_vector for DataRoomDocumentChunk rows that have NULL search_vector.

Usage:
    python manage.py backfill_search_vectors
    python manage.py backfill_search_vectors --batch-size 200
"""
from __future__ import annotations

from django.contrib.postgres.search import SearchVector
from django.core.management.base import BaseCommand

from documents.models import DataRoomDocument, DataRoomDocumentChunk


class Command(BaseCommand):
    help = "Populate search_vector for existing chunks that have not been indexed for full-text search."

    def add_arguments(self, parser):
        parser.add_argument(
            "--batch-size",
            type=int,
            default=500,
            help="Number of chunks to update per batch (default: 500).",
        )

    def handle(self, *args, **options):
        batch_size = options["batch_size"]

        documents = (
            DataRoomDocument.objects.filter(
                status=DataRoomDocument.Status.READY,
                chunks__search_vector__isnull=True,
            )
            .distinct()
            .values_list("id", flat=True)
        )
        doc_ids = list(documents)

        if not doc_ids:
            self.stdout.write(self.style.SUCCESS("All chunks already have search vectors. Nothing to do."))
            return

        self.stdout.write(f"Backfilling search vectors for {len(doc_ids)} document(s)...")

        total_updated = 0
        for doc_id in doc_ids:
            doc_updated = 0
            while True:
                chunk_ids = list(
                    DataRoomDocumentChunk.objects.filter(
                        document_id=doc_id,
                        search_vector__isnull=True,
                    ).values_list("pk", flat=True)[:batch_size]
                )
                if not chunk_ids:
                    break
                updated = DataRoomDocumentChunk.objects.filter(
                    pk__in=chunk_ids,
                ).update(
                    search_vector=(
                        SearchVector("heading", weight="A", config="english")
                        + SearchVector("text", weight="B", config="english")
                    )
                )
                doc_updated += updated
                if len(chunk_ids) < batch_size:
                    break
            total_updated += doc_updated
            self.stdout.write(f"  document {doc_id}: {doc_updated} chunk(s) updated")

        self.stdout.write(self.style.SUCCESS(f"Done. {total_updated} chunk(s) backfilled across {len(doc_ids)} document(s)."))
