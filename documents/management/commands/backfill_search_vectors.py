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

        # Chunks belong to versions now; update any chunk missing a search vector
        # directly, in batches, regardless of which version it belongs to.
        total_updated = 0
        while True:
            chunk_ids = list(
                DataRoomDocumentChunk.objects.filter(
                    search_vector__isnull=True,
                ).values_list("pk", flat=True)[:batch_size]
            )
            if not chunk_ids:
                break
            updated = DataRoomDocumentChunk.objects.filter(pk__in=chunk_ids).update(
                search_vector=(
                    SearchVector("heading", weight="A", config="english")
                    + SearchVector("text", weight="B", config="english")
                )
            )
            total_updated += updated
            if len(chunk_ids) < batch_size:
                break

        if total_updated:
            self.stdout.write(self.style.SUCCESS(f"Done. {total_updated} chunk(s) backfilled."))
        else:
            self.stdout.write(self.style.SUCCESS("All chunks already have search vectors. Nothing to do."))
