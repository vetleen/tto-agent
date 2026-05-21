"""
Backfill file_metadata_date for READY documents by re-reading file metadata.

Usage:
    python manage.py backfill_file_dates
    python manage.py backfill_file_dates --doc-ids 30 33
"""
from __future__ import annotations

from django.core.management.base import BaseCommand

from documents.models import DataRoomDocument


class Command(BaseCommand):
    help = "Extract file metadata dates for READY documents that are missing them."

    def add_arguments(self, parser):
        parser.add_argument(
            "--doc-ids",
            nargs="*",
            type=int,
            help="Specific document IDs to process (default: all READY docs without file_metadata_date).",
        )

    def handle(self, *args, **options):
        qs = DataRoomDocument.objects.filter(status=DataRoomDocument.Status.READY)
        if options["doc_ids"]:
            qs = qs.filter(pk__in=options["doc_ids"])
        else:
            qs = qs.filter(file_metadata_date__isnull=True)

        docs = list(qs.exclude(original_file="").order_by("id"))
        if not docs:
            self.stdout.write(self.style.SUCCESS("No documents need file date extraction. Nothing to do."))
            return

        self.stdout.write(f"Extracting file metadata dates for {len(docs)} document(s)...")

        from documents.services.chunking import extract_file_metadata_date
        from documents.services.storage_utils import local_copy

        success = 0
        for doc in docs:
            try:
                ext = (doc.original_filename or "").rsplit(".", 1)[-1].lower() or "txt"
                with local_copy(doc.original_file) as file_path:
                    file_date = extract_file_metadata_date(file_path, ext)

                if file_date:
                    doc.file_metadata_date = file_date
                    doc.save(update_fields=["file_metadata_date", "updated_at"])
                    success += 1
                    self.stdout.write(f"  doc {doc.id}: OK (date={file_date})")
                else:
                    self.stdout.write(f"  doc {doc.id}: no date found")
            except Exception as e:
                self.stderr.write(f"  doc {doc.id}: FAILED ({e})")

        self.stdout.write(self.style.SUCCESS(f"Done. {success}/{len(docs)} document(s) updated."))
