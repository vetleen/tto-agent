"""
Backfill PII category tags for READY documents.

Usage:
    python manage.py backfill_pii_tags
    python manage.py backfill_pii_tags --doc-ids 30 33
"""
from __future__ import annotations

from django.core.management.base import BaseCommand

from documents.models import DataRoomDocument, DataRoomDocumentTag
from documents.services.pii_scan import PII_CATEGORIES


class Command(BaseCommand):
    help = "Run PII category classification on READY documents that have no PII tags."

    def add_arguments(self, parser):
        parser.add_argument(
            "--doc-ids",
            nargs="*",
            type=int,
            help="Specific document IDs to process (default: all READY docs without PII tags).",
        )

    def handle(self, *args, **options):
        qs = DataRoomDocument.objects.filter(status=DataRoomDocument.Status.READY)
        if options["doc_ids"]:
            qs = qs.filter(pk__in=options["doc_ids"])
        else:
            already_tagged = DataRoomDocumentTag.objects.filter(
                key__in=PII_CATEGORIES,
            ).values_list("version__document_id", flat=True)
            qs = qs.exclude(pk__in=already_tagged)

        docs = list(qs.order_by("id"))
        if not docs:
            self.stdout.write(self.style.SUCCESS("No documents need PII scanning. Nothing to do."))
            return

        self.stdout.write(f"Scanning {len(docs)} document(s) for PII categories...")

        from accounts.models import Membership
        from documents.services.chunking import clean_extracted_text, load_documents
        from documents.services.pii_scan import scan_pii_categories
        from documents.services.storage_utils import local_copy

        success = 0
        for doc in docs:
            try:
                version = doc.active_searchable_version or doc.current_version
                if version is None:
                    self.stdout.write(f"  doc {doc.id}: skipped (no version)")
                    continue
                text = ""
                if doc.original_file:
                    try:
                        with local_copy(doc.original_file) as file_path:
                            ext = (doc.original_filename or "").rsplit(".", 1)[-1].lower() or "txt"
                            raw_docs = load_documents(file_path, ext)
                            combined = "\n\n".join(getattr(d, "page_content", "") or "" for d in raw_docs)
                            text = clean_extracted_text(combined)
                    except Exception:
                        text = ""
                if not text:
                    chunks = version.chunks.order_by("chunk_index")
                    text = "\n\n".join(c.text for c in chunks)

                if not text.strip():
                    self.stdout.write(f"  doc {doc.id}: skipped (no text)")
                    continue

                org_id = None
                if doc.uploaded_by_id:
                    org_id = Membership.objects.filter(user_id=doc.uploaded_by_id).values_list("org_id", flat=True).first()

                result = scan_pii_categories(
                    text, user_id=doc.uploaded_by_id,
                    data_room_id=doc.data_room_id, org_id=org_id,
                )
                for category in result:
                    DataRoomDocumentTag.objects.update_or_create(
                        version=version, key=category,
                        defaults={"value": "true"},
                    )
                success += 1
                detected = list(result.keys())
                self.stdout.write(f"  doc {doc.id}: OK (detected={detected})")
            except Exception as e:
                self.stderr.write(f"  doc {doc.id}: FAILED ({e})")

        self.stdout.write(self.style.SUCCESS(f"Done. {success}/{len(docs)} document(s) scanned."))
