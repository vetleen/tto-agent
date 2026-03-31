"""
Backfill descriptions for DataRoomDocument rows that are READY but have no description.

Usage:
    python manage.py backfill_descriptions
    python manage.py backfill_descriptions --doc-ids 30 33
"""
from __future__ import annotations

import sys
from pathlib import Path

from django.core.management.base import BaseCommand

from documents.models import DataRoomDocument, DataRoomDocumentTag


class Command(BaseCommand):
    help = "Generate descriptions for READY documents that are missing them."

    def add_arguments(self, parser):
        parser.add_argument(
            "--doc-ids",
            nargs="*",
            type=int,
            help="Specific document IDs to process (default: all READY docs without descriptions).",
        )

    def handle(self, *args, **options):
        from django.conf import settings

        if not getattr(settings, "LLM_DEFAULT_CHEAP_MODEL", ""):
            self.stderr.write(self.style.ERROR("LLM_DEFAULT_CHEAP_MODEL is not set. Cannot generate descriptions."))
            sys.exit(1)

        qs = DataRoomDocument.objects.filter(status=DataRoomDocument.Status.READY)
        if options["doc_ids"]:
            qs = qs.filter(pk__in=options["doc_ids"])
        else:
            qs = qs.filter(description="")

        docs = list(qs.order_by("id"))
        if not docs:
            self.stdout.write(self.style.SUCCESS("No documents need descriptions. Nothing to do."))
            return

        self.stdout.write(f"Generating descriptions for {len(docs)} document(s)...")

        from documents.services.chunking import clean_extracted_text, load_documents
        from documents.services.description import generate_description_and_tags_from_text
        from documents.services.storage_utils import local_copy

        success = 0
        for doc in docs:
            try:
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
                    chunks = doc.chunks.order_by("chunk_index")
                    text = "\n\n".join(c.text for c in chunks)

                if not text.strip():
                    self.stdout.write(f"  doc {doc.id}: skipped (no text)")
                    continue

                result = generate_description_and_tags_from_text(
                    text, user_id=doc.uploaded_by_id, data_room_id=doc.data_room_id
                )
                doc.description = result["description"]
                doc.save(update_fields=["description", "updated_at"])
                for tag_key, tag_value in result.get("tags", {}).items():
                    DataRoomDocumentTag.objects.update_or_create(
                        document=doc, key=tag_key,
                        defaults={"value": tag_value},
                    )
                success += 1
                self.stdout.write(f"  doc {doc.id}: OK (desc_len={len(doc.description)})")
            except Exception as e:
                self.stderr.write(f"  doc {doc.id}: FAILED ({e})")

        self.stdout.write(self.style.SUCCESS(f"Done. {success}/{len(docs)} document(s) updated."))
