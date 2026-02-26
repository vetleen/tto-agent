from celery import shared_task
from documents.services.process_document import process_document


@shared_task
def process_document_task(document_id: int) -> None:
    process_document(document_id)
