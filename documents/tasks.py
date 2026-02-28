from celery import shared_task
from documents.services.process_document import process_document


@shared_task(
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 5},
    time_limit=600,
    soft_time_limit=540,
)
def process_document_task(document_id: int) -> None:
    process_document(document_id)
