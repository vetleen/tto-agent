"""Generate a short description for a data room using a cheap LLM."""

from __future__ import annotations

import logging

from django.conf import settings

from documents.models import DataRoom, DataRoomDocument

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Based on the data room name and the document descriptions below, "
    "write one to two sentences with a concise description (50 tokens) of what this data room contains "
    "and what its purpose is. Output ONLY the description paragraph."
)


def generate_data_room_description(data_room_id: int, user_id: int | None = None) -> str:
    """Generate a description for a data room based on its documents.

    Fetches the room name and up to 10 document descriptions, then asks
    the LLM to synthesize a room-level description.
    """
    from llm import get_llm_service
    from llm.types import ChatRequest, Message, RunContext

    room = DataRoom.objects.get(pk=data_room_id)

    # Fetch up to 10 random document descriptions
    doc_descriptions = list(
        DataRoomDocument.objects.filter(
            data_room_id=data_room_id,
            status=DataRoomDocument.Status.READY,
            is_archived=False,
        )
        .exclude(description="")
        .order_by("?")[:10]
        .values_list("original_filename", "description")
    )

    if not doc_descriptions:
        return ""

    doc_lines = "\n".join(
        f'- "{fname}": {desc}' for fname, desc in doc_descriptions
    )
    user_content = f"Data room name: {room.name}\n\nDocuments:\n{doc_lines}"

    if room.description:
        user_content += (
            f"\n\nCurrent description: {room.description}\n"
            "The user has asked you to suggest a new description, so don't "
            "feel bound by the current one — but it's included for reference, "
            "since it may contain information that isn't apparent from the "
            "documents in the room alone."
        )

    context = RunContext.create(
        user_id=user_id,
        conversation_id=data_room_id,
    )
    request = ChatRequest(
        messages=[
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(role="user", content=user_content),
        ],
        model=settings.LLM_DEFAULT_CHEAP_MODEL,
        stream=False,
        tools=[],
        context=context,
    )

    service = get_llm_service()
    response = service.run("simple_chat", request)
    description = response.message.content.strip()[:1000]

    logger.info(
        "generate_data_room_description: data_room_id=%s len=%s",
        data_room_id, len(description),
    )
    return description
