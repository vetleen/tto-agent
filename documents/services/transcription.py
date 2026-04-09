"""Audio transcription convenience wrapper.

Delegates to llm.service.transcription_service.TranscriptionService for
the actual API calls, splitting, cost tracking, and logging.
"""

from __future__ import annotations

from pathlib import Path


def transcribe_audio(file_path: Path, model_id: str, user=None, prompt: str | None = None) -> str:
    """Transcribe an audio file, returning the transcript text.

    This is a convenience wrapper around TranscriptionService.transcribe()
    that handles RunContext creation from the user.

    ``prompt`` is forwarded to the transcription API to bias the model toward
    proper nouns / vocabulary in the prompt. None means no prompt is sent.
    """
    from llm.service.transcription_service import get_transcription_service
    from llm.types.context import RunContext

    context = RunContext.create(user_id=user.pk if user else None)
    service = get_transcription_service()
    result = service.transcribe(file_path, model_id, context=context, prompt=prompt)
    return result.text
