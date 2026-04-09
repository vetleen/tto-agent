"""Tests for meetings.tasks Celery tasks.

Mocks ``documents.services.transcription.transcribe_audio`` (the wrapper
that the tasks call) so we can assert prompt seeding and orchestration
wiring without touching the real OpenAI API.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from meetings.models import Meeting, MeetingTranscriptSegment
from meetings.tasks import (
    transcribe_meeting_chunk_task,
    transcribe_uploaded_audio_task,
)

User = get_user_model()


def _write_temp_audio(suffix: str = ".webm") -> str:
    """Create a placeholder temp audio file and return its absolute path."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, prefix="task_test_")
    tmp.write(b"\x00" * 64)
    tmp.close()
    return tmp.name


class TranscribeMeetingChunkTaskPromptTests(TestCase):
    """Live-path chunk task should seed the transcription prompt with meeting
    metadata + tail of the existing transcript."""

    def setUp(self):
        self.user = User.objects.create_user(email="ct@example.com", password="pw")
        self.meeting = Meeting.objects.create(
            name="Live meeting",
            slug="m-live-task",
            agenda="Discuss OncoBio Therapeutics",
            participants="Alice, Bob",
            transcript="…earlier we said the deadline would slip to next quarter.",
            created_by=self.user,
        )

    @patch("documents.services.transcription.transcribe_audio", return_value="hello segment")
    def test_live_chunk_task_passes_prompt_with_metadata_and_tail(self, mock_transcribe):
        path = _write_temp_audio()
        try:
            transcribe_meeting_chunk_task(
                meeting_id=self.meeting.pk,
                segment_index=0,
                temp_path=path,
                mime="audio/webm",
                model_id="openai/gpt-4o-mini-transcribe",
                user_id=self.user.pk,
                start_offset_seconds=0.0,
            )
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

        self.assertTrue(mock_transcribe.called)
        kwargs = mock_transcribe.call_args.kwargs
        prompt = kwargs.get("prompt")
        self.assertIsNotNone(prompt)
        self.assertIn("Live meeting", prompt)
        self.assertIn("OncoBio Therapeutics", prompt)
        self.assertIn("Alice, Bob", prompt)
        # Tail of the existing transcript should be appended as carryover.
        self.assertIn("Previous transcript excerpt", prompt)
        self.assertIn("deadline would slip", prompt)

        # Segment was persisted.
        seg = MeetingTranscriptSegment.objects.get(meeting=self.meeting, segment_index=0)
        self.assertEqual(seg.status, MeetingTranscriptSegment.Status.READY)
        self.assertEqual(seg.text, "hello segment")


@override_settings(MEETING_CHUNK_TEMP_DIR=tempfile.gettempdir())
class TranscribeUploadedAudioTaskTests(TestCase):
    """Upload-path task should delegate to the orchestrator and clean up."""

    def setUp(self):
        self.user = User.objects.create_user(email="ut@example.com", password="pw")
        self.meeting = Meeting.objects.create(
            name="Upload meeting",
            slug="m-upload-task",
            created_by=self.user,
        )

    @patch("meetings.services.audio_transcription.orchestrate_upload_transcription")
    def test_happy_path_calls_orchestrator_and_unlinks_temp(self, mock_orch):
        path = _write_temp_audio()
        mock_orch.return_value = "ok"

        try:
            transcribe_uploaded_audio_task(
                meeting_id=self.meeting.pk,
                temp_path=path,
                model_id="openai/gpt-4o-mini-transcribe",
                user_id=self.user.pk,
            )
        finally:
            # Defensive — should already be gone via cleanup_temp.
            try:
                os.unlink(path)
            except OSError:
                pass

        self.assertTrue(mock_orch.called)
        kwargs = mock_orch.call_args.kwargs
        self.assertEqual(kwargs["meeting_id"], self.meeting.pk)
        self.assertEqual(kwargs["model_id"], "openai/gpt-4o-mini-transcribe")
        self.assertEqual(kwargs["user_id"], self.user.pk)
        # cleanup_temp should have removed the file.
        self.assertFalse(Path(path).exists())

    @patch("meetings.services.audio_transcription.orchestrate_upload_transcription")
    def test_orchestrator_exception_marks_meeting_failed_no_reraise(self, mock_orch):
        """A pre-orchestrator crash (or one that didn't already finalize the
        meeting) is caught here and the meeting is marked FAILED defensively.
        The task does NOT re-raise (no useful Celery retry path)."""
        path = _write_temp_audio()
        mock_orch.side_effect = RuntimeError("pydub blew up")

        # Should NOT raise out of the task.
        try:
            transcribe_uploaded_audio_task(
                meeting_id=self.meeting.pk,
                temp_path=path,
                model_id="openai/gpt-4o-mini-transcribe",
                user_id=self.user.pk,
            )
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

        self.meeting.refresh_from_db()
        self.assertEqual(self.meeting.status, Meeting.Status.FAILED)
        self.assertIn("pydub blew up", self.meeting.transcription_error)
        self.assertEqual(self.meeting.transcription_chunks_total, 0)
        self.assertEqual(self.meeting.transcription_chunks_done, 0)
        # Temp file unlinked.
        self.assertFalse(Path(path).exists())
