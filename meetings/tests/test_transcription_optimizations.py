"""Tests for the PR 1 transcription optimizations.

Covers:
- Registry capability flags on transcription models.
- ffmpeg speed-up factor passthrough in ``ffmpeg_extract_chunk``.
- Transcription service passing ``language`` / ``on_delta`` through.
- Orchestrator wiring the partial-transcript flusher so polling sees
  progressive text updates during upload transcription.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from llm.service._audio_subprocess import (
    _resolve_speed_up_factor,
    ffmpeg_extract_chunk,
)
from llm.transcription_registry import (
    get_all_transcription_models,
    get_transcription_model_info,
)
from meetings.models import Meeting
from meetings.services.audio_transcription import (
    ChunkSpec,
    _PartialTranscriptFlusher,
    orchestrate_upload_transcription,
)

User = get_user_model()


class RegistryCapabilityFlagTests(TestCase):
    def test_all_registered_models_declare_capability_flags(self):
        for model_id, info in get_all_transcription_models().items():
            with self.subTest(model=model_id):
                self.assertIsInstance(info.supports_live_streaming, bool)
                self.assertIsInstance(info.supports_output_streaming, bool)
                self.assertIsInstance(info.supports_diarization, bool)

    def test_mini_and_full_4o_are_live_and_streamable(self):
        mini = get_transcription_model_info("openai/gpt-4o-mini-transcribe")
        full = get_transcription_model_info("openai/gpt-4o-transcribe")
        self.assertTrue(mini.supports_live_streaming)
        self.assertTrue(mini.supports_output_streaming)
        self.assertFalse(mini.supports_diarization)
        self.assertTrue(full.supports_live_streaming)
        self.assertTrue(full.supports_output_streaming)
        self.assertFalse(full.supports_diarization)

    def test_diarize_is_batch_only(self):
        diarize = get_transcription_model_info("openai/gpt-4o-transcribe-diarize")
        self.assertIsNotNone(diarize)
        self.assertFalse(diarize.supports_live_streaming)
        self.assertFalse(diarize.supports_output_streaming)
        self.assertTrue(diarize.supports_diarization)


class SpeedUpFactorTests(TestCase):
    def test_resolve_speed_up_clamps_to_range(self):
        self.assertEqual(_resolve_speed_up_factor(0.1), 0.5)
        self.assertEqual(_resolve_speed_up_factor(5.0), 3.0)
        self.assertEqual(_resolve_speed_up_factor(2.0), 2.0)
        self.assertEqual(_resolve_speed_up_factor(1.0), 1.0)

    @override_settings(MEETING_UPLOAD_SPEED_UP_FACTOR=2.5)
    def test_resolve_speed_up_reads_setting_when_none_explicit(self):
        self.assertEqual(_resolve_speed_up_factor(None), 2.5)

    def test_ffmpeg_command_includes_atempo_when_factor_above_one(self):
        """Verify the atempo filter is present in the ffmpeg args for speed-up."""
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as src:
            src.write(b"\x00" * 16)
            src_path = Path(src.name)

        try:
            with patch("llm.service._audio_subprocess.subprocess.run", side_effect=fake_run):
                out = ffmpeg_extract_chunk(
                    src_path, 0, 10_000, 0, speed_up_factor=2.0,
                )
            cmd = captured["cmd"]
            self.assertIn("-filter:a", cmd)
            idx = cmd.index("-filter:a")
            self.assertIn("atempo=", cmd[idx + 1])
            self.assertIn("2.000", cmd[idx + 1])
        finally:
            src_path.unlink(missing_ok=True)
            Path(out).unlink(missing_ok=True)

    def test_ffmpeg_command_omits_atempo_when_factor_is_one(self):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as src:
            src.write(b"\x00" * 16)
            src_path = Path(src.name)

        try:
            with patch("llm.service._audio_subprocess.subprocess.run", side_effect=fake_run):
                out = ffmpeg_extract_chunk(
                    src_path, 0, 10_000, 0, speed_up_factor=1.0,
                )
            self.assertNotIn("-filter:a", captured["cmd"])
        finally:
            src_path.unlink(missing_ok=True)
            Path(out).unlink(missing_ok=True)


class TranscriptionServiceStreamingTests(TestCase):
    """_consume_streaming should drain events and invoke the delta callback."""

    def test_consume_streaming_forwards_deltas_and_returns_done_event(self):
        from llm.service.transcription_service import TranscriptionService

        delta1 = MagicMock()
        delta1.type = "transcript.text.delta"
        delta1.delta = "Hello "
        delta2 = MagicMock()
        delta2.type = "transcript.text.delta"
        delta2.delta = "world."
        done = MagicMock()
        done.type = "transcript.text.done"
        done.text = "Hello world."

        service = TranscriptionService()
        received = []
        result = service._consume_streaming(iter([delta1, delta2, done]), received.append)

        self.assertEqual(received, ["Hello ", "world."])
        self.assertEqual(result.text, "Hello world.")

    def test_consume_streaming_survives_callback_raising(self):
        from llm.service.transcription_service import TranscriptionService

        delta = MagicMock()
        delta.type = "transcript.text.delta"
        delta.delta = "x"
        done = MagicMock()
        done.type = "transcript.text.done"
        done.text = "x"

        def broken(_):
            raise RuntimeError("boom")

        service = TranscriptionService()
        # A faulty callback must not break the stream drain.
        result = service._consume_streaming(iter([delta, done]), broken)
        self.assertEqual(result.text, "x")


class PartialTranscriptFlusherTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="flush@example.com", password="pw")
        self.meeting = Meeting.objects.create(
            name="Flush meeting", slug="m-flush", created_by=self.user,
        )

    def test_flusher_persists_seed_plus_delta_buffer(self):
        flusher = _PartialTranscriptFlusher(self.meeting.pk, "existing text")
        # Drive the internal clock past the flush interval so every delta flushes.
        flusher._last_flush = 0.0
        flusher.on_delta("hello ")
        flusher.on_delta("world")
        # Force-flush so we can assert without sleeping.
        flusher._last_flush = 0.0
        flusher._flush()

        fresh = Meeting.objects.get(pk=self.meeting.pk)
        self.assertIn("existing text", fresh.transcript)
        self.assertIn("hello world", fresh.transcript)

    def test_flusher_ignores_empty_delta(self):
        flusher = _PartialTranscriptFlusher(self.meeting.pk, "")
        flusher.on_delta("")
        # Empty delta should not touch the buffer or DB.
        self.assertEqual(flusher._buffer, "")


class OrchestratorLanguageAndDeltaTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="lang@example.com", password="pw")
        self.meeting = Meeting.objects.create(
            name="Lang meeting",
            slug="m-lang",
            forced_language="no",
            created_by=self.user,
        )

    def _make_spec(self) -> ChunkSpec:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, prefix="lang_")
        tmp.write(b"\x00" * 16)
        tmp.close()
        return ChunkSpec(path=Path(tmp.name), index=0, start_ms=0, end_ms=1000)

    @patch("meetings.services.audio_transcription.split_audio_with_overlap")
    def test_orchestrator_passes_forced_language_and_delta_callback(self, mock_split):
        spec = self._make_spec()
        mock_split.return_value = [spec]

        captured_kwargs = {}

        def fake_transcribe(file_path, model_id, *, context, prompt, language, on_delta):
            captured_kwargs["language"] = language
            captured_kwargs["on_delta"] = on_delta
            # Simulate some deltas so the flusher gets exercised.
            if on_delta is not None:
                on_delta("streamed text")
            result = MagicMock()
            result.text = "streamed text"
            return result

        service = MagicMock()
        service.transcribe.side_effect = fake_transcribe

        orchestrate_upload_transcription(
            meeting_id=self.meeting.pk,
            temp_path=Path("/fake.mp3"),
            model_id="openai/gpt-4o-mini-transcribe",
            user_id=self.user.pk,
            service=service,
        )

        self.assertEqual(captured_kwargs["language"], "no")
        self.assertIsNotNone(captured_kwargs["on_delta"])
