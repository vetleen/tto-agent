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
    DEFAULT_TARGET_CHUNK_SECONDS,
    ChunkBoundary,
    _PartialTranscriptFlusher,
    orchestrate_upload_transcription,
    plan_chunk_boundaries,
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

    def test_consume_streaming_returns_assembled_text_for_frozen_done_event(self):
        """A frozen/slotted done event (setattr would raise) must not lose the
        transcript — the assembled deltas are returned instead of an empty text."""
        from llm.service.transcription_service import TranscriptionService

        class _FrozenDone:
            __slots__ = ("type",)  # no settable/readable .text attribute

            def __init__(self):
                self.type = "transcript.text.done"

        delta = MagicMock()
        delta.type = "transcript.text.delta"
        delta.delta = "Hello world."

        service = TranscriptionService()
        result = service._consume_streaming(iter([delta, _FrozenDone()]), None)
        self.assertEqual(result.text, "Hello world.")


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


class ChunkPlannerSpeedUpAwarenessTests(TestCase):
    """plan_chunk_boundaries divides by duration, and respects the speed-up
    factor when the API size limit (not the 5-min chunk target) is the binding
    constraint.

    With the default 5-min target the count is simply ceil(duration / 300s):
    300 s of source is well under the size/duration cap at any factor, so the
    factor only changes the count when a larger target lets the size cap bite.
    """

    def _run_planner(
        self,
        duration_seconds: float,
        factor: float,
        target_chunk_seconds: int = DEFAULT_TARGET_CHUNK_SECONDS,
    ) -> int:
        from llm.service import _audio_subprocess

        # plan_chunk_boundaries is pure math after a metadata-only probe; it
        # imports _resolve_speed_up_factor lazily, so patch it on the source
        # module (the local import resolves the attribute at call time).
        with patch.object(_audio_subprocess, "_resolve_speed_up_factor", return_value=factor):
            boundaries = plan_chunk_boundaries(
                int(duration_seconds * 1000),
                target_chunk_seconds=target_chunk_seconds,
                max_bytes=25_000_000,
                max_seconds=1400,
            )
        return len(boundaries)

    def test_default_target_chunks_by_duration(self):
        # 5-min chunks: a 60-min file -> 12 chunks, a 90-min file -> 18.
        self.assertEqual(self._run_planner(duration_seconds=3600, factor=2.0), 12)
        self.assertEqual(self._run_planner(duration_seconds=5400, factor=2.0), 18)

    def test_short_file_is_single_chunk(self):
        self.assertEqual(self._run_planner(duration_seconds=300, factor=2.0), 1)

    def test_speed_up_factor_reduces_chunks_when_size_bound(self):
        # With a large chunk target the API size limit is the binding cap, so a
        # 2x speed-up (half the bytes per source-second) yields fewer chunks.
        chunks_1x = self._run_planner(duration_seconds=3600, factor=1.0, target_chunk_seconds=3600)
        chunks_2x = self._run_planner(duration_seconds=3600, factor=2.0, target_chunk_seconds=3600)
        self.assertLess(chunks_2x, chunks_1x)


class OrchestratorLanguageAndDeltaTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="lang@example.com", password="pw")
        self.meeting = Meeting.objects.create(
            name="Lang meeting",
            slug="m-lang",
            forced_language="no",
            created_by=self.user,
        )

    def _fake_extract(self, source_path, boundary, max_bytes):
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, prefix="lang_")
        tmp.write(b"\x00" * 16)
        tmp.close()
        return Path(tmp.name)

    @patch("meetings.services.audio_transcription._extract_chunk")
    @patch("meetings.services.audio_transcription._plan_upload_chunks")
    def test_orchestrator_passes_forced_language_and_delta_callback(self, mock_plan, mock_extract):
        mock_plan.return_value = [ChunkBoundary(index=0, start_ms=0, end_ms=1000)]
        mock_extract.side_effect = self._fake_extract

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
