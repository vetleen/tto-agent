"""Unit tests for meetings.services.audio_transcription.

Covers the prompt builder, the overlap-aware audio splitter (with ffmpeg
mocked), the fuzzy transcript stitcher, and the top-level orchestrator
that drives the upload-path Celery task. The shared TranscriptionService
is always mocked here — its own behaviour is tested in
``documents/tests/test_transcription.py``.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase

from meetings.models import Meeting
from meetings.services.audio_transcription import (
    AUDIO_SPLIT_TIMEOUT_SECONDS,
    CHARS_PER_OVERLAP_SECOND,
    DEFAULT_OVERLAP_SECONDS,
    AudioSplitTimeoutError,
    ChunkBoundary,
    _extract_chunk,
    _plan_upload_chunks,
    build_transcription_prompt,
    orchestrate_upload_transcription,
    plan_chunk_boundaries,
    stitch_transcripts,
)

User = get_user_model()


# ---------------------------------------------------------------------------
# build_transcription_prompt
# ---------------------------------------------------------------------------


class BuildTranscriptionPromptTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="bp@example.com", password="pw")

    def _meeting(self, **kwargs):
        defaults = dict(name="M", slug="m-prompt", created_by=self.user)
        defaults.update(kwargs)
        return Meeting.objects.create(**defaults)

    def test_full_metadata_no_tail(self):
        m = self._meeting(
            name="Q1 board meeting",
            agenda="Review pipeline; OncoBio Therapeutics update",
            participants="Alice, Bob, Carol",
            description="Quarterly review with the founders.",
        )
        prompt = build_transcription_prompt(m)
        self.assertIn("business meeting", prompt)
        self.assertIn("Meeting: Q1 board meeting", prompt)
        self.assertIn("Agenda: Review pipeline", prompt)
        self.assertIn("OncoBio Therapeutics", prompt)
        self.assertIn("Participants: Alice, Bob, Carol", prompt)
        self.assertIn("Notes: Quarterly review", prompt)
        self.assertNotIn("Previous transcript excerpt", prompt)

    def test_empty_fields_are_skipped(self):
        m = self._meeting(name="Just a name")
        prompt = build_transcription_prompt(m)
        self.assertIn("Meeting: Just a name", prompt)
        # No empty label lines.
        self.assertNotIn("Agenda:", prompt)
        self.assertNotIn("Participants:", prompt)
        self.assertNotIn("Notes:", prompt)

    def test_with_prior_tail(self):
        m = self._meeting(name="Continuing meeting")
        prompt = build_transcription_prompt(m, prior_tail="…and then we agreed to extend the deadline.")
        self.assertIn("Previous transcript excerpt (for context only, do not repeat):", prompt)
        self.assertIn("agreed to extend the deadline", prompt)

    def test_long_agenda_passes_through_unchanged(self):
        """No preemptive truncation — the API gets exactly what we built."""
        long_agenda = "topic; " * 5000
        m = self._meeting(agenda=long_agenda)
        prompt = build_transcription_prompt(m)
        self.assertIn(long_agenda.strip(), prompt)


# ---------------------------------------------------------------------------
# plan_chunk_boundaries (pure math) + probe/extract helpers
# ---------------------------------------------------------------------------


class PlanChunkBoundariesTests(TestCase):
    """Boundary planning is pure arithmetic — no ffmpeg, no disk."""

    def test_short_file_single_chunk_no_overlap(self):
        boundaries = plan_chunk_boundaries(
            60_000,  # 60 s
            target_chunk_seconds=900,
            overlap_seconds=15,
            max_bytes=25_000_000,
            max_seconds=1400,
        )
        self.assertEqual(len(boundaries), 1)
        self.assertEqual(boundaries[0].start_ms, 0)
        self.assertEqual(boundaries[0].end_ms, 60_000)

    def test_two_chunks_with_correct_overlap(self):
        # 2 * target = 1800 s. We expect 2 chunks; the second should start
        # at 900_000 - 15_000 = 885_000 ms (15 s leading overlap).
        total_ms = 1_800_000
        boundaries = plan_chunk_boundaries(
            total_ms,
            target_chunk_seconds=900,
            overlap_seconds=15,
            max_bytes=25_000_000,
            max_seconds=1400,
        )
        self.assertEqual(len(boundaries), 2)
        self.assertEqual(boundaries[0].start_ms, 0)
        self.assertEqual(boundaries[0].end_ms, 900_000)
        self.assertEqual(boundaries[1].start_ms, 900_000 - 15_000)
        self.assertEqual(boundaries[1].end_ms, total_ms)

    def test_four_chunks_for_3_5x_target(self):
        # 3.5 * target = 3150 s -> ceil(3150/900) = 4 chunks
        total_ms = 3_150_000
        boundaries = plan_chunk_boundaries(
            total_ms,
            target_chunk_seconds=900,
            overlap_seconds=15,
            max_bytes=25_000_000,
            max_seconds=1400,
        )
        self.assertEqual(len(boundaries), 4)
        self.assertEqual(boundaries[0].start_ms, 0)
        for i in range(1, 4):
            # Each subsequent chunk starts 15 s before the previous chunk's
            # nominal end.
            self.assertEqual(boundaries[i].start_ms, i * 900_000 - 15_000)
        # Last chunk ends at the file end.
        self.assertEqual(boundaries[-1].end_ms, total_ms)

    def test_non_positive_duration_yields_no_chunks(self):
        self.assertEqual(
            plan_chunk_boundaries(0, max_bytes=25_000_000, max_seconds=1400), []
        )

    def test_degenerate_overlap_raises(self):
        with self.assertRaises(ValueError):
            plan_chunk_boundaries(
                60_000,
                target_chunk_seconds=10,
                overlap_seconds=10,  # 2*overlap >= target
                max_bytes=25_000_000,
                max_seconds=1400,
            )

    def test_negative_overlap_raises(self):
        with self.assertRaises(ValueError):
            plan_chunk_boundaries(
                60_000,
                target_chunk_seconds=900,
                overlap_seconds=-1,
                max_bytes=25_000_000,
                max_seconds=1400,
            )


class _FakeInfo:
    """Minimal transcription-model info stand-in for _plan_upload_chunks."""
    max_file_size_bytes = 25_000_000
    max_duration_seconds = 1400


class PlanUploadChunksTests(TestCase):
    """The probe wrapper falls back to a single direct pass when it can't split."""

    def test_no_ffmpeg_returns_none_for_small_file(self):
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(b"\x00" * 1024)
            tmp = Path(f.name)
        try:
            with patch("llm.service._audio_subprocess.ffmpeg_available", return_value=False):
                result = _plan_upload_chunks(
                    tmp, _FakeInfo(),
                    target_chunk_seconds=300, overlap_seconds=15, meeting_id=1,
                )
            self.assertIsNone(result)
        finally:
            tmp.unlink(missing_ok=True)

    def test_unprobeable_duration_returns_none(self):
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(b"\x00" * 1024)
            tmp = Path(f.name)
        try:
            with patch("llm.service._audio_subprocess.ffmpeg_available", return_value=True), \
                 patch("llm.service._audio_subprocess.ffprobe_duration_ms", return_value=None):
                result = _plan_upload_chunks(
                    tmp, _FakeInfo(),
                    target_chunk_seconds=300, overlap_seconds=15, meeting_id=1,
                )
            self.assertIsNone(result)
        finally:
            tmp.unlink(missing_ok=True)

    def test_oversized_unsplittable_file_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(b"\x00" * 2048)
            tmp = Path(f.name)
        try:
            with patch("llm.service._audio_subprocess.ffmpeg_available", return_value=False), \
                 patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value = MagicMock(st_size=99_000_000)
                with self.assertRaises(RuntimeError) as ctx:
                    _plan_upload_chunks(
                        tmp, _FakeInfo(),
                        target_chunk_seconds=300, overlap_seconds=15, meeting_id=1,
                    )
            self.assertIn("exceeds", str(ctx.exception))
        finally:
            tmp.unlink(missing_ok=True)


class ExtractChunkTests(TestCase):
    @patch("meetings.services.audio_transcription.AUDIO_SPLIT_TIMEOUT_SECONDS", 1)
    def test_extract_timeout_raises_and_logs(self):
        import subprocess as _sp

        boundary = ChunkBoundary(index=0, start_ms=0, end_ms=1000)
        with patch(
            "llm.service._audio_subprocess.ffmpeg_extract_chunk",
            side_effect=_sp.TimeoutExpired(cmd="ffmpeg", timeout=1),
        ):
            with self.assertLogs("meetings.services.audio_transcription", level="ERROR") as cm:
                with self.assertRaises(AudioSplitTimeoutError) as ctx:
                    _extract_chunk(Path("/fake.mp3"), boundary, 25_000_000)
        self.assertIn("timed out", str(ctx.exception))
        self.assertTrue(any("extraction timed out" in line for line in cm.output))


# ---------------------------------------------------------------------------
# stitch_transcripts
# ---------------------------------------------------------------------------


class StitchTranscriptsTests(TestCase):
    def test_clean_exact_overlap(self):
        prev = "We started by reviewing the agenda and then the team discussed the budget"
        nxt = "the team discussed the budget for next quarter and the new hire plan"
        result = stitch_transcripts(prev, nxt, expected_overlap_chars=40)
        # The overlap should be deduped — we should not see two copies of
        # "the team discussed the budget".
        self.assertEqual(result.count("the team discussed the budget"), 1)
        self.assertIn("for next quarter", result)
        self.assertIn("reviewing the agenda", result)

    def test_fuzzy_overlap_with_substitution(self):
        prev = "Then we covered the new product roadmap and shipping timeline today"
        nxt = "the new product roadmap and shipping timetable next year and the marketing plan"
        result = stitch_transcripts(prev, nxt, expected_overlap_chars=40)
        # Both halves should be present.
        self.assertIn("Then we covered", result)
        self.assertIn("marketing plan", result)
        # No trivial duplication of the matching phrase.
        self.assertEqual(result.count("the new product roadmap"), 1)

    def test_no_match_falls_back_to_drop(self):
        prev = "Completely unrelated text about coffee preferences in the office today."
        nxt = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, totally different content here."
        result = stitch_transcripts(prev, nxt, expected_overlap_chars=20)
        # Fallback drops first 20 chars from next.
        self.assertIn("coffee preferences", result)
        # The first 20 chars of next ("Lorem ipsum dolor si") should NOT be in
        # the merged result.
        self.assertNotIn("Lorem ipsum dolor si", result)

    def test_empty_prev_returns_next(self):
        self.assertEqual(stitch_transcripts("", "hello world", expected_overlap_chars=10), "hello world")

    def test_empty_next_returns_prev(self):
        self.assertEqual(stitch_transcripts("hello world", "", expected_overlap_chars=10), "hello world")

    def test_both_empty(self):
        self.assertEqual(stitch_transcripts("", "", expected_overlap_chars=10), "")

    def test_zero_expected_overlap_concatenates(self):
        result = stitch_transcripts("abc", "def", expected_overlap_chars=0)
        self.assertEqual(result, "abc def")


# ---------------------------------------------------------------------------
# orchestrate_upload_transcription
# ---------------------------------------------------------------------------


def _make_boundary(index: int) -> ChunkBoundary:
    return ChunkBoundary(index=index, start_ms=index * 1000, end_ms=(index + 1) * 1000)


def _fake_extract_chunk(source_path, boundary, max_bytes):
    """Stand-in for _extract_chunk: write a tiny temp file and return its Path.

    Signature mirrors meetings.services.audio_transcription._extract_chunk so it
    can be used as a ``side_effect`` for that patch.
    """
    tmp = tempfile.NamedTemporaryFile(
        suffix=".mp3", delete=False, prefix=f"orch_test_seg{boundary.index}_",
    )
    tmp.write(b"\x00" * 16)
    tmp.close()
    return Path(tmp.name)


class _FakeResult:
    def __init__(self, text: str):
        self.text = text


class OrchestratorTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="orch@example.com", password="pw")
        self.meeting = Meeting.objects.create(
            name="Orchestrated meeting",
            slug="m-orch",
            agenda="Discuss OncoBio Therapeutics partnership",
            participants="Alice, Bob",
            created_by=self.user,
        )

    def _fresh(self) -> Meeting:
        return Meeting.objects.get(pk=self.meeting.pk)

    @patch("meetings.services.audio_transcription._extract_chunk", side_effect=_fake_extract_chunk)
    @patch("meetings.services.audio_transcription._plan_upload_chunks")
    def test_single_chunk_path_skips_progress_fields(self, mock_plan, mock_extract):
        mock_plan.return_value = [_make_boundary(0)]

        service = MagicMock()
        service.transcribe.return_value = _FakeResult("only chunk text")

        text = orchestrate_upload_transcription(
            meeting_id=self.meeting.pk,
            temp_path=Path("/fake.mp3"),
            model_id="openai/gpt-4o-mini-transcribe",
            user_id=self.user.pk,
            service=service,
        )
        self.assertEqual(text, "only chunk text")

        m = self._fresh()
        self.assertEqual(m.status, Meeting.Status.READY)
        self.assertEqual(m.transcript, "only chunk text")
        self.assertEqual(m.transcription_chunks_total, 0)
        self.assertEqual(m.transcription_chunks_done, 0)
        # Single transcribe call with the meta prompt (no prior tail).
        self.assertEqual(service.transcribe.call_count, 1)
        prompt_arg = service.transcribe.call_args.kwargs["prompt"]
        self.assertIn("Orchestrated meeting", prompt_arg)
        self.assertIn("OncoBio Therapeutics", prompt_arg)
        self.assertNotIn("Previous transcript excerpt", prompt_arg)

    @patch("meetings.services.audio_transcription._extract_chunk", side_effect=_fake_extract_chunk)
    @patch("meetings.services.audio_transcription._plan_upload_chunks")
    def test_multi_chunk_happy_path_progress_and_carryover(self, mock_plan, mock_extract):
        mock_plan.return_value = [_make_boundary(i) for i in range(3)]

        # Each chunk's "transcript" includes a long, distinctive overlap phrase
        # with the next chunk so the stitcher's fuzzy matcher splices them
        # cleanly. The overlap phrases are intentionally well over the
        # confident-match floor (max(20, expected_overlap_chars // 3)).
        overlap_one_two = (
            "OncoBio Therapeutics partnership terms in great detail and the "
            "new product roadmap announcements coming up next quarter"
        )
        overlap_two_three = (
            "we discussed the marketing budget for the Q3 launch event in "
            "some detail and the projected ROI for the campaign"
        )
        chunk_texts = [
            "First chunk introduction text and then the team reviewed " + overlap_one_two,
            overlap_one_two + " and then " + overlap_two_three,
            overlap_two_three + " and finally we wrapped up with the sales target review for the second half of the year",
        ]
        service = MagicMock()
        service.transcribe.side_effect = [_FakeResult(t) for t in chunk_texts]

        orchestrate_upload_transcription(
            meeting_id=self.meeting.pk,
            temp_path=Path("/fake.mp3"),
            model_id="openai/gpt-4o-mini-transcribe",
            user_id=self.user.pk,
            service=service,
        )

        m = self._fresh()
        self.assertEqual(m.status, Meeting.Status.READY)
        self.assertEqual(m.transcription_chunks_total, 0)  # reset on success
        self.assertEqual(m.transcription_chunks_done, 0)
        self.assertEqual(m.transcription_model, "openai/gpt-4o-mini-transcribe")
        self.assertIn("First chunk introduction", m.transcript)
        self.assertIn("sales target review", m.transcript)
        # No duplication of the long overlap phrases.
        self.assertEqual(m.transcript.count(overlap_one_two), 1)
        self.assertEqual(m.transcript.count(overlap_two_three), 1)

        # Each transcribe call got a prompt; chunks 1 and 2 had a prior tail.
        self.assertEqual(service.transcribe.call_count, 3)
        first_prompt = service.transcribe.call_args_list[0].kwargs["prompt"]
        second_prompt = service.transcribe.call_args_list[1].kwargs["prompt"]
        third_prompt = service.transcribe.call_args_list[2].kwargs["prompt"]
        self.assertNotIn("Previous transcript excerpt", first_prompt)
        self.assertIn("Previous transcript excerpt", second_prompt)
        self.assertIn("Previous transcript excerpt", third_prompt)

    @patch("meetings.services.audio_transcription._extract_chunk", side_effect=_fake_extract_chunk)
    @patch("meetings.services.audio_transcription._plan_upload_chunks")
    def test_chunk_failure_persists_partial_and_resets_progress(self, mock_plan, mock_extract):
        mock_plan.return_value = [_make_boundary(i) for i in range(3)]

        service = MagicMock()
        service.transcribe.side_effect = [
            _FakeResult("first chunk transcript"),
            RuntimeError("boom"),
        ]

        with self.assertRaises(RuntimeError):
            orchestrate_upload_transcription(
                meeting_id=self.meeting.pk,
                temp_path=Path("/fake.mp3"),
                model_id="openai/gpt-4o-mini-transcribe",
                user_id=self.user.pk,
                service=service,
            )

        m = self._fresh()
        self.assertEqual(m.status, Meeting.Status.FAILED)
        self.assertEqual(m.transcript, "first chunk transcript")
        self.assertIn("Failed on chunk 2/3", m.transcription_error)
        self.assertIn("partial transcript saved", m.transcription_error)
        # Progress fields reset so the polling UI does not show a stale bar.
        self.assertEqual(m.transcription_chunks_total, 0)
        self.assertEqual(m.transcription_chunks_done, 0)

    @patch("meetings.services.audio_transcription._extract_chunk", side_effect=_fake_extract_chunk)
    @patch("meetings.services.audio_transcription._plan_upload_chunks")
    def test_cancellation_bails_before_next_chunk_and_preserves_partial(self, mock_plan, mock_extract):
        """When the user clicks Stop mid-way, the cancel view flips status to
        FAILED. The orchestrator must detect that before starting the next
        chunk, persist whatever was stitched so far, and return instead of
        raising."""
        mock_plan.return_value = [_make_boundary(i) for i in range(3)]

        service = MagicMock()

        # Simulate the user clicking Stop after chunk 0 returns. The cancel
        # view sets status=FAILED; subsequent iterations of the orchestrator
        # loop should observe that and bail.
        def transcribe_side_effect(*args, **kwargs):
            call_num = service.transcribe.call_count
            if call_num == 1:
                # Before the first return, flip status to FAILED to simulate
                # the cancel endpoint firing while this chunk was in flight.
                Meeting.objects.filter(pk=self.meeting.pk).update(
                    status=Meeting.Status.FAILED,
                    transcription_error="Cancelled by user",
                )
                return _FakeResult("first chunk text that survives cancellation")
            raise AssertionError(
                "orchestrator called transcribe after cancellation "
                "— should have bailed before starting next chunk"
            )

        service.transcribe.side_effect = transcribe_side_effect

        # Must return normally (not raise) — cancellation is not an error.
        text = orchestrate_upload_transcription(
            meeting_id=self.meeting.pk,
            temp_path=Path("/fake.mp3"),
            model_id="openai/gpt-4o-mini-transcribe",
            user_id=self.user.pk,
            service=service,
        )

        # Only chunk 0 should have been transcribed; chunks 1 and 2 skipped.
        self.assertEqual(service.transcribe.call_count, 1)
        self.assertIn("first chunk text that survives cancellation", text)

        m = self._fresh()
        # Cancel view already set status=FAILED; the orchestrator must not
        # overwrite it back to READY.
        self.assertEqual(m.status, Meeting.Status.FAILED)
        self.assertEqual(m.transcription_error, "Cancelled by user")
        # Partial transcript is preserved.
        self.assertIn("first chunk text that survives cancellation", m.transcript)
        # Progress fields reset so the polling UI does not show a stale bar.
        self.assertEqual(m.transcription_chunks_total, 0)
        self.assertEqual(m.transcription_chunks_done, 0)

    @patch("meetings.services.audio_transcription._extract_chunk", side_effect=_fake_extract_chunk)
    @patch("meetings.services.audio_transcription._plan_upload_chunks")
    def test_cancellation_before_first_chunk_preserves_existing_transcript(self, mock_plan, mock_extract):
        """If cancellation happens before ANY chunk completes (edge case: user
        is fast, or the first chunk hasn't been dispatched yet), the meeting's
        existing transcript must survive untouched."""
        self.meeting.transcript = "PRIOR TRANSCRIPT FROM FILE ONE"
        self.meeting.save(update_fields=["transcript"])

        mock_plan.return_value = [_make_boundary(i) for i in range(3)]

        # Flip status before the loop even starts.
        Meeting.objects.filter(pk=self.meeting.pk).update(
            status=Meeting.Status.FAILED,
            transcription_error="Cancelled by user",
        )
        # But the multi-chunk branch also resets status to LIVE_TRANSCRIBING
        # at the top of the loop (line ~437 in the orchestrator). Simulate a
        # cancel that races in AFTER that reset by using a service that flips
        # status to FAILED *during* the status check itself — but simpler: we
        # just let the orchestrator start normally, and the cancel-check at
        # the top of the loop iteration catches it when we flip status inside
        # the first transcribe call.
        service = MagicMock()

        def flip_and_raise(*args, **kwargs):
            # Before any chunk returns, flip status to FAILED. Since this is
            # inside the first transcribe call, the orchestrator will still
            # receive this result and advance to the next iteration, where
            # it should see FAILED and bail.
            Meeting.objects.filter(pk=self.meeting.pk).update(
                status=Meeting.Status.FAILED,
                transcription_error="Cancelled by user",
            )
            return _FakeResult("")  # empty result — user cancelled immediately

        service.transcribe.side_effect = flip_and_raise

        text = orchestrate_upload_transcription(
            meeting_id=self.meeting.pk,
            temp_path=Path("/fake.mp3"),
            model_id="openai/gpt-4o-mini-transcribe",
            user_id=self.user.pk,
            service=service,
        )

        m = self._fresh()
        self.assertEqual(m.status, Meeting.Status.FAILED)
        # Existing transcript MUST still be present even though the new
        # chunk produced empty text.
        self.assertIn("PRIOR TRANSCRIPT FROM FILE ONE", m.transcript)
        self.assertIn("PRIOR TRANSCRIPT FROM FILE ONE", text)

    @patch("meetings.services.audio_transcription._extract_chunk", side_effect=_fake_extract_chunk)
    @patch("meetings.services.audio_transcription._plan_upload_chunks")
    def test_single_chunk_appends_to_existing_transcript(self, mock_plan, mock_extract):
        """Re-uploading a short (single-chunk) file to a meeting that already
        has a transcript must append the new text, never replace it."""
        self.meeting.transcript = "PRIOR TRANSCRIPT FROM FILE ONE — do not lose."
        self.meeting.save(update_fields=["transcript"])

        mock_plan.return_value = [_make_boundary(0)]
        service = MagicMock()
        service.transcribe.return_value = _FakeResult("fresh content from file two")

        text = orchestrate_upload_transcription(
            meeting_id=self.meeting.pk,
            temp_path=Path("/fake.mp3"),
            model_id="openai/gpt-4o-mini-transcribe",
            user_id=self.user.pk,
            service=service,
        )

        m = self._fresh()
        # Existing content MUST still be present.
        self.assertIn("PRIOR TRANSCRIPT FROM FILE ONE", m.transcript)
        # New content MUST also be present.
        self.assertIn("fresh content from file two", m.transcript)
        # Order: existing first, new second.
        self.assertLess(
            m.transcript.index("PRIOR TRANSCRIPT"),
            m.transcript.index("fresh content"),
        )
        # Return value matches the persisted transcript.
        self.assertEqual(m.transcript, text)
        self.assertEqual(m.status, Meeting.Status.READY)

        # The single-chunk prompt should have carried the existing transcript
        # tail as context for proper-noun continuity.
        prompt_arg = service.transcribe.call_args.kwargs["prompt"]
        self.assertIn("Previous transcript excerpt", prompt_arg)
        self.assertIn("do not lose", prompt_arg)

    @patch("meetings.services.audio_transcription._extract_chunk", side_effect=_fake_extract_chunk)
    @patch("meetings.services.audio_transcription._plan_upload_chunks")
    def test_multi_chunk_appends_to_existing_transcript(self, mock_plan, mock_extract):
        """Re-uploading a multi-chunk file must preserve the prior transcript
        AND stitch all new chunks — not just the last chunk."""
        self.meeting.transcript = (
            "PRIOR TRANSCRIPT HEADER\n\n"
            "This is file one's full content with meaningful text inside."
        )
        self.meeting.save(update_fields=["transcript"])

        mock_plan.return_value = [_make_boundary(i) for i in range(3)]

        # Build overlap phrases long enough for the fuzzy stitcher's confident
        # match floor (same pattern as the existing happy-path test).
        overlap_one_two = (
            "the partnership structure between the two parties was reviewed "
            "and then we discussed the budget allocation for next quarter"
        )
        overlap_two_three = (
            "the marketing plan for the product launch event in Stockholm "
            "and then the Q3 sales targets for the new territory"
        )
        chunk_texts = [
            "FILE TWO CHUNK ZERO intro text and " + overlap_one_two,
            overlap_one_two + " and then " + overlap_two_three,
            overlap_two_three + " and FILE TWO CHUNK THREE closing remarks and thank you",
        ]
        service = MagicMock()
        service.transcribe.side_effect = [_FakeResult(t) for t in chunk_texts]

        orchestrate_upload_transcription(
            meeting_id=self.meeting.pk,
            temp_path=Path("/fake.mp3"),
            model_id="openai/gpt-4o-mini-transcribe",
            user_id=self.user.pk,
            service=service,
        )

        m = self._fresh()
        self.assertEqual(m.status, Meeting.Status.READY)

        # The prior transcript MUST still be there in full.
        self.assertIn("PRIOR TRANSCRIPT HEADER", m.transcript)
        self.assertIn("meaningful text inside", m.transcript)

        # EVERY chunk's distinctive content should be in the final transcript,
        # not just the last one. This is the regression we're guarding.
        self.assertIn("FILE TWO CHUNK ZERO intro text", m.transcript)
        self.assertIn("FILE TWO CHUNK THREE closing remarks", m.transcript)

        # Order: existing first, then the new stitched text.
        self.assertLess(
            m.transcript.index("PRIOR TRANSCRIPT HEADER"),
            m.transcript.index("FILE TWO CHUNK ZERO"),
        )
        self.assertLess(
            m.transcript.index("FILE TWO CHUNK ZERO"),
            m.transcript.index("FILE TWO CHUNK THREE"),
        )

        # No duplication of the internal overlap phrases.
        self.assertEqual(m.transcript.count(overlap_one_two), 1)
        self.assertEqual(m.transcript.count(overlap_two_three), 1)

        # First new chunk's prompt should pull context from the EXISTING
        # transcript (not from nothing); later chunks pull from running tail.
        first_prompt = service.transcribe.call_args_list[0].kwargs["prompt"]
        self.assertIn("Previous transcript excerpt", first_prompt)
        self.assertIn("meaningful text inside", first_prompt)

    @patch("meetings.services.audio_transcription._extract_chunk", side_effect=_fake_extract_chunk)
    @patch("meetings.services.audio_transcription._plan_upload_chunks")
    def test_multi_chunk_append_failure_preserves_existing(self, mock_plan, mock_extract):
        """If a chunk fails mid-upload, the existing transcript must still be
        present (alongside any partial new content)."""
        self.meeting.transcript = "KEEP THIS PRIOR TRANSCRIPT SAFE"
        self.meeting.save(update_fields=["transcript"])

        mock_plan.return_value = [_make_boundary(i) for i in range(3)]

        service = MagicMock()
        service.transcribe.side_effect = [
            _FakeResult("first new chunk content"),
            RuntimeError("network went away"),
        ]

        with self.assertRaises(RuntimeError):
            orchestrate_upload_transcription(
                meeting_id=self.meeting.pk,
                temp_path=Path("/fake.mp3"),
                model_id="openai/gpt-4o-mini-transcribe",
                user_id=self.user.pk,
                service=service,
            )

        m = self._fresh()
        self.assertEqual(m.status, Meeting.Status.FAILED)
        # Prior content MUST still be present — a failed re-upload is not
        # allowed to eat the previous transcript.
        self.assertIn("KEEP THIS PRIOR TRANSCRIPT SAFE", m.transcript)
        # Partial new content is also persisted.
        self.assertIn("first new chunk content", m.transcript)

    @patch("meetings.services.audio_transcription.time.sleep", return_value=None)
    @patch("meetings.services.audio_transcription._extract_chunk", side_effect=_fake_extract_chunk)
    @patch("meetings.services.audio_transcription._plan_upload_chunks")
    def test_transient_error_is_retried(self, mock_plan, mock_extract, mock_sleep):
        try:
            from openai import RateLimitError
        except Exception:  # pragma: no cover
            self.skipTest("openai library missing")

        mock_plan.return_value = [_make_boundary(0), _make_boundary(1)]

        rate_limited = RateLimitError(
            message="429 rate-limited",
            response=MagicMock(),
            body=None,
        )
        # Use long, distinctive texts so the stitcher's fallback doesn't drop
        # everything in chunk 2 — the test is about retry, not stitching, but
        # we still need the orchestrator to reach READY successfully.
        long_overlap = "the discussion of the partnership terms and the planned roadmap for the next two quarters"
        service = MagicMock()
        service.transcribe.side_effect = [
            _FakeResult("First chunk content followed by " + long_overlap),
            rate_limited,  # transient on chunk 1
            _FakeResult(long_overlap + " and then second chunk content after retry"),
        ]

        orchestrate_upload_transcription(
            meeting_id=self.meeting.pk,
            temp_path=Path("/fake.mp3"),
            model_id="openai/gpt-4o-mini-transcribe",
            user_id=self.user.pk,
            service=service,
        )

        # Three total calls: chunk 0 (ok), chunk 1 (fail), chunk 1 (retry ok).
        self.assertEqual(service.transcribe.call_count, 3)
        m = self._fresh()
        self.assertEqual(m.status, Meeting.Status.READY)
        self.assertIn("First chunk content", m.transcript)
        self.assertIn("second chunk content after retry", m.transcript)
        # We slept for the first backoff entry.
        self.assertTrue(mock_sleep.called)

    @patch("meetings.services.audio_transcription._extract_chunk", side_effect=_fake_extract_chunk)
    @patch("meetings.services.audio_transcription._plan_upload_chunks")
    def test_chunk_plan_is_logged(self, mock_plan, mock_extract):
        """The orchestrator logs the planned chunk count at INFO level."""
        mock_plan.return_value = [_make_boundary(0)]

        service = MagicMock()
        service.transcribe.return_value = _FakeResult("ok")

        with self.assertLogs("meetings.services.audio_transcription", level="INFO") as cm:
            orchestrate_upload_transcription(
                meeting_id=self.meeting.pk,
                temp_path=Path("/fake.mp3"),
                model_id="openai/gpt-4o-mini-transcribe",
                user_id=self.user.pk,
                service=service,
            )

        plan_log = [line for line in cm.output if "planned" in line and "chunk" in line]
        self.assertEqual(len(plan_log), 1)
        self.assertIn("1 chunk", plan_log[0])

    @patch("meetings.services.audio_transcription._extract_chunk")
    @patch("meetings.services.audio_transcription._plan_upload_chunks")
    def test_single_chunk_extract_timeout_propagates(self, mock_plan, mock_extract):
        """An extraction timeout surfaces as AudioSplitTimeoutError to the caller
        (the Celery task wrapper then marks the meeting FAILED)."""
        mock_plan.return_value = [_make_boundary(0)]
        mock_extract.side_effect = AudioSplitTimeoutError(
            "Audio chunk 0 extraction timed out after 1s."
        )

        with self.assertRaises(AudioSplitTimeoutError) as ctx:
            orchestrate_upload_transcription(
                meeting_id=self.meeting.pk,
                temp_path=Path("/fake.mp3"),
                model_id="openai/gpt-4o-mini-transcribe",
                user_id=self.user.pk,
                service=MagicMock(),
            )
        self.assertIn("timed out", str(ctx.exception))

    @patch("meetings.services.audio_transcription._extract_chunk")
    @patch("meetings.services.audio_transcription._plan_upload_chunks")
    def test_extract_failure_mid_upload_preserves_partial(self, mock_plan, mock_extract):
        """An extraction failure on a later chunk persists the partial transcript
        and marks the meeting FAILED — same contract as a transcribe failure."""
        mock_plan.return_value = [_make_boundary(i) for i in range(3)]

        def extract_side_effect(source_path, boundary, max_bytes):
            if boundary.index == 0:
                return _fake_extract_chunk(source_path, boundary, max_bytes)
            raise AudioSplitTimeoutError("Audio chunk 1 extraction timed out after 1s.")
        mock_extract.side_effect = extract_side_effect

        service = MagicMock()
        service.transcribe.return_value = _FakeResult("first chunk transcript")

        with self.assertRaises(AudioSplitTimeoutError):
            orchestrate_upload_transcription(
                meeting_id=self.meeting.pk,
                temp_path=Path("/fake.mp3"),
                model_id="openai/gpt-4o-mini-transcribe",
                user_id=self.user.pk,
                service=service,
            )

        m = self._fresh()
        self.assertEqual(m.status, Meeting.Status.FAILED)
        self.assertIn("first chunk transcript", m.transcript)
        self.assertIn("Failed on chunk 2/3", m.transcription_error)
        self.assertEqual(m.transcription_chunks_total, 0)
        self.assertEqual(m.transcription_chunks_done, 0)
