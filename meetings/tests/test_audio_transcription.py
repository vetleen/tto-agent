"""Unit tests for meetings.services.audio_transcription.

Covers the prompt builder, the overlap-aware audio splitter (with pydub
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
    ChunkSpec,
    build_transcription_prompt,
    orchestrate_upload_transcription,
    split_audio_with_overlap,
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
# split_audio_with_overlap
# ---------------------------------------------------------------------------


class FakeAudio:
    """Minimal pydub.AudioSegment stand-in."""

    def __init__(self, total_ms: int):
        self.total_ms = total_ms

    def __len__(self):
        return self.total_ms

    def __getitem__(self, item):
        # Slicing returns another FakeAudio whose length matches the slice.
        if isinstance(item, slice):
            start = item.start or 0
            stop = item.stop if item.stop is not None else self.total_ms
            return FakeAudio(max(0, min(stop, self.total_ms) - max(0, start)))
        raise TypeError("FakeAudio only supports slicing")

    def export(self, path, format="mp3"):
        # Write a tiny placeholder so .stat().st_size > 0 but well under any limit.
        with open(path, "wb") as f:
            f.write(b"\x00" * 32)


def _patch_pydub(total_ms: int):
    """Patch pydub.AudioSegment.from_file to return a FakeAudio of the given length."""
    fake = FakeAudio(total_ms)
    pydub_mod = MagicMock()
    pydub_mod.AudioSegment.from_file.return_value = fake
    return patch.dict("sys.modules", {"pydub": pydub_mod})


class SplitAudioWithOverlapTests(TestCase):
    """Splitter tests use FakeAudio to avoid touching real audio files."""

    def test_short_file_single_chunk_no_overlap(self):
        with _patch_pydub(total_ms=60_000):  # 60 s
            specs = split_audio_with_overlap(
                Path("/fake.mp3"),
                target_chunk_seconds=900,
                overlap_seconds=15,
                max_bytes=25_000_000,
                max_seconds=1400,
            )
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].start_ms, 0)
        self.assertEqual(specs[0].end_ms, 60_000)
        for s in specs:
            s.path.unlink(missing_ok=True)

    def test_two_chunks_with_correct_overlap(self):
        # 2 * target = 1800 s. We expect 2 chunks; the second should start
        # at 900_000 - 15_000 = 885_000 ms (15 s leading overlap).
        total_ms = 1_800_000
        with _patch_pydub(total_ms=total_ms):
            specs = split_audio_with_overlap(
                Path("/fake.mp3"),
                target_chunk_seconds=900,
                overlap_seconds=15,
                max_bytes=25_000_000,
                max_seconds=1400,
            )
        try:
            self.assertEqual(len(specs), 2)
            self.assertEqual(specs[0].start_ms, 0)
            self.assertEqual(specs[0].end_ms, 900_000)
            self.assertEqual(specs[1].start_ms, 900_000 - 15_000)
            self.assertEqual(specs[1].end_ms, total_ms)
        finally:
            for s in specs:
                s.path.unlink(missing_ok=True)

    def test_four_chunks_for_3_5x_target(self):
        # 3.5 * target = 3150 s -> ceil(3150/900) = 4 chunks
        total_ms = 3_150_000
        with _patch_pydub(total_ms=total_ms):
            specs = split_audio_with_overlap(
                Path("/fake.mp3"),
                target_chunk_seconds=900,
                overlap_seconds=15,
                max_bytes=25_000_000,
                max_seconds=1400,
            )
        try:
            self.assertEqual(len(specs), 4)
            self.assertEqual(specs[0].start_ms, 0)
            for i in range(1, 4):
                # Each subsequent chunk starts 15 s before the previous chunk's
                # nominal end.
                self.assertEqual(specs[i].start_ms, i * 900_000 - 15_000)
            # Last chunk ends at the file end.
            self.assertEqual(specs[-1].end_ms, total_ms)
        finally:
            for s in specs:
                s.path.unlink(missing_ok=True)

    def test_degenerate_overlap_raises(self):
        with self.assertRaises(ValueError):
            split_audio_with_overlap(
                Path("/fake.mp3"),
                target_chunk_seconds=10,
                overlap_seconds=10,  # 2*overlap >= target
                max_bytes=25_000_000,
                max_seconds=1400,
            )

    def test_negative_overlap_raises(self):
        with self.assertRaises(ValueError):
            split_audio_with_overlap(
                Path("/fake.mp3"),
                target_chunk_seconds=900,
                overlap_seconds=-1,
                max_bytes=25_000_000,
                max_seconds=1400,
            )

    def test_ffprobe_json_error_falls_back_to_single_chunk(self):
        """When ffprobe returns empty output, AudioSegment.from_file raises
        JSONDecodeError. The splitter should fall back to a single chunk
        pointing at the original file instead of crashing."""
        import json

        pydub_mod = MagicMock()
        pydub_mod.AudioSegment.from_file.side_effect = json.JSONDecodeError(
            "Expecting value", "", 0
        )

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(b"\x00" * 1024)
            tmp = Path(f.name)
        try:
            with patch.dict("sys.modules", {"pydub": pydub_mod}), \
                 patch("shutil.which", return_value="/usr/bin/ffmpeg"):
                specs = split_audio_with_overlap(
                    tmp,
                    target_chunk_seconds=900,
                    overlap_seconds=15,
                    max_bytes=25_000_000,
                    max_seconds=1400,
                )
            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0].path, tmp)
            self.assertEqual(specs[0].index, 0)
        finally:
            tmp.unlink(missing_ok=True)


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


def _make_chunk_spec(index: int) -> ChunkSpec:
    """Create a real (but tiny) temp file and return a ChunkSpec for it."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, prefix=f"orch_test_seg{index}_")
    tmp.write(b"\x00" * 16)
    tmp.close()
    return ChunkSpec(path=Path(tmp.name), index=index, start_ms=index * 1000, end_ms=(index + 1) * 1000)


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

    @patch("meetings.services.audio_transcription.split_audio_with_overlap")
    def test_single_chunk_path_skips_progress_fields(self, mock_split):
        spec = _make_chunk_spec(0)
        mock_split.return_value = [spec]

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

    @patch("meetings.services.audio_transcription.split_audio_with_overlap")
    def test_multi_chunk_happy_path_progress_and_carryover(self, mock_split):
        specs = [_make_chunk_spec(i) for i in range(3)]
        mock_split.return_value = specs

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

    @patch("meetings.services.audio_transcription.split_audio_with_overlap")
    def test_chunk_failure_persists_partial_and_resets_progress(self, mock_split):
        specs = [_make_chunk_spec(i) for i in range(3)]
        mock_split.return_value = specs

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

    @patch("meetings.services.audio_transcription.split_audio_with_overlap")
    def test_single_chunk_appends_to_existing_transcript(self, mock_split):
        """Re-uploading a short (single-chunk) file to a meeting that already
        has a transcript must append the new text, never replace it."""
        self.meeting.transcript = "PRIOR TRANSCRIPT FROM FILE ONE — do not lose."
        self.meeting.save(update_fields=["transcript"])

        spec = _make_chunk_spec(0)
        mock_split.return_value = [spec]
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

    @patch("meetings.services.audio_transcription.split_audio_with_overlap")
    def test_multi_chunk_appends_to_existing_transcript(self, mock_split):
        """Re-uploading a multi-chunk file must preserve the prior transcript
        AND stitch all new chunks — not just the last chunk."""
        self.meeting.transcript = (
            "PRIOR TRANSCRIPT HEADER\n\n"
            "This is file one's full content with meaningful text inside."
        )
        self.meeting.save(update_fields=["transcript"])

        specs = [_make_chunk_spec(i) for i in range(3)]
        mock_split.return_value = specs

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

    @patch("meetings.services.audio_transcription.split_audio_with_overlap")
    def test_multi_chunk_append_failure_preserves_existing(self, mock_split):
        """If a chunk fails mid-upload, the existing transcript must still be
        present (alongside any partial new content)."""
        self.meeting.transcript = "KEEP THIS PRIOR TRANSCRIPT SAFE"
        self.meeting.save(update_fields=["transcript"])

        specs = [_make_chunk_spec(i) for i in range(3)]
        mock_split.return_value = specs

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
    @patch("meetings.services.audio_transcription.split_audio_with_overlap")
    def test_transient_error_is_retried(self, mock_split, mock_sleep):
        try:
            from openai import RateLimitError
        except Exception:  # pragma: no cover
            self.skipTest("openai library missing")

        specs = [_make_chunk_spec(0), _make_chunk_spec(1)]
        mock_split.return_value = specs

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

    @patch("meetings.services.audio_transcription.split_audio_with_overlap")
    def test_split_duration_is_logged(self, mock_split):
        """The orchestrator logs the split duration at INFO level."""
        spec = _make_chunk_spec(0)
        mock_split.return_value = [spec]

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

        split_log = [line for line in cm.output if "audio split completed" in line]
        self.assertEqual(len(split_log), 1)
        self.assertIn("1 chunks", split_log[0])

    @patch("meetings.services.audio_transcription.AUDIO_SPLIT_TIMEOUT_SECONDS", 0.1)
    @patch("meetings.services.audio_transcription.split_audio_with_overlap")
    def test_split_timeout_raises_error(self, mock_split):
        """When the split hangs beyond the timeout, AudioSplitTimeoutError is raised."""
        import threading

        def hang_forever(*args, **kwargs):
            threading.Event().wait(timeout=10)
            return []

        mock_split.side_effect = hang_forever

        with self.assertRaises(AudioSplitTimeoutError) as ctx:
            orchestrate_upload_transcription(
                meeting_id=self.meeting.pk,
                temp_path=Path("/fake.mp3"),
                model_id="openai/gpt-4o-mini-transcribe",
                user_id=self.user.pk,
                service=MagicMock(),
            )

        self.assertIn("timed out", str(ctx.exception))

    @patch("meetings.services.audio_transcription.AUDIO_SPLIT_TIMEOUT_SECONDS", 0.1)
    @patch("meetings.services.audio_transcription.split_audio_with_overlap")
    def test_split_timeout_logs_error(self, mock_split):
        """The orchestrator logs an ERROR when the split times out."""
        import threading

        def hang_forever(*args, **kwargs):
            threading.Event().wait(timeout=10)
            return []

        mock_split.side_effect = hang_forever

        with self.assertLogs("meetings.services.audio_transcription", level="ERROR") as cm:
            with self.assertRaises(AudioSplitTimeoutError):
                orchestrate_upload_transcription(
                    meeting_id=self.meeting.pk,
                    temp_path=Path("/fake.mp3"),
                    model_id="openai/gpt-4o-mini-transcribe",
                    user_id=self.user.pk,
                    service=MagicMock(),
                )

        timeout_log = [line for line in cm.output if "audio split timed out" in line]
        self.assertEqual(len(timeout_log), 1)
