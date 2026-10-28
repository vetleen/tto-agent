"""Unit tests for transcription service and convenience wrapper."""

import tempfile
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings

from documents.services.transcription import transcribe_audio
from llm.service.transcription_service import (
    TranscriptionService,
    _split_audio_file,
    get_transcription_service,
)
from llm.transcription_registry import TranscriptionModelInfo


def _mock_response(text="Hello transcript."):
    """Create a mock OpenAI `json` transcription response (just `.text`)."""
    resp = MagicMock()
    resp.text = text
    return resp


class TranscriptionServiceTests(TestCase):
    """Tests for llm.service.transcription_service.TranscriptionService."""

    def setUp(self):
        super().setUp()
        p = patch("llm.service.transcription_service._ffmpeg_available", return_value=False)
        p.start()
        self.addCleanup(p.stop)

    @patch("llm.service.transcription_service._get_audio_duration_seconds", return_value=15.5)
    @patch("llm.service.transcription_service.log_transcription")
    @patch("openai.OpenAI")
    def test_transcribe_single_file(self, mock_openai_cls, mock_log, mock_duration):
        """Small file transcribed directly; LLMCallLog entry written."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.transcriptions.create.return_value = _mock_response(
            "Hello, this is a test.",
        )

        service = TranscriptionService()
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"\x00" * 1000)
            f.flush()
            result = service.transcribe(Path(f.name), "openai/gpt-transcribe")

        self.assertEqual(result.text, "Hello, this is a test.")
        self.assertAlmostEqual(result.audio_duration_seconds, 15.5)
        self.assertEqual(result.segments, 1)
        self.assertIsNotNone(result.cost_usd)

        # Verify json format used (only whisper-1 ever accepted verbose_json).
        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        self.assertEqual(call_kwargs["response_format"], "json")

        # Verify logging called
        mock_log.assert_called_once()
        log_kwargs = mock_log.call_args[1]
        self.assertEqual(log_kwargs["model"], "openai/gpt-transcribe")
        self.assertAlmostEqual(log_kwargs["audio_duration_seconds"], 15.5)

    @patch("llm.service.transcription_service._get_audio_duration_seconds", return_value=5.0)
    @patch("llm.service.transcription_service.log_transcription")
    @patch("openai.OpenAI")
    def test_retired_and_live_ids_use_the_file_model(self, mock_openai_cls, mock_log, mock_duration):
        """A retired ID (or the live model) handed to the file path calls and
        logs the file model that serves it."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.transcriptions.create.return_value = _mock_response("ok")
        service = TranscriptionService()
        for model_id in ("openai/gpt-4o-transcribe-diarize", "openai/gpt-live-transcribe"):
            with self.subTest(model=model_id):
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    f.write(b"\x00" * 1000)
                    f.flush()
                    result = service.transcribe(Path(f.name), model_id)
                self.assertEqual(result.model, "openai/gpt-transcribe")
                self.assertEqual(
                    mock_client.audio.transcriptions.create.call_args.kwargs["model"], "gpt-transcribe",
                )
                self.assertEqual(mock_log.call_args.kwargs["model"], "openai/gpt-transcribe")

    @override_settings(AUDIO_UPLOAD_MAX_SIZE_BYTES=500)
    @patch("llm.service.transcription_service._get_audio_duration_seconds", side_effect=[300.0, 280.0])
    @patch("llm.service.transcription_service.log_transcription")
    @patch("openai.OpenAI")
    @patch("llm.service.transcription_service._split_audio_file")
    @patch("llm.service.transcription_service.get_transcription_model_info")
    def test_transcribe_splits_large_file(self, mock_get_info, mock_split, mock_openai_cls, mock_log, mock_duration):
        """Files exceeding API limit are split, transcribed per-segment, and joined."""
        mock_get_info.return_value = TranscriptionModelInfo(
            display_name="Test", provider="openai", api_model="test-model",
            price_per_minute=Decimal("0.06"), max_file_size_bytes=100,
            supports_file_transcription=True,
        )

        seg1 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        seg1.write(b"\x00" * 40)
        seg1.close()
        seg2 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        seg2.write(b"\x00" * 40)
        seg2.close()
        mock_split.return_value = [Path(seg1.name), Path(seg2.name)]

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.transcriptions.create.side_effect = [
            _mock_response("First segment."),
            _mock_response("Second segment."),
        ]

        service = TranscriptionService()
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"\x00" * 150)
            f.flush()
            result = service.transcribe(Path(f.name), "openai/gpt-transcribe")

        self.assertEqual(result.text, "First segment. Second segment.")
        self.assertEqual(result.segments, 2)
        self.assertAlmostEqual(result.audio_duration_seconds, 580.0)
        self.assertEqual(mock_client.audio.transcriptions.create.call_count, 2)
        # One log entry per segment
        self.assertEqual(mock_log.call_count, 2)
        # Temp files cleaned up
        self.assertFalse(Path(seg1.name).exists())
        self.assertFalse(Path(seg2.name).exists())

    @override_settings(AUDIO_UPLOAD_MAX_SIZE_BYTES=500)
    @patch("llm.service.transcription_service.log_transcription_error")
    @patch("llm.service.transcription_service.log_transcription")
    @patch("openai.OpenAI")
    @patch("llm.service.transcription_service._split_audio_file")
    @patch("llm.service.transcription_service.get_transcription_model_info")
    def test_transcribe_cleans_up_on_api_error(self, mock_get_info, mock_split, mock_openai_cls, mock_log, mock_log_err):
        """Temp segment files cleaned up and error logged when API call fails."""
        mock_get_info.return_value = TranscriptionModelInfo(
            display_name="Test", provider="openai", api_model="test-model",
            price_per_minute=Decimal("0.06"), max_file_size_bytes=100,
            supports_file_transcription=True,
        )

        seg1 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        seg1.write(b"\x00" * 40)
        seg1.close()
        seg2 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        seg2.write(b"\x00" * 40)
        seg2.close()
        mock_split.return_value = [Path(seg1.name), Path(seg2.name)]

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.transcriptions.create.side_effect = [
            _mock_response("First."),
            RuntimeError("API error"),
        ]

        service = TranscriptionService()
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"\x00" * 150)
            f.flush()
            from llm.service.errors import LLMProviderError
            with self.assertRaises(LLMProviderError):
                service.transcribe(Path(f.name), "openai/gpt-transcribe")

        self.assertFalse(Path(seg1.name).exists())
        self.assertFalse(Path(seg2.name).exists())
        mock_log_err.assert_called_once()

    def test_transcribe_file_not_found(self):
        service = TranscriptionService()
        with self.assertRaises(FileNotFoundError):
            service.transcribe(Path("/nonexistent/file.mp3"), "openai/gpt-transcribe")

    def test_transcribe_unknown_model(self):
        service = TranscriptionService()
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"\x00" * 100)
            f.flush()
            with self.assertRaises(ValueError) as ctx:
                service.transcribe(Path(f.name), "nonexistent/model")
            self.assertIn("Unknown transcription model", str(ctx.exception))

    @override_settings(AUDIO_UPLOAD_MAX_SIZE_BYTES=100)
    def test_transcribe_file_too_large(self):
        service = TranscriptionService()
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"\x00" * 200)
            f.flush()
            with self.assertRaises(ValueError) as ctx:
                service.transcribe(Path(f.name), "openai/gpt-transcribe")
            self.assertIn("too large", str(ctx.exception))

    @patch("openai.OpenAI")
    @patch("llm.service.transcription_service._split_audio_file")
    def test_no_split_under_api_limit(self, mock_split, mock_openai_cls):
        """Files under the API limit are transcribed directly without splitting."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.transcriptions.create.return_value = _mock_response()

        service = TranscriptionService()
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"\x00" * 1000)
            f.flush()
            service.transcribe(Path(f.name), "openai/gpt-transcribe")

        mock_split.assert_not_called()

    @patch("llm.service.transcription_service._ffmpeg_available", return_value=True)
    @patch("llm.service._audio_subprocess.ffprobe_duration_ms", return_value=None)
    def test_transcribe_rejects_corrupted_audio(self, mock_probe, mock_ffmpeg):
        """Corrupted files (ffprobe returns None) are rejected before hitting the API."""
        service = TranscriptionService()
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(b"\x00" * 100)
            f.flush()
            with self.assertRaises(ValueError) as ctx:
                service.transcribe(Path(f.name), "openai/gpt-transcribe")
            self.assertIn("corrupted", str(ctx.exception))


class TranscribeAudioWrapperTests(TestCase):
    """Tests for the convenience wrapper in documents.services.transcription."""

    def setUp(self):
        super().setUp()
        p = patch("llm.service.transcription_service._ffmpeg_available", return_value=False)
        p.start()
        self.addCleanup(p.stop)

    @patch("llm.service.transcription_service.log_transcription")
    @patch("openai.OpenAI")
    def test_wrapper_returns_text(self, mock_openai_cls, mock_log):
        """transcribe_audio() returns just the text string."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.transcriptions.create.return_value = _mock_response("Wrapper test.")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"\x00" * 1000)
            f.flush()
            result = transcribe_audio(Path(f.name), "openai/gpt-transcribe")

        self.assertEqual(result, "Wrapper test.")

    @patch("llm.service.transcription_service.log_transcription")
    @patch("openai.OpenAI")
    def test_wrapper_passes_user(self, mock_openai_cls, mock_log):
        """User is passed through to RunContext for attribution."""
        from django.contrib.auth import get_user_model
        User = get_user_model()
        user = User.objects.create_user(email="tx@example.com", password="testpass")

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.transcriptions.create.return_value = _mock_response()

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"\x00" * 1000)
            f.flush()
            transcribe_audio(Path(f.name), "openai/gpt-transcribe", user=user)

        # Check the context passed to log has the user_id
        log_kwargs = mock_log.call_args[1]
        self.assertEqual(log_kwargs["context"].user_id, str(user.pk))


class TranscriptionPromptTests(TestCase):
    """Tests for the optional ``prompt`` parameter on the transcription stack."""

    def setUp(self):
        super().setUp()
        p = patch("llm.service.transcription_service._ffmpeg_available", return_value=False)
        p.start()
        self.addCleanup(p.stop)

    @patch("llm.service.transcription_service.log_transcription")
    @patch("openai.OpenAI")
    def test_prompt_omitted_when_none(self, mock_openai_cls, mock_log):
        """prompt=None (default) does NOT include 'prompt' in the API call kwargs."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.transcriptions.create.return_value = _mock_response("hi")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"\x00" * 1000)
            f.flush()
            transcribe_audio(Path(f.name), "openai/gpt-transcribe")

        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        self.assertNotIn("prompt", call_kwargs)

    @patch("llm.service.transcription_service.log_transcription")
    @patch("openai.OpenAI")
    def test_language_sent_as_languages_list_never_singular(self, mock_openai_cls, mock_log):
        """gpt-transcribe takes ``languages`` (a list); sending the singular
        ``language`` alongside it is rejected, so it is never sent."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.transcriptions.create.return_value = _mock_response("hei")

        for language, expected in (("no", ["no"]), (None, None)):
            with self.subTest(language=language):
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    f.write(b"\x00" * 1000)
                    f.flush()
                    transcribe_audio(Path(f.name), "openai/gpt-transcribe", language=language)
                call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
                self.assertNotIn("language", call_kwargs)
                self.assertEqual(call_kwargs.get("languages"), expected)
                self.assertEqual(call_kwargs["response_format"], "json")
                self.assertEqual(call_kwargs["chunking_strategy"], "auto")

    @patch("llm.service.transcription_service._get_audio_duration_seconds", return_value=0.0)
    @patch("llm.service.transcription_service.log_transcription")
    @patch("openai.OpenAI")
    def test_duration_usage_drives_cost(self, mock_openai_cls, mock_log, mock_duration):
        """The billed seconds in a duration usage price the call even when our
        own ffprobe duration is unavailable (0.0)."""
        class _Usage:
            type = "duration"
            seconds = 120.0

        class _Resp:
            text = "billed by the minute"
            usage = _Usage()

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.transcriptions.create.return_value = _Resp()

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"\x00" * 1000)
            f.flush()
            result = TranscriptionService().transcribe(Path(f.name), "openai/gpt-transcribe")
        self.assertAlmostEqual(float(result.cost_usd), 0.009, places=6)

    def test_billed_seconds_extraction(self):
        from llm.service.transcription_service import _extract_billed_seconds

        class _Usage:
            type = "duration"
            seconds = 9

        class _Resp:
            usage = _Usage()

        self.assertEqual(_extract_billed_seconds(_Resp()), 9.0)
        self.assertEqual(
            _extract_billed_seconds(MagicMock(usage={"type": "duration", "seconds": 4})), 4.0,
        )
        self.assertIsNone(_extract_billed_seconds(MagicMock()))
        self.assertIsNone(_extract_billed_seconds(MagicMock(usage=None)))

    @patch("llm.service.transcription_service.log_transcription")
    @patch("openai.OpenAI")
    def test_prompt_forwarded_when_set(self, mock_openai_cls, mock_log):
        """A non-empty prompt is forwarded to the API call kwargs."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.transcriptions.create.return_value = _mock_response("hi")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"\x00" * 1000)
            f.flush()
            transcribe_audio(
                Path(f.name),
                "openai/gpt-transcribe",
                prompt="OncoBio Therapeutics",
            )

        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        self.assertEqual(call_kwargs.get("prompt"), "OncoBio Therapeutics")

    @patch("llm.service.transcription_service.log_transcription")
    @patch("openai.OpenAI")
    def test_prompt_bad_request_falls_back_without_prompt(self, mock_openai_cls, mock_log):
        """If the API rejects the prompt, retry once with prompt stripped."""
        from openai import BadRequestError

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # First call: BadRequestError mentioning the prompt; second call: success.
        bad_request = BadRequestError(
            message="Invalid 'prompt': value too long",
            response=MagicMock(),
            body=None,
        )
        mock_client.audio.transcriptions.create.side_effect = [
            bad_request,
            _mock_response("recovered"),
        ]

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"\x00" * 1000)
            f.flush()
            text = transcribe_audio(
                Path(f.name),
                "openai/gpt-transcribe",
                prompt="some very long prompt",
            )

        self.assertEqual(text, "recovered")
        # Two API calls: one with prompt, one without.
        self.assertEqual(mock_client.audio.transcriptions.create.call_count, 2)
        first_kwargs = mock_client.audio.transcriptions.create.call_args_list[0][1]
        second_kwargs = mock_client.audio.transcriptions.create.call_args_list[1][1]
        self.assertEqual(first_kwargs.get("prompt"), "some very long prompt")
        self.assertNotIn("prompt", second_kwargs)

    @patch("llm.service.transcription_service.log_transcription")
    @patch("openai.OpenAI")
    def test_keywords_forwarded_and_dropped_on_bad_request(self, mock_openai_cls, mock_log):
        """Keywords go out as a list; if the API rejects them, retry once
        without them (keeping the prompt)."""
        from openai import BadRequestError

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.transcriptions.create.side_effect = [
            BadRequestError(message="Invalid 'keywords'", response=MagicMock(), body=None),
            _mock_response("recovered"),
        ]
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"\x00" * 1000)
            f.flush()
            text = transcribe_audio(
                Path(f.name), "openai/gpt-transcribe", prompt="ctx",
                keywords=["OncoBio Therapeutics", "Vetle"],
            )
        self.assertEqual(text, "recovered")
        first, second = (c.kwargs for c in mock_client.audio.transcriptions.create.call_args_list)
        self.assertEqual(first["keywords"], ["OncoBio Therapeutics", "Vetle"])
        self.assertNotIn("keywords", second)
        self.assertEqual(second["prompt"], "ctx")


class SplitAudioFileTests(TestCase):
    """Tests for the _split_audio_file helper."""

    def test_ffmpeg_not_available(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"\x00" * 100)
            f.flush()
            with patch(
                "llm.service._audio_subprocess.ffmpeg_available",
                return_value=False,
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    _split_audio_file(Path(f.name), 50)
                self.assertIn("ffmpeg", str(ctx.exception).lower())


class TranscriptionCostTests(TestCase):
    """Tests for transcription cost calculation."""

    def test_per_minute_pricing_for_both_models(self):
        from llm.service.pricing import calculate_transcription_cost

        # gpt-transcribe: $0.0045/min; gpt-live-transcribe: $0.017/min.
        self.assertAlmostEqual(float(calculate_transcription_cost("openai/gpt-transcribe", 120.0)), 0.009, places=6)
        self.assertAlmostEqual(float(calculate_transcription_cost("openai/gpt-live-transcribe", 60.0)), 0.017, places=6)

    def test_retired_id_is_priced_as_its_successor(self):
        from llm.service.pricing import calculate_transcription_cost

        self.assertEqual(
            calculate_transcription_cost("openai/gpt-4o-mini-transcribe", 120.0),
            calculate_transcription_cost("openai/gpt-transcribe", 120.0),
        )

    def test_calculate_transcription_cost_unknown_model(self):
        from llm.service.pricing import calculate_transcription_cost

        cost = calculate_transcription_cost("nonexistent/model", 60.0)
        self.assertIsNone(cost)

    def test_tokens_without_token_rates_use_per_minute(self):
        from llm.service.pricing import calculate_transcription_cost

        cost = calculate_transcription_cost(
            "openai/gpt-transcribe", audio_duration_seconds=120.0, input_tokens=10_000, output_tokens=2_000,
        )
        self.assertAlmostEqual(float(cost), 0.009, places=6)

    def test_token_rates_when_a_model_has_them(self):
        """A model with token rates (none today) bills per token."""
        from llm.service.pricing import calculate_transcription_cost

        info = TranscriptionModelInfo(
            display_name="T", provider="openai", api_model="t", price_per_minute=Decimal("0.006"),
            input_price_per_1m_tokens=Decimal("2.50"), output_price_per_1m_tokens=Decimal("10.00"),
            supports_file_transcription=True,
        )
        with patch("llm.transcription_registry.get_transcription_model_info", return_value=info):
            cost = calculate_transcription_cost("t", 60.0, input_tokens=10_000, output_tokens=2_000)
            partial = calculate_transcription_cost("t", 120.0, input_tokens=10_000)
        self.assertAlmostEqual(float(cost), 0.045, places=5)
        self.assertAlmostEqual(float(partial), 0.012, places=5)


class TranscriptionUsageExtractionTests(TestCase):
    """Tests for _extract_transcription_usage and the full single-file path.

    These use the in-repo response-usage reader directly with hand-rolled
    stand-ins so the test doesn't depend on pulling in real OpenAI SDK
    BaseModel classes. The stand-ins mimic the SDK shape exactly.
    """

    def setUp(self):
        super().setUp()
        p = patch("llm.service.transcription_service._ffmpeg_available", return_value=False)
        p.start()
        self.addCleanup(p.stop)

    def test_extract_usage_tokens_variant(self):
        from llm.service.transcription_service import _extract_transcription_usage

        class _InputDetails:
            audio_tokens = 400
            text_tokens = 50

        class _Usage:
            type = "tokens"
            input_tokens = 450
            output_tokens = 120
            total_tokens = 570
            input_token_details = _InputDetails()

        class _Resp:
            text = "ok"
            usage = _Usage()

        inp, out, tot, aud = _extract_transcription_usage(_Resp())
        self.assertEqual(inp, 450)
        self.assertEqual(out, 120)
        self.assertEqual(tot, 570)
        self.assertEqual(aud, 400)

    def test_extract_usage_duration_variant_returns_none(self):
        """UsageDuration responses (whisper-1) should yield all-None so the
        caller falls back to per-minute billing."""
        from llm.service.transcription_service import _extract_transcription_usage

        class _Usage:
            type = "duration"
            seconds = 30.0

        class _Resp:
            text = "ok"
            usage = _Usage()

        self.assertEqual(
            _extract_transcription_usage(_Resp()),
            (None, None, None, None),
        )

    def test_extract_usage_missing(self):
        from llm.service.transcription_service import _extract_transcription_usage

        class _Resp:
            text = "ok"
            usage = None

        self.assertEqual(
            _extract_transcription_usage(_Resp()),
            (None, None, None, None),
        )

    def test_extract_usage_magicmock_response_falls_back(self):
        """A plain MagicMock (as in existing tests) auto-vivifies attributes
        but they aren't real strings/ints — must fall back to all-None."""
        from llm.service.transcription_service import _extract_transcription_usage

        resp = MagicMock()
        resp.text = "ok"
        self.assertEqual(
            _extract_transcription_usage(resp),
            (None, None, None, None),
        )

    @patch("llm.service.transcription_service._get_audio_duration_seconds", return_value=60.0)
    @patch("llm.service.transcription_service.log_transcription")
    @patch("openai.OpenAI")
    def test_transcribe_single_forwards_token_usage_but_bills_per_minute(
        self, mock_openai_cls, mock_log, mock_duration,
    ):
        """End-to-end: token counts in a response are logged; the current
        models have no token rates, so cost stays per minute."""
        class _InputDetails:
            audio_tokens = 800
            text_tokens = 200

        class _Usage:
            type = "tokens"
            input_tokens = 1000
            output_tokens = 250
            total_tokens = 1250
            input_token_details = _InputDetails()

        class _Resp:
            text = "billed by token"
            usage = _Usage()

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.transcriptions.create.return_value = _Resp()

        service = TranscriptionService()
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"\x00" * 1000)
            f.flush()
            result = service.transcribe(Path(f.name), "openai/gpt-transcribe")

        self.assertEqual(result.text, "billed by token")
        # 1 minute * $0.0045
        self.assertAlmostEqual(float(result.cost_usd), 0.0045, places=6)

        log_kwargs = mock_log.call_args.kwargs
        self.assertEqual(log_kwargs["input_tokens"], 1000)
        self.assertEqual(log_kwargs["output_tokens"], 250)
        self.assertEqual(log_kwargs["total_tokens"], 1250)
        self.assertEqual(log_kwargs["audio_tokens"], 800)


class SharedOpenAIClientTests(TestCase):
    """Tests for the process-wide cached OpenAI client (_get_openai_client)."""

    def test_client_cached_per_class_and_rebuilt_when_class_changes(self):
        from llm.service import transcription_service as ts

        with patch("openai.OpenAI") as first_cls:
            c1 = ts._get_openai_client()
            c2 = ts._get_openai_client()
            # Same class -> one construction, same cached instance.
            self.assertIs(c1, c2)
            self.assertIs(c1, first_cls.return_value)
            first_cls.assert_called_once()
            # Explicit timeouts + retries are pinned at construction: the SDK
            # default 600s read timeout would let one wedged request burn most
            # of a Celery task's budget.
            kwargs = first_cls.call_args.kwargs
            self.assertEqual(kwargs["max_retries"], 2)
            self.assertEqual(kwargs["timeout"].read, 300.0)
            self.assertEqual(kwargs["timeout"].connect, 10.0)

        with patch("openai.OpenAI") as second_cls:
            # A different class object (a fresh test patch) must NOT be served
            # the stale cached client — it gets one built from the new class.
            c3 = ts._get_openai_client()
            self.assertIs(c3, second_cls.return_value)
            self.assertIsNot(c3, c1)
