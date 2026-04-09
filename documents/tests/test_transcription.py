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
            result = service.transcribe(Path(f.name), "openai/gpt-4o-mini-transcribe")

        self.assertEqual(result.text, "Hello, this is a test.")
        self.assertAlmostEqual(result.audio_duration_seconds, 15.5)
        self.assertEqual(result.segments, 1)
        self.assertIsNotNone(result.cost_usd)

        # Verify json format used (gpt-4o-* transcribe models reject verbose_json).
        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        self.assertEqual(call_kwargs["response_format"], "json")

        # Verify logging called
        mock_log.assert_called_once()
        log_kwargs = mock_log.call_args[1]
        self.assertEqual(log_kwargs["model"], "openai/gpt-4o-mini-transcribe")
        self.assertAlmostEqual(log_kwargs["audio_duration_seconds"], 15.5)

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
            result = service.transcribe(Path(f.name), "openai/gpt-4o-mini-transcribe")

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
                service.transcribe(Path(f.name), "openai/gpt-4o-mini-transcribe")

        self.assertFalse(Path(seg1.name).exists())
        self.assertFalse(Path(seg2.name).exists())
        mock_log_err.assert_called_once()

    def test_transcribe_file_not_found(self):
        service = TranscriptionService()
        with self.assertRaises(FileNotFoundError):
            service.transcribe(Path("/nonexistent/file.mp3"), "openai/gpt-4o-mini-transcribe")

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
                service.transcribe(Path(f.name), "openai/gpt-4o-mini-transcribe")
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
            service.transcribe(Path(f.name), "openai/gpt-4o-mini-transcribe")

        mock_split.assert_not_called()


class TranscribeAudioWrapperTests(TestCase):
    """Tests for the convenience wrapper in documents.services.transcription."""

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
            result = transcribe_audio(Path(f.name), "openai/gpt-4o-mini-transcribe")

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
            transcribe_audio(Path(f.name), "openai/gpt-4o-mini-transcribe", user=user)

        # Check the context passed to log has the user_id
        log_kwargs = mock_log.call_args[1]
        self.assertEqual(log_kwargs["context"].user_id, str(user.pk))


class TranscriptionPromptTests(TestCase):
    """Tests for the optional ``prompt`` parameter on the transcription stack."""

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
            transcribe_audio(Path(f.name), "openai/gpt-4o-mini-transcribe")

        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        self.assertNotIn("prompt", call_kwargs)

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
                "openai/gpt-4o-mini-transcribe",
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
                "openai/gpt-4o-mini-transcribe",
                prompt="some very long prompt",
            )

        self.assertEqual(text, "recovered")
        # Two API calls: one with prompt, one without.
        self.assertEqual(mock_client.audio.transcriptions.create.call_count, 2)
        first_kwargs = mock_client.audio.transcriptions.create.call_args_list[0][1]
        second_kwargs = mock_client.audio.transcriptions.create.call_args_list[1][1]
        self.assertEqual(first_kwargs.get("prompt"), "some very long prompt")
        self.assertNotIn("prompt", second_kwargs)


class SplitAudioFileTests(TestCase):
    """Tests for the _split_audio_file helper."""

    def test_pydub_not_installed(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"\x00" * 100)
            f.flush()
            with patch.dict("sys.modules", {"pydub": None}):
                with self.assertRaises(RuntimeError) as ctx:
                    _split_audio_file(Path(f.name), 50)
                self.assertIn("pydub is required", str(ctx.exception))


class TranscriptionCostTests(TestCase):
    """Tests for transcription cost calculation."""

    def test_calculate_transcription_cost(self):
        from llm.service.pricing import calculate_transcription_cost

        # gpt-4o-mini-transcribe: $0.03/min
        cost = calculate_transcription_cost("openai/gpt-4o-mini-transcribe", 120.0)
        self.assertIsNotNone(cost)
        # 2 minutes * $0.03 = $0.06
        self.assertAlmostEqual(float(cost), 0.06, places=4)

    def test_calculate_transcription_cost_unknown_model(self):
        from llm.service.pricing import calculate_transcription_cost

        cost = calculate_transcription_cost("nonexistent/model", 60.0)
        self.assertIsNone(cost)
