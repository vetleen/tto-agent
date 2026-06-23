"""Tests for the image-generation registry, cost calc, and service.

The Gemini SDK is never called for real here — ``_call_gemini`` is patched (or
the pure response-parsing helpers are exercised with fake response objects), so
these run without ``google-genai`` installed or any network access.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

from django.test import TestCase

from llm.image_generation_registry import (
    ImageGenModelInfo,
    get_image_generation_model_info,
    get_image_generation_models,
)
from llm.service.image_generation_service import (
    ImageGenerationError,
    ImageGenerationService,
    InputImage,
    _extract_image,
    _extract_usage,
    _no_image_reason,
)
from llm.service.pricing import calculate_image_generation_cost
from llm.types.context import RunContext


class RegistryTests(TestCase):
    def test_known_model(self):
        info = get_image_generation_model_info("gemini/gemini-2.5-flash-image")
        self.assertIsNotNone(info)
        self.assertEqual(info.provider, "google_genai")
        self.assertEqual(info.api_model, "gemini-2.5-flash-image")
        self.assertTrue(info.supports_editing)

    def test_unknown_model(self):
        self.assertIsNone(get_image_generation_model_info("nope/nope"))

    def test_registry_listing(self):
        self.assertIn("gemini/gemini-2.5-flash-image", get_image_generation_models())


class CostTests(TestCase):
    def test_per_image_price(self):
        cost = calculate_image_generation_cost("gemini/gemini-2.5-flash-image")
        self.assertEqual(cost, Decimal("0.039"))

    def test_per_image_multiple(self):
        cost = calculate_image_generation_cost("gemini/gemini-2.5-flash-image", n_images=3)
        self.assertEqual(cost, Decimal("0.117"))

    def test_unknown_model_none(self):
        self.assertIsNone(calculate_image_generation_cost("nope/nope"))

    def test_token_based_path(self):
        fake = ImageGenModelInfo(
            display_name="Token Model",
            provider="openai",
            api_model="gpt-image-2",
            price_per_image=None,
            input_price_per_1m_tokens=Decimal("5.00"),
            output_image_price_per_1m_tokens=Decimal("30.00"),
        )
        with patch(
            "llm.image_generation_registry.get_image_generation_model_info",
            return_value=fake,
        ):
            cost = calculate_image_generation_cost(
                "openai/gpt-image-2", input_tokens=1_000_000, output_tokens=1_000_000
            )
        self.assertEqual(cost, Decimal("35.00"))


class ResponseHelperTests(TestCase):
    def test_extract_image(self):
        part = SimpleNamespace(inline_data=SimpleNamespace(data=b"abc", mime_type="image/png"))
        response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])
        data, mime = _extract_image(response)
        self.assertEqual(data, b"abc")
        self.assertEqual(mime, "image/png")

    def test_extract_image_none_when_only_text(self):
        part = SimpleNamespace(inline_data=None, text="hello")
        response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])
        data, mime = _extract_image(response)
        self.assertIsNone(data)

    def test_no_image_reason_safety(self):
        response = SimpleNamespace(
            prompt_feedback=SimpleNamespace(block_reason="SAFETY"), candidates=[]
        )
        self.assertIn("safety", _no_image_reason(response).lower())

    def test_extract_usage(self):
        usage = SimpleNamespace(
            prompt_token_count=10, candidates_token_count=1290, total_token_count=1300
        )
        self.assertEqual(_extract_usage(usage), (10, 1290, 1300))

    def test_extract_usage_none(self):
        self.assertEqual(_extract_usage(None), (None, None, None))


class ServiceTests(TestCase):
    def _usage(self):
        return SimpleNamespace(
            prompt_token_count=10, candidates_token_count=1290, total_token_count=1300
        )

    def test_generate_success_logs_call(self):
        from llm.models import LLMCallLog

        svc = ImageGenerationService()
        ctx = RunContext.create()
        with patch.object(
            ImageGenerationService,
            "_call_gemini",
            return_value=(b"PNGBYTES", "image/png", self._usage()),
        ):
            result = svc.generate("a cat", "gemini/gemini-2.5-flash-image", context=ctx)

        self.assertEqual(result.media_type, "image/png")
        self.assertEqual(result.cost_usd, Decimal("0.039"))
        self.assertFalse(result.is_edit)
        log = LLMCallLog.objects.filter(run_id=ctx.run_id).first()
        self.assertIsNotNone(log)
        self.assertEqual(log.status, LLMCallLog.Status.SUCCESS)
        self.assertEqual(log.cost_usd, Decimal("0.039"))

    def test_generate_edit_flag(self):
        svc = ImageGenerationService()
        with patch.object(
            ImageGenerationService,
            "_call_gemini",
            return_value=(b"PNGBYTES", "image/png", self._usage()),
        ):
            result = svc.generate(
                "make it blue",
                "gemini/gemini-2.5-flash-image",
                input_images=[InputImage(data=b"x", mime_type="image/png")],
            )
        self.assertTrue(result.is_edit)

    def test_generate_safety_block_logs_error(self):
        from llm.models import LLMCallLog

        svc = ImageGenerationService()
        ctx = RunContext.create()
        with patch.object(
            ImageGenerationService,
            "_call_gemini",
            side_effect=ImageGenerationError("blocked by safety"),
        ):
            with self.assertRaises(ImageGenerationError):
                svc.generate("bad prompt", "gemini/gemini-2.5-flash-image", context=ctx)
        log = LLMCallLog.objects.filter(run_id=ctx.run_id).first()
        self.assertIsNotNone(log)
        self.assertEqual(log.status, LLMCallLog.Status.ERROR)

    def test_unknown_model_raises(self):
        svc = ImageGenerationService()
        with self.assertRaises(ValueError):
            svc.generate("x", "nope/nope")
